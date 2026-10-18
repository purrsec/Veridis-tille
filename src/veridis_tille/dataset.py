"""Prepare the flat POC dataset and validate its audio, analysis and pair splits."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from .preprocessing import ALL_STEMS, prepare_audio

SPLIT_NAMES = ("train", "validation", "test")
CSV_FIELDS = ["track_1", "track_2", "is_plagiarism"]
FILENAME_PATTERN = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*-(?:original|AI-[A-Za-z0-9]+)\.wav$"
)
NOTE_EVENT_FIELDS = [
    "start_time_s",
    "end_time_s",
    "pitch_midi",
    "velocity",
    "pitch_bend",
]


def decoded_sha256(path: Path) -> str:
    """Decode the complete audio stream and hash its canonical PCM samples."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-c:a",
        "pcm_s16le",
        "-f",
        "hash",
        "-hash",
        "sha256",
        "-",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is required but was not found") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"cannot fully decode {path.name}: {message}") from error

    algorithm, separator, digest = result.stdout.strip().partition("=")
    if algorithm != "SHA256" or separator != "=" or len(digest) != 64:
        raise RuntimeError(f"cannot hash decoded audio for {path.name}")
    return digest


def probe(path: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,sample_rate,channels:format=duration:format_tags",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffprobe is required but was not found") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"cannot decode {path.name}: {message}") from error
    return json.loads(result.stdout)


def validate_audio(dataset_dir: Path, errors: list[str]) -> list[Path]:
    if not dataset_dir.is_dir():
        errors.append(f"missing dataset directory: {dataset_dir}")
        return []

    entries = sorted(dataset_dir.iterdir())
    audio_files: list[Path] = []
    hashes: dict[str, Path] = {}

    for path in entries:
        if path.is_dir():
            continue
        if not FILENAME_PATTERN.fullmatch(path.name):
            errors.append(f"invalid dataset filename: {path.name}")
            continue

        audio_files.append(path)
        try:
            details = probe(path)
        except (RuntimeError, json.JSONDecodeError) as error:
            errors.append(str(error))
            continue

        streams = details.get("streams", [])
        if len(streams) != 1 or streams[0].get("codec_type") != "audio":
            errors.append(f"{path.name}: expected exactly one stream containing audio")
            continue

        stream = streams[0]
        expected = {
            "codec_name": "pcm_s16le",
            "sample_rate": "44100",
            "channels": 2,
        }
        for field, value in expected.items():
            if stream.get(field) != value:
                errors.append(
                    f"{path.name}: expected {field}={value}, got {stream.get(field)}"
                )

        try:
            duration = float(details["format"]["duration"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{path.name}: missing or invalid duration")
        else:
            if duration <= 0:
                errors.append(f"{path.name}: duration must be positive")

        tags = details.get("format", {}).get("tags", {})
        if tags:
            errors.append(f"{path.name}: embedded metadata must be removed")

        try:
            digest = decoded_sha256(path)
        except RuntimeError as error:
            errors.append(str(error))
            continue
        if digest in hashes:
            errors.append(
                f"duplicate decoded audio: {path.name} and {hashes[digest].name}"
            )
        else:
            hashes[digest] = path

    return audio_files


def validate_analysis_directories(
    dataset_dir: Path, audio_files: list[Path], errors: list[str],
) -> None:
    if not dataset_dir.is_dir():
        return
    source_by_stem = {path.stem: path for path in audio_files}
    directories = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    directory_names = {path.name for path in directories}

    for source_stem in sorted(source_by_stem.keys() - directory_names):
        errors.append(f"missing analysis directory for {source_stem}.wav")

    for directory in directories:
        if directory.name not in source_by_stem:
            errors.append(
                f"analysis directory has no matching source WAV: {directory.name}"
            )
            continue

        expected = {
            directory / f"{stem}.wav" for stem in ALL_STEMS
        } | {
            directory / f"{stem}_basic_pitch.{extension}"
            for stem in ALL_STEMS
            for extension in ("mid", "csv")
        }
        entries = set(directory.iterdir())
        for path in sorted(expected - entries):
            errors.append(f"{directory.name}: missing analysis file {path.name}")
        for path in sorted(entries - expected):
            errors.append(f"{directory.name}: unexpected analysis file {path.name}")
        for path in entries:
            if path.is_dir():
                errors.append(
                    f"{directory.name}: nested directory is not allowed: {path.name}"
                )
            elif path.stat().st_size == 0:
                errors.append(f"{directory.name}: empty analysis file {path.name}")

        for stem in ALL_STEMS:
            stem_path = directory / f"{stem}.wav"
            if not stem_path.is_file():
                continue
            try:
                details = probe(stem_path)
            except (RuntimeError, json.JSONDecodeError) as error:
                errors.append(str(error))
                continue
            streams = details.get("streams", [])
            if len(streams) != 1 or streams[0].get("codec_type") != "audio":
                errors.append(f"{directory.name}/{stem_path.name}: expected one audio stream")
                continue
            stream = streams[0]
            for field, value in {
                "codec_name": "pcm_s16le",
                "sample_rate": "44100",
                "channels": 2,
            }.items():
                if stream.get(field) != value:
                    errors.append(
                        f"{directory.name}/{stem_path.name}: expected {field}={value}, "
                        f"got {stream.get(field)}"
                    )

            midi_path = directory / f"{stem}_basic_pitch.mid"
            if midi_path.is_file():
                with midi_path.open("rb") as stream:
                    if stream.read(4) != b"MThd":
                        errors.append(
                            f"{directory.name}/{midi_path.name}: invalid MIDI header"
                        )

            notes_path = directory / f"{stem}_basic_pitch.csv"
            if notes_path.is_file():
                with notes_path.open(newline="", encoding="utf-8") as stream:
                    reader = csv.reader(stream)
                    if next(reader, None) != NOTE_EVENT_FIELDS:
                        errors.append(
                            f"{directory.name}/{notes_path.name}: invalid note CSV header"
                        )


def validate_splits(data_dir: Path, audio_files: list[Path], errors: list[str]) -> None:
    filenames = {path.name for path in audio_files}
    track_splits: dict[str, str] = {}
    pairs: set[tuple[str, str]] = set()

    for split in SPLIT_NAMES:
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.is_file():
            errors.append(f"missing split file: {csv_path.name}")
            continue

        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != CSV_FIELDS:
                errors.append(
                    f"{csv_path.name}: expected header {','.join(CSV_FIELDS)}"
                )
                continue

            for line_number, row in enumerate(reader, start=2):
                first = row["track_1"]
                second = row["track_2"]
                label = row["is_plagiarism"]

                if first == second:
                    errors.append(f"{csv_path.name}:{line_number}: self-pair")
                if label not in {"true", "false"}:
                    errors.append(
                        f"{csv_path.name}:{line_number}: invalid boolean {label!r}"
                    )

                for filename in (first, second):
                    if Path(filename).name != filename:
                        errors.append(
                            f"{csv_path.name}:{line_number}: use a filename, not a path"
                        )
                    elif filename not in filenames:
                        errors.append(
                            f"{csv_path.name}:{line_number}: missing audio {filename}"
                        )

                    previous_split = track_splits.setdefault(filename, split)
                    if previous_split != split:
                        errors.append(
                            f"{filename} occurs in both {previous_split} and {split}"
                        )

                pair = tuple(sorted((first, second)))
                if pair in pairs:
                    errors.append(
                        f"{csv_path.name}:{line_number}: duplicate pair {pair[0]}, {pair[1]}"
                    )
                else:
                    pairs.add(pair)


def prepare_dataset(data_dir: str | Path = "data", *, force: bool = False) -> None:
    """Prepare all masters in data_dir/dataset without modifying their audio."""
    dataset_dir = Path(data_dir) / "dataset"
    sources = sorted(dataset_dir.glob("*.wav"))
    if not sources:
        raise FileNotFoundError(f"no WAV sources in {dataset_dir}")
    prepare_audio(sources, force=force, complete=True)
    print(f"Complete POC dataset: {len(sources)} sources.")


def validate_dataset(data_dir: str | Path = "data") -> int:
    """Validate one data directory; return a shell exit status and report errors."""
    data_dir = Path(data_dir)
    dataset_dir = data_dir / "dataset"
    errors: list[str] = []
    audio_files = validate_audio(dataset_dir, errors)
    validate_analysis_directories(dataset_dir, audio_files, errors)
    validate_splits(data_dir, audio_files, errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"Validation failed with {len(errors)} error(s).", file=sys.stderr)
        return 1

    print(
        f"Validated {len(audio_files)} source files, "
        f"{len(audio_files) * len(ALL_STEMS)} stems, and "
        f"{len(SPLIT_NAMES)} split CSV files."
    )
    return 0
