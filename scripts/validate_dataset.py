#!/usr/bin/env python3
"""Validate sanitized audio files and pair split CSVs."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
SPLIT_NAMES = ("train", "validation", "test")
CSV_FIELDS = ["track_1", "track_2", "is_plagiarism"]
FILENAME_PATTERN = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*-(?:original|AI-[A-Za-z0-9]+)\.wav$"
)


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


def validate_audio(errors: list[str]) -> list[Path]:
    if not DATASET_DIR.is_dir():
        errors.append(f"missing dataset directory: {DATASET_DIR}")
        return []

    entries = sorted(DATASET_DIR.iterdir())
    audio_files: list[Path] = []
    hashes: dict[str, Path] = {}

    for path in entries:
        if not path.is_file():
            errors.append(f"dataset must be flat; found directory: {path.name}")
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


def validate_splits(audio_files: list[Path], errors: list[str]) -> None:
    filenames = {path.name for path in audio_files}
    track_splits: dict[str, str] = {}
    pairs: set[tuple[str, str]] = set()

    for split in SPLIT_NAMES:
        csv_path = DATA_DIR / f"{split}.csv"
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


def main() -> int:
    errors: list[str] = []
    audio_files = validate_audio(errors)
    validate_splits(audio_files, errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"Validation failed with {len(errors)} error(s).", file=sys.stderr)
        return 1

    print(
        f"Validated {len(audio_files)} audio files and "
        f"{len(SPLIT_NAMES)} split CSV files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
