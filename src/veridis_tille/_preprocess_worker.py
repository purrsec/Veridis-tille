# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "basic-pitch==0.4.0",
#   "setuptools<81",
#   "spleeter==2.4.2",
# ]
# ///
"""Build flat Spleeter and Basic Pitch outputs beside each dataset WAV."""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
_checkout_cache = ROOT / "sandbox" / "spleeter-models"
MODEL_DIR = (_checkout_cache if _checkout_cache.is_dir() else
             Path(os.environ.get("XDG_CACHE_HOME", str(Path.home()/".cache"))) / "veridis-tille" / "spleeter")
SOURCES: list[Path] = []
STEMS = ("vocals", "drums", "bass", "piano", "other")


def source_files() -> list[Path]:
    return SOURCES


def output_directory(source: Path) -> Path:
    return source.with_suffix("")


def stem_files(source: Path) -> list[Path]:
    directory = output_directory(source)
    return [directory / f"{stem}.wav" for stem in STEMS]


def basic_pitch_files(stem: Path) -> tuple[Path, Path]:
    prefix = stem.with_suffix("")
    return (
        prefix.with_name(f"{prefix.name}_basic_pitch.mid"),
        prefix.with_name(f"{prefix.name}_basic_pitch.csv"),
    )


def run_spleeter(*, force: bool) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("MODEL_PATH", str(MODEL_DIR))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    pending = [
        source
        for source in source_files()
        if force
        or not all(
            path.is_file() and path.stat().st_size > 0
            for path in stem_files(source)
        )
    ]
    if not pending:
        print("Spleeter: all stems already exist.", flush=True)
        return

    import soundfile as sf
    from spleeter.audio import Codec
    from spleeter.separator import Separator

    print(f"Spleeter: separating {len(pending)} source files.", flush=True)
    separator = Separator("spleeter:5stems", multiprocess=False)
    for index, source in enumerate(pending, start=1):
        directory = output_directory(source)
        directory.mkdir(exist_ok=True)
        print(f"[{index}/{len(pending)}] Spleeter: {source.name}", flush=True)
        separator.separate_to_file(
            str(source),
            str(directory),
            codec=Codec.WAV,
            filename_format="{instrument}.{codec}",
            synchronous=True,
            duration=sf.info(str(source)).duration,
        )

    del separator
    gc.collect()


def run_basic_pitch(*, force: bool) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict, save_note_events

    pending: list[Path] = []
    for source in source_files():
        for stem in stem_files(source):
            if not stem.is_file():
                raise FileNotFoundError(f"missing Spleeter stem: {stem}")
            midi, notes = basic_pitch_files(stem)
            if force or not (
                midi.is_file()
                and midi.stat().st_size > 0
                and notes.is_file()
                and notes.stat().st_size > 0
            ):
                pending.append(stem)

    if not pending:
        print("Basic Pitch: all MIDI and note CSV files already exist.", flush=True)
        return

    print(f"Basic Pitch: transcribing {len(pending)} stems.", flush=True)
    model = Model(ICASSP_2022_MODEL_PATH)
    for index, stem in enumerate(pending, start=1):
        midi_path, notes_path = basic_pitch_files(stem)
        frame_threshold = 0.35 if stem.stem == "bass" else 0.30
        print(
            f"[{index}/{len(pending)}] Basic Pitch: {stem.parent.name}/{stem.name}",
            flush=True,
        )
        _, midi_data, note_events = predict(
            stem,
            model,
            frame_threshold=frame_threshold,
        )
        midi_data.write(str(midi_path))
        save_note_events(note_events, notes_path)


def verify_outputs() -> None:
    errors: list[str] = []
    sources = source_files()
    for source in sources:
        directory = output_directory(source)
        expected = set(stem_files(source))
        for stem in stem_files(source):
            expected.update(basic_pitch_files(stem))
        missing = [
            path.name
            for path in sorted(expected)
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            errors.append(f"{directory.name}: missing {', '.join(missing)}")
        if directory.is_dir():
            unexpected = [
                path.name for path in directory.iterdir() if path not in expected
            ]
            if unexpected:
                errors.append(f"{directory.name}: unexpected {', '.join(sorted(unexpected))}")

    if errors:
        raise RuntimeError("\n".join(errors))
    print(f"Complete POC dataset: {len(sources)} sources, {len(sources) * 5} stems.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument(
        "--force", action="store_true", help="regenerate existing outputs"
    )
    parser.add_argument(
        "--stage",
        choices=("spleeter", "basic-pitch"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    SOURCES[:] = [path.resolve() for path in args.sources]
    if args.stage == "spleeter":
        run_spleeter(force=args.force)
        return 0
    if args.stage == "basic-pitch":
        run_basic_pitch(force=args.force)
        return 0

    common = [sys.executable, str(Path(__file__).resolve()), *map(str, SOURCES)]
    force = ["--force"] if args.force else []
    subprocess.run([*common, "--stage", "spleeter", *force], check=True)
    subprocess.run([*common, "--stage", "basic-pitch", *force], check=True)
    verify_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
