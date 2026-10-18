"""Command-line interface for the Veridis-tille prototype."""

from __future__ import annotations

import argparse
import json
import subprocess
import wave
from collections.abc import Sequence
from pathlib import Path

from .dataset import prepare_dataset, validate_dataset
from .midi_similarity import SimilarityConfig, compare_midi
from .pipeline import Calibration, compare_tracks
from .preprocessing import prepare_audio


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veridis-tille")
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare-midi", help="compare and locate similar MIDI passages")
    compare.add_argument("query")
    compare.add_argument("reference")
    compare.add_argument("--window-seconds", type=float, default=10.0)
    compare.add_argument("--hop-seconds", type=float, default=2.0)
    compare.add_argument("--frame-seconds", type=float, default=0.5)
    compare.add_argument("--cell-threshold", type=float, default=0.72)
    compare.add_argument("--match-threshold", type=float, default=0.78)
    compare.add_argument("--minimum-aligned-windows", type=int, default=2)
    compare.add_argument("--minimum-match-seconds", type=float, default=10.0)
    compare.add_argument("--max-matches", type=int, default=5)
    compare.add_argument(
        "--tempo-ratios",
        type=float,
        nargs="+",
        default=[0.8, 0.9, 1.0, 1.125, 1.2, 1.25],
        help="reference/query tempo ratios to try",
    )

    pair = commands.add_parser("compare", help="compare two WAV masters with the retained POC pipeline")
    pair.add_argument("query")
    pair.add_argument("reference")
    pair.add_argument("--calibration", help="JSON thresholds; default: bundled exploratory POC profile")
    pair.add_argument("--no-prepare", action="store_true", help="require existing companion MIDI files")
    prepare = commands.add_parser("prepare", help="generate flat Spleeter and Basic Pitch outputs")
    prepare.add_argument("sources", nargs="+")
    prepare.add_argument("--force", action="store_true")

    dataset = commands.add_parser("dataset", help="prepare and validate a POC dataset")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_prepare = dataset_commands.add_parser("prepare", help="prepare all masters in data/dataset")
    dataset_prepare.add_argument("data_dir", nargs="?", type=Path, default=Path("data"),
                                 help="directory containing dataset/ and split CSVs (default: ./data)")
    dataset_prepare.add_argument("--force", action="store_true")
    dataset_validate = dataset_commands.add_parser("validate", help="check audio, flat analysis and splits")
    dataset_validate.add_argument("data_dir", nargs="?", type=Path, default=Path("data"),
                                  help="directory containing dataset/ and split CSVs (default: ./data)")
    return parser


def _run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "compare-midi":
        config = SimilarityConfig(
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
            frame_seconds=args.frame_seconds,
            cell_threshold=args.cell_threshold,
            match_threshold=args.match_threshold,
            minimum_aligned_windows=args.minimum_aligned_windows,
            minimum_match_seconds=args.minimum_match_seconds,
            max_matches=args.max_matches,
            tempo_ratios=tuple(args.tempo_ratios),
        )
        result = compare_midi(args.query, args.reference, config=config)
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    if args.command == "compare":
        result = compare_tracks(args.query, args.reference,
                                calibration=Calibration.load(args.calibration),
                                prepare=not args.no_prepare)
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    if args.command == "prepare":
        prepare_audio(args.sources, force=args.force, complete=True)
        return 0
    if args.command == "dataset":
        if args.dataset_command == "prepare":
            prepare_dataset(args.data_dir, force=args.force)
            return 0
        return validate_dataset(args.data_dir)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(argv)
    except (OSError, ValueError, EOFError, wave.Error, subprocess.CalledProcessError) as error:
        _parser().exit(2, f"veridis-tille: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
