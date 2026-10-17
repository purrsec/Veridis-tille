"""Command-line interface for the Veridis-tille prototype."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .drum_similarity import DrumSimilarityConfig, compare_drum_audio
from .midi_similarity import SimilarityConfig, compare_midi


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

    drums = commands.add_parser(
        "compare-drums",
        help="compare isolated drum WAVs without using pitch",
    )
    drums.add_argument("query")
    drums.add_argument("reference")
    drums.add_argument("--window-seconds", type=float, default=10.0)
    drums.add_argument("--hop-seconds", type=float, default=2.0)
    drums.add_argument("--frame-seconds", type=float, default=0.1)
    drums.add_argument("--cell-threshold", type=float, default=0.78)
    drums.add_argument("--match-threshold", type=float, default=0.82)
    drums.add_argument("--minimum-aligned-windows", type=int, default=2)
    drums.add_argument("--minimum-match-seconds", type=float, default=10.0)
    drums.add_argument("--max-matches", type=int, default=5)
    drums.add_argument(
        "--tempo-ratios",
        type=float,
        nargs="+",
        default=[0.8, 0.9, 1.0, 1.125, 1.2, 1.25],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
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
    if args.command == "compare-drums":
        config = DrumSimilarityConfig(
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
        result = compare_drum_audio(args.query, args.reference, config=config)
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
