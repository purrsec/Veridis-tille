#!/usr/bin/env python3
"""Compare every AI excerpt with every original across the five POC stems."""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, median

from veridis_tille.drum_similarity import (
    DrumComparisonResult,
    RhythmEvent,
    compare_rhythm_events,
    load_wav_onsets,
)
from veridis_tille.midi_similarity import (
    ComparisonResult,
    NoteEvent,
    compare_events,
    load_midi_events,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "dataset"
OUTPUT_DIR = ROOT / "output" / "research"
RESULTS_PATH = OUTPUT_DIR / "similarity-baseline.csv"
REPORT_PATH = ROOT / "docs" / "experiments" / "vectorization-baseline.md"
MIDI_STEMS = ("bass", "piano", "vocals", "other")
FEATURES = (*MIDI_STEMS, "drums")
VARIANT_SUFFIX = re.compile(r"-(?:original|AI-[A-Za-z0-9]+)$")
FIELDS = (
    "query",
    "reference",
    "expected_pair",
    "feature",
    "score",
    "is_similar",
    "match_count",
    "query_start",
    "query_end",
    "reference_start",
    "reference_end",
    "tempo_ratio",
    "transposition",
    "aligned_windows",
    "tempo_similarity",
    "rhythm_similarity",
)


def track_key(path: Path) -> str:
    return VARIANT_SUFFIX.sub("", path.name)


def analysis_directories() -> tuple[list[Path], list[Path]]:
    directories = sorted(path for path in DATASET_DIR.iterdir() if path.is_dir())
    queries = [path for path in directories if "-AI-" in path.name]
    references = [path for path in directories if path.name.endswith("-original")]
    if not queries or not references:
        raise RuntimeError("the dataset must contain AI and original analysis directories")
    return queries, references


def load_midi_cache(
    directories: Iterable[Path],
) -> dict[tuple[str, str], tuple[list[NoteEvent], float]]:
    cache: dict[tuple[str, str], tuple[list[NoteEvent], float]] = {}
    for directory in directories:
        for stem in MIDI_STEMS:
            events = load_midi_events(directory / f"{stem}_basic_pitch.mid")
            duration = max((event.end for event in events), default=0.0)
            cache[(directory.name, stem)] = (events, duration)
    return cache


def load_drum_cache(
    directories: Iterable[Path],
) -> dict[str, tuple[list[RhythmEvent], float]]:
    return {
        directory.name: load_wav_onsets(directory / "drums.wav")
        for directory in directories
    }


def result_row(
    *,
    query: Path,
    reference: Path,
    feature: str,
    result: ComparisonResult | DrumComparisonResult,
) -> dict[str, object]:
    best = result.matches[0] if result.matches else None
    return {
        "query": query.name,
        "reference": reference.name,
        "expected_pair": str(track_key(query) == track_key(reference)).lower(),
        "feature": feature,
        "score": f"{result.score:.9f}",
        "is_similar": str(result.is_similar).lower(),
        "match_count": len(result.matches),
        "query_start": "" if best is None else f"{best.query_start:.6f}",
        "query_end": "" if best is None else f"{best.query_end:.6f}",
        "reference_start": "" if best is None else f"{best.reference_start:.6f}",
        "reference_end": "" if best is None else f"{best.reference_end:.6f}",
        "tempo_ratio": "" if best is None else f"{best.tempo_ratio:.6f}",
        "transposition": "" if best is None or feature == "drums" else best.transposition,
        "aligned_windows": "" if best is None else best.aligned_windows,
        "tempo_similarity": (
            "" if best is None or feature != "drums" else f"{best.tempo_similarity:.9f}"
        ),
        "rhythm_similarity": (
            "" if best is None or feature != "drums" else f"{best.rhythm_similarity:.9f}"
        ),
    }


def compare_all() -> list[dict[str, object]]:
    queries, references = analysis_directories()
    directories = [*queries, *references]
    midi_cache = load_midi_cache(directories)
    drum_cache = load_drum_cache(directories)
    rows: list[dict[str, object]] = []

    for index, query in enumerate(queries, start=1):
        print(f"[{index}/{len(queries)}] Comparing {query.name}", flush=True)
        for reference in references:
            for stem in MIDI_STEMS:
                query_events, query_duration = midi_cache[(query.name, stem)]
                reference_events, reference_duration = midi_cache[(reference.name, stem)]
                result = compare_events(
                    query_events,
                    reference_events,
                    query_duration=query_duration,
                    reference_duration=reference_duration,
                )
                rows.append(
                    result_row(
                        query=query,
                        reference=reference,
                        feature=stem,
                        result=result,
                    )
                )

            query_events, query_duration = drum_cache[query.name]
            reference_events, reference_duration = drum_cache[reference.name]
            drum_result = compare_rhythm_events(
                query_events,
                reference_events,
                query_duration=query_duration,
                reference_duration=reference_duration,
            )
            rows.append(
                result_row(
                    query=query,
                    reference=reference,
                    feature="drums",
                    result=drum_result,
                )
            )

    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv() -> list[dict[str, object]]:
    with RESULTS_PATH.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def score(row: dict[str, object]) -> float:
    return float(row["score"])


def flag(row: dict[str, object], name: str) -> bool:
    return row[name] == "true"


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(rows: list[dict[str, object]]) -> None:
    query_names = sorted({str(row["query"]) for row in rows})
    summary_rows: list[tuple[object, ...]] = []
    for feature in FEATURES:
        feature_rows = [row for row in rows if row["feature"] == feature]
        positives = [row for row in feature_rows if flag(row, "expected_pair")]
        negatives = [row for row in feature_rows if not flag(row, "expected_pair")]
        rank_one = 0
        margins: list[float] = []
        for query in query_names:
            query_rows = [row for row in feature_rows if row["query"] == query]
            positive = next(row for row in query_rows if flag(row, "expected_pair"))
            best_negative = max(
                score(row) for row in query_rows if not flag(row, "expected_pair")
            )
            margin = score(positive) - best_negative
            margins.append(margin)
            rank_one += margin > 1e-9

        summary_rows.append(
            (
                feature,
                f"{median(map(score, positives)):.3f}",
                f"{median(map(score, negatives)):.3f}",
                f"{sum(flag(row, 'is_similar') for row in positives)}/{len(positives)}",
                f"{sum(flag(row, 'is_similar') for row in negatives)}/{len(negatives)}",
                f"{rank_one}/{len(query_names)}",
                f"{mean(margins):+.3f}",
            )
        )

    bass_rows = [row for row in rows if row["feature"] == "bass"]
    piano_rows = [row for row in rows if row["feature"] == "piano"]
    vocal_rows = [row for row in rows if row["feature"] == "vocals"]
    drum_rows = [row for row in rows if row["feature"] == "drums"]
    expected_drums = [row for row in drum_rows if flag(row, "expected_pair")]
    other_drums = [row for row in drum_rows if not flag(row, "expected_pair")]
    matched_expected_drums = [row for row in expected_drums if row["tempo_similarity"]]
    matched_other_drums = [row for row in other_drums if row["tempo_similarity"]]
    drum_expected_tempo = mean(
        float(row["tempo_similarity"]) for row in matched_expected_drums
    )
    drum_other_tempo = mean(
        float(row["tempo_similarity"]) for row in matched_other_drums
    )
    drum_expected_rhythm = mean(
        float(row["rhythm_similarity"]) for row in matched_expected_drums
    )
    drum_other_rhythm = mean(
        float(row["rhythm_similarity"]) for row in matched_other_drums
    )
    bass_false_positives = sum(
        flag(row, "is_similar") and not flag(row, "expected_pair")
        for row in bass_rows
    )
    piano_false_positives = sum(
        flag(row, "is_similar") and not flag(row, "expected_pair")
        for row in piano_rows
    )
    vocal_expected_rank_one = 0
    for query in query_names:
        query_rows = [row for row in vocal_rows if row["query"] == query]
        positive = next(row for row in query_rows if flag(row, "expected_pair"))
        vocal_expected_rank_one += score(positive) > max(
            score(row) for row in query_rows if not flag(row, "expected_pair")
        ) + 1e-9

    positive_rows: list[tuple[object, ...]] = []
    for query in query_names:
        query_rows = {
            str(row["feature"]): row
            for row in rows
            if row["query"] == query and flag(row, "expected_pair")
        }
        positive_rows.append(
            (
                track_key(Path(query)),
                *(f"{score(query_rows[feature]):.3f}" for feature in FEATURES),
            )
        )

    content = f"""# Vectorization baseline on the POC dataset

## Scope

This baseline compares each of the 8 AI excerpts against each of the 8 full
original recordings. It produces 64 pairs per feature: 8 expected pairs and
56 different-song pairs. All comparisons use the unchanged default parameters
from the passage-similarity prototypes.

The MIDI vectors are evaluated separately for bass, piano, vocals, and other.
Drums use the pitch-free onset and tempo representation. No combined score is
computed at this stage.

## Summary

{markdown_table(
    (
        "Feature",
        "Expected median",
        "Other median",
        "Expected detected",
        "Other detected",
        "Expected ranks first",
        "Mean rank margin",
    ),
    summary_rows,
)}

`Expected detected` and `Other detected` apply the current prototype thresholds.
`Expected ranks first` checks whether the correct original receives the highest
score for each AI query. The mean rank margin is the expected-pair score minus
the strongest different-song score.

## Initial reading

- Bass is the strongest current MIDI representation. It ranks the expected
  original first for 5 of 8 excerpts, with {bass_false_positives} false positives
  among the 56 different-song pairs.
- Vocals do not cross the current decision threshold, but still rank the
  expected original first for {vocal_expected_rank_one} of 8 excerpts. This points
  to useful ordering information with an unsuitable absolute threshold.
- Piano produces {piano_false_positives} false positives. The worst cases come from
  extremely sparse AI piano transcriptions: after normalization and free
  transposition, a window containing only one or two notes can match many songs
  almost perfectly.
- Drums produce {sum(flag(row, "is_similar") for row in other_drums)} false positives
  among 56 different-song pairs. Their mean tempo similarity is
  {drum_expected_tempo:.3f} for expected pairs and {drum_other_tempo:.3f} for other
  pairs; mean rhythm similarity is {drum_expected_rhythm:.3f} and
  {drum_other_rhythm:.3f}, respectively, among threshold-matched passages. The
  current drum vector therefore carries almost no discriminating information on
  this corpus.

These results should be treated as feature diagnostics rather than an estimate
of final detector accuracy. The corpus contains only eight expected pairs, and
the parameters were initially chosen on synthetic examples.

## Expected-pair scores

{markdown_table(("Track", *FEATURES), positive_rows)}

The complete pair-level results, including best passage locations, tempo ratio,
transposition, and drum sub-scores, are stored in
`output/research/similarity-baseline.csv`.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="regenerate the Markdown report from the existing result CSV",
    )
    args = parser.parse_args()

    if args.report_only:
        rows = read_csv()
    else:
        rows = compare_all()
        write_csv(rows)
    write_report(rows)
    if not args.report_only:
        print(f"Wrote {len(rows)} comparisons to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote report to {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
