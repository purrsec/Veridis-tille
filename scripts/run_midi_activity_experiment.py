#!/usr/bin/env python3
"""Evaluate MIDI window activity filters on every POC query/reference pair."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import numpy as np

from veridis_tille.midi_similarity import (
    NoteEvent,
    SimilarityConfig,
    build_window_vectors,
    compare_events,
    load_midi_events,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "dataset"
OUTPUT_DIR = ROOT / "output" / "research"
RESULTS_PATH = OUTPUT_DIR / "midi-activity-experiment.csv"
REPORT_PATH = ROOT / "docs" / "experiments" / "midi-activity-experiment.md"
STEMS = ("bass", "piano", "vocals", "other")
VARIANT_SUFFIX = re.compile(r"-(?:original|AI-[A-Za-z0-9]+)$")
FIELDS = (
    "experiment",
    "minimum_window_onsets",
    "activity_target_onsets",
    "activity_weight",
    "query",
    "reference",
    "expected_pair",
    "feature",
    "query_eligible",
    "score",
    "is_similar",
    "match_count",
)


@dataclass(frozen=True)
class Experiment:
    name: str
    minimum_window_onsets: int = 0
    activity_target_onsets: int = 0
    activity_weight: float = 0.0

    def config(self) -> SimilarityConfig:
        return SimilarityConfig(
            minimum_window_onsets=self.minimum_window_onsets,
            activity_target_onsets=self.activity_target_onsets,
            activity_weight=self.activity_weight,
        )


EXPERIMENTS = (
    Experiment("baseline"),
    Experiment("gate-3", minimum_window_onsets=3),
    Experiment("gate-5", minimum_window_onsets=5),
    Experiment("gate-8", minimum_window_onsets=8),
    Experiment("density-light", minimum_window_onsets=3, activity_weight=0.5),
    Experiment(
        "confidence-8-light",
        minimum_window_onsets=3,
        activity_target_onsets=8,
        activity_weight=0.5,
    ),
    Experiment(
        "confidence-12-light",
        minimum_window_onsets=3,
        activity_target_onsets=12,
        activity_weight=0.5,
    ),
    Experiment(
        "confidence-12-strong",
        minimum_window_onsets=3,
        activity_target_onsets=12,
        activity_weight=1.0,
    ),
)


def track_key(path: Path) -> str:
    return VARIANT_SUFFIX.sub("", path.name)


def directories() -> tuple[list[Path], list[Path]]:
    paths = sorted(path for path in DATASET_DIR.iterdir() if path.is_dir())
    queries = [path for path in paths if "-AI-" in path.name]
    references = [path for path in paths if path.name.endswith("-original")]
    return queries, references


def load_cache(
    paths: list[Path],
) -> dict[tuple[str, str], tuple[list[NoteEvent], float]]:
    cache: dict[tuple[str, str], tuple[list[NoteEvent], float]] = {}
    for path in paths:
        for stem in STEMS:
            events = load_midi_events(path / f"{stem}_basic_pitch.mid")
            duration = max((event.end for event in events), default=0.0)
            cache[(path.name, stem)] = (events, duration)
    return cache


def maximum_window_onsets(events: list[NoteEvent], duration: float) -> int:
    windows = build_window_vectors(
        events,
        duration=duration,
        window_seconds=10.0,
        hop_seconds=2.0,
        frame_seconds=0.5,
    )
    counts = np.count_nonzero(windows.values[:, :, 1, :], axis=(1, 2))
    return int(counts.max(initial=0))


def compare_all() -> list[dict[str, object]]:
    queries, references = directories()
    cache = load_cache([*queries, *references])
    maximum_activity = {
        key: maximum_window_onsets(*value)
        for key, value in cache.items()
        if "-AI-" in key[0]
    }
    rows: list[dict[str, object]] = []

    for experiment_index, experiment in enumerate(EXPERIMENTS, start=1):
        print(
            f"[{experiment_index}/{len(EXPERIMENTS)}] {experiment.name}",
            flush=True,
        )
        config = experiment.config()
        for query in queries:
            for reference in references:
                for stem in STEMS:
                    query_events, query_duration = cache[(query.name, stem)]
                    reference_events, reference_duration = cache[(reference.name, stem)]
                    result = compare_events(
                        query_events,
                        reference_events,
                        query_duration=query_duration,
                        reference_duration=reference_duration,
                        config=config,
                    )
                    rows.append(
                        {
                            "experiment": experiment.name,
                            "minimum_window_onsets": experiment.minimum_window_onsets,
                            "activity_target_onsets": experiment.activity_target_onsets,
                            "activity_weight": experiment.activity_weight,
                            "query": query.name,
                            "reference": reference.name,
                            "expected_pair": str(
                                track_key(query) == track_key(reference)
                            ).lower(),
                            "feature": stem,
                            "query_eligible": str(
                                maximum_activity[(query.name, stem)]
                                >= experiment.minimum_window_onsets
                            ).lower(),
                            "score": f"{result.score:.9f}",
                            "is_similar": str(result.is_similar).lower(),
                            "match_count": len(result.matches),
                        }
                    )
    return rows


def flag(row: dict[str, object], name: str) -> bool:
    return row[name] == "true"


def score(row: dict[str, object]) -> float:
    return float(row["score"])


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def write_results(rows: list[dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, object]]) -> None:
    table_rows: list[tuple[object, ...]] = []
    for experiment in EXPERIMENTS:
        for stem in STEMS:
            selected = [
                row
                for row in rows
                if row["experiment"] == experiment.name
                and row["feature"] == stem
                and flag(row, "query_eligible")
            ]
            queries = sorted({str(row["query"]) for row in selected})
            positives = [row for row in selected if flag(row, "expected_pair")]
            negatives = [row for row in selected if not flag(row, "expected_pair")]
            rank_one = 0
            margins: list[float] = []
            for query in queries:
                query_rows = [row for row in selected if row["query"] == query]
                positive = next(
                    row for row in query_rows if flag(row, "expected_pair")
                )
                strongest_negative = max(
                    score(row)
                    for row in query_rows
                    if not flag(row, "expected_pair")
                )
                margin = score(positive) - strongest_negative
                margins.append(margin)
                rank_one += margin > 1e-9

            table_rows.append(
                (
                    experiment.name,
                    stem,
                    f"{len(queries)}/8",
                    f"{rank_one}/{len(queries)}" if queries else "0/0",
                    f"{mean(margins):+.3f}" if margins else "n/a",
                    f"{sum(flag(row, 'is_similar') for row in positives)}/{len(positives)}",
                    f"{sum(flag(row, 'is_similar') for row in negatives)}/{len(negatives)}",
                    f"{median(map(score, positives)):.3f}" if positives else "n/a",
                    f"{median(map(score, negatives)):.3f}" if negatives else "n/a",
                )
            )

    content = f"""# MIDI activity confidence experiment

## Scope

This experiment tests whether MIDI comparisons improve when windows with too
few note onsets are rejected or down-weighted. It uses the same 8 AI excerpts,
8 originals, four MIDI stems, window size, tempo ratios, and match thresholds
as the vectorization baseline.

`Coverage` counts AI queries with at least one eligible 10-second window.
Ranking and detection statistics exclude uncovered queries instead of treating
missing musical evidence as a negative match.

## Results

{markdown_table(
    (
        "Experiment",
        "Stem",
        "Coverage",
        "Ranks first",
        "Mean margin",
        "Expected detected",
        "Other detected",
        "Expected median",
        "Other median",
    ),
    table_rows,
)}

## Conclusion

The activity hypothesis is useful as an evidence check, but not as a general
similarity multiplier:

- onset gates remove the sparse-piano false positives by withholding a result,
  yet the covered piano queries still almost never rank the expected original
  first;
- activity weighting reduces bass rank-one results from 5/8 to 4/8, even when
  its mean rank margin increases slightly;
- the light 12-onset confidence improves the vocal mean margin from `+0.026`
  to `+0.031` while preserving 5/8 rank-one results, which is too small a gain
  on eight examples to justify changing the default;
- every weighted variant degrades the `other` stem.

The new parameters therefore remain disabled by default. A minimum activity
gate can later support an explicit `insufficient evidence` state, especially
for piano, but it should not silently turn missing notes into a negative match.
The next MIDI experiment should add melodic interval or contour information to
the promising bass and vocal representations.

The complete results are stored in
`output/research/midi-activity-experiment.csv`.
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> int:
    rows = compare_all()
    write_results(rows)
    write_report(rows)
    print(f"Wrote {len(rows)} comparisons to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote report to {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
