#!/usr/bin/env python3
"""Compare late score fusion with time-synchronized multi-stem fusion."""

from __future__ import annotations

import csv
import re
import wave
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from veridis_tille.midi_similarity import (
    NoteEvent,
    WindowVectors,
    build_window_vectors,
    compute_similarity_matrix,
    load_midi_events,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "dataset"
OUTPUT_DIR = ROOT / "output" / "research"
RESULTS_PATH = OUTPUT_DIR / "multistem-fusion-experiment.csv"
REPORT_PATH = ROOT / "docs" / "experiments" / "multistem-fusion-experiment.md"
STEMS = ("bass", "piano", "vocals", "other")
TEMPO_RATIOS = (0.8, 0.9, 1.0, 1.125, 1.2, 1.25)
VARIANT_SUFFIX = re.compile(r"-(?:original|AI-[A-Za-z0-9]+)$")
FusionMode = Literal["late_mean", "synchronized_mean", "synchronized_top2", "synchronized_second", "synchronized_geometric"]
FIELDS = (
    "strategy",
    "query",
    "reference",
    "expected_pair",
    "score",
    "query_start",
    "reference_start",
    "tempo_ratio",
    "bass_score",
    "piano_score",
    "vocals_score",
    "other_score",
)


@dataclass(frozen=True)
class Strategy:
    name: str
    mode: FusionMode
    stems: tuple[str, ...]


@dataclass(frozen=True)
class PairwiseMetrics:
    true_positives: int
    false_positives: int
    positive_evaluations: int
    negative_evaluations: int
    thresholds: tuple[float, ...]
    detections_by_family: dict[str, int]

    @property
    def precision(self) -> float:
        predicted_positives = self.true_positives + self.false_positives
        return self.true_positives / predicted_positives

    @property
    def recall(self) -> float:
        return self.true_positives / self.positive_evaluations

    @property
    def f1(self) -> float:
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def false_positive_rate(self) -> float:
        return self.false_positives / self.negative_evaluations


STRATEGIES = (
    Strategy("late-mean-all", "late_mean", STEMS),
    Strategy("late-mean-bass-vocals-other", "late_mean", ("bass", "vocals", "other")),
    Strategy("late-mean-bass-other", "late_mean", ("bass", "other")),
    Strategy("sync-mean-all", "synchronized_mean", STEMS),
    Strategy("sync-mean-bass-vocals-other", "synchronized_mean", ("bass", "vocals", "other")),
    Strategy("sync-mean-bass-other", "synchronized_mean", ("bass", "other")),
    Strategy("sync-top2-bass-vocals-other", "synchronized_top2", ("bass", "vocals", "other")),
    Strategy("sync-second-bass-vocals-other", "synchronized_second", ("bass", "vocals", "other")),
    Strategy("sync-geometric-bass-vocals-other", "synchronized_geometric", ("bass", "vocals", "other")),
)


def track_key(path: Path | str) -> str:
    name = path.name if isinstance(path, Path) else path
    return VARIANT_SUFFIX.sub("", name)


def directories() -> tuple[list[Path], list[Path]]:
    paths = sorted(path for path in DATASET_DIR.iterdir() if path.is_dir())
    return (
        [path for path in paths if "-AI-" in path.name],
        [path for path in paths if path.name.endswith("-original")],
    )


def audio_duration(directory: Path) -> float:
    with wave.open(str(DATASET_DIR / f"{directory.name}.wav"), "rb") as stream:
        return stream.getnframes() / stream.getframerate()


def load_event_cache(
    directories_: list[Path],
) -> dict[tuple[str, str], list[NoteEvent]]:
    return {
        (directory.name, stem): load_midi_events(
            directory / f"{stem}_basic_pitch.mid"
        )
        for directory in directories_
        for stem in STEMS
    }


def build_vector_cache(
    queries: list[Path],
    references: list[Path],
    events: dict[tuple[str, str], list[NoteEvent]],
) -> dict[tuple[str, str, float], WindowVectors]:
    cache: dict[tuple[str, str, float], WindowVectors] = {}
    for directory in [*queries, *references]:
        duration = audio_duration(directory)
        ratios = (1.0,) if directory in queries else TEMPO_RATIOS
        for stem in STEMS:
            for ratio in ratios:
                cache[(directory.name, stem, ratio)] = build_window_vectors(
                    events[(directory.name, stem)],
                    duration=duration,
                    window_seconds=10.0 * ratio,
                    hop_seconds=2.0 * ratio,
                    frame_seconds=0.5 * ratio,
                )
    return cache


def fuse_matrices(
    matrices: dict[str, NDArray[np.float32]],
    strategy: Strategy,
) -> NDArray[np.float32]:
    selected = np.stack([matrices[stem] for stem in strategy.stems])
    if strategy.mode == "synchronized_mean":
        return np.mean(selected, axis=0)
    if strategy.mode == "synchronized_top2":
        return np.mean(np.sort(selected, axis=0)[-2:], axis=0)
    if strategy.mode == "synchronized_second":
        return np.sort(selected, axis=0)[-2]
    if strategy.mode == "synchronized_geometric":
        return np.prod(np.maximum(selected, 1e-6), axis=0) ** (
            1.0 / len(strategy.stems)
        )
    raise ValueError(f"{strategy.name} is not a synchronized strategy")


def empty_best() -> dict[str, object]:
    return {
        "score": 0.0,
        "query_start": None,
        "reference_start": None,
        "tempo_ratio": None,
        "component_scores": {stem: 0.0 for stem in STEMS},
    }


def compare_pair(
    query: Path,
    reference: Path,
    vectors: dict[tuple[str, str, float], WindowVectors],
) -> dict[str, dict[str, object]]:
    stem_maxima = {stem: 0.0 for stem in STEMS}
    synchronized = {
        strategy.name: empty_best()
        for strategy in STRATEGIES
        if strategy.mode != "late_mean"
    }

    for ratio in TEMPO_RATIOS:
        matrices: dict[str, NDArray[np.float32]] = {}
        for stem in STEMS:
            matrix, _ = compute_similarity_matrix(
                vectors[(query.name, stem, 1.0)],
                vectors[(reference.name, stem, ratio)],
            )
            matrices[stem] = matrix
            stem_maxima[stem] = max(
                stem_maxima[stem], float(np.max(matrix, initial=0.0))
            )

        for strategy in STRATEGIES:
            if strategy.mode == "late_mean":
                continue
            fused = fuse_matrices(matrices, strategy)
            flat_index = int(np.argmax(fused))
            query_index, reference_index = np.unravel_index(flat_index, fused.shape)
            candidate_score = float(fused[query_index, reference_index])
            best = synchronized[strategy.name]
            if candidate_score <= float(best["score"]):
                continue
            best["score"] = candidate_score
            best["query_start"] = float(
                vectors[(query.name, STEMS[0], 1.0)].starts_seconds[query_index]
            )
            best["reference_start"] = float(
                vectors[
                    (reference.name, STEMS[0], ratio)
                ].starts_seconds[reference_index]
            )
            best["tempo_ratio"] = ratio
            best["component_scores"] = {
                stem: float(matrix[query_index, reference_index])
                for stem, matrix in matrices.items()
            }

    results = dict(synchronized)
    for strategy in STRATEGIES:
        if strategy.mode != "late_mean":
            continue
        results[strategy.name] = {
            "score": mean(stem_maxima[stem] for stem in strategy.stems),
            "query_start": None,
            "reference_start": None,
            "tempo_ratio": None,
            "component_scores": stem_maxima.copy(),
        }
    return results


def result_row(
    *,
    strategy: str,
    query: str,
    reference: str,
    result: dict[str, object],
) -> dict[str, object]:
    components = result["component_scores"]
    assert isinstance(components, dict)
    return {
        "strategy": strategy,
        "query": query,
        "reference": reference,
        "expected_pair": str(track_key(query) == track_key(reference)).lower(),
        "score": f"{float(result['score']):.9f}",
        "query_start": (
            ""
            if result["query_start"] is None
            else f"{float(result['query_start']):.6f}"
        ),
        "reference_start": (
            ""
            if result["reference_start"] is None
            else f"{float(result['reference_start']):.6f}"
        ),
        "tempo_ratio": (
            ""
            if result["tempo_ratio"] is None
            else f"{float(result['tempo_ratio']):.6f}"
        ),
        **{
            f"{stem}_score": f"{float(components[stem]):.9f}"
            for stem in STEMS
        },
    }


def compare_all() -> list[dict[str, object]]:
    queries, references = directories()
    events = load_event_cache([*queries, *references])
    vectors = build_vector_cache(queries, references, events)
    rows: list[dict[str, object]] = []
    for index, query in enumerate(queries, start=1):
        print(f"[{index}/{len(queries)}] query={query.name}", flush=True)
        for reference in references:
            results = compare_pair(query, reference, vectors)
            for strategy in STRATEGIES:
                rows.append(
                    result_row(
                        strategy=strategy.name,
                        query=query.name,
                        reference=reference.name,
                        result=results[strategy.name],
                    )
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


def strategy_metrics(
    rows: list[dict[str, object]], strategy: str
) -> tuple[int, float, float, float]:
    selected = [row for row in rows if row["strategy"] == strategy]
    queries = sorted({str(row["query"]) for row in selected})
    margins: list[float] = []
    for query in queries:
        query_rows = [row for row in selected if row["query"] == query]
        positive = next(row for row in query_rows if flag(row, "expected_pair"))
        best_negative = max(
            score(row) for row in query_rows if not flag(row, "expected_pair")
        )
        margins.append(score(positive) - best_negative)
    positives = [score(row) for row in selected if flag(row, "expected_pair")]
    negatives = [score(row) for row in selected if not flag(row, "expected_pair")]
    return (
        sum(margin > 1e-9 for margin in margins),
        mean(margins),
        median(positives),
        median(negatives),
    )


def query_rank(
    rows: list[dict[str, object]], strategy: str, query: str
) -> tuple[int, float, float, str]:
    selected = sorted(
        (
            row
            for row in rows
            if row["strategy"] == strategy and row["query"] == query
        ),
        key=score,
        reverse=True,
    )
    rank = next(
        index
        for index, row in enumerate(selected, start=1)
        if flag(row, "expected_pair")
    )
    positive = next(score(row) for row in selected if flag(row, "expected_pair"))
    best_negative_row = next(row for row in selected if not flag(row, "expected_pair"))
    return (
        rank,
        positive,
        positive - score(best_negative_row),
        str(best_negative_row["reference"]),
    )


def best_balanced_threshold(rows: list[dict[str, object]]) -> float:
    values = sorted({score(row) for row in rows})
    candidates = [values[0] - 1e-9, values[-1] + 1e-9]
    candidates.extend((left + right) / 2 for left, right in zip(values, values[1:]))
    positives = sum(flag(row, "expected_pair") for row in rows)
    negatives = len(rows) - positives

    def metric(threshold: float) -> tuple[float, int, float]:
        true_positives = sum(
            flag(row, "expected_pair") and score(row) >= threshold for row in rows
        )
        false_positives = sum(
            not flag(row, "expected_pair") and score(row) >= threshold for row in rows
        )
        sensitivity = true_positives / positives
        specificity = 1.0 - false_positives / negatives
        return (mean((sensitivity, specificity)), -false_positives, threshold)

    return max(candidates, key=metric)


def leave_two_families_out(
    rows: list[dict[str, object]], strategy: str
) -> PairwiseMetrics:
    selected = [row for row in rows if row["strategy"] == strategy]
    families = sorted({track_key(str(row["query"])) for row in selected})
    true_positives = 0
    false_positives = 0
    positive_evaluations = 0
    negative_evaluations = 0
    thresholds: list[float] = []
    detections_by_family = {family: 0 for family in families}
    for held_out in combinations(families, 2):
        training = [
            row
            for row in selected
            if track_key(str(row["query"])) not in held_out
            and track_key(str(row["reference"])) not in held_out
        ]
        testing = [
            row
            for row in selected
            if track_key(str(row["query"])) in held_out
            and track_key(str(row["reference"])) in held_out
        ]
        threshold = best_balanced_threshold(training)
        thresholds.append(threshold)
        for row in testing:
            prediction = score(row) >= threshold
            if flag(row, "expected_pair"):
                positive_evaluations += 1
                true_positives += prediction
                if prediction:
                    detections_by_family[track_key(str(row["query"]))] += 1
            else:
                negative_evaluations += 1
                false_positives += prediction
    return PairwiseMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        positive_evaluations=positive_evaluations,
        negative_evaluations=negative_evaluations,
        thresholds=tuple(thresholds),
        detections_by_family=detections_by_family,
    )


def write_results(rows: list[dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, object]]) -> None:
    pairwise_rows: list[tuple[object, ...]] = []
    pairwise_metrics: dict[str, PairwiseMetrics] = {}
    for strategy in STRATEGIES:
        metrics = leave_two_families_out(rows, strategy.name)
        pairwise_metrics[strategy.name] = metrics
        pairwise_rows.append(
            (
                strategy.name,
                f"{metrics.true_positives}/{metrics.positive_evaluations}",
                f"{metrics.false_positives}/{metrics.negative_evaluations}",
                f"{metrics.precision:.3f}",
                f"{metrics.recall:.3f}",
                f"{metrics.f1:.3f}",
                f"{metrics.detections_by_family['rasputin-boney-m']}/7",
                f"{min(metrics.thresholds):.3f}-{max(metrics.thresholds):.3f}",
            )
        )

    summary_rows: list[tuple[object, ...]] = []
    for strategy in STRATEGIES:
        ranks, margin, positive_median, negative_median = strategy_metrics(
            rows, strategy.name
        )
        summary_rows.append(
            (
                strategy.name,
                f"{ranks}/8",
                f"{margin:+.3f}",
                f"{positive_median:.3f}",
                f"{negative_median:.3f}",
            )
        )

    query_rows: list[tuple[object, ...]] = []
    query_names = sorted({str(row["query"]) for row in rows})
    for query in query_names:
        rank, positive, margin, strongest_negative = query_rank(
            rows, "late-mean-all", query
        )
        query_rows.append(
            (
                track_key(query),
                f"{rank}/8",
                f"{positive:.3f}",
                f"{margin:+.3f}",
                track_key(strongest_negative),
            )
        )

    rasputin_rows: list[tuple[object, ...]] = []
    for strategy in STRATEGIES:
        rank, positive, margin, _ = query_rank(
            rows, strategy.name, "rasputin-boney-m-AI-GEMA"
        )
        rasputin_rows.append(
            (strategy.name, f"{rank}/8", f"{positive:.3f}", f"{margin:+.3f}")
        )

    content = f"""# Multi-stem fusion experiment

## Scope

This experiment combines the bass, piano, vocal, and other MIDI similarity
matrices in two ways:

- late fusion averages each stem's best score even when the evidence comes
  from different source times or tempo ratios;
- synchronized fusion combines matrix cells before taking the maximum, so all
  contributing stems must support the same query window, reference window,
  and tempo ratio.

All stems use one common duration read from the source WAV. This gives their
window matrices identical time axes. Each stem may still select its own pitch
transposition at a synchronized cell.

## Primary result: pairwise plagiarism decision

The target task receives one AI recording and one proposed original, then
returns plagiarism or no plagiarism without consulting a catalog. Thresholds
are evaluated with leave-two-families-out cross-validation. Each fold removes
two complete song families, including both their AI and original tracks, fits
the threshold on the other six families, then tests the four AI/original
combinations between the held-out families: two positives and two negatives.

Across all 28 possible held-out family pairs this produces 56 positive and 56
negative evaluations. Each positive song pair is evaluated seven times under
different training folds; each directed negative pair is evaluated once.

{markdown_table(
    (
        "Strategy",
        "TP",
        "FP",
        "Precision",
        "Recall",
        "F1",
        "Rasputin folds",
        "Threshold range",
    ),
    pairwise_rows,
)}

## Secondary diagnostic: catalog retrieval

For each of the eight AI excerpts, the experiment compares it with all eight
original recordings and sorts the originals by decreasing similarity. A
`7/8` result means that the corresponding original is ranked first for seven
AI excerpts out of eight. This measures source retrieval in a known catalog;
it does not mean that seven pairs have passed a plagiarism threshold.

{markdown_table(
    ("Strategy", "Ranks first", "Mean margin", "Expected median", "Other median"),
    summary_rows,
)}

## Late mean of all stems by query

{markdown_table(
    ("Query", "Expected rank", "Score", "Margin", "Strongest alternative"),
    query_rows,
)}

## Rasputin

{markdown_table(
    ("Strategy", "Expected rank", "Score", "Margin"),
    rasputin_rows,
)}

## Conclusion

For the intended pairwise task, `sync-mean-bass-other` is the strongest rule.
It detects 40/56 positive evaluations with 7/56 false positives, giving
precision `0.851`, recall `0.714`, and F1 `0.777`. Rasputin is detected in six
of its seven held-out folds. Requiring bass and other to support the same time
region is therefore useful for a binary decision even though it is weaker for
catalog ranking.

The result is promising but incomplete. Brother Louie and Mambo No. 5 are
never detected in their seven folds, and one in eight negative evaluations is
still a false alert. The official train, validation, and test CSV files are
also empty, so these 28 cross-validation folds remain exploratory rather than
an immutable final test.

Catalog retrieval is retained only as a diagnostic table. It is not a success
criterion for the first research milestone. Future experiments should be
selected by pairwise precision, recall, F1, false-positive rate, and stability
across song-family folds.

The complete results are stored in
`output/research/multistem-fusion-experiment.csv`.
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
