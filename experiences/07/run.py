#!/usr/bin/env python3
"""Evaluate local melodic sequence alignment for pairwise plagiarism detection."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
import re
from dataclasses import dataclass
from itertools import combinations
from statistics import mean
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from experiences._shared.midi_similarity import NoteEvent, load_midi_events


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data" / "dataset"
FUSION_RESULTS_PATH = ROOT / "experiences" / "06" / "multistem-fusion-experiment.csv"
RESULTS_PATH = ROOT / "experiences" / "07" / "sequence-alignment-experiment.csv"
REPORT_PATH = ROOT / "experiences" / "07" / "rapport.md"
STEMS = ("bass", "vocals")
MODES = ("exact", "direction")
TEMPO_RATIOS = (0.8, 0.9, 1.0, 1.125, 1.2, 1.25)
VARIANT_SUFFIX = re.compile(r"-(?:original|AI-[A-Za-z0-9]+)$")
AlignmentMode = Literal["exact", "direction"]
FIELDS = (
    "mode",
    "stem",
    "query",
    "reference",
    "expected_pair",
    "score",
    "aligned_transitions",
    "query_start",
    "reference_start",
    "tempo_ratio",
)


@dataclass(frozen=True)
class MelodicSequence:
    intervals: NDArray[np.float64]
    inter_onset_times: NDArray[np.float64]
    onsets: NDArray[np.float64]


@dataclass(frozen=True)
class AlignmentResult:
    score: float
    aligned_transitions: int
    query_start: float
    reference_start: float
    tempo_ratio: float


@dataclass
class PairwiseMetrics:
    true_positives: int = 0
    false_positives: int = 0
    positive_evaluations: int = 0
    negative_evaluations: int = 0
    detections_by_family: dict[str, int] | None = None

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.positive_evaluations

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


def track_key(path: Path | str) -> str:
    name = path.name if isinstance(path, Path) else path
    return VARIANT_SUFFIX.sub("", name)


def directories() -> tuple[list[Path], list[Path]]:
    paths = sorted(path for path in DATASET_DIR.iterdir() if path.is_dir())
    return (
        [path for path in paths if "-AI-" in path.name],
        [path for path in paths if path.name.endswith("-original")],
    )


def melody_notes(
    events: list[NoteEvent],
    simultaneous_tolerance: float = 0.08,
) -> list[NoteEvent]:
    ordered = sorted(events, key=lambda event: (event.start, -event.velocity, -event.end))
    groups: list[list[NoteEvent]] = []
    for event in ordered:
        if (
            not groups
            or event.start - groups[-1][0].start > simultaneous_tolerance
        ):
            groups.append([event])
        else:
            groups[-1].append(event)
    return [
        max(group, key=lambda event: (event.velocity, event.end - event.start))
        for group in groups
    ]


def build_sequence(events: list[NoteEvent]) -> MelodicSequence:
    notes = melody_notes(events)
    return MelodicSequence(
        intervals=np.asarray(
            [following.pitch - previous.pitch for previous, following in zip(notes, notes[1:])],
            dtype=np.float64,
        ),
        inter_onset_times=np.asarray(
            [following.start - previous.start for previous, following in zip(notes, notes[1:])],
            dtype=np.float64,
        ),
        onsets=np.asarray([note.start for note in notes[1:]], dtype=np.float64),
    )


def interval_similarities(
    query: NDArray[np.float64],
    reference: NDArray[np.float64],
    mode: AlignmentMode,
) -> NDArray[np.float64]:
    if mode == "exact":
        return np.exp(-np.abs(query[:, None] - reference[None, :]) / 2.0)

    def category(intervals: NDArray[np.float64]) -> NDArray[np.int8]:
        return np.where(
            intervals <= -3,
            0,
            np.where(
                intervals < 0,
                1,
                np.where(intervals == 0, 2, np.where(intervals < 3, 3, 4)),
            ),
        ).astype(np.int8)

    difference = np.abs(category(query)[:, None] - category(reference)[None, :])
    return np.where(difference == 0, 1.0, np.where(difference == 1, 0.55, 0.05))


def smith_waterman(
    query: MelodicSequence,
    reference: MelodicSequence,
    *,
    tempo_ratio: float,
    mode: AlignmentMode,
) -> AlignmentResult:
    if not len(query.intervals) or not len(reference.intervals):
        return AlignmentResult(0.0, 0, 0.0, 0.0, tempo_ratio)

    pitch_similarity = interval_similarities(
        query.intervals, reference.intervals, mode
    )
    normalized_reference_times = np.maximum(
        reference.inter_onset_times / tempo_ratio, 0.04
    )
    rhythm_similarity = np.exp(
        -np.abs(
            np.log(
                np.maximum(query.inter_onset_times[:, None], 0.04)
                / normalized_reference_times[None, :]
            )
        )
        / 0.5
    )
    similarities = 0.78 * pitch_similarity + 0.22 * rhythm_similarity

    reference_length = len(reference.intervals)
    previous = np.zeros(reference_length + 1, dtype=np.float32)
    previous_counts = np.zeros(reference_length + 1, dtype=np.int16)
    previous_starts = np.zeros(reference_length + 1, dtype=np.int32)
    best_score = 0.0
    best_count = 0
    best_query_start = 0
    best_reference_start = 0
    gap_penalty = 0.16
    match_floor = 0.55

    for query_index in range(1, len(query.intervals) + 1):
        current = np.zeros(reference_length + 1, dtype=np.float32)
        current_counts = np.zeros(reference_length + 1, dtype=np.int16)
        current_starts = np.zeros(reference_length + 1, dtype=np.int32)
        for reference_index in range(1, reference_length + 1):
            options = (
                0.0,
                float(
                    previous[reference_index - 1]
                    + similarities[query_index - 1, reference_index - 1]
                    - match_floor
                ),
                float(previous[reference_index] - gap_penalty),
                float(current[reference_index - 1] - gap_penalty),
            )
            counts = (
                0,
                int(previous_counts[reference_index - 1]) + 1,
                int(previous_counts[reference_index]),
                int(current_counts[reference_index - 1]),
            )
            choice = max(range(4), key=lambda index: (options[index], counts[index]))
            current[reference_index] = options[choice]
            current_counts[reference_index] = counts[choice]
            if choice == 0:
                current_starts[reference_index] = reference_index - 1
            elif choice == 1 and previous_counts[reference_index - 1] == 0:
                current_starts[reference_index] = reference_index - 1
            elif choice in {1, 2}:
                current_starts[reference_index] = previous_starts[reference_index - (choice == 1)]
            else:
                current_starts[reference_index] = current_starts[reference_index - 1]

            if (current[reference_index], current_counts[reference_index]) > (
                best_score,
                best_count,
            ):
                best_score = float(current[reference_index])
                best_count = int(current_counts[reference_index])
                best_query_start = max(0, query_index - best_count)
                best_reference_start = int(current_starts[reference_index])
        previous = current
        previous_counts = current_counts
        previous_starts = current_starts

    if best_count == 0:
        return AlignmentResult(0.0, 0, 0.0, 0.0, tempo_ratio)
    quality = best_score / ((1.0 - match_floor) * best_count)
    length_confidence = min(1.0, best_count / 8.0)
    return AlignmentResult(
        score=float(np.clip(quality * length_confidence, 0.0, 1.0)),
        aligned_transitions=best_count,
        query_start=float(query.onsets[best_query_start]),
        reference_start=float(reference.onsets[best_reference_start]),
        tempo_ratio=tempo_ratio,
    )


def compare_sequences(
    query: MelodicSequence,
    reference: MelodicSequence,
    mode: AlignmentMode,
) -> AlignmentResult:
    return max(
        (
            smith_waterman(
                query,
                reference,
                tempo_ratio=ratio,
                mode=mode,
            )
            for ratio in TEMPO_RATIOS
        ),
        key=lambda result: (result.score, result.aligned_transitions),
    )


def compare_all() -> list[dict[str, object]]:
    queries, references = directories()
    paths = [*queries, *references]
    sequences = {
        (path.name, stem): build_sequence(
            load_midi_events(path / f"{stem}_basic_pitch.mid")
        )
        for path in paths
        for stem in STEMS
    }
    rows: list[dict[str, object]] = []
    for mode in MODES:
        for index, query in enumerate(queries, start=1):
            print(f"[{mode} {index}/{len(queries)}] query={query.name}", flush=True)
            for reference in references:
                for stem in STEMS:
                    result = compare_sequences(
                        sequences[(query.name, stem)],
                        sequences[(reference.name, stem)],
                        mode,
                    )
                    rows.append(
                        {
                            "mode": mode,
                            "stem": stem,
                            "query": query.name,
                            "reference": reference.name,
                            "expected_pair": str(
                                track_key(query) == track_key(reference)
                            ).lower(),
                            "score": f"{result.score:.9f}",
                            "aligned_transitions": result.aligned_transitions,
                            "query_start": f"{result.query_start:.6f}",
                            "reference_start": f"{result.reference_start:.6f}",
                            "tempo_ratio": f"{result.tempo_ratio:.6f}",
                        }
                    )
    return rows


def score(row: dict[str, object]) -> float:
    return float(row["score"])


def expected(row: dict[str, object]) -> bool:
    return row["expected_pair"] == "true"


def feature_scores(
    rows: list[dict[str, object]],
) -> tuple[list[str], dict[tuple[str, str], dict[str, float]]]:
    with FUSION_RESULTS_PATH.open(newline="", encoding="utf-8") as stream:
        fusion_rows = list(csv.DictReader(stream))
    pairs: dict[tuple[str, str], dict[str, float]] = {}
    for row in fusion_rows:
        if row["strategy"] != "sync-mean-bass-other":
            continue
        pairs[(row["query"], row["reference"])] = {
            "baseline": score(row),
            "expected": 1.0 if expected(row) else 0.0,
        }
    for row in rows:
        pairs[(row["query"], row["reference"])][
            f"{row['mode']}-{row['stem']}"
        ] = score(row)
    return sorted({track_key(query) for query, _ in pairs}), pairs


def threshold_candidates(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    return [
        unique[0] - 1e-9,
        *((left + right) / 2 for left, right in zip(unique, unique[1:])),
        unique[-1] + 1e-9,
    ]


def classification_metric(
    labels: list[bool], predictions: list[bool]
) -> tuple[float, float, int]:
    positives = sum(labels)
    negatives = len(labels) - positives
    true_positives = sum(label and prediction for label, prediction in zip(labels, predictions))
    false_positives = sum(not label and prediction for label, prediction in zip(labels, predictions))
    recall = true_positives / positives
    specificity = 1.0 - false_positives / negatives
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return mean((recall, specificity)), f1, -false_positives


def best_scalar_threshold(
    values: list[float], labels: list[bool]
) -> tuple[float, tuple[float, float, int]]:
    return max(
        (
            (
                threshold,
                classification_metric(
                    labels, [value >= threshold for value in values]
                ),
            )
            for threshold in threshold_candidates(values)
        ),
        key=lambda item: (*item[1], item[0]),
    )


def update_metrics(
    metrics: PairwiseMetrics,
    test_rows: list[tuple[str, str, dict[str, float]]],
    predictions: list[bool],
) -> None:
    assert metrics.detections_by_family is not None
    for (query, _, features), prediction in zip(test_rows, predictions):
        label = bool(features["expected"])
        if label:
            metrics.positive_evaluations += 1
            metrics.true_positives += prediction
            if prediction:
                metrics.detections_by_family[track_key(query)] += 1
        else:
            metrics.negative_evaluations += 1
            metrics.false_positives += prediction


def evaluate_scalar(
    families: list[str],
    pairs: dict[tuple[str, str], dict[str, float]],
    feature: str,
) -> PairwiseMetrics:
    metrics = PairwiseMetrics(detections_by_family={family: 0 for family in families})
    for held_out in combinations(families, 2):
        training, testing = split_fold(pairs, held_out)
        values = [features[feature] for _, _, features in training]
        labels = [bool(features["expected"]) for _, _, features in training]
        threshold, _ = best_scalar_threshold(values, labels)
        update_metrics(
            metrics,
            testing,
            [features[feature] >= threshold for _, _, features in testing],
        )
    return metrics


def split_fold(
    pairs: dict[tuple[str, str], dict[str, float]],
    held_out: tuple[str, str],
) -> tuple[
    list[tuple[str, str, dict[str, float]]],
    list[tuple[str, str, dict[str, float]]],
]:
    training: list[tuple[str, str, dict[str, float]]] = []
    testing: list[tuple[str, str, dict[str, float]]] = []
    for (query, reference), features in pairs.items():
        query_family = track_key(query)
        reference_family = track_key(reference)
        row = (query, reference, features)
        if query_family in held_out and reference_family in held_out:
            testing.append(row)
        elif query_family not in held_out and reference_family not in held_out:
            training.append(row)
    return training, testing


def evaluate_nested_alpha(
    families: list[str],
    pairs: dict[tuple[str, str], dict[str, float]],
) -> PairwiseMetrics:
    metrics = PairwiseMetrics(detections_by_family={family: 0 for family in families})
    for held_out in combinations(families, 2):
        training, testing = split_fold(pairs, held_out)
        labels = [bool(features["expected"]) for _, _, features in training]
        candidates: list[tuple[tuple[float, float, int], float, float]] = []
        for alpha_index in range(11):
            alpha = alpha_index / 10.0
            values = [
                (1.0 - alpha) * features["baseline"]
                + alpha * features["exact-vocals"]
                for _, _, features in training
            ]
            threshold, metric = best_scalar_threshold(values, labels)
            candidates.append((metric, -alpha, threshold))
        _, negative_alpha, threshold = max(candidates)
        alpha = -negative_alpha
        predictions = [
            (1.0 - alpha) * features["baseline"]
            + alpha * features["exact-vocals"]
            >= threshold
            for _, _, features in testing
        ]
        update_metrics(metrics, testing, predictions)
    return metrics


def evaluate_two_thresholds(
    families: list[str],
    pairs: dict[tuple[str, str], dict[str, float]],
    operation: Literal["or", "and"],
) -> PairwiseMetrics:
    metrics = PairwiseMetrics(detections_by_family={family: 0 for family in families})
    for held_out in combinations(families, 2):
        training, testing = split_fold(pairs, held_out)
        labels = [bool(features["expected"]) for _, _, features in training]
        baseline_thresholds = threshold_candidates(
            [features["baseline"] for _, _, features in training]
        )
        sequence_thresholds = threshold_candidates(
            [features["exact-vocals"] for _, _, features in training]
        )
        candidates: list[tuple[tuple[float, float, int], float, float]] = []
        for baseline_threshold in baseline_thresholds:
            for sequence_threshold in sequence_thresholds:
                predictions = [
                    (
                        features["baseline"] >= baseline_threshold
                        or features["exact-vocals"] >= sequence_threshold
                    )
                    if operation == "or"
                    else (
                        features["baseline"] >= baseline_threshold
                        and features["exact-vocals"] >= sequence_threshold
                    )
                    for _, _, features in training
                ]
                candidates.append(
                    (
                        classification_metric(labels, predictions),
                        baseline_threshold,
                        sequence_threshold,
                    )
                )
        _, baseline_threshold, sequence_threshold = max(candidates)
        predictions = [
            (
                features["baseline"] >= baseline_threshold
                or features["exact-vocals"] >= sequence_threshold
            )
            if operation == "or"
            else (
                features["baseline"] >= baseline_threshold
                and features["exact-vocals"] >= sequence_threshold
            )
            for _, _, features in testing
        ]
        update_metrics(metrics, testing, predictions)
    return metrics


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def write_results(rows: list[dict[str, object]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, object]]) -> None:
    families, pairs = feature_scores(rows)
    classifiers = {
        "baseline-sync-bass-other": evaluate_scalar(families, pairs, "baseline"),
        "sequence-exact-bass": evaluate_scalar(families, pairs, "exact-bass"),
        "sequence-exact-vocals": evaluate_scalar(families, pairs, "exact-vocals"),
        "sequence-direction-bass": evaluate_scalar(families, pairs, "direction-bass"),
        "sequence-direction-vocals": evaluate_scalar(
            families, pairs, "direction-vocals"
        ),
        "nested-weight-baseline-exact-vocals": evaluate_nested_alpha(
            families, pairs
        ),
        "nested-OR-baseline-exact-vocals": evaluate_two_thresholds(
            families, pairs, "or"
        ),
        "nested-AND-baseline-exact-vocals": evaluate_two_thresholds(
            families, pairs, "and"
        ),
    }
    table_rows = []
    for name, metrics in classifiers.items():
        assert metrics.detections_by_family is not None
        table_rows.append(
            (
                name,
                f"{metrics.true_positives}/{metrics.positive_evaluations}",
                f"{metrics.false_positives}/{metrics.negative_evaluations}",
                f"{metrics.precision:.3f}",
                f"{metrics.recall:.3f}",
                f"{metrics.f1:.3f}",
                f"{metrics.detections_by_family['rasputin-boney-m']}/7",
                f"{metrics.detections_by_family['brother-louie-modern-talking']}/7",
                f"{metrics.detections_by_family['mambo-no-5-lou-bega']}/7",
            )
        )

    baseline = classifiers["baseline-sync-bass-other"]
    nested_or = classifiers["nested-OR-baseline-exact-vocals"]
    assert baseline.detections_by_family is not None
    assert nested_or.detections_by_family is not None
    family_rows = [
        (
            family,
            f"{baseline.detections_by_family[family]}/7",
            f"{nested_or.detections_by_family[family]}/7",
        )
        for family in families
    ]

    content = f"""# Local melodic sequence alignment experiment

## Scope

The fixed-window vectors require notes to fall into the same 500 ms frames.
This experiment replaces that comparison for bass and vocals with a local
Smith-Waterman alignment of consecutive note transitions. The representation
is invariant to a constant pitch shift and tests six tempo ratios. It tolerates
inserted or deleted notes with a gap penalty and requires eight aligned
transitions for full confidence.

Two pitch comparisons are tested:

- `exact`: signed semitone intervals with exponentially decreasing tolerance;
- `direction`: five coarse descending, repeated, and ascending classes.

Near-simultaneous Basic Pitch notes are reduced to the strongest melodic
candidate. All classifier combinations are selected inside each training fold
and evaluated with the same leave-two-families-out protocol as the fusion
experiment.

## Pairwise results

{markdown_table(
    (
        "Classifier",
        "TP",
        "FP",
        "Precision",
        "Recall",
        "F1",
        "Rasputin",
        "Brother Louie",
        "Mambo No. 5",
    ),
    table_rows,
)}

## Detection stability by song family

{markdown_table(
    ("Song family", "Fusion baseline", "Nested OR"),
    family_rows,
)}

## Conclusion

The nested OR rule is the best pairwise result so far. It accepts a pair when
either synchronized bass/other fusion or exact vocal sequence alignment passes
its threshold selected inside the training fold. F1 rises from `0.777` to
`0.800`: true positives increase from 40/56 to 42/56 while false positives stay
at 7/56. Precision is `0.857` and recall is `0.750`.

The vocal sequence adds genuinely complementary evidence. It recovers Brother
Louie in 7/7 folds and Mambo No. 5 in 2/7 while preserving Rasputin at 6/7.
The gain is not yet stable across songs: Daddy Cool falls from 6/7 to 0/7 and
You're My Heart falls from 7/7 to 6/7. On only eight families, the aggregate
F1 improvement is therefore promising rather than conclusive.

The nested OR rule becomes the new experimental pairwise candidate. The next
experiment should work directly from audio chroma to add evidence which does
not inherit Spleeter and Basic Pitch note errors.

The complete alignment results are stored in
`experiences/07/sequence-alignment-experiment.csv`.
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> int:
    if not FUSION_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            "run experiences/06/run.py before this experiment"
        )
    rows = compare_all()
    write_results(rows)
    write_report(rows)
    print(f"Wrote {len(rows)} comparisons to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote report to {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
