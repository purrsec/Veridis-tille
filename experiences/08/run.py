#!/usr/bin/env python3
"""Evaluate full-mix audio chroma against the current stem/MIDI pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
import re
import wave
from dataclasses import dataclass
from itertools import combinations
from statistics import mean
from typing import Literal

import numpy as np
from numpy.typing import NDArray


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data" / "dataset"
FUSION_RESULTS_PATH = ROOT / "experiences" / "06" / "multistem-fusion-experiment.csv"
SEQUENCE_RESULTS_PATH = ROOT / "experiences" / "07" / "sequence-alignment-experiment.csv"
RESULTS_PATH = ROOT / "experiences" / "08" / "full-mix-chroma-experiment.csv"
REPORT_PATH = ROOT / "experiences" / "08" / "rapport.md"
TEMPO_RATIOS = (0.8, 0.9, 1.0, 1.125, 1.2, 1.25)
STRATEGIES = ("chroma-max", "chroma-diagonal-2", "chroma-diagonal-3")
VARIANT_SUFFIX = re.compile(r"-(?:original|AI-[A-Za-z0-9]+)$")
FIELDS = (
    "strategy",
    "query",
    "reference",
    "expected_pair",
    "score",
    "query_start",
    "reference_start",
    "tempo_ratio",
    "transposition",
    "aligned_windows",
)


@dataclass(frozen=True)
class ChromaTrack:
    values: NDArray[np.float32]
    frame_seconds: float
    duration: float


@dataclass(frozen=True)
class ChromaWindows:
    starts: NDArray[np.float64]
    values: NDArray[np.float32]


@dataclass(frozen=True)
class LocalResult:
    score: float
    query_start: float
    reference_start: float
    tempo_ratio: float
    transposition: int
    aligned_windows: int


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

    @property
    def false_positive_rate(self) -> float:
        return self.false_positives / self.negative_evaluations


def track_key(path: Path | str) -> str:
    name = path.name if isinstance(path, Path) else path
    return VARIANT_SUFFIX.sub("", name.removesuffix(".wav"))


def audio_paths() -> tuple[list[Path], list[Path]]:
    paths = sorted(DATASET_DIR.glob("*.wav"))
    return (
        [path for path in paths if "-AI-" in path.stem],
        [path for path in paths if path.stem.endswith("-original")],
    )


def read_wav_mono(path: Path) -> tuple[NDArray[np.float32], int]:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        sample_rate = stream.getframerate()
        frame_count = stream.getnframes()
        payload = stream.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"{path.name}: expected 16-bit PCM")
    audio = np.frombuffer(payload, dtype="<i2").reshape(-1, channels)
    mono = audio.astype(np.float32).mean(axis=1) / 32768.0
    return mono, sample_rate


def chroma_projection(sample_rate: int, fft_size: int) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    selected = np.flatnonzero((frequencies >= 65.0) & (frequencies <= 4200.0))
    midi = 69.0 + 12.0 * np.log2(frequencies[selected] / 440.0)
    lower = np.floor(midi).astype(np.int64)
    fraction = (midi - lower).astype(np.float32)
    projection = np.zeros((len(selected), 12), dtype=np.float32)
    indices = np.arange(len(selected))
    projection[indices, lower % 12] += 1.0 - fraction
    projection[indices, (lower + 1) % 12] += fraction
    return selected, projection


def extract_chroma(
    path: Path,
    *,
    frame_seconds: float = 0.5,
    fft_size: int = 8192,
    hop_size: int = 2048,
) -> ChromaTrack:
    audio, sample_rate = read_wav_mono(path)
    return extract_chroma_samples(
        audio, sample_rate, frame_seconds=frame_seconds,
        fft_size=fft_size, hop_size=hop_size,
    )


def extract_chroma_samples(
    audio: NDArray[np.float32],
    sample_rate: int,
    *,
    frame_seconds: float = 0.5,
    fft_size: int = 8192,
    hop_size: int = 2048,
) -> ChromaTrack:
    """Apply the identical extractor to in-memory, potentially floating WAV audio."""
    duration = len(audio) / sample_rate
    if len(audio) < fft_size:
        audio = np.pad(audio, (0, fft_size - len(audio)))
    frame_count = 1 + (len(audio) - fft_size) // hop_size
    starts = np.arange(frame_count, dtype=np.int64) * hop_size
    selected, projection = chroma_projection(sample_rate, fft_size)
    window = np.hanning(fft_size).astype(np.float32)
    time_bins = max(1, int(np.ceil(duration / frame_seconds)))
    accumulated = np.zeros((time_bins, 12), dtype=np.float32)
    counts = np.zeros(time_bins, dtype=np.int32)

    for batch_start in range(0, frame_count, 256):
        batch_starts = starts[batch_start : batch_start + 256]
        frames = np.stack([audio[start : start + fft_size] for start in batch_starts])
        magnitudes = np.abs(np.fft.rfft(frames * window, axis=1))[:, selected]
        # Square-root compression prevents a few loud partials from dominating.
        chroma = np.sqrt(magnitudes, dtype=np.float32) @ projection
        norms = np.linalg.norm(chroma, axis=1, keepdims=True)
        chroma = np.divide(chroma, norms, out=np.zeros_like(chroma), where=norms > 0)
        centers = (batch_starts + fft_size / 2) / sample_rate
        bin_indices = np.minimum((centers / frame_seconds).astype(np.int64), time_bins - 1)
        np.add.at(accumulated, bin_indices, chroma)
        np.add.at(counts, bin_indices, 1)

    accumulated = np.divide(
        accumulated,
        counts[:, None],
        out=np.zeros_like(accumulated),
        where=counts[:, None] > 0,
    )
    # Remove part of the broadband floor before the final frame normalization.
    accumulated = np.maximum(
        accumulated - 0.5 * np.median(accumulated, axis=1, keepdims=True),
        0.0,
    )
    norms = np.linalg.norm(accumulated, axis=1, keepdims=True)
    accumulated = np.divide(
        accumulated,
        norms,
        out=np.zeros_like(accumulated),
        where=norms > 0,
    )
    return ChromaTrack(accumulated, frame_seconds, duration)


def window_starts(duration: float, window_seconds: float, hop_seconds: float) -> NDArray[np.float64]:
    if duration <= window_seconds:
        return np.asarray([0.0], dtype=np.float64)
    count = int(np.ceil((duration - window_seconds) / hop_seconds)) + 1
    return np.arange(count, dtype=np.float64) * hop_seconds


def build_windows(track: ChromaTrack, tempo_ratio: float) -> ChromaWindows:
    window_seconds = 10.0 * tempo_ratio
    hop_seconds = 2.0 * tempo_ratio
    starts = window_starts(track.duration, window_seconds, hop_seconds)
    sample_offsets = (np.arange(20, dtype=np.float64) + 0.5) * 0.5 * tempo_ratio
    source_times = (np.arange(len(track.values), dtype=np.float64) + 0.5) * track.frame_seconds
    windows = np.empty((len(starts), 20, 12), dtype=np.float32)
    for window_index, start in enumerate(starts):
        targets = start + sample_offsets
        for pitch_class in range(12):
            windows[window_index, :, pitch_class] = np.interp(
                targets,
                source_times,
                track.values[:, pitch_class],
                left=0.0,
                right=0.0,
            )
    return ChromaWindows(starts, windows)


def similarity_matrix(
    query: ChromaWindows,
    reference: ChromaWindows,
) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
    reference_flat = reference.values.reshape(len(reference.starts), -1)
    reference_norm = np.linalg.norm(reference_flat, axis=1, keepdims=True)
    reference_unit = np.divide(
        reference_flat,
        reference_norm,
        out=np.zeros_like(reference_flat),
        where=reference_norm > 0,
    )
    best_scores = np.zeros((len(query.starts), len(reference.starts)), dtype=np.float32)
    best_shifts = np.zeros(best_scores.shape, dtype=np.int8)
    for shift in range(12):
        shifted = np.roll(query.values, shift=shift, axis=-1).reshape(len(query.starts), -1)
        query_norm = np.linalg.norm(shifted, axis=1, keepdims=True)
        query_unit = np.divide(
            shifted,
            query_norm,
            out=np.zeros_like(shifted),
            where=query_norm > 0,
        )
        scores = np.clip(query_unit @ reference_unit.T, 0.0, 1.0)
        improved = scores > best_scores
        best_scores[improved] = scores[improved]
        best_shifts[improved] = shift if shift <= 6 else shift - 12
    return best_scores, best_shifts


def best_diagonal_run(
    scores: NDArray[np.float32],
    shifts: NDArray[np.int8],
    length: int,
) -> tuple[float, int, int, int]:
    best = (0.0, 0, 0, 0)
    query_count, reference_count = scores.shape
    for offset in range(-query_count + 1, reference_count):
        query_start = max(0, -offset)
        query_end = min(query_count, reference_count - offset)
        if query_end - query_start < length:
            continue
        diagonal_scores = np.asarray(
            [scores[index, index + offset] for index in range(query_start, query_end)]
        )
        means = np.convolve(diagonal_scores, np.ones(length) / length, mode="valid")
        local_index = int(np.argmax(means))
        score = float(means[local_index])
        query_index = query_start + local_index
        reference_index = query_index + offset
        run_shifts = shifts[
            query_index : query_index + length,
            reference_index : reference_index + length,
        ].diagonal()
        transposition = int(round(float(np.median(run_shifts))))
        if score > best[0]:
            best = (score, query_index, reference_index, transposition)
    return best


def compare_pair(
    query: ChromaWindows,
    references: dict[float, ChromaWindows],
) -> dict[str, LocalResult]:
    best = {
        strategy: LocalResult(0.0, 0.0, 0.0, 1.0, 0, int(strategy.rsplit("-", 1)[-1]) if "diagonal" in strategy else 1)
        for strategy in STRATEGIES
    }
    for ratio, reference in references.items():
        scores, shifts = similarity_matrix(query, reference)
        flat_index = int(np.argmax(scores))
        query_index, reference_index = np.unravel_index(flat_index, scores.shape)
        candidates = {
            "chroma-max": (
                float(scores[query_index, reference_index]),
                query_index,
                reference_index,
                int(shifts[query_index, reference_index]),
                1,
            )
        }
        for length in (2, 3):
            score, run_query, run_reference, shift = best_diagonal_run(scores, shifts, length)
            candidates[f"chroma-diagonal-{length}"] = (
                score,
                run_query,
                run_reference,
                shift,
                length,
            )
        for strategy, (score, qi, ri, shift, length) in candidates.items():
            if score <= best[strategy].score:
                continue
            best[strategy] = LocalResult(
                score,
                float(query.starts[qi]),
                float(reference.starts[ri]),
                ratio,
                shift,
                length,
            )
    return best


def compare_all() -> list[dict[str, object]]:
    queries, references = audio_paths()
    tracks: dict[str, ChromaTrack] = {}
    for index, path in enumerate([*queries, *references], start=1):
        print(f"[chroma {index}/{len(queries) + len(references)}] {path.name}", flush=True)
        tracks[path.name] = extract_chroma(path)
    query_windows = {path.name: build_windows(tracks[path.name], 1.0) for path in queries}
    reference_windows = {
        path.name: {ratio: build_windows(tracks[path.name], ratio) for ratio in TEMPO_RATIOS}
        for path in references
    }
    rows: list[dict[str, object]] = []
    for index, query_path in enumerate(queries, start=1):
        print(f"[compare {index}/{len(queries)}] {query_path.stem}", flush=True)
        for reference_path in references:
            results = compare_pair(
                query_windows[query_path.name], reference_windows[reference_path.name]
            )
            for strategy, result in results.items():
                rows.append(
                    {
                        "strategy": strategy,
                        "query": query_path.stem,
                        "reference": reference_path.stem,
                        "expected_pair": str(
                            track_key(query_path) == track_key(reference_path)
                        ).lower(),
                        "score": f"{result.score:.9f}",
                        "query_start": f"{result.query_start:.6f}",
                        "reference_start": f"{result.reference_start:.6f}",
                        "tempo_ratio": f"{result.tempo_ratio:.6f}",
                        "transposition": result.transposition,
                        "aligned_windows": result.aligned_windows,
                    }
                )
    return rows


def threshold_candidates(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    return [
        unique[0] - 1e-9,
        *((left + right) / 2 for left, right in zip(unique, unique[1:])),
        unique[-1] + 1e-9,
    ]


def classification_metric(labels: list[bool], predictions: list[bool]) -> tuple[float, float, int]:
    positives = sum(labels)
    negatives = len(labels) - positives
    true_positives = sum(label and prediction for label, prediction in zip(labels, predictions))
    false_positives = sum(not label and prediction for label, prediction in zip(labels, predictions))
    recall = true_positives / positives
    specificity = 1.0 - false_positives / negatives
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return mean((recall, specificity)), f1, -false_positives


def split_fold(
    pairs: dict[tuple[str, str], dict[str, float]], held_out: tuple[str, str]
) -> tuple[list[tuple[str, str, dict[str, float]]], list[tuple[str, str, dict[str, float]]]]:
    training = []
    testing = []
    for (query, reference), features in pairs.items():
        query_family = track_key(query)
        reference_family = track_key(reference)
        row = (query, reference, features)
        if query_family in held_out and reference_family in held_out:
            testing.append(row)
        elif query_family not in held_out and reference_family not in held_out:
            training.append(row)
    return training, testing


def update_metrics(
    metrics: PairwiseMetrics,
    rows: list[tuple[str, str, dict[str, float]]],
    predictions: list[bool],
) -> None:
    assert metrics.detections_by_family is not None
    for (query, _, features), prediction in zip(rows, predictions):
        label = bool(features["expected"])
        if label:
            metrics.positive_evaluations += 1
            metrics.true_positives += prediction
            if prediction:
                metrics.detections_by_family[track_key(query)] += 1
        else:
            metrics.negative_evaluations += 1
            metrics.false_positives += prediction


def best_threshold(values: list[float], labels: list[bool]) -> float:
    return max(
        threshold_candidates(values),
        key=lambda threshold: (
            *classification_metric(labels, [value >= threshold for value in values]),
            threshold,
        ),
    )


def evaluate_scalar(
    families: list[str], pairs: dict[tuple[str, str], dict[str, float]], feature: str
) -> PairwiseMetrics:
    metrics = PairwiseMetrics(detections_by_family={family: 0 for family in families})
    for held_out in combinations(families, 2):
        training, testing = split_fold(pairs, held_out)
        labels = [bool(features["expected"]) for _, _, features in training]
        threshold = best_threshold(
            [features[feature] for _, _, features in training], labels
        )
        update_metrics(
            metrics,
            testing,
            [features[feature] >= threshold for _, _, features in testing],
        )
    return metrics


def evaluate_or(
    families: list[str],
    pairs: dict[tuple[str, str], dict[str, float]],
    features: tuple[str, ...],
) -> PairwiseMetrics:
    metrics = PairwiseMetrics(detections_by_family={family: 0 for family in families})
    for held_out in combinations(families, 2):
        training, testing = split_fold(pairs, held_out)
        labels = [bool(row["expected"]) for _, _, row in training]
        threshold_sets = [
            threshold_candidates([row[feature] for _, _, row in training])
            for feature in features
        ]
        candidates: list[tuple[tuple[float, float, int], tuple[float, ...]]] = []
        if len(features) == 2:
            for first in threshold_sets[0]:
                for second in threshold_sets[1]:
                    predictions = [
                        row[features[0]] >= first or row[features[1]] >= second
                        for _, _, row in training
                    ]
                    candidates.append((classification_metric(labels, predictions), (first, second)))
        elif len(features) == 3:
            for first in threshold_sets[0]:
                for second in threshold_sets[1]:
                    for third in threshold_sets[2]:
                        predictions = [
                            row[features[0]] >= first
                            or row[features[1]] >= second
                            or row[features[2]] >= third
                            for _, _, row in training
                        ]
                        candidates.append(
                            (classification_metric(labels, predictions), (first, second, third))
                        )
        else:
            raise ValueError("OR evaluation supports two or three features")
        _, thresholds = max(candidates, key=lambda item: (*item[0], *item[1]))
        predictions = [
            any(row[feature] >= threshold for feature, threshold in zip(features, thresholds))
            for _, _, row in testing
        ]
        update_metrics(metrics, testing, predictions)
    return metrics


def load_feature_scores(rows: list[dict[str, object]]) -> tuple[list[str], dict[tuple[str, str], dict[str, float]]]:
    pairs: dict[tuple[str, str], dict[str, float]] = {}
    with FUSION_RESULTS_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["strategy"] != "sync-mean-bass-other":
                continue
            pairs[(row["query"], row["reference"])] = {
                "expected": 1.0 if row["expected_pair"] == "true" else 0.0,
                "stems": float(row["score"]),
            }
    with SEQUENCE_RESULTS_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["mode"] == "exact" and row["stem"] == "vocals":
                pairs[(row["query"], row["reference"])]["vocals"] = float(row["score"])
    for row in rows:
        pairs[(str(row["query"]), str(row["reference"]))][str(row["strategy"])] = float(row["score"])
    return sorted({track_key(query) for query, _ in pairs}), pairs


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
    families, pairs = load_feature_scores(rows)
    classifiers = {
        "stems-sync-bass-other": evaluate_scalar(families, pairs, "stems"),
        "chroma-max": evaluate_scalar(families, pairs, "chroma-max"),
        "chroma-diagonal-2": evaluate_scalar(families, pairs, "chroma-diagonal-2"),
        "chroma-diagonal-3": evaluate_scalar(families, pairs, "chroma-diagonal-3"),
        "previous-OR-stems-vocals": evaluate_or(families, pairs, ("stems", "vocals")),
        "OR-stems-chroma-diagonal-3": evaluate_or(
            families, pairs, ("stems", "chroma-diagonal-3")
        ),
        "OR-stems-vocals-chroma-diagonal-3": evaluate_or(
            families, pairs, ("stems", "vocals", "chroma-diagonal-3")
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
                f"{metrics.false_positive_rate:.3f}",
                f"{metrics.detections_by_family['rasputin-boney-m']}/7",
            )
        )
    selected = classifiers["chroma-diagonal-3"]
    previous = classifiers["previous-OR-stems-vocals"]
    combined = classifiers["OR-stems-chroma-diagonal-3"]
    extended = classifiers["OR-stems-vocals-chroma-diagonal-3"]
    assert selected.detections_by_family is not None
    assert previous.detections_by_family is not None
    assert combined.detections_by_family is not None
    family_rows = [
        (
            family,
            f"{selected.detections_by_family[family]}/7",
            f"{previous.detections_by_family[family]}/7",
            f"{combined.detections_by_family[family]}/7",
        )
        for family in families
    ]
    best_chroma_name = max(
        (name for name in STRATEGIES), key=lambda name: classifiers[name].f1
    )
    best_chroma = classifiers[best_chroma_name]
    if best_chroma.f1 > classifiers["stems-sync-bass-other"].f1:
        ablation = "The full-mix control outperforms the synchronized stem/MIDI baseline on this corpus."
    else:
        ablation = "The synchronized stem/MIDI baseline remains stronger than the full-mix chroma control."
    if combined.f1 > previous.f1:
        fusion = "Adding full-mix chroma improves the previous candidate under nested threshold selection."
    elif combined.f1 == previous.f1:
        fusion = "Adding full-mix chroma does not change the previous candidate's aggregate F1."
    else:
        fusion = "Adding full-mix chroma degrades the previous candidate under nested threshold selection and should not be retained."

    content = f"""# Full-mix chroma ablation experiment

## Scope

This experiment is the direct control required by the research hypothesis. It
extracts 12-bin chroma from each complete master WAV and does not use Spleeter
or Basic Pitch. The signal is downmixed to mono for analysis only. An 8192
sample Hann-windowed spectrum is computed every 2048 samples; spectral bins
from 65 Hz to 4.2 kHz are assigned linearly between adjacent pitch classes.
The resulting chroma is aggregated into 500 ms frames and normalized.

Ten-second query windows are compared with reference windows under all twelve
pitch transpositions and six tempo ratios. Three fixed scores are evaluated:
the best isolated window, and the best mean over two or three consecutive
diagonal windows. Requiring a diagonal run tests whether the evidence persists
over roughly 12 or 14 seconds instead of relying on one local maximum.

Thresholds and OR combinations are selected inside each training fold. The
evaluation uses the same 28 leave-two-families-out folds as the earlier MIDI
experiments.

## Pairwise results

{markdown_table(
    ("Classifier", "TP", "FP", "Precision", "Recall", "F1", "FPR", "Rasputin"),
    table_rows,
)}

## Stability by family

{markdown_table(
    ("Song family", "Chroma diag-3", "Previous candidate", "Stems + chroma"),
    family_rows,
)}

## Interpretation

The best fixed full-mix chroma rule is `{best_chroma_name}` with F1
`{best_chroma.f1:.3f}`, precision `{best_chroma.precision:.3f}`, recall
`{best_chroma.recall:.3f}`, and false-positive rate
`{best_chroma.false_positive_rate:.3f}`. Its positive scores are highly
compressed and overlap the negative scores, so its threshold is fragile.
{ablation} This is not a strict test of H2 because the two branches change both
the signal input and the representation: full-mix spectral chroma is compared
with Spleeter followed by Basic Pitch MIDI. A chroma-on-mix versus
chroma-on-stems experiment is still required to isolate the effect of source
separation.

The previous stem/MIDI candidate has F1 `{previous.f1:.3f}`. The pre-specified
stems-plus-chroma OR has F1 `{combined.f1:.3f}`, precision
`{combined.precision:.3f}`, and recall `{combined.recall:.3f}`. {fusion} Adding
the vocal interval branch as a third OR reduces F1 to `{extended.f1:.3f}` and
is not retained.

These are exploratory repeated-fold counts over eight song families, not
independent test-set estimates. The experiment isolates the full-mix control,
but H2 can only be accepted or rejected after the same comparison on a larger
family-disjoint test set with hard negatives.

The complete local scores and candidate locations are stored in
`experiences/08/full-mix-chroma-experiment.csv`.
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> int:
    for dependency in (FUSION_RESULTS_PATH, SEQUENCE_RESULTS_PATH):
        if not dependency.is_file():
            raise FileNotFoundError(f"missing prerequisite experiment: {dependency}")
    rows = compare_all()
    write_results(rows)
    write_report(rows)
    print(f"Wrote {len(rows)} comparisons to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote report to {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
