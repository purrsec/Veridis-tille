#!/usr/bin/env python3
"""Isolate source separation by applying one chroma method to mix and stems."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import numpy as np
from numpy.typing import NDArray

from run_full_mix_chroma_experiment import (
    DATASET_DIR,
    ROOT,
    TEMPO_RATIOS,
    ChromaWindows,
    LocalResult,
    audio_paths,
    best_diagonal_run,
    build_windows,
    evaluate_scalar,
    extract_chroma,
    markdown_table,
    similarity_matrix,
    track_key,
)


FULL_MIX_RESULTS_PATH = ROOT / "output" / "research" / "full-mix-chroma-experiment.csv"
MIDI_RESULTS_PATH = ROOT / "output" / "research" / "multistem-fusion-experiment.csv"
RESULTS_PATH = ROOT / "output" / "research" / "stem-chroma-ablation.csv"
REPORT_PATH = ROOT / "docs" / "experiments" / "stem-chroma-ablation.md"
STEMS = ("bass", "drums", "piano", "vocals", "other")
FUSIONS = {
    "sync-chroma-bass-other": ("bass", "other"),
    "sync-chroma-bass-vocals-other": ("bass", "vocals", "other"),
    "sync-chroma-all-tonal": ("bass", "piano", "vocals", "other"),
}
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


def source_directories() -> tuple[list[Path], list[Path]]:
    queries, references = audio_paths()
    return (
        [DATASET_DIR / path.stem for path in queries],
        [DATASET_DIR / path.stem for path in references],
    )


def build_cache(
    queries: list[Path], references: list[Path]
) -> dict[tuple[str, str, float], ChromaWindows]:
    cache: dict[tuple[str, str, float], ChromaWindows] = {}
    paths = [*queries, *references]
    total = len(paths) * len(STEMS)
    index = 0
    for directory in paths:
        ratios = (1.0,) if directory in queries else TEMPO_RATIOS
        for stem in STEMS:
            index += 1
            print(f"[stem chroma {index}/{total}] {directory.name}/{stem}.wav", flush=True)
            track = extract_chroma(directory / f"{stem}.wav")
            for ratio in ratios:
                cache[(directory.name, stem, ratio)] = build_windows(track, ratio)
    return cache


def update_best(
    current: LocalResult,
    matrix: NDArray[np.float32],
    shifts: NDArray[np.int8] | None,
    query: ChromaWindows,
    reference: ChromaWindows,
    ratio: float,
) -> LocalResult:
    score, query_index, reference_index, transposition = best_diagonal_run(
        matrix,
        shifts if shifts is not None else np.zeros(matrix.shape, dtype=np.int8),
        3,
    )
    if score <= current.score:
        return current
    return LocalResult(
        score=score,
        query_start=float(query.starts[query_index]),
        reference_start=float(reference.starts[reference_index]),
        tempo_ratio=ratio,
        transposition=transposition if shifts is not None else 0,
        aligned_windows=3,
    )


def compare_pair(
    query_name: str,
    reference_name: str,
    cache: dict[tuple[str, str, float], ChromaWindows],
) -> dict[str, LocalResult]:
    names = tuple(f"stem-chroma-{stem}" for stem in STEMS) + tuple(FUSIONS)
    best = {name: LocalResult(0.0, 0.0, 0.0, 1.0, 0, 3) for name in names}
    for ratio in TEMPO_RATIOS:
        matrices: dict[str, NDArray[np.float32]] = {}
        shifts: dict[str, NDArray[np.int8]] = {}
        for stem in STEMS:
            query = cache[(query_name, stem, 1.0)]
            reference = cache[(reference_name, stem, ratio)]
            matrix, shift_matrix = similarity_matrix(query, reference)
            matrices[stem] = matrix
            shifts[stem] = shift_matrix
            name = f"stem-chroma-{stem}"
            best[name] = update_best(
                best[name], matrix, shift_matrix, query, reference, ratio
            )

        for name, selected_stems in FUSIONS.items():
            fused = np.mean(np.stack([matrices[stem] for stem in selected_stems]), axis=0)
            query = cache[(query_name, selected_stems[0], 1.0)]
            reference = cache[(reference_name, selected_stems[0], ratio)]
            best[name] = update_best(best[name], fused, None, query, reference, ratio)
    return best


def compare_all() -> list[dict[str, object]]:
    queries, references = source_directories()
    cache = build_cache(queries, references)
    rows: list[dict[str, object]] = []
    for index, query in enumerate(queries, start=1):
        print(f"[compare {index}/{len(queries)}] {query.name}", flush=True)
        for reference in references:
            results = compare_pair(query.name, reference.name, cache)
            for strategy, result in results.items():
                rows.append(
                    {
                        "strategy": strategy,
                        "query": query.name,
                        "reference": reference.name,
                        "expected_pair": str(
                            track_key(query.name) == track_key(reference.name)
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


def write_results(rows: list[dict[str, object]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_pairs(rows: list[dict[str, object]]) -> tuple[list[str], dict[tuple[str, str], dict[str, float]]]:
    pairs: dict[tuple[str, str], dict[str, float]] = {}
    with FULL_MIX_RESULTS_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["strategy"] != "chroma-diagonal-3":
                continue
            pairs[(row["query"], row["reference"])] = {
                "expected": 1.0 if row["expected_pair"] == "true" else 0.0,
                "full-mix-chroma": float(row["score"]),
            }
    with MIDI_RESULTS_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["strategy"] == "sync-mean-bass-other":
                pairs[(row["query"], row["reference"])]["midi-bass-other"] = float(
                    row["score"]
                )
    for row in rows:
        pairs[(str(row["query"]), str(row["reference"]))][str(row["strategy"])] = float(
            row["score"]
        )
    families = sorted({track_key(query) for query, _ in pairs})
    return families, pairs


def write_report(rows: list[dict[str, object]]) -> None:
    families, pairs = load_pairs(rows)
    strategies = (
        "full-mix-chroma",
        *(f"stem-chroma-{stem}" for stem in STEMS),
        *FUSIONS,
        "midi-bass-other",
    )
    metrics = {
        strategy: evaluate_scalar(families, pairs, strategy) for strategy in strategies
    }
    table_rows = [
        (
            strategy,
            f"{result.true_positives}/{result.positive_evaluations}",
            f"{result.false_positives}/{result.negative_evaluations}",
            f"{result.precision:.3f}",
            f"{result.recall:.3f}",
            f"{result.f1:.3f}",
            f"{result.false_positive_rate:.3f}",
            f"{result.detections_by_family['rasputin-boney-m']}/7",
        )
        for strategy, result in metrics.items()
    ]
    mix = metrics["full-mix-chroma"]
    separated = metrics["sync-chroma-bass-other"]
    midi = metrics["midi-bass-other"]
    if separated.f1 > mix.f1:
        h2_result = (
            "The same chroma representation performs better after the pre-selected "
            "bass/other separation, which provisionally supports H2 on this corpus."
        )
    elif separated.f1 == mix.f1:
        h2_result = (
            "The same chroma representation is unchanged by the pre-selected "
            "bass/other separation, so this experiment does not support H2."
        )
    else:
        h2_result = (
            "The same chroma representation performs worse after the pre-selected "
            "bass/other separation, so H2 is not supported in this setting."
        )

    content = f"""# Stem chroma ablation

## Scope

This experiment isolates source separation more strictly than the preceding
full-mix control. The exact same spectral chroma extractor, 10-second windows,
three-window diagonal continuity, twelve transpositions, six tempo ratios,
folds, and threshold-selection rule are applied to the master and to each
Spleeter stem. The pre-selected primary stem condition synchronously averages
the pitch-invariant `bass` and `other` similarity matrices before selecting a
local three-window run.

The comparison still evaluates estimated sources rather than studio stems. It
measures the utility of this specific Spleeter preprocessing for this POC, not
source-reconstruction accuracy.

## Pairwise results

{markdown_table(
    ("Representation", "TP", "FP", "Precision", "Recall", "F1", "FPR", "Rasputin"),
    table_rows,
)}

## Interpretation

Full-mix chroma reaches F1 `{mix.f1:.3f}` with false-positive rate
`{mix.false_positive_rate:.3f}`. The pre-selected synchronized bass/other stem
chroma reaches F1 `{separated.f1:.3f}` with false-positive rate
`{separated.false_positive_rate:.3f}`. {h2_result}

The MIDI representation on the same bass/other stem families reaches F1
`{midi.f1:.3f}`. Its difference from stem chroma concerns the representation
and transcription stage rather than separation. Results remain exploratory:
the prior selection of bass/other and all thresholds were developed around the
same eight song families, and no untouched final test exists yet.

The complete local scores are stored in
`output/research/stem-chroma-ablation.csv`.
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> int:
    for dependency in (FULL_MIX_RESULTS_PATH, MIDI_RESULTS_PATH):
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
