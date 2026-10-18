"""Verify that promoted code reproduces the historical 64 pairwise scores."""

from __future__ import annotations

import csv
from importlib import import_module
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from veridis_tille.pipeline import compare_features, load_features


def main() -> None:
    experiment = import_module("experiences.08.run")
    with (Path(__file__).parent / "full-mix-chroma-experiment.csv").open() as stream:
        original = list(csv.DictReader(stream))
    families, old_pairs = experiment.load_feature_scores(original)
    dataset = ROOT / "data" / "dataset"
    queries = {q: load_features(dataset / f"{q}.wav") for q in {q for q, _ in old_pairs}}
    references = {r: load_features(dataset / f"{r}.wav", reference=True) for r in {r for _, r in old_pairs}}
    pairs = {}
    errors = {"midi": 0.0, "chroma": 0.0}
    for (query, reference), old in old_pairs.items():
        result = compare_features(queries[query], references[reference])
        errors["midi"] = max(errors["midi"], abs(result.midi.score - old["stems"]))
        errors["chroma"] = max(errors["chroma"], abs(result.chroma.score - old["chroma-diagonal-3"]))
        pairs[query, reference] = {"expected": old["expected"], "stems": round(result.midi.score, 9),
                                  "chroma": round(result.chroma.score, 9)}
    assert max(errors.values()) < 1e-6, errors
    metrics = experiment.evaluate_or(families, pairs, ("stems", "chroma"))
    assert (metrics.true_positives, metrics.false_positives) == (47, 8)
    evidence = dict(pairs=len(pairs), max_absolute_score_error=errors,
                    true_positives=metrics.true_positives, false_positives=metrics.false_positives,
                    positive_evaluations=metrics.positive_evaluations,
                    negative_evaluations=metrics.negative_evaluations,
                    precision=metrics.precision, recall=metrics.recall, f1=metrics.f1,
                    note="Regression check with historical family folds; no new performance claim.")
    (Path(__file__).parent / "runtime-verification.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
