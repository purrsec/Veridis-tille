"""Fit a deployable POC profile on all development pairs, outside the runtime."""

from __future__ import annotations

import csv
from importlib import import_module
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
experiment = import_module("experiences.08.run")


def main() -> None:
    with (Path(__file__).parent / "full-mix-chroma-experiment.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    families, pairs = experiment.load_feature_scores(rows)
    values = list(pairs.values())
    labels = [bool(row["expected"]) for row in values]
    candidates = []
    for midi in experiment.threshold_candidates([r["stems"] for r in values]):
        for chroma in experiment.threshold_candidates([r["chroma-diagonal-3"] for r in values]):
            predictions = [r["stems"] >= midi or r["chroma-diagonal-3"] >= chroma for r in values]
            candidates.append((*experiment.classification_metric(labels, predictions), midi, chroma))
    *_, midi, chroma = max(candidates)
    profile = {
        "name": "poc-eight-families",
        "midi_threshold": midi,
        "chroma_threshold": chroma,
        "source_experiment": "08",
        "fitting_families": families,
        "fitting_pairs": len(values),
        "scope": "Development calibration on the complete POC corpus. This is not an independent test; the 0.847 F1 refers to fold-specific calibration, not this fixed profile.",
    }
    path = ROOT / "src" / "veridis_tille" / "profiles" / "poc.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(profile, indent=2) + "\n")
    print(f"Exported {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
