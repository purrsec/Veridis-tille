# /// script
# requires-python = ">=3.13"
# dependencies = ["matplotlib==3.10.7"]
# ///
"""Create standalone figures from the ablation and robustness CSV decisions."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPRESENTATION = ROOT / "experiences" / "10"
ROBUSTNESS = ROOT / "experiences" / "12"
COLORS = ("#2468a2", "#d58b16", "#288b63")


def read(name):
    directory = ROBUSTNESS if "robustness" in name else REPRESENTATION
    with (directory / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    directory = ROBUSTNESS if "transformation" in name else REPRESENTATION
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ablation():
    rows = read("pitch-ablation-decisions.csv")
    strategies = ("stems-cqt", "stems-probability", "stems-midi", "mix-probability", "mix-midi")
    labels = ("CQT / stems", "Probas / stems", "MIDI / stems", "Probas / master", "MIDI / master")
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    positions = np.arange(len(strategies))
    for offset, positive, label, color in ((-0.18, True, "Rappel", COLORS[0]),
                                           (0.18, False, "Faux positifs", COLORS[1])):
        values = []
        for strategy in strategies:
            selected = [r for r in rows if r["strategy"] == strategy and
                        r["policy"] == "balanced" and (r["expected_pair"] == "true") == positive]
            values.append(sum(r["predicted"] == "true" for r in selected) / len(selected))
        bars = ax.bar(positions + offset, values, width=0.34, label=label, color=color)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in values], fontsize=9, padding=3)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1.13)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_title("Représentation et séparation — seuils calibrés par famille")
    ax.legend(loc="upper right", ncols=2)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.supxlabel("8 familles ; 28 folds ; les décisions répétées ne sont pas indépendantes", fontsize=9)
    save(fig, "pitch-representation-ablation")


def plot_robustness():
    rows = read("pitch-robustness-summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row", layout="constrained")
    for column, (group, title) in enumerate((("global-pitch", "Transposition globale"),
                                            ("vocal-pitch", "Transposition de la voix"),
                                            ("tempo", "Changement de tempo"))):
        control = "reconstructed" if group == "vocal-pitch" else "unchanged"
        for branch, color in zip(("midi", "chroma", "OR"), COLORS):
            selected = [r for r in rows if r["branch"] == branch and
                        (r["group"] == group or r["condition"] == control)]
            def coordinate(row):
                return (1.0 if group == "tempo" else 0.0) if row["condition"] == control else float(row["value"])
            selected.sort(key=coordinate)
            x = [coordinate(r) for r in selected]
            for line, metric in enumerate(("recall", "fpr")):
                axes[line, column].plot(x, [float(r[metric]) for r in selected],
                                       marker="o", label=branch, color=color)
        axes[0, column].set_title(title)
        axes[1, column].set_xlabel("Facteur de vitesse" if group == "tempo" else "Demi-tons")
        for ax in axes[:, column]:
            ax.grid(alpha=0.25)
            ax.set_ylim(-0.04, 1.04)
            ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    axes[0, 0].set_ylabel("Rappel")
    axes[1, 0].set_ylabel("Taux de faux positifs")
    axes[0, 2].legend()
    fig.suptitle("Robustesse — seuils GEMA figés, pipeline complet réexécuté")
    fig.supxlabel("8 familles, 1 extrait de 20 s par original ; aucune estimation d'incertitude indépendante", fontsize=9)
    save(fig, "pitch-transformation-robustness")


if __name__ == "__main__":
    plot_ablation()
    plot_robustness()
