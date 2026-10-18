"""Evaluate separation using the shared, already computed Basic Pitch inference."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiences._shared.pitch_ablation import count_metrics, markdown_table, read_csv, write_csv


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    source = ROOT / "experiences" / "10"
    output = Path(__file__).parent
    strategies = ("mix-probability", "stems-probability", "mix-midi", "stems-midi")
    scores = [r for r in read_csv(source / "pitch-ablation-scores.csv") if r["strategy"] in strategies]
    decisions = [r for r in read_csv(source / "pitch-ablation-decisions.csv") if r["strategy"] in strategies]
    write_csv(output / "separation-scores.csv", scores)
    write_csv(output / "separation-decisions.csv", decisions)
    table = []
    for strategy in strategies:
        rows = [r for r in decisions if r["strategy"] == strategy and r["policy"] == "balanced"]
        tp, fp, p, n, f1 = count_metrics(rows)
        table.append((strategy, f"{tp}/{p}", f"{fp}/{n}", f"{f1:.3f}"))
    content = """# Effet de Spleeter à représentation identique

Même inférence, grille temporelle, fenêtres, comparateur et folds que dans
l'[expérience 10](../10/rapport.md). Les MIDI utilisent le seuil de trame 0.30
partout, afin de ne pas confondre séparation et réglage propre à la basse.
Ce script extrait les mesures communes ; il ne relance pas l'inférence et
ne constitue pas un nouveau jeu d'évaluation indépendant.

""" + markdown_table(("Variante", "VP", "FP", "F1"), table) + """

La séparation améliore le MIDI dans ce contrôle, mais dégrade les probabilités
continues. Aucune de ces variantes ne remplace la fusion retenue dans
l'expérience 08. Les seuils sont ajustés seulement sur les familles
d'entraînement ; les comptes répètent huit correspondances positives sur
28 folds. La portée reste exploratoire.

Sources : [protocole commun](../10/protocole.md),
[crédits](../../CREDITS.md).
"""
    (output / "rapport.md").write_text(content)
    print(markdown_table(("Variante", "VP", "FP", "F1"), table))


if __name__ == "__main__":
    main()
