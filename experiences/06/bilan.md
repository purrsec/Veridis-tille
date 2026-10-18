# 06 — Fusion synchronisée des stems

**A-t-on progressé ? Oui.**

## Question

Exiger que les stems rapprochent les mêmes passages aux mêmes tempos.

## Protocole

Comparer fusion tardive et fusion des cellules bass/other avant le maximum local.

## Résultat et bilan

La fusion synchronisée bass + other atteint 40/56 VP et 7/56 FP, F1 0,777 ; la moyenne tardive bass/other atteint 0,729.

## Décision pour src/

Retenir la moyenne synchronisée bass + other.

Code retenu : `src/veridis_tille/pipeline.py`

## Reproduction

```sh
uv run python experiences/06/run.py
```

[Rapport détaillé](rapport.md) · [Journal des expériences](../README.md) · [Crédits et sources](../../CREDITS.md).
