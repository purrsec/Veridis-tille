# 09 — Même chroma sur master et stems

**A-t-on progressé ? Non.**

## Question

Isoler l’effet de la séparation, sans changer la représentation.

## Protocole

Même extracteur chroma, mêmes fenêtres et même évaluation pour le mix et les stems.

## Résultat et bilan

F1 0,660 sur le mix contre 0,614 sur bass + other séparés. La séparation dégrade ce contrôle.

## Décision pour src/

Conserver la chroma du master ; ne pas ajouter de chroma sur stems.

Code retenu : `Aucun ajout.`

## Reproduction

```sh
uv run python experiences/09/run.py
```

[Rapport détaillé](rapport.md) · [Journal des expériences](../README.md) · [Crédits et sources](../../CREDITS.md).
