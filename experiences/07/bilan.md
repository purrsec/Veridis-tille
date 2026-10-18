# 07 — Alignement de séquences vocales

**A-t-on progressé ? Oui, puis remplacé.**

## Question

Compléter la fusion instrumentale avec un alignement mélodique local.

## Protocole

Alignement Smith–Waterman d’intervalles et règle OU avec le score bass/other.

## Résultat et bilan

Gain intermédiaire de F1 0,777 à 0,800 : 42/56 VP et 7/56 FP. L’expérience 08 le dépasse ; ajouter ensuite cette troisième branche dégrade le F1 de 0,847 à 0,836.

## Décision pour src/

Archiver la branche vocale, remplacée dans la meilleure fusion actuelle. Un ancien gain ne justifie pas de conserver indéfiniment sa complexité.

Code retenu : `Aucun dans le pipeline actuel.`

## Reproduction

```sh
uv run python experiences/07/run.py
```

[Rapport détaillé](rapport.md) · [Journal des expériences](../README.md) · [Crédits et sources](../../CREDITS.md).
