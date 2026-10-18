# 03 — Pondération par la vélocité

**A-t-on progressé ? Non.**

## Question

Vérifier si des poids binaires ou une compression améliorent les vecteurs.

## Protocole

Comparer linear, binary et sqrt avec toutes les autres options identiques.

## Résultat et bilan

Le binaire dégrade plusieurs stems et la racine carrée n’apporte pas de gain cohérent.

## Décision pour src/

Conserver la pondération linéaire. Les autres modes restent dans le code figé des expériences.

Code retenu : `src/veridis_tille/midi_similarity.py`

## Reproduction

```sh
uv run python experiences/03/run.py
```

[Rapport détaillé](rapport.md) · [Journal des expériences](../README.md) · [Crédits et sources](../../CREDITS.md).
