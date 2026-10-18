# 04 — Confiance par activité des notes

**A-t-on progressé ? Non.**

## Question

Réduire les faux rapprochements issus de fenêtres peu actives.

## Protocole

Seuils de densité et facteurs de confiance sur les mêmes fenêtres MIDI.

## Résultat et bilan

Les pondérations dégradent notamment bass et other. Les portes sur le piano retirent surtout des fenêtres sans résoudre la discrimination.

## Décision pour src/

Ne pas ajouter de pondération ni de porte d’activité au pipeline retenu.

Code retenu : `Aucun ajout.`

## Reproduction

```sh
uv run python experiences/04/run.py
```

[Rapport détaillé](rapport.md) · [Journal des expériences](../README.md) · [Crédits et sources](../../CREDITS.md).
