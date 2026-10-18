# 02 — Vectorisation MIDI et témoin batterie

**A-t-on progressé ? Oui, socle MIDI.**

## Question

Comparer des passages à partir des notes et de leurs attaques.

## Protocole

Fenêtres de 10 s, pas de 2 s, classes de hauteur, vélocité linéaire, douze transpositions et six rapports temporels. Le témoin batterie est testé séparément.

## Résultat et bilan

Les représentations MIDI constituent le socle des expériences suivantes. Le classement de catalogue historique est un diagnostic, pas notre objectif final. Aucun gain batterie n’est retenu.

## Décision pour src/

Conserver les primitives MIDI ; laisser le prototype batterie dans cette expérience.

Code retenu : `src/veridis_tille/midi_similarity.py`

## Reproduction

```sh
uv run python experiences/02/run.py
```

[Rapport détaillé](rapport.md) · [Journal des expériences](../README.md) · [Crédits et sources](../../CREDITS.md).
