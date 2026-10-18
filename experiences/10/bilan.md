# 10 — CQT, probabilités et MIDI

**A-t-on progressé ? Non pour le détecteur complet.**

## Question

Identifier l’apport du réseau Basic Pitch et du décodage en événements.

## Protocole

Même grille de 500 ms ; CQT, probabilités et MIDI repliés sur les mêmes classes de hauteur. Contrôle des attaques.

## Résultat et bilan

Sur stems : F1 0,473 → 0,680 → 0,716. Le MIDI conserve les 34 VP des probabilités et réduit les FP de 10 à 5. Cette agrégation diffère de la référence ; aucun nouveau meilleur pipeline n’est établi.

## Décision pour src/

Garder ces représentations et leurs utilitaires dans les expériences. Conserver la branche MIDI retenue auparavant.

Code retenu : `Aucun ajout.`

## Reproduction

```sh
uv run --script experiences/10/run.py
```

Les expériences 10, 11 et 12 partagent leur [protocole initial](../10/protocole.md) et des utilitaires figés dans `experiences/_shared/`. Les audios et caches volumineux restent dans `sandbox/`.

[Rapport détaillé](rapport.md) · [Journal des expériences](../README.md) · [Crédits et sources](../../CREDITS.md).
