# 11 — Effet de Spleeter avec Basic Pitch

**A-t-on progressé ? Non pour le détecteur complet.**

## Question

Tester si séparer aide à représentation Basic Pitch identique.

## Protocole

Réutiliser l’inférence commune de 10 ; comparer master et bass/other sans changer le seuil MIDI de trame 0,30.

## Résultat et bilan

La séparation aide le MIDI (F1 0,589 → 0,716), mais dégrade les probabilités (0,810 → 0,680). Il n’existe pas de bénéfice universel de Spleeter.

## Décision pour src/

Conserver les décisions précédentes ; ne pas remplacer le pipeline par des probabilités sur le master.

Code retenu : `Aucun ajout.`

## Reproduction

```sh
uv run python experiences/11/run.py
```

Les expériences 10, 11 et 12 partagent leur [protocole initial](../10/protocole.md) et des utilitaires figés dans `experiences/_shared/`. Les audios et caches volumineux restent dans `sandbox/`.

[Rapport détaillé](rapport.md) · [Journal des expériences](../README.md) · [Crédits et sources](../../CREDITS.md).
