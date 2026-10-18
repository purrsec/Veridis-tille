# 08 — Chroma du master et fusion OU

**A-t-on progressé ? Oui — référence actuelle.**

## Question

Ajouter une information qui ne dépend pas de la transcription MIDI.

## Protocole

Chroma directement sur le master, continuité sur trois fenêtres, décision positive si MIDI synchronisé OU chroma dépasse son seuil. Calibration par famille.

## Résultat et bilan

47/56 VP, 8/56 FP : précision 0,855, rappel 0,839, F1 0,847, contre 0,800 pour le candidat précédent. Huit familles seulement.

## Décision pour src/

Retenir les deux branches. Le portage est vérifié sur les 64 paires, avec les mêmes folds et les mêmes décisions.

Code retenu : `src/veridis_tille/chroma.py ; src/veridis_tille/pipeline.py`

## Reproduction

```sh
uv run python experiences/08/run.py
```

Le [contrôle de portage](runtime-verification.json) mesure un écart de score
inférieur à 5 × 10⁻¹⁰ sur les 64 paires et retrouve les comptes 47/56 et 8/56.
Le profil fixe livré dans `src/veridis_tille/profiles/poc.json` est ajusté
sur les 64 paires de développement. Le F1 0,847 correspond aux seuils propres
aux folds historiques, pas à une validation indépendante de ce profil fixe.

```sh
uv run python experiences/08/verify_runtime.py
uv run python experiences/08/export_calibration.py
```

[Rapport détaillé](rapport.md) · [Journal des expériences](../README.md) · [Crédits et sources](../../CREDITS.md).
