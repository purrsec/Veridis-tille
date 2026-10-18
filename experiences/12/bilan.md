# 12 — Robustesse aux transpositions et au tempo

**A-t-on progressé ? Non — diagnostic.**

## Question

Mesurer les limites du pipeline retenu avec des transformations connues.

## Protocole

18 conditions sur un extrait fixé de 20 s par original ; Spleeter et Basic Pitch réexécutés ; seuils appris sur les familles non transformées puis figés.

## Résultat et bilan

La fusion conserve les positifs sur les tempos testés. Les transpositions globales de ±4 demi-tons font perdre Brother Louie et You’re My Heart. Rasputin reste détecté dans ces contrôles. Des faux positifs subsistent.

## Décision pour src/

Documenter ces limites et garder les contrôles dans les expériences. Aucune amélioration du détecteur n’est revendiquée.

Code retenu : `Aucun ajout.`

## Reproduction

```sh
uv run --script experiences/12/run.py
```

Les expériences 10, 11 et 12 partagent leur [protocole initial](../10/protocole.md) et des utilitaires figés dans `experiences/_shared/`. Les audios et caches volumineux restent dans `sandbox/`.

[Rapport détaillé](rapport.md) · [Journal des expériences](../README.md) · [Crédits et sources](../../CREDITS.md).
