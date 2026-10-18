# 01 — Réglage de Basic Pitch sur la basse

**A-t-on progressé ? Oui, provisoire.**

## Question

Rendre la transcription de basse moins sensible aux notes parasites.

## Protocole

Comparaison des seuils d’attaque, de trame, de durée et de fréquence sur Big in Japan. Écoute manuelle et décompte de notes.

## Résultat et bilan

Le seuil de trame 0,35 a été préféré à 0,30 sur cet exemple. Ce résultat manuel avec CoreML ne démontre pas un gain global.

## Décision pour src/

Conserver 0,35 uniquement pour la basse dans la préparation du POC.

Code retenu : `src/veridis_tille/_preprocess_worker.py`

## Reproduction

```sh
uv run --script experiences/01/run.py
```

Le script de comptage a été reconstruit lors du rangement du dépôt. Le bilan historique conserve les écoutes CoreML d’origine ; une exécution TensorFlow peut donner des nombres différents.

[Rapport détaillé](rapport.md) · [Journal des expériences](../README.md) · [Crédits et sources](../../CREDITS.md).
