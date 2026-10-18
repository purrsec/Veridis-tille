# Effet de Spleeter à représentation identique

Même inférence, grille temporelle, fenêtres, comparateur et folds que dans
l'[expérience 10](../10/rapport.md). Les MIDI utilisent le seuil de trame 0.30
partout, afin de ne pas confondre séparation et réglage propre à la basse.
Ce script extrait les mesures communes ; il ne relance pas l'inférence et
ne constitue pas un nouveau jeu d'évaluation indépendant.

| Variante | VP | FP | F1 |
| --- | --- | --- | --- |
| mix-probability | 47/56 | 13/56 | 0.810 |
| stems-probability | 34/56 | 10/56 | 0.680 |
| mix-midi | 28/56 | 11/56 | 0.589 |
| stems-midi | 34/56 | 5/56 | 0.716 |

La séparation améliore le MIDI dans ce contrôle, mais dégrade les probabilités
continues. Aucune de ces variantes ne remplace la fusion retenue dans
l'expérience 08. Les seuils sont ajustés seulement sur les familles
d'entraînement ; les comptes répètent huit correspondances positives sur
28 folds. La portée reste exploratoire.

Sources : [protocole commun](../10/protocole.md),
[crédits](../../CREDITS.md).
