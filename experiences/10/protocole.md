# Protocole : représentation, séparation et robustesse

Protocole fixé avant de calculer les nouveaux scores, le 18 octobre 2026.
Ces expériences prolongent le POC de comparaison de deux chansons. Aucun
réentraînement de Basic Pitch ou de Spleeter n'est prévu.

## 1. Représentation

Sur les mêmes stems Spleeter `bass` et `other`, comparer :

- la CQT logarithmique calculée par le frontal de Basic Pitch, avant le réseau ;
- la sortie continue `note` du modèle, avant décodage ;
- les événements MIDI décodés à partir de cette même inférence.

La CQT utilise les mêmes fenêtres audio, rééchantillonnage, hop, noyaux et
normalisation logarithmique que Basic Pitch. Ses trois bandes par demi-ton
sont moyennées ; seules les 88 hauteurs de sortie du réseau sont conservées.
Pour chaque représentation, les octaves sont regroupées par maximum, puis
l'activité est moyennée sur la même grille de 500 ms. Les événements MIDI
sont échantillonnés aux mêmes instants que les sorties du modèle, avec leur
vélocité divisée par 127. Les vecteurs sont comparés par cosinus, sans
normalisation spécifique à une seule représentation.

Le contrôle secondaire ajoute le canal d'attaques : sortie `onset` continue
pour le réseau, impulsion pondérée par la vélocité pour le MIDI. Ce contrôle
ne sert pas à attribuer tout le gain au seul réseau, car les deux canaux
d'attaques ont des distributions différentes.

## 2. Séparation

Appliquer les mêmes représentations Basic Pitch au master, et aux stems
`bass` + `other`. Comparer chaque représentation à elle-même. Garder le
réseau, le rééchantillonnage et le calcul de similarité identiques.

Le contrôle principal décode tous les MIDI avec le même seuil de trame
`0.30`. Un contrôle secondaire applique `0.35` à la basse comme dans le POC
historique. Cela distingue la séparation du réglage spécifique à la basse.
Les stems restent des estimations Spleeter, pas des pistes studio.

## Comparaison et évaluation communes

- Fenêtres de 10 s, pas de 2 s, 20 trames par fenêtre.
- Douze transpositions et rapports temporels historiques :
  `0.8, 0.9, 1.0, 1.125, 1.2, 1.25`.
- Fusion synchronisée : moyenne des matrices `bass` et `other`, puis
  maximum local, comme dans la baseline MIDI. Chaque stem peut choisir sa
  transposition comme dans cette baseline ; cette liberté reste une limite.
- Pas de modification du score d'alignement pendant cette expérience.
- Les huit familles donnent 64 paires uniques : 8 correspondances attendues,
  56 paires entre morceaux différents.
- Conserver les 28 folds laissant deux familles de côté ; le seuil de chaque
  représentation maximise l'exactitude équilibrée sur les six familles
  d'entraînement, avec les mêmes règles de départage que l'expérience chroma.
- Publier les décisions par fold, les seuils, faux positifs, faux négatifs,
  F1 et résultats par famille. Les 56 décisions positives répètent huit
  paires ; elles ne sont pas 56 exemples indépendants.
- Publier aussi le rappel avec un seuil choisi pour un FPR d'entraînement
  inférieur ou égal à 10 %. Le FPR tenu à l'écart peut dépasser cette cible.
- Aucune variante ne sera déclarée validée à partir d'un petit écart de F1.

## 3. Contrôles de robustesse

Prendre un extrait de 20 s à 30 s dans chaque original, sans choisir les
positions selon les scores. Comparer chaque extrait aux huit masters.
Créer les conditions suivantes avec Rubber Band :

- extrait inchangé ;
- somme des cinq stems existants, sans transformation ;
- transposition globale : -4, -2, -1, +1, +2, +4 demi-tons ;
- transposition de la voix seule aux mêmes valeurs, autres stems inchangés ;
- tempo global : 0.8, 0.9, 1.1, 1.2, hauteur conservée.

Le contrôle voix est comparé à la somme inchangée des stems, afin de séparer
l'effet du remixage de celui de la transposition. Aucun ajustement des gains,
normalisation ou écrêtage volontaire ; stockage intermédiaire en WAV flottant.
Ces sons restent dans `sandbox/`, en dehors du dataset de référence.

Chaque extrait transformé repasse dans Spleeter puis Basic Pitch : on ne
transpose pas simplement les vecteurs finaux. Évaluer la branche MIDI
historique synchronisée, la chroma du master et leur règle OR historique.
Les seuils sont appris uniquement sur les paires GEMA non transformées des
familles d'entraînement, puis figés pour toutes les transformations des
familles laissées de côté. Les deux familles exclues restent entièrement
absentes de la calibration ; toutes leurs variantes sont exclues ensemble.

Publier rappel, FPR et différence de score contre le témoin correspondant,
par transformation et par famille, dont Rasputin. Une transformation qui
fait baisser le rappel ou monter le FPR est un échec de robustesse. Les
transformations synthétiques ne constituent pas de nouvelles chansons
indépendantes et ne prouvent pas la détection d'une génération IA.

## Reproductibilité et sources

Les scores, décisions et rapports initialement écrits dans `output/research/` et
`docs/experiments/` ont été regroupés dans `experiences/10`, `11` et `12`.
Le protocole numérique est conservé ; caches et audios restent dans `sandbox/`.
Les sources audio et leurs crédits sont détaillés dans [CREDITS.md](../../CREDITS.md).

- [Basic Pitch : code et sorties brutes](https://github.com/spotify/basic-pitch).
- [Bittner et al., 2022, article Basic Pitch](https://arxiv.org/abs/2203.09893).
- [Spleeter](https://github.com/deezer/spleeter).
- [Rubber Band](https://breakfastquay.com/rubberband/).
- [Corpus d'extraits GEMA](https://www.gema.de/en/news/ai-and-music/ai-lawsuit/audio-samples-suno).
