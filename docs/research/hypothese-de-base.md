---
title: "Détection paire-à-paire de réutilisation musicale dans des masters audio"
subtitle: "Note d’hypothèse, modèle de mesure et protocole de falsification"
author: "Projet Veridis-tille"
date: "Document de travail — 18 octobre 2026"
lang: fr-FR
---

<div class="status-line"><span>POC exploratoire</span><span>Version 3</span><span>Décision assistée</span></div>

## Résumé

Ce travail étudie une question technique limitée : **étant donnés deux
enregistrements audio finis, peut-on détecter et localiser une réutilisation
musicale entre eux ?** Le système ne recherche pas un titre dans un catalogue
et ne prononce pas un plagiat au sens juridique. Il calcule des indices de
similarité suffisamment précis pour aider un expert à examiner une paire
proposée : un morceau de référence et un morceau suspect.

L’intuition initiale doit être formulée avec prudence. Un master est le résultat
observable d’un mixage de sources musicales latentes. Une séparation de
sources telle que Spleeter ne restitue pas les pistes de studio : elle en
produit des **estimations**, susceptibles de contenir des fuites et des
artefacts. De même, une transcription automatique fournit des événements de
notes estimés, et non une partition vraie. La chaîne « séparation puis pitch »
est donc une méthode candidate. Son utilité doit être démontrée par comparaison
avec une méthode qui travaille directement sur le mix complet.

L’hypothèse principale est falsifiable : si deux enregistrements partagent un
passage musical issu d’une même matière source, des représentations robustes au
timbre, à la transposition et à des variations modérées de tempo devraient
produire une similarité locale plus élevée que pour des paires indépendantes,
y compris des négatifs musicalement proches. Une deuxième hypothèse teste si
la séparation en sources améliore réellement cette discrimination. Le
protocole exige des familles de chansons disjointes entre apprentissage,
validation et test, des négatifs difficiles, des ablations et un test final
conservé intact.

## 1. Objet scientifique et limites de l’inférence

### 1.1 Ce qui est observé

Le fichier distribué est un signal stéréo final, noté *x(t)*. On peut le
représenter schématiquement comme une combinaison de sources latentes
*s₁(t), …, sₖ(t)*, transformées par le mixage, les effets et le mastering :

<div class="equation">x(t) = M(s₁(t), …, sₖ(t)) + ε(t)</div>

La fonction *M* n’est ni connue ni, en général, inversible. Plusieurs ensembles
de sources peuvent expliquer le même mix. Une séparation calcule donc
*ŝₖ(t)*, une estimation conditionnée par son modèle et ses données
d’entraînement. Elle ne permet pas d’affirmer que *ŝₖ = sₖ*.

### 1.2 Ce que la méthode peut conclure

Pour deux fichiers *A* et *B*, la sortie recherchée est :

- un score de similarité paire-à-paire ;
- une décision au seuil, calibrée sur des données distinctes ;
- un ou plusieurs couples d’intervalles temporels qui expliquent le score ;
- les transformations estimées, par exemple transposition et rapport de tempo.

Cette sortie constitue un **indice technique de similarité musicale**. Le mot
« plagiat » désigne ici l’étiquette du jeu de données. Une qualification
juridique demande d’autres éléments : originalité du matériau, accès à
l’œuvre, importance qualitative de l’emprunt et contexte de création.

### 1.3 Ce que l’on sait sur les données génératives

Suno déclare entraîner ses modèles sur des dizaines de millions de fichiers
audio musicaux publics et leurs métadonnées, ainsi que sur certaines données
post-entraînement. Cette publication établit l’usage de fichiers audio, mais
elle ne documente ni leur statut exact de master, ni l’éventuelle présence de
stems, ni l’architecture et les représentations internes du modèle [1].

L’entraînement sur des fichiers audio complets **n’implique pas** qu’une sortie
générée contienne une piste source récupérable. Un modèle peut apprendre des
régularités générales, mémoriser certains fragments, les transformer, ou
combiner plusieurs influences. La persistance d’un motif identifiable dans une
sortie est précisément la question empirique ; elle ne doit pas servir de
prémisse à la démonstration.

## 2. Hypothèses falsifiables

<div class="hypothesis null">
<strong>H₀ — hypothèse nulle.</strong> À difficulté musicale comparable, les
scores des paires liées et indépendantes suivent la même distribution. Les
performances observées sur le petit corpus proviennent du hasard, d’artefacts
de génération, d’un réglage excessif ou de variables parasites.
</div>

<div class="hypothesis">
<strong>H₁ — persistance d’un motif local.</strong> Une paire liée contient au
moins un passage dont la structure de hauteurs et/ou d’attaques reste plus
similaire que celle de paires indépendantes, après compensation d’une
transposition et d’une variation modérée de tempo.
</div>

<div class="hypothesis">
<strong>H₂ — valeur ajoutée de la séparation.</strong> Les représentations
calculées sur des sources estimées améliorent la discrimination hors
échantillon par rapport aux mêmes familles d’indices calculées sur le mix
complet. Si elles n’apportent aucun gain stable, la séparation n’est pas
justifiée pour ce POC.
</div>

<div class="hypothesis">
<strong>H₃ — complémentarité.</strong> Les erreurs de la transcription MIDI et
des descripteurs audio continus ne sont pas totalement corrélées. Leur fusion,
réglée sans voir le test, améliore le compromis précision–rappel sans accroître
de manière disproportionnée les faux positifs.
</div>

Ces hypothèses ne prétendent pas déduire le mécanisme interne du générateur.
Elles testent uniquement une relation mesurable entre deux signaux de sortie.

## 3. Modèle de mesure

<div class="flow-diagram">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 570" role="img" aria-label="Processus complet : deux masters, branche chroma directe, branche Spleeter et Basic Pitch, fenêtrage, comparaison locale, invariances, calibration et verdict.">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L10,5 L0,10 z" fill="#607987"/></marker>
    <style>.t{font-family:Noto Sans,sans-serif}.h{font-size:15px;font-weight:700;fill:#173b57}.s{font-size:12px;fill:#405662}.l{font-size:12px;fill:#e7f0f4}.w{fill:#fff}.n{font-size:12px;font-weight:700;fill:#fff}</style>
  </defs>
  <circle cx="38" cy="35" r="17" fill="#176b68"/><text x="38" y="40" text-anchor="middle" class="t n">1</text>
  <rect x="70" y="8" width="390" height="56" rx="9" fill="#edf5f4" stroke="#2d8981" stroke-width="2"/>
  <rect x="620" y="8" width="390" height="56" rx="9" fill="#edf5f4" stroke="#2d8981" stroke-width="2"/>
  <text x="265" y="31" text-anchor="middle" class="t h">Master A — référence proposée</text><text x="265" y="51" text-anchor="middle" class="t s">WAV PCM 16 bits · stéréo · 44,1 kHz</text>
  <text x="815" y="31" text-anchor="middle" class="t h">Master B — morceau suspect</text><text x="815" y="51" text-anchor="middle" class="t s">WAV PCM 16 bits · stéréo · 44,1 kHz</text>
  <path d="M265 64 V77 H540" fill="none" stroke="#607987" stroke-width="2"/><path d="M815 64 V77 H540" fill="none" stroke="#607987" stroke-width="2"/>
  <path d="M540 77 H297 V91" fill="none" stroke="#607987" stroke-width="2" marker-end="url(#arrow)"/><path d="M540 77 H807 V91" fill="none" stroke="#607987" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="450" y="68" width="180" height="18" rx="8" fill="#fff"/><text x="540" y="81" text-anchor="middle" class="t" font-size="10" fill="#687781">traités séparément</text>
  <circle cx="38" cy="133" r="17" fill="#176b68"/><text x="38" y="138" text-anchor="middle" class="t n">2</text>
  <rect x="70" y="94" width="455" height="94" rx="9" fill="#f3f6f8" stroke="#9eafb9" stroke-width="2"/>
  <text x="297" y="119" text-anchor="middle" class="t h">A et B — branche directe : mix complet</text>
  <text x="297" y="142" text-anchor="middle" class="t s">mono d’analyse → STFT → énergie spectrale</text>
  <text x="297" y="162" text-anchor="middle" class="t s">→ chroma : 12 classes de hauteur par trame</text>
  <text x="297" y="181" text-anchor="middle" class="t" font-size="11" fill="#9a5f18">risque : tonalité ou accords banals</text>
  <circle cx="572" cy="133" r="17" fill="#176b68"/><text x="572" y="138" text-anchor="middle" class="t n">3</text>
  <rect x="605" y="94" width="405" height="94" rx="9" fill="#f3f6f8" stroke="#9eafb9" stroke-width="2"/>
  <text x="807" y="119" text-anchor="middle" class="t h">A et B — branche symbolique : sources estimées</text>
  <text x="807" y="142" text-anchor="middle" class="t s">Spleeter → voix · batterie · basse · piano · other</text>
  <text x="807" y="162" text-anchor="middle" class="t s">Basic Pitch → début · fin · hauteur · vélocité</text>
  <text x="807" y="181" text-anchor="middle" class="t" font-size="11" fill="#9a5f18">risques : fuites de stems + erreurs de notes</text>
  <path d="M297 188 V216" fill="none" stroke="#607987" stroke-width="2" marker-end="url(#arrow)"/><path d="M807 188 V216" fill="none" stroke="#607987" stroke-width="2" marker-end="url(#arrow)"/>
  <circle cx="38" cy="253" r="17" fill="#176b68"/><text x="38" y="258" text-anchor="middle" class="t n">4</text>
  <rect x="70" y="219" width="940" height="68" rx="9" fill="#edf5f4" stroke="#2d8981" stroke-width="2"/>
  <text x="540" y="244" text-anchor="middle" class="t h">Représentations temporelles comparables</text>
  <text x="540" y="266" text-anchor="middle" class="t s">fenêtres de 10 s · pas de 2 s · trames de 0,5 s · sustain/onset pondérés par vélocité · intervalles mélodiques</text>
  <path d="M540 287 V315" fill="none" stroke="#607987" stroke-width="2" marker-end="url(#arrow)"/>
  <circle cx="38" cy="351" r="17" fill="#176b68"/><text x="38" y="356" text-anchor="middle" class="t n">5</text>
  <rect x="70" y="318" width="455" height="75" rx="9" fill="#f3f6f8" stroke="#9eafb9" stroke-width="2"/>
  <text x="297" y="342" text-anchor="middle" class="t h">Comparaison locale A × B</text>
  <text x="297" y="363" text-anchor="middle" class="t s">une cellule = similarité entre deux fenêtres datées</text>
  <text x="297" y="382" text-anchor="middle" class="t s">matrices chroma, bass, other et voix</text>
  <rect x="605" y="318" width="405" height="75" rx="9" fill="#fff4dc" stroke="#d39a2c" stroke-width="2"/>
  <text x="807" y="342" text-anchor="middle" class="t h">Invariances testées</text>
  <text x="807" y="363" text-anchor="middle" class="t s">12 transpositions · tempo 0,80 à 1,25</text>
  <text x="807" y="382" text-anchor="middle" class="t s">continuité diagonale · alignement local</text>
  <path d="M525 355 H601" fill="none" stroke="#607987" stroke-width="2" marker-end="url(#arrow)"/><path d="M807 393 V421" fill="none" stroke="#607987" stroke-width="2" marker-end="url(#arrow)"/>
  <circle cx="38" cy="458" r="17" fill="#176b68"/><text x="38" y="463" text-anchor="middle" class="t n">6</text>
  <rect x="70" y="424" width="285" height="70" rx="9" fill="#173b57" stroke="#173b57" stroke-width="2"/>
  <text x="212" y="450" text-anchor="middle" class="t h w">Scores localisés</text><text x="212" y="472" text-anchor="middle" class="t l">mix chroma · bass/other · voix</text>
  <path d="M355 459 H395" fill="none" stroke="#607987" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="400" y="424" width="285" height="70" rx="9" fill="#173b57" stroke="#173b57" stroke-width="2"/>
  <text x="542" y="450" text-anchor="middle" class="t h w">Calibration imbriquée</text><text x="542" y="472" text-anchor="middle" class="t l">seuils appris sur les familles train</text>
  <path d="M685 459 H725" fill="none" stroke="#607987" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="730" y="424" width="280" height="70" rx="9" fill="#edf5f4" stroke="#2d8981" stroke-width="2"/>
  <text x="870" y="450" text-anchor="middle" class="t h">Verdict paire-à-paire</text><text x="870" y="472" text-anchor="middle" class="t s">score + passages + transformations</text>
  <circle cx="38" cy="535" r="17" fill="#176b68"/><text x="38" y="540" text-anchor="middle" class="t n">7</text>
  <rect x="70" y="512" width="940" height="46" rx="9" fill="#fff" stroke="#9eafb9" stroke-width="2"/>
  <text x="540" y="532" text-anchor="middle" class="t h">Évaluation par familles jamais vues dans le fold</text>
  <text x="540" y="550" text-anchor="middle" class="t s">précision · rappel · F1 · faux positifs · stabilité par chanson · localisation</text>
</svg>
</div>

### 3.1 Du fichier audio au verdict

**1 — Entrées et provenance.** Le système reçoit exactement deux masters : un
enregistrement de référence proposé et un enregistrement suspect. Ils sont
convertis sans modification musicale en WAV d’analyse homogènes. Les extraits
IA du corpus actuel proviennent de la page d’échantillons publiée par GEMA dans
le cadre de son action contre Suno [2]. Les liens vers les sources audio et les
logiciels sont consignés dans `CREDITS.md` [3].

**2 — Branche directe.** Chaque master est ramené en mono uniquement pour le
calcul. Une transformée de Fourier à court terme mesure son énergie spectrale,
puis les fréquences sont repliées dans douze classes de hauteur : do, do♯, …,
si, indépendamment de l’octave. On obtient un chromagramme, c’est-à-dire une
suite de douze valeurs au cours du temps. Cette branche conserve le contexte du
mix, mais peut confondre une reprise avec une tonalité ou des accords courants.

**3 — Branche symbolique.** Spleeter [4, 5] estime cinq familles de sources pour
chacun des deux masters. Basic Pitch [6, 7] transcrit chaque stem en événements
décrits par un début, une fin, une hauteur MIDI et une vélocité. La
vectorisation construit ensuite deux canaux par classe de hauteur : la note
tenue (*sustain*) et son attaque (*onset*), pondérées par la vélocité. Pour la
voix, une seconde représentation conserve l’ordre des intervalles entre notes.
Les erreurs de séparation et de transcription peuvent ici se cumuler.

**4 — Fenêtrage.** Les suites sont découpées en fenêtres locales de dix
secondes, espacées de deux secondes et décrites par des trames de 500 ms. Une
reprise courte peut ainsi être détectée sans exiger que les morceaux entiers
aient la même durée ou la même structure.

**5 — Matrice de comparaison.** Chaque fenêtre de A est comparée à chaque
fenêtre de B. Une cellule de la matrice contient donc un score et deux positions
temporelles précises. La comparaison est répétée pour les douze transpositions
et six rapports de tempo, de 0,80 à 1,25. Les maxima isolés sont fragiles ; le
pipeline favorise plusieurs cellules cohérentes sur une diagonale ou un
alignement local continu de notes.

**6 — Fusion et calibration.** Les branches produisent des scores distincts :
chroma du mix, similarité synchronisée `bass` + `other`, et intervalles vocaux.
Les seuils et la règle de fusion sont choisis uniquement sur les familles du
fold d’apprentissage. La règle candidate actuelle déclenche si le score MIDI
`bass` + `other` **ou** le chroma continu dépasse son seuil. Elle retourne aussi
les passages, la transposition et le rapport de tempo qui soutiennent le score.

**7 — Évaluation.** Deux familles complètes sont retirées à chaque fold. Les
seuils sont réglés sur les six autres, puis évalués sur les deux familles
inédites. Cela empêche qu’un même morceau serve simultanément au réglage et à la
mesure, sans rendre les 112 décisions indépendantes : le corpus ne contient
toujours que huit familles.

### 3.2 Représentations candidates

| Branche | Mesure | Invariance recherchée | Risque principal |
| --- | --- | --- | --- |
| Mix complet | chroma à 12 classes, énergie d’attaque | timbre, octave, transposition | accords banals et tonalité commune |
| Sources estimées | activité de notes, vélocité, intervalles | timbre, gain, transposition | erreurs Spleeter puis Basic Pitch |
| Alignement temporel | fenêtres locales, rapports de tempo | décalage et tempo modéré | maximum fort obtenu par hasard |
| Fusion | combinaison de scores complémentaires | type de transformation inconnu | surapprentissage des seuils |

Spleeter fournit des modèles pré-entraînés de séparation à deux, quatre ou cinq
stems. Son article décrit des masques spectraux estimés par U-Net et propose la
séparation comme prétraitement possible pour la transcription, l’extraction de
mélodie et la détection de reprises [5]. Basic Pitch est un modèle de
transcription polyphonique, indépendant de l’instrument, qui estime notamment
les attaques, le multipitch et les activations de notes [7]. Ces travaux
justifient la faisabilité technique des mesures ; ils ne prouvent pas H₂ sur
notre corpus.

Le chroma regroupe l’énergie spectrale dans les douze classes de hauteur en
repliant les octaves. Cette représentation a déjà été employée pour comparer
des versions d’un même morceau malgré des différences d’instrumentation et de
tonalité, avec alignement temporel et examen des douze transpositions [8]. Elle
constitue donc un contrôle direct sur le mix complet.

### 3.3 Variable cible

L’unité statistique est la **paire de morceaux**, et l’unité d’indépendance est
la **famille de chanson**. Une paire positive relie un original documenté à une
version dérivée. Une paire négative relie deux familles indépendantes. Les
évaluations répétées d’une même paire dans plusieurs folds ne sont pas huit
nouveaux morceaux : elles mesurent la stabilité du réglage.

## 4. Protocole expérimental

### 4.1 Constitution des données

1. Définir les paires et leur provenance avant le calcul des scores.
2. Grouper un original et toutes ses variantes dans une même famille.
3. Séparer les familles entre apprentissage, validation et test ; aucun fichier
   audio ne doit apparaître dans deux ensembles.
4. Inclure des négatifs faciles et des **négatifs difficiles** partageant genre,
   tonalité, tempo, progression harmonique, époque ou timbres.
5. Documenter, lorsque c’est possible, les intervalles réellement repris afin
   d’évaluer la localisation, pas seulement la décision globale.

Avec seulement huit familles, le protocole actuel écarte deux familles à la
fois et règle les paramètres sur les six restantes. Il convient à
l’exploration et à l’élimination d’idées faibles. Il ne remplace pas un jeu de
test indépendant.

### 4.2 Comparaisons nécessaires

| Expérience | Question isolée | Comparaison décisive |
| --- | --- | --- |
| A — mix complet | existe-t-il un signal sans séparation ? | chroma du mix contre H₀ |
| B — ablation Spleeter | la séparation apporte-t-elle de l’information ? | mix seul contre stems seuls, mêmes folds |
| C — ablation de transcription | le MIDI aide-t-il malgré ses erreurs ? | chroma de stems contre notes Basic Pitch |
| D — invariances | quel changement est réellement toléré ? | sans/avec transposition et rapports de tempo |
| E — fusion | les indices sont-ils complémentaires ? | meilleur indice seul contre combinaison réglée en interne |
| F — robustesse | le score capte-t-il autre chose que le style ? | négatifs difficiles et générateurs nouveaux |

Les paramètres — taille de fenêtre, seuils, pondérations et règle de fusion —
sont choisis uniquement sur le sous-ensemble d’apprentissage de chaque fold.
Une fois le protocole arrêté et le corpus agrandi, ces paramètres sont figés et
évalués une seule fois sur le test conservé à part.

### 4.3 Mesures et incertitude

La décision paire-à-paire est évaluée par la précision, le rappel, le F1, le
taux de faux positifs et l’aire sous la courbe précision–rappel. La localisation
demande une mesure séparée de recouvrement temporel. Les résultats doivent être
rapportés par famille, accompagnés d’intervalles de confiance par bootstrap au
niveau des familles lorsque le corpus sera assez grand.

Le classement dans un catalogue peut rester un diagnostic, mais il ne mesure
pas l’objectif principal. Être classé premier parmi huit candidats ne démontre
pas qu’une paire dépasse un seuil fiable face à des morceaux inconnus.

### 4.4 Critères de réfutation et de décision

- **Rejeter H₁** si, sur un test indépendant avec négatifs difficiles, les
  scores ne discriminent pas les classes au-delà de l’incertitude attendue.
- **Rejeter H₂** si les stems n’améliorent pas de façon stable le mix complet,
  ou si le gain disparaît sur de nouvelles familles ou un nouveau générateur.
- **Rejeter H₃** si la fusion choisie en validation accroît surtout les faux
  positifs ou si son gain ne se reproduit pas sur le test.
- **Réviser la représentation** si les passages les mieux notés ne recouvrent
  pas les passages annotés, même lorsque la décision globale est correcte.

## 5. État des observations du POC

Le corpus de travail reprend huit comparaisons documentées par GEMA, chacune
avec un original et un extrait généré [2, 3]. Les expériences ont testé une vectorisation MIDI
par fenêtres, la vélocité, l’activité, les contours, la fusion de stems et un
alignement local de séquences mélodiques.

Le meilleur candidat exploratoire accepte une paire si la fusion synchronisée
des stems `bass` et `other`, ou l’alignement exact des intervalles vocaux,
dépasse son seuil appris dans le fold. Agrégée sur les 28 choix de deux familles
tenues à l’écart, cette règle produit :

<div class="metrics"><div><strong>0,800</strong><span>F1</span></div><div><strong>0,857</strong><span>précision</span></div><div><strong>0,750</strong><span>rappel</span></div><div><strong>7 / 56</strong><span>faux positifs</span></div></div>

Ces 112 décisions ne sont pas 112 observations indépendantes : chaque paire
positive est réévaluée sept fois avec des ensembles d’apprentissage différents.
Le résultat décrit la stabilité du pipeline sur **huit familles seulement**.
Il est utile pour choisir la prochaine ablation, mais sa précision numérique ne
doit pas être présentée comme la performance attendue en production.

Les résultats par famille montrent aussi une forte hétérogénéité. La fusion
MIDI détecte Rasputin dans six folds sur sept ; l’ajout des intervalles vocaux
récupère Brother Louie, mais peut perdre Daddy Cool selon les seuils choisis.
Cette instabilité est compatible aussi bien avec un signal réel mais fragile
qu’avec un réglage trop adapté au corpus.

## 6. Contrôle réalisé sur le mix complet

Un chromagramme à douze classes a été calculé directement sur chaque WAV
master, sans Spleeter ni Basic Pitch. Des fenêtres de dix secondes ont été
comparées sous les douze transpositions et six rapports de tempo. Le score le
plus robuste de cette branche impose une continuité sur deux fenêtres voisines.

Utilisé seul, ce contrôle obtient un F1 exploratoire de **0,679**, contre
**0,777** pour la fusion MIDI synchronisée `bass` + `other`. Les scores chroma
sont fortement comprimés et se recouvrent entre paires positives et négatives :
le seuil est donc fragile. Cette comparaison ne permet pas encore d’attribuer
l’écart à Spleeter, car elle change simultanément le signal d’entrée et la
représentation — chroma spectral du mix d’un côté, stems puis MIDI de l’autre.
Une comparaison chroma-sur-mix contre chroma-sur-stems reste nécessaire pour
isoler l’effet de la séparation et tester strictement H₂.

Le chroma fournit néanmoins une information complémentaire. Une règle choisie
dans chaque fold, positive si la branche `bass` + `other` **ou** le chroma
continu dépasse son seuil, obtient :

<div class="metrics"><div><strong>0,847</strong><span>F1</span></div><div><strong>0,855</strong><span>précision</span></div><div><strong>0,839</strong><span>rappel</span></div><div><strong>8 / 56</strong><span>faux positifs</span></div></div>

Elle détecte Rasputin dans six folds sur sept. Le gain global vient surtout de
Mambo No. 5 et Daddy Cool, mais la règle perd Brother Louie dans ces folds.
L’ajout d’une troisième branche vocale réduit le F1 à 0,836 ; elle n’est donc
pas retenue. La fusion à deux branches devient le meilleur candidat du POC,
sans constituer une estimation finale de performance.

Une seconde ablation applique ensuite **le même chroma** au master et aux stems
Spleeter. Avec la même continuité de trois fenêtres, le mix obtient un F1 de
0,660 et un taux de faux positifs de 0,268. La fusion chroma synchronisée
`bass` + `other` tombe à 0,614 et 0,411. H₂ n’est donc pas soutenue pour cette
représentation : la séparation dégrade le chroma au lieu de l’améliorer. Le bon
résultat du pipeline MIDI dépend plutôt de la transcription symbolique, ou de
son interaction avec les stems, que de la séparation seule.

<div class="page-break"></div>

## 7. Conclusion opérationnelle

L’hypothèse défendable n’est pas que l’on peut « retrouver les pistes
d’origine » d’un master. Elle est que certaines relations musicales locales
peuvent rester mesurables entre deux enregistrements transformés, et que la
séparation de sources pourrait — ou non — rendre ces relations plus faciles à
détecter. Le protocole est construit pour départager ces possibilités par
ablation, sur des familles inédites et avec des négatifs difficiles.

À ce stade, le POC fournit un signal exploratoire, un meilleur candidat de
fusion et des localisations temporelles. Il montre aussi que Spleeter n’aide
pas toutes les représentations : son intérêt éventuel doit être établi pour
chaque mesure. Le POC ne permet encore ni d’attribuer l’origine exacte d’un
morceau généré, ni d’estimer une performance générale, ni de conclure
juridiquement à un plagiat. L’étape décisive suivante est un corpus plus large,
avec négatifs difficiles et test par familles conservé intact.

## Références

<div class="references" markdown="1">

1. Suno, « [Suno — CA AB 2013 Disclosure](https://help.suno.com/en/articles/9709569) », mise à jour du 1er janvier 2026.
2. GEMA, « [Audio samples: How Suno copies famous songs](https://www.gema.de/en/news/ai-and-music/ai-lawsuit/audio-samples-suno) », échantillons audio et leadsheets utilisés pour constituer le corpus du POC, consulté le 18 octobre 2026.
3. Projet Veridis-tille, « [CREDITS.md — Credits and sources](https://github.com/purrsec/Veridis-tille/blob/main/CREDITS.md) », provenance détaillée des fichiers audio et logiciels du projet.
4. Deezer Research, « [Spleeter — source separation library](https://github.com/deezer/spleeter) », logiciel libre de séparation utilisé pour produire les cinq stems.
5. R. Hennequin, A. Khlif, F. Voituret et M. Moussallam, « [Spleeter: a fast and efficient music source separation tool with pre-trained models](https://doi.org/10.21105/joss.02154) », *Journal of Open Source Software*, 5(50), 2154, 2020.
6. Spotify, « [Basic Pitch — Audio-to-MIDI converter](https://github.com/spotify/basic-pitch) », logiciel libre de transcription utilisé pour produire les événements MIDI et CSV.
7. R. M. Bittner et al., « [A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation](https://arxiv.org/abs/2203.09893) », *ICASSP*, 2022.
8. D. P. W. Ellis et G. E. Poliner, « [Identifying ‘Cover Songs’ with Chroma Features and Dynamic Programming Beat Tracking](https://doi.org/10.1109/ICASSP.2007.367348) », *ICASSP*, 2007.

</div>
