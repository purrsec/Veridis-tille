# Découvrir les critères utiles de similarité musicale

## Objectif

Le modèle supervisé n'est pas le produit final. Il sert uniquement d'outil de
recherche pour déterminer quelles propriétés mesurables permettent réellement
de rapprocher deux passages musicaux.

Le système livré devra rester déterministe et explicable : il calculera des
critères explicites, produira un score de similarité et localisera les passages
correspondants. Une fois les critères et leurs interactions identifiés, le
modèle d'exploration pourra être écarté.

## Données nécessaires

Le corpus doit contenir plusieurs couples indépendants `(original, copie)` et,
pour chaque plagiat connu, les intervalles temporels correspondants dans les
deux morceaux. Il faut également inclure :

- des morceaux sans lien, pour les exemples négatifs simples ;
- des morceaux musicalement proches mais non copiés, pour les négatifs
  difficiles ;
- différentes transformations d'un même passage : décalage temporel,
  transposition, changement de tempo, réorchestration ou modification du
  rythme.

Les fichiers `train.csv`, `validation.csv` et `test.csv` restent réservés aux
couples de morceaux et à leur étiquette globale. Les bornes exactes des
passages devront être stockées dans des annotations séparées afin de ne pas
modifier leur schéma.

Le découpage doit se faire par morceau source complet, jamais par fragment. Un
même morceau, ou une variation dérivée de ce morceau, ne doit pas apparaître
dans plusieurs partitions. Cette règle évite que le modèle reconnaisse une
production plutôt qu'un critère général de copie.

## Unité d'analyse

Un exemple d'apprentissage est une paire de fenêtres temporelles candidates,
par exemple de 10 à 20 secondes. Pour chaque paire, le pipeline calcule un
vecteur de caractéristiques interprétables. Les fenêtres annotées comme
correspondantes sont positives ; les autres sont négatives.

Le vecteur doit exposer le plus de familles de critères raisonnables possible,
sans les fusionner prématurément :

- tempo et rapport de tempo ;
- positions des attaques et intervalles entre attaques ;
- densité et régularité rythmiques ;
- motifs de batterie indépendants de la hauteur ;
- hauteur absolue, classe de hauteur et intervalles mélodiques ;
- chroma, contour mélodique et profil harmonique ;
- ligne de basse et mouvements relatifs ;
- durée des notes, silences et articulations ;
- continuité d'un alignement entre fenêtres voisines ;
- coût des insertions, suppressions et substitutions d'événements.

Toutes les caractéristiques doivent être normalisées avant l'entraînement. La
vitesse de baisse de la loss ou la taille brute d'un gradient ne suffisent pas
pour juger l'importance d'un critère : elles dépendent aussi de son échelle et
de ses corrélations avec les autres entrées.

## Modèle d'exploration

La première référence doit être un modèle parcimonieux, par exemple une
régression logistique régularisée avec un poids par caractéristique. Il permet
de lire directement le sens et l'ordre de grandeur des contributions.

Un petit réseau de neurones peut ensuite tester les interactions non linéaires
entre critères. Un modèle d'arbres peut servir de contrôle indépendant. Ces
modèles consomment uniquement les caractéristiques explicites : l'objectif
n'est pas d'entraîner un détecteur opaque directement sur l'audio.

## Identifier les critères réellement utiles

La sélection ne repose pas sur une seule mesure. Un critère est retenu s'il
reste utile selon plusieurs tests :

1. **Ablation** : le retirer dégrade les résultats sur la validation.
2. **Permutation** : mélanger sa valeur dégrade les résultats.
3. **Stabilité** : son importance reste comparable entre plusieurs morceaux,
   folds et initialisations.
4. **Généralisation** : il améliore les résultats sur des morceaux sources
   entièrement inconnus pendant l'entraînement.
5. **Interaction** : sa combinaison avec un autre critère apporte un gain que
   chacun ne produit pas seul.

Une caractéristique très forte sur un seul couple mais instable ailleurs est
un signe d'overfitting, pas un critère à conserver.

## Distillation vers l'algorithme final

Les critères stables, leurs transformations et leurs interactions sont ensuite
traduits en une formule explicite, par exemple :

```text
score = w_rythme * similarité_rythmique
      + w_contour * similarité_du_contour
      + w_harmonie * similarité_harmonique
      + bonus_continuité * continuité_de_l_alignement
      - pénalité_édition * coût_des_modifications
```

Les poids, seuils et invariances retenus proviennent des expériences, mais le
calcul final ne dépend plus du modèle entraîné. La formule distillée doit être
réévaluée seule sur le jeu de test, sans réutiliser ce dernier pour la régler.

## Mesures de réussite

Deux problèmes doivent être évalués séparément :

- **Décision globale** : précision, rappel, F1 et courbe précision-rappel pour
  répondre « ces morceaux contiennent-ils une copie ? » ;
- **Localisation** : intersection sur union des intervalles, erreur sur les
  bornes temporelles et rappel des passages copiés pour répondre « où se trouve
  la copie dans chacun des morceaux ? ».

Il faut aussi suivre le taux de faux positifs sur les négatifs difficiles et la
stabilité des critères sélectionnés. Une bonne classification globale ne
suffit pas si la zone mise en évidence est incorrecte ou trop large.

## Plan expérimental

1. Constituer quelques couples précisément annotés, avec plusieurs morceaux
   sources et des négatifs difficiles.
2. Générer les fenêtres candidates et enregistrer toutes les caractéristiques
   sans poids manuel définitif.
3. Entraîner les modèles d'exploration sur `train`, choisir les critères avec
   `validation`, puis figer la méthode.
4. Distiller les résultats en score déterministe et en règle d'alignement.
5. Mesurer une seule fois la décision et la localisation finales sur `test`.

Cette démarche utilise donc l'apprentissage automatique comme microscope pour
comprendre le signal, et non comme composant obligatoire du détecteur final.
