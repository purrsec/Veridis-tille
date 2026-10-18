# Journal des expériences

Le code retenu vit dans [`src/`](../src/veridis_tille/). Chaque expérience
reste dans son dossier numéroté avec son `run.py`, ses mesures et son
`bilan.md`. Les décisions ne sont pas effacées quand une piste est abandonnée.

| N° | Expérience | Progrès | Décision actuelle |
| --- | --- | --- | --- |
| [01](01/bilan.md) | Réglage de Basic Pitch sur la basse | Oui, provisoire | Conserver 0,35 uniquement pour la basse dans la préparation du POC. |
| [02](02/bilan.md) | Vectorisation MIDI et témoin batterie | Oui, socle MIDI | Conserver les primitives MIDI ; laisser le prototype batterie dans cette expérience. |
| [03](03/bilan.md) | Pondération par la vélocité | Non | Conserver la pondération linéaire. Les autres modes restent dans le code figé des expériences. |
| [04](04/bilan.md) | Confiance par activité des notes | Non | Ne pas ajouter de pondération ni de porte d’activité au pipeline retenu. |
| [05](05/bilan.md) | Intervalles et contours mélodiques | Non | Conserver les classes de hauteur ; archiver les contours. |
| [06](06/bilan.md) | Fusion synchronisée des stems | Oui | Retenir la moyenne synchronisée bass + other. |
| [07](07/bilan.md) | Alignement de séquences vocales | Oui, puis remplacé | Archiver la branche vocale, remplacée dans la meilleure fusion actuelle. Un ancien gain ne justifie pas de conserver indéfiniment sa complexité. |
| [08](08/bilan.md) | Chroma du master et fusion OU | Oui — référence actuelle | Retenir les deux branches. Le portage est vérifié sur les 64 paires, avec les mêmes folds et les mêmes décisions. |
| [09](09/bilan.md) | Même chroma sur master et stems | Non | Conserver la chroma du master ; ne pas ajouter de chroma sur stems. |
| [10](10/bilan.md) | CQT, probabilités et MIDI | Non pour le détecteur complet | Garder ces représentations et leurs utilitaires dans les expériences. Conserver la branche MIDI retenue auparavant. |
| [11](11/bilan.md) | Effet de Spleeter avec Basic Pitch | Non pour le détecteur complet | Conserver les décisions précédentes ; ne pas remplacer le pipeline par des probabilités sur le master. |
| [12](12/bilan.md) | Robustesse aux transpositions et au tempo | Non — diagnostic | Documenter ces limites et garder les contrôles dans les expériences. Aucune amélioration du détecteur n’est revendiquée. |

## Référence à battre

La fusion de l’[expérience 08](08/bilan.md) : MIDI bass/other synchronisé OU
chroma continu du master. Sur les folds historiques : précision **85,5 %**,
rappel **83,9 %**, F1 **0,847**. Huit familles et 28 folds exploratoires ; les
56 décisions positives répètent huit paires. Aucun test final indépendant
n’a été réalisé. Les premiers classements de catalogue ne sont pas le critère
actuel de réussite.

## Règle de travail

1. Créer `experiences/13/`, puis les numéros suivants, sans modifier la référence.
2. Écrire la question, le protocole, la référence et le critère de réussite
   dans `bilan.md` avant de calculer les résultats.
3. Comparer sur les mêmes familles et le même protocole ; publier précision,
   rappel, F1, faux positifs et résultats par chanson. Une meilleure précision
   accompagnée de nombreux faux négatifs ne suffit pas.
4. **Progrès démontré dans le POC :** porter seulement la variante retenue dans
   `src/`, ajouter une vérification de non-régression et actualiser la référence.
5. **Pas de progrès :** garder les résultats et le code dans cette expérience,
   écrire pourquoi, puis essayer une autre piste. Ne pas ajouter la variante
   au code courant. Un diagnostic utile peut faire progresser la recherche
   sans améliorer le détecteur.
6. Quand une nouvelle combinaison remplace un ancien gain, retirer la branche
   devenue inutile du runtime et garder son histoire ici.

`_shared/` contient les implémentations historiques figées nécessaires pour
reproduire les expériences, notamment les options rejetées. Elles ne sont
jamais importées par `src/`. Cela évite qu’un changement du runtime réécrive
silencieusement la référence d’une ancienne expérience.

## Commandes

Les expériences 02 à 09 s’exécutent avec `uv run python experiences/NN/run.py`.
Les expériences 01, 10 et 12 utilisent `uv run --script` pour isoler les
versions Python 3.11 de Spleeter et Basic Pitch. L’expérience 11 réutilise les
mesures communes de 10. L’inférence n’a donc pas besoin d’être relancée pour
recalculer cette comparaison.

```sh
uv run --script experiences/_shared/plot_pitch.py
uv run pytest
uv run python experiences/08/verify_runtime.py
```

Les CSV d’origine ont été déplacés sans changer leurs octets. Le
[manifeste d’archive](archive-manifest.json) conserve leurs anciennes adresses
et empreintes, ainsi que celles des scripts et rapports avant adaptation des
chemins. Les empreintes du runner enregistrées dans les anciens environnements
décrivent l’exécution historique, avant ce rangement.

[Hypothèse et méthode](../docs/research/hypothese-de-base.pdf) · [Crédits](../CREDITS.md).
