# Veridis-tille

POC de comparaison de deux chansons : détecter une similarité musicale entre
une chanson à tester et un original proposé, puis localiser les passages.

## Utilisation

```sh
uv sync --dev
uv run veridis-tille compare chanson-a-tester.wav original.wav
```

La commande retourne du JSON : décision `is_similar`, branches qui ont
déclenché (`matched_by`), scores MIDI/chroma et positions dans les deux audios.
Les scores sont des similarités, pas des probabilités de plagiat.
Les masters doivent être des WAV PCM 16 bits ; le dataset du POC utilise
44,1 kHz et deux canaux.

Si les fichiers dérivés manquent, Spleeter et Basic Pitch sont exécutés dans
un environnement Python 3.11 isolé par `uv`. Le master WAV reste intact.
Les cinq stems et leurs MIDI/CSV sont écrits dans un seul dossier compagnon
plat, portant le même nom que le master.

```sh
# Préparer explicitement tous les fichiers dérivés.
uv run veridis-tille prepare chanson-a-tester.wav original.wav

# Comparer des fichiers déjà préparés sans lancer les modèles.
uv run veridis-tille compare chanson-a-tester.wav original.wav --no-prepare

# Fournir d'autres seuils de décision.
uv run veridis-tille compare chanson-a-tester.wav original.wav --calibration profil.json
```

Le profil JSON contient `name`, `midi_threshold` et `chroma_threshold`.
Le profil fourni par défaut est une calibration de développement sur les
64 paires du POC. Il n'a pas fait l'objet d'un test indépendant.

Pour le dataset complet :

```sh
uv run veridis-tille dataset prepare
uv run veridis-tille dataset validate
```

Ces commandes utilisent `./data/` : les masters et leurs analyses dans
`data/dataset/`, les trois CSV de paires directement dans `data/`.
Un autre répertoire peut être passé en argument, par exemple
`uv run veridis-tille dataset validate /chemin/vers/data`.
La préparation accepte `--force` pour recalculer les fichiers dérivés.

## Organisation

```text
src/veridis_tille/       code retenu et commande utilisable
  pipeline.py           fusion MIDI bass/other synchronisé OU chroma du master
  midi_similarity.py    notes, vélocité linéaire, attaques et fenêtres
  chroma.py             extraction et continuité sur trois fenêtres
  dataset.py            préparation et validation du dataset et de ses paires
  preprocessing.py     préparation des fichiers manquants
  _preprocess_worker.py Spleeter / Basic Pitch en Python 3.11
  profiles/poc.json     seuils explicites et provenance
experiences/
  01/ … 12/             run.py, résultats, rapport.md et bilan.md
  _shared/              outils de recherche et implémentations historiques figées
tests/                  tests du code retenu
data/dataset/           masters WAV et dossiers compagnons plats
docs/research/          hypothèse, méthode et PDF
sandbox/                caches et audios temporaires ignorés par Git
```

Les prototypes non retenus sont dans les expériences. Le runtime n'importe
aucun de leurs modules et ne lit aucun CSV de résultats pour prendre sa décision.

## Avancer par expériences

Le [journal des expériences](experiences/README.md) donne les douze bilans,
les gains retenus et les pistes écartées. Pour la suite : créer le prochain
dossier, fixer son protocole, comparer à la référence, puis décider :

- **Progrès :** intégrer la variante retenue dans `src/`, vérifier les résultats
  et actualiser la référence.
- **Pas de progrès :** conserver le bilan dans l'expérience et essayer une
  autre hypothèse, sans ajouter la variante au pipeline courant.

La référence actuelle vient de l'[expérience 08](experiences/08/bilan.md) :
précision **85,5 %**, rappel **83,9 %**, F1 **0,847** avec les seuils ajustés
dans chaque fold. Ces comptes répètent huit familles de chansons ; ils restent
exploratoires et ne constituent pas l'évaluation du profil fixe livré.
Le [contrôle du portage](experiences/08/runtime-verification.json) confirme que
le code de `src/` reproduit les scores et les résultats historiques.

## Vérifications

```sh
uv run pytest
uv run python experiences/08/verify_runtime.py
uv run veridis-tille dataset validate
```

Les tests du runtime sont dans `tests/`. Les tests des prototypes restent
avec leurs expériences et sont aussi exécutés par `pytest`.

[Hypothèse et méthode en PDF](docs/research/hypothese-de-base.pdf) ·
[Crédits des audios et logiciels](CREDITS.md).
