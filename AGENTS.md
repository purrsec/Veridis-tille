# Code and research workflow

Keep the retained pairwise detection pipeline in `src/veridis_tille/`.
Keep dataset preparation and validation in `src/veridis_tille/dataset.py`,
exposed through `veridis-tille dataset prepare` and `veridis-tille dataset validate`.
Do not recreate a general-purpose `scripts/` directory; place code with the
domain or numbered experiment it belongs to.
Put each research experiment in `experiences/NN/`, with its runnable code,
measurements, and `bilan.md` describing the question, protocol, baseline,
results, and explicit promotion/rejection decision. Use the next available
number; the current archive ends at 12.

Promote an experimental change into `src/` only when its comparison with the
current baseline demonstrates progress on the pairwise task. Check precision,
recall, false positives, F1, and results by family together. If there is no
progress, keep the variant in the experiment and try a different hypothesis.
Retire runtime branches superseded by a better combination, while preserving
their historical experiment. Runtime code must never import `experiences/`
or read experiment result CSVs. Preserve frozen historical helpers so that
future runtime edits do not silently change old experiments.

When promoting or refactoring retained code, run the tests and the relevant
numerical regression check. Experiment 08 is the present reference; its
verification command is `uv run python experiences/08/verify_runtime.py`.
Its historical cross-validation metrics must not be attributed to the fixed
development calibration profile without a separate evaluation.

# Dataset conventions

When adding music files to `data/dataset/`, name original recordings `<title>-<artist>-original.<extension>` and AI-generated recordings `<title>-<artist>-AI-<source>.<extension>`. Always include the artist in the filename. Write titles and artist names in lowercase ASCII, replace spaces with hyphens, remove accents and punctuation, and keep extensions lowercase. Preserve the uppercase `AI` marker and the agreed capitalization of the source name; use `GEMA` for AI samples supplied by GEMA. Declare pairs only in `train.csv`, `validation.csv`, and `test.csv` with the columns `track_1,track_2,is_plagiarism`; write labels as lowercase `true` or `false`, and never place the same audio track in more than one split.

Store analysis-ready audio in `data/dataset/` as metadata-free WAV files using 16-bit signed PCM, stereo, and a 44.1 kHz sample rate. Never apply lossy-to-lossy transcoding, loudness normalization, silence trimming, or other signal-altering processing during dataset sanitation. Run `uv run veridis-tille dataset validate` after changing dataset files or split CSVs.

For the POC analysis, keep each source WAV directly in `data/dataset/` and put its derived files in one companion directory with the same basename. Keep that directory flat: the five Spleeter WAV stems (`vocals`, `drums`, `bass`, `piano`, and `other`) and the Basic Pitch MIDI and note CSV for each stem all belong directly inside it. Do not create nested stem or Basic Pitch directories.
