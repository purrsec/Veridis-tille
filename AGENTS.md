# Dataset conventions

When adding music files to `data/dataset/`, name original recordings `<title>__<artist>__original.<extension>` and AI-generated recordings `<title>__ai-<provider>.<extension>`. Use lowercase ASCII only, replace spaces with hyphens, remove accents, keep extensions lowercase, and use normalized provider names such as `suno` or `udio`. Declare pairs only in `train.csv`, `validation.csv`, and `test.csv` with the columns `track_1,track_2,is_plagiarism`; write labels as lowercase `true` or `false`, and never place the same audio track in more than one split.
