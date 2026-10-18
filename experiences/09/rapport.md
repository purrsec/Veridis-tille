# Stem chroma ablation

## Scope

This experiment isolates source separation more strictly than the preceding
full-mix control. The exact same spectral chroma extractor, 10-second windows,
three-window diagonal continuity, twelve transpositions, six tempo ratios,
folds, and threshold-selection rule are applied to the master and to each
Spleeter stem. The pre-selected primary stem condition synchronously averages
the pitch-invariant `bass` and `other` similarity matrices before selecting a
local three-window run.

The comparison still evaluates estimated sources rather than studio stems. It
measures the utility of this specific Spleeter preprocessing for this POC, not
source-reconstruction accuracy.

## Pairwise results

| Representation | TP | FP | Precision | Recall | F1 | FPR | Rasputin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full-mix-chroma | 35/56 | 15/56 | 0.700 | 0.625 | 0.660 | 0.268 | 0/7 |
| stem-chroma-bass | 35/56 | 26/56 | 0.574 | 0.625 | 0.598 | 0.464 | 0/7 |
| stem-chroma-drums | 41/56 | 39/56 | 0.512 | 0.732 | 0.603 | 0.696 | 2/7 |
| stem-chroma-piano | 13/56 | 15/56 | 0.464 | 0.232 | 0.310 | 0.268 | 3/7 |
| stem-chroma-vocals | 22/56 | 22/56 | 0.500 | 0.393 | 0.440 | 0.393 | 2/7 |
| stem-chroma-other | 20/56 | 17/56 | 0.541 | 0.357 | 0.430 | 0.304 | 0/7 |
| sync-chroma-bass-other | 35/56 | 23/56 | 0.603 | 0.625 | 0.614 | 0.411 | 3/7 |
| sync-chroma-bass-vocals-other | 29/56 | 23/56 | 0.558 | 0.518 | 0.537 | 0.411 | 1/7 |
| sync-chroma-all-tonal | 33/56 | 30/56 | 0.524 | 0.589 | 0.555 | 0.536 | 5/7 |
| midi-bass-other | 40/56 | 7/56 | 0.851 | 0.714 | 0.777 | 0.125 | 6/7 |

## Interpretation

Full-mix chroma reaches F1 `0.660` with false-positive rate
`0.268`. The pre-selected synchronized bass/other stem
chroma reaches F1 `0.614` with false-positive rate
`0.411`. The same chroma representation performs worse after the pre-selected bass/other separation, so H2 is not supported in this setting.

The MIDI representation on the same bass/other stem families reaches F1
`0.777`. Its difference from stem chroma concerns the representation
and transcription stage rather than separation. Results remain exploratory:
the prior selection of bass/other and all thresholds were developed around the
same eight song families, and no untouched final test exists yet.

The complete local scores are stored in
`experiences/09/stem-chroma-ablation.csv`.
