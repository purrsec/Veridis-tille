# Full-mix chroma ablation experiment

## Scope

This experiment is the direct control required by the research hypothesis. It
extracts 12-bin chroma from each complete master WAV and does not use Spleeter
or Basic Pitch. The signal is downmixed to mono for analysis only. An 8192
sample Hann-windowed spectrum is computed every 2048 samples; spectral bins
from 65 Hz to 4.2 kHz are assigned linearly between adjacent pitch classes.
The resulting chroma is aggregated into 500 ms frames and normalized.

Ten-second query windows are compared with reference windows under all twelve
pitch transpositions and six tempo ratios. Three fixed scores are evaluated:
the best isolated window, and the best mean over two or three consecutive
diagonal windows. Requiring a diagonal run tests whether the evidence persists
over roughly 12 or 14 seconds instead of relying on one local maximum.

Thresholds and OR combinations are selected inside each training fold. The
evaluation uses the same 28 leave-two-families-out folds as the earlier MIDI
experiments.

## Pairwise results

| Classifier | TP | FP | Precision | Recall | F1 | FPR | Rasputin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stems-sync-bass-other | 40/56 | 7/56 | 0.851 | 0.714 | 0.777 | 0.125 | 6/7 |
| chroma-max | 35/56 | 15/56 | 0.700 | 0.625 | 0.660 | 0.268 | 1/7 |
| chroma-diagonal-2 | 37/56 | 16/56 | 0.698 | 0.661 | 0.679 | 0.286 | 0/7 |
| chroma-diagonal-3 | 35/56 | 15/56 | 0.700 | 0.625 | 0.660 | 0.268 | 0/7 |
| previous-OR-stems-vocals | 42/56 | 7/56 | 0.857 | 0.750 | 0.800 | 0.125 | 6/7 |
| OR-stems-chroma-diagonal-3 | 47/56 | 8/56 | 0.855 | 0.839 | 0.847 | 0.143 | 6/7 |
| OR-stems-vocals-chroma-diagonal-3 | 46/56 | 8/56 | 0.852 | 0.821 | 0.836 | 0.143 | 6/7 |

## Stability by family

| Song family | Chroma diag-3 | Previous candidate | Stems + chroma |
| --- | --- | --- | --- |
| big-in-japan-alphaville | 3/7 | 7/7 | 7/7 |
| brother-louie-modern-talking | 7/7 | 7/7 | 0/7 |
| cheri-cheri-lady-modern-talking | 7/7 | 7/7 | 7/7 |
| daddy-cool-boney-m | 0/7 | 0/7 | 6/7 |
| forever-young-alphaville | 4/7 | 7/7 | 7/7 |
| mambo-no-5-lou-bega | 7/7 | 2/7 | 7/7 |
| rasputin-boney-m | 0/7 | 6/7 | 6/7 |
| youre-my-heart-youre-my-soul-modern-talking | 7/7 | 6/7 | 7/7 |

## Interpretation

The best fixed full-mix chroma rule is `chroma-diagonal-2` with F1
`0.679`, precision `0.698`, recall
`0.661`, and false-positive rate
`0.286`. Its positive scores are highly
compressed and overlap the negative scores, so its threshold is fragile.
The synchronized stem/MIDI baseline remains stronger than the full-mix chroma control. This is not a strict test of H2 because the two branches change both
the signal input and the representation: full-mix spectral chroma is compared
with Spleeter followed by Basic Pitch MIDI. A chroma-on-mix versus
chroma-on-stems experiment is still required to isolate the effect of source
separation.

The previous stem/MIDI candidate has F1 `0.800`. The pre-specified
stems-plus-chroma OR has F1 `0.847`, precision
`0.855`, and recall `0.839`. Adding full-mix chroma improves the previous candidate under nested threshold selection. Adding
the vocal interval branch as a third OR reduces F1 to `0.836` and
is not retained.

These are exploratory repeated-fold counts over eight song families, not
independent test-set estimates. The experiment isolates the full-mix control,
but H2 can only be accepted or rejected after the same comparison on a larger
family-disjoint test set with hard negatives.

The complete local scores and candidate locations are stored in
`output/research/full-mix-chroma-experiment.csv`.
