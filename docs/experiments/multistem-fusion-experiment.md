# Multi-stem fusion experiment

## Scope

This experiment combines the bass, piano, vocal, and other MIDI similarity
matrices in two ways:

- late fusion averages each stem's best score even when the evidence comes
  from different source times or tempo ratios;
- synchronized fusion combines matrix cells before taking the maximum, so all
  contributing stems must support the same query window, reference window,
  and tempo ratio.

All stems use one common duration read from the source WAV. This gives their
window matrices identical time axes. Each stem may still select its own pitch
transposition at a synchronized cell.

## Primary result: pairwise plagiarism decision

The target task receives one AI recording and one proposed original, then
returns plagiarism or no plagiarism without consulting a catalog. Thresholds
are evaluated with leave-two-families-out cross-validation. Each fold removes
two complete song families, including both their AI and original tracks, fits
the threshold on the other six families, then tests the four AI/original
combinations between the held-out families: two positives and two negatives.

Across all 28 possible held-out family pairs this produces 56 positive and 56
negative evaluations. Each positive song pair is evaluated seven times under
different training folds; each directed negative pair is evaluated once.

| Strategy | TP | FP | Precision | Recall | F1 | Rasputin folds | Threshold range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| late-mean-all | 19/56 | 14/56 | 0.576 | 0.339 | 0.427 | 0/7 | 0.595-0.782 |
| late-mean-bass-vocals-other | 34/56 | 14/56 | 0.708 | 0.607 | 0.654 | 0/7 | 0.622-0.700 |
| late-mean-bass-other | 35/56 | 5/56 | 0.875 | 0.625 | 0.729 | 0/7 | 0.689-0.730 |
| sync-mean-all | 27/56 | 5/56 | 0.844 | 0.482 | 0.614 | 0/7 | 0.495-0.532 |
| sync-mean-bass-vocals-other | 27/56 | 4/56 | 0.871 | 0.482 | 0.621 | 0/7 | 0.485-0.589 |
| sync-mean-bass-other | 40/56 | 7/56 | 0.851 | 0.714 | 0.777 | 6/7 | 0.623-0.663 |
| sync-top2-bass-vocals-other | 35/56 | 11/56 | 0.761 | 0.625 | 0.686 | 0/7 | 0.584-0.665 |
| sync-second-bass-vocals-other | 33/56 | 7/56 | 0.825 | 0.589 | 0.688 | 0/7 | 0.594-0.658 |
| sync-geometric-bass-vocals-other | 26/56 | 9/56 | 0.743 | 0.464 | 0.571 | 0/7 | 0.467-0.564 |

## Secondary diagnostic: catalog retrieval

For each of the eight AI excerpts, the experiment compares it with all eight
original recordings and sorts the originals by decreasing similarity. A
`7/8` result means that the corresponding original is ranked first for seven
AI excerpts out of eight. This measures source retrieval in a known catalog;
it does not mean that seven pairs have passed a plagiarism threshold.

| Strategy | Ranks first | Mean margin | Expected median | Other median |
| --- | --- | --- | --- | --- |
| late-mean-all | 7/8 | +0.021 | 0.657 | 0.635 |
| late-mean-bass-vocals-other | 6/8 | +0.043 | 0.694 | 0.593 |
| late-mean-bass-other | 6/8 | +0.015 | 0.731 | 0.616 |
| sync-mean-all | 4/8 | +0.002 | 0.516 | 0.449 |
| sync-mean-bass-vocals-other | 6/8 | +0.024 | 0.571 | 0.479 |
| sync-mean-bass-other | 3/8 | +0.004 | 0.655 | 0.560 |
| sync-top2-bass-vocals-other | 4/8 | +0.020 | 0.669 | 0.567 |
| sync-second-bass-vocals-other | 4/8 | -0.006 | 0.653 | 0.528 |
| sync-geometric-bass-vocals-other | 5/8 | +0.019 | 0.542 | 0.463 |

## Late mean of all stems by query

| Query | Expected rank | Score | Margin | Strongest alternative |
| --- | --- | --- | --- | --- |
| big-in-japan-alphaville | 1/8 | 0.778 | +0.079 | forever-young-alphaville |
| brother-louie-modern-talking | 8/8 | 0.598 | -0.125 | rasputin-boney-m |
| cheri-cheri-lady-modern-talking | 1/8 | 0.622 | +0.005 | mambo-no-5-lou-bega |
| daddy-cool-boney-m | 1/8 | 0.820 | +0.038 | big-in-japan-alphaville |
| forever-young-alphaville | 1/8 | 0.826 | +0.134 | youre-my-heart-youre-my-soul-modern-talking |
| mambo-no-5-lou-bega | 1/8 | 0.629 | +0.012 | big-in-japan-alphaville |
| rasputin-boney-m | 1/8 | 0.639 | +0.001 | youre-my-heart-youre-my-soul-modern-talking |
| youre-my-heart-youre-my-soul-modern-talking | 1/8 | 0.675 | +0.022 | cheri-cheri-lady-modern-talking |

## Rasputin

| Strategy | Expected rank | Score | Margin |
| --- | --- | --- | --- |
| late-mean-all | 1/8 | 0.639 | +0.001 |
| late-mean-bass-vocals-other | 2/8 | 0.623 | -0.027 |
| late-mean-bass-other | 1/8 | 0.706 | +0.003 |
| sync-mean-all | 4/8 | 0.455 | -0.039 |
| sync-mean-bass-vocals-other | 5/8 | 0.489 | -0.044 |
| sync-mean-bass-other | 2/8 | 0.638 | -0.036 |
| sync-top2-bass-vocals-other | 2/8 | 0.638 | -0.036 |
| sync-second-bass-vocals-other | 3/8 | 0.599 | -0.059 |
| sync-geometric-bass-vocals-other | 4/8 | 0.469 | -0.056 |

## Conclusion

For the intended pairwise task, `sync-mean-bass-other` is the strongest rule.
It detects 40/56 positive evaluations with 7/56 false positives, giving
precision `0.851`, recall `0.714`, and F1 `0.777`. Rasputin is detected in six
of its seven held-out folds. Requiring bass and other to support the same time
region is therefore useful for a binary decision even though it is weaker for
catalog ranking.

The result is promising but incomplete. Brother Louie and Mambo No. 5 are
never detected in their seven folds, and one in eight negative evaluations is
still a false alert. The official train, validation, and test CSV files are
also empty, so these 28 cross-validation folds remain exploratory rather than
an immutable final test.

Catalog retrieval is retained only as a diagnostic table. It is not a success
criterion for the first research milestone. Future experiments should be
selected by pairwise precision, recall, F1, false-positive rate, and stability
across song-family folds.

The complete results are stored in
`output/research/multistem-fusion-experiment.csv`.
