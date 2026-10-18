# Vectorization baseline on the POC dataset

## Scope

This baseline compares each of the 8 AI excerpts against each of the 8 full
original recordings. It produces 64 pairs per feature: 8 expected pairs and
56 different-song pairs. All comparisons use the unchanged default parameters
from the passage-similarity prototypes.

The MIDI vectors are evaluated separately for bass, piano, vocals, and other.
Drums use the pitch-free onset and tempo representation. No combined score is
computed at this stage.

## Summary

| Feature | Expected median | Other median | Expected detected | Other detected | Expected ranks first | Mean rank margin |
| --- | --- | --- | --- | --- | --- | --- |
| bass | 0.730 | 0.613 | 3/8 | 2/56 | 5/8 | +0.011 |
| piano | 0.707 | 0.708 | 1/8 | 8/56 | 0/8 | -0.100 |
| vocals | 0.587 | 0.509 | 0/8 | 0/56 | 5/8 | +0.026 |
| other | 0.681 | 0.614 | 1/8 | 0/56 | 4/8 | -0.026 |
| drums | 0.906 | 0.902 | 7/8 | 51/56 | 1/8 | -0.023 |

`Expected detected` and `Other detected` apply the current prototype thresholds.
`Expected ranks first` checks whether the correct original receives the highest
score for each AI query. The mean rank margin is the expected-pair score minus
the strongest different-song score.

## Initial reading

- Bass is the strongest current MIDI representation. It ranks the expected
  original first for 5 of 8 excerpts, with 2 false positives
  among the 56 different-song pairs.
- Vocals do not cross the current decision threshold, but still rank the
  expected original first for 5 of 8 excerpts. This points
  to useful ordering information with an unsuitable absolute threshold.
- Piano produces 8 false positives. The worst cases come from
  extremely sparse AI piano transcriptions: after normalization and free
  transposition, a window containing only one or two notes can match many songs
  almost perfectly.
- Drums produce 51 false positives
  among 56 different-song pairs. Their mean tempo similarity is
  0.997 for expected pairs and 0.995 for other
  pairs; mean rhythm similarity is 0.731 and
  0.705, respectively, among threshold-matched passages. The
  current drum vector therefore carries almost no discriminating information on
  this corpus.

These results should be treated as feature diagnostics rather than an estimate
of final detector accuracy. The corpus contains only eight expected pairs, and
the parameters were initially chosen on synthetic examples.

## Expected-pair scores

| Track | bass | piano | vocals | other | drums |
| --- | --- | --- | --- | --- | --- |
| big-in-japan-alphaville | 0.801 | 1.000 | 0.584 | 0.660 | 0.920 |
| brother-louie-modern-talking | 0.537 | 0.727 | 0.544 | 0.524 | 0.890 |
| cheri-cheri-lady-modern-talking | 0.888 | 0.381 | 0.514 | 0.664 | 0.920 |
| daddy-cool-boney-m | 0.731 | 0.880 | 0.701 | 0.731 | 0.921 |
| forever-young-alphaville | 0.807 | 0.849 | 0.762 | 0.808 | 0.825 |
| mambo-no-5-lou-bega | 0.624 | 0.533 | 0.730 | 0.565 | 0.893 |
| rasputin-boney-m | 0.713 | 0.687 | 0.456 | 0.699 | 0.868 |
| youre-my-heart-youre-my-soul-modern-talking | 0.729 | 0.647 | 0.591 | 0.735 | 0.950 |

The complete pair-level results, including best passage locations, tempo ratio,
transposition, and drum sub-scores, are stored in
`output/research/similarity-baseline.csv`.
