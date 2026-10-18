# MIDI activity confidence experiment

## Scope

This experiment tests whether MIDI comparisons improve when windows with too
few note onsets are rejected or down-weighted. It uses the same 8 AI excerpts,
8 originals, four MIDI stems, window size, tempo ratios, and match thresholds
as the vectorization baseline.

`Coverage` counts AI queries with at least one eligible 10-second window.
Ranking and detection statistics exclude uncovered queries instead of treating
missing musical evidence as a negative match.

## Results

| Experiment | Stem | Coverage | Ranks first | Mean margin | Expected detected | Other detected | Expected median | Other median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | bass | 8/8 | 5/8 | +0.011 | 3/8 | 2/56 | 0.730 | 0.613 |
| baseline | piano | 8/8 | 0/8 | -0.100 | 1/8 | 8/56 | 0.707 | 0.708 |
| baseline | vocals | 8/8 | 5/8 | +0.026 | 0/8 | 0/56 | 0.587 | 0.509 |
| baseline | other | 8/8 | 4/8 | -0.026 | 1/8 | 0/56 | 0.681 | 0.614 |
| gate-3 | bass | 8/8 | 5/8 | +0.011 | 3/8 | 2/56 | 0.730 | 0.613 |
| gate-3 | piano | 7/8 | 1/7 | -0.169 | 0/7 | 0/49 | 0.496 | 0.552 |
| gate-3 | vocals | 8/8 | 5/8 | +0.026 | 0/8 | 0/56 | 0.587 | 0.504 |
| gate-3 | other | 8/8 | 4/8 | -0.026 | 1/8 | 0/56 | 0.681 | 0.614 |
| gate-5 | bass | 8/8 | 5/8 | +0.011 | 3/8 | 2/56 | 0.730 | 0.613 |
| gate-5 | piano | 6/8 | 0/6 | -0.131 | 0/6 | 0/42 | 0.424 | 0.490 |
| gate-5 | vocals | 8/8 | 5/8 | +0.026 | 0/8 | 0/56 | 0.587 | 0.501 |
| gate-5 | other | 8/8 | 4/8 | -0.026 | 1/8 | 0/56 | 0.681 | 0.614 |
| gate-8 | bass | 8/8 | 5/8 | +0.011 | 3/8 | 2/56 | 0.730 | 0.610 |
| gate-8 | piano | 6/8 | 1/6 | -0.225 | 0/6 | 0/42 | 0.306 | 0.361 |
| gate-8 | vocals | 8/8 | 4/8 | +0.007 | 0/8 | 0/56 | 0.555 | 0.489 |
| gate-8 | other | 8/8 | 4/8 | -0.026 | 1/8 | 0/56 | 0.681 | 0.614 |
| density-light | bass | 8/8 | 4/8 | +0.007 | 1/8 | 1/56 | 0.680 | 0.570 |
| density-light | piano | 7/8 | 0/7 | -0.223 | 0/7 | 0/49 | 0.392 | 0.492 |
| density-light | vocals | 8/8 | 4/8 | +0.018 | 0/8 | 0/56 | 0.546 | 0.442 |
| density-light | other | 8/8 | 1/8 | -0.058 | 0/8 | 0/56 | 0.608 | 0.576 |
| confidence-8-light | bass | 8/8 | 4/8 | +0.007 | 1/8 | 1/56 | 0.680 | 0.570 |
| confidence-8-light | piano | 7/8 | 1/7 | -0.162 | 0/7 | 0/49 | 0.339 | 0.394 |
| confidence-8-light | vocals | 8/8 | 4/8 | +0.020 | 0/8 | 0/56 | 0.526 | 0.440 |
| confidence-8-light | other | 8/8 | 1/8 | -0.058 | 0/8 | 0/56 | 0.608 | 0.576 |
| confidence-12-light | bass | 8/8 | 4/8 | +0.006 | 1/8 | 1/56 | 0.680 | 0.569 |
| confidence-12-light | piano | 7/8 | 1/7 | -0.167 | 0/7 | 0/49 | 0.282 | 0.323 |
| confidence-12-light | vocals | 8/8 | 5/8 | +0.031 | 0/8 | 0/56 | 0.514 | 0.431 |
| confidence-12-light | other | 8/8 | 1/8 | -0.058 | 0/8 | 0/56 | 0.608 | 0.576 |
| confidence-12-strong | bass | 8/8 | 4/8 | +0.021 | 1/8 | 1/56 | 0.680 | 0.559 |
| confidence-12-strong | piano | 7/8 | 1/7 | -0.175 | 0/7 | 0/49 | 0.203 | 0.230 |
| confidence-12-strong | vocals | 8/8 | 4/8 | +0.019 | 0/8 | 0/56 | 0.499 | 0.423 |
| confidence-12-strong | other | 8/8 | 1/8 | -0.072 | 0/8 | 0/56 | 0.568 | 0.559 |

## Conclusion

The activity hypothesis is useful as an evidence check, but not as a general
similarity multiplier:

- onset gates remove the sparse-piano false positives by withholding a result,
  yet the covered piano queries still almost never rank the expected original
  first;
- activity weighting reduces bass rank-one results from 5/8 to 4/8, even when
  its mean rank margin increases slightly;
- the light 12-onset confidence improves the vocal mean margin from `+0.026`
  to `+0.031` while preserving 5/8 rank-one results, which is too small a gain
  on eight examples to justify changing the default;
- every weighted variant degrades the `other` stem.

The new parameters therefore remain disabled by default. A minimum activity
gate can later support an explicit `insufficient evidence` state, especially
for piano, but it should not silently turn missing notes into a negative match.
The next MIDI experiment should add melodic interval or contour information to
the promising bass and vocal representations.

The complete results are stored in
`output/research/midi-activity-experiment.csv`.
