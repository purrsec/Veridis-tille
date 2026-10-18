# Melodic interval and contour experiment

## Scope

The pitch-class baseline already searches all twelve transpositions. This
experiment tests whether a more explicit melody representation can tolerate
pitching and Basic Pitch errors better. Notes starting within 80 ms are reduced
to the strongest melodic candidate. Six representations are compared with the
same 10-second windows, 2-second hops, tempo ratios, 64 query/reference pairs,
and four MIDI stems:

- `direction`: five descending, repeated, or ascending movement classes;
- `coarse_interval`: nine signed interval ranges;
- `exact_interval`: signed semitone intervals clipped to one octave;
- `direction_exact`: direction and exact interval channels together;
- `relative_first`: every pitch relative to the first note in its window;
- `relative_median_coarse`: coarse pitch position around the window median.

Contour scores are maximum window cosine similarities. They are evaluated as
retrieval signals only: no detection threshold was fitted on this eight-pair
POC.

## Full-dataset results

| Representation | Stem | Ranks first | Mean margin | Expected median | Other median |
| --- | --- | --- | --- | --- | --- |
| pitch_class | bass | 5/8 | +0.011 | 0.730 | 0.613 |
| pitch_class | piano | 0/8 | -0.100 | 0.707 | 0.708 |
| pitch_class | vocals | 5/8 | +0.026 | 0.587 | 0.509 |
| pitch_class | other | 4/8 | -0.026 | 0.681 | 0.614 |
| direction | bass | 2/8 | -0.083 | 0.627 | 0.636 |
| direction | piano | 0/8 | -0.153 | 0.696 | 0.736 |
| direction | vocals | 2/8 | -0.038 | 0.554 | 0.519 |
| direction | other | 2/8 | -0.068 | 0.738 | 0.730 |
| coarse_interval | bass | 1/8 | -0.116 | 0.560 | 0.594 |
| coarse_interval | piano | 1/8 | -0.061 | 0.696 | 0.620 |
| coarse_interval | vocals | 3/8 | -0.043 | 0.486 | 0.453 |
| coarse_interval | other | 1/8 | -0.089 | 0.559 | 0.572 |
| exact_interval | bass | 1/8 | -0.120 | 0.558 | 0.558 |
| exact_interval | piano | 0/8 | -0.175 | 0.493 | 0.538 |
| exact_interval | vocals | 2/8 | -0.066 | 0.407 | 0.426 |
| exact_interval | other | 2/8 | -0.072 | 0.464 | 0.487 |
| direction_exact | bass | 1/8 | -0.107 | 0.568 | 0.571 |
| direction_exact | piano | 0/8 | -0.149 | 0.504 | 0.540 |
| direction_exact | vocals | 2/8 | -0.035 | 0.445 | 0.445 |
| direction_exact | other | 1/8 | -0.073 | 0.557 | 0.573 |
| relative_first | bass | 5/8 | -0.015 | 0.638 | 0.568 |
| relative_first | piano | 0/8 | -0.132 | 0.742 | 0.745 |
| relative_first | vocals | 2/8 | -0.034 | 0.626 | 0.550 |
| relative_first | other | 1/8 | -0.040 | 0.708 | 0.704 |
| relative_median_coarse | bass | 4/8 | -0.019 | 0.610 | 0.549 |
| relative_median_coarse | piano | 0/8 | -0.218 | 0.554 | 0.717 |
| relative_median_coarse | vocals | 3/8 | +0.021 | 0.578 | 0.519 |
| relative_median_coarse | other | 0/8 | -0.074 | 0.512 | 0.512 |

## Rasputin

| Representation | Stem | Expected rank | Expected score |
| --- | --- | --- | --- |
| pitch_class | bass | 3/8 | 0.713 |
| pitch_class | piano | 6/8 | 0.687 |
| pitch_class | vocals | 8/8 | 0.456 |
| pitch_class | other | 1/8 | 0.699 |
| direction | bass | 3/8 | 0.707 |
| direction | piano | 7/8 | 0.602 |
| direction | vocals | 2/8 | 0.581 |
| direction | other | 3/8 | 0.789 |
| coarse_interval | bass | 4/8 | 0.663 |
| coarse_interval | piano | 7/8 | 0.546 |
| coarse_interval | vocals | 3/8 | 0.457 |
| coarse_interval | other | 3/8 | 0.593 |
| exact_interval | bass | 3/8 | 0.663 |
| exact_interval | piano | 6/8 | 0.508 |
| exact_interval | vocals | 8/8 | 0.371 |
| exact_interval | other | 1/8 | 0.559 |
| direction_exact | bass | 2/8 | 0.685 |
| direction_exact | piano | 6/8 | 0.508 |
| direction_exact | vocals | 4/8 | 0.449 |
| direction_exact | other | 3/8 | 0.623 |
| relative_first | bass | 1/8 | 0.698 |
| relative_first | piano | 8/8 | 0.691 |
| relative_first | vocals | 8/8 | 0.627 |
| relative_first | other | 4/8 | 0.799 |
| relative_median_coarse | bass | 1/8 | 0.698 |
| relative_median_coarse | piano | 7/8 | 0.541 |
| relative_median_coarse | vocals | 2/8 | 0.520 |
| relative_median_coarse | other | 5/8 | 0.546 |

## Conclusion

The explicit contour hypothesis does not improve the detector globally. The
pitch-class baseline remains strongest with 5/8 rank-one results for bass and
vocals. Directional contour moves the Rasputin vocal pair from rank 8 to rank
2, and `relative_median_coarse` also places it second, but those
representations retrieve only 2/8 and 3/8 expected vocal pairs respectively.
The apparent Rasputin gain therefore does not generalize.

`relative_first` keeps bass at 5/8 and places Rasputin bass first, but its mean
margin is negative. Exact intervals amplify transcription errors and perform
worse than coarse direction for the difficult vocal example.

These results support keeping pitch-class vectors as the melodic baseline.
The useful Rasputin evidence still comes from combining non-vocal stems; the
next experiment should evaluate calibrated multi-stem score fusion rather than
replace the current representation with contours.

The complete results are stored in
`experiences/05/melodic-contour-experiment.csv`.
