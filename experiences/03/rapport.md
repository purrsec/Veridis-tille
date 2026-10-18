# MIDI velocity ablation

## Scope

Basic Pitch converts its estimated note amplitude to MIDI velocity. The current
vectors use that value linearly in both sustain and onset channels. This
experiment compares three otherwise identical representations:

- `linear`: current velocity divided by 127;
- `binary`: every detected note has weight 1;
- `sqrt`: square-root compression retains accents while reducing their range.

Each mode is evaluated on all 64 query/reference pairs for bass, piano, vocals,
and other. Activity confidence is disabled, so velocity is the only changed
variable.

## Results

| Velocity | Stem | Ranks first | Mean margin | Expected detected | Other detected | Expected median | Other median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear | bass | 5/8 | +0.011 | 3/8 | 2/56 | 0.730 | 0.613 |
| linear | piano | 0/8 | -0.100 | 1/8 | 8/56 | 0.707 | 0.708 |
| linear | vocals | 5/8 | +0.026 | 0/8 | 0/56 | 0.587 | 0.509 |
| linear | other | 4/8 | -0.026 | 1/8 | 0/56 | 0.681 | 0.614 |
| binary | bass | 4/8 | +0.004 | 3/8 | 2/56 | 0.698 | 0.591 |
| binary | piano | 1/8 | -0.110 | 1/8 | 7/56 | 0.637 | 0.667 |
| binary | vocals | 4/8 | +0.032 | 0/8 | 0/56 | 0.556 | 0.484 |
| binary | other | 2/8 | -0.028 | 1/8 | 0/56 | 0.669 | 0.619 |
| sqrt | bass | 5/8 | +0.006 | 3/8 | 2/56 | 0.718 | 0.598 |
| sqrt | piano | 0/8 | -0.113 | 1/8 | 8/56 | 0.662 | 0.688 |
| sqrt | vocals | 5/8 | +0.029 | 0/8 | 0/56 | 0.577 | 0.495 |
| sqrt | other | 4/8 | -0.029 | 1/8 | 0/56 | 0.679 | 0.617 |

## Conclusion

Velocity carries useful information in this POC:

- binary weighting reduces rank-one retrieval from 5/8 to 4/8 for bass, from
  5/8 to 4/8 for vocals, and from 4/8 to 2/8 for other;
- square-root compression preserves the linear rank-one counts, but lowers the
  bass mean margin from `+0.011` to `+0.006` and the other margin from `-0.026`
  to `-0.029`;
- square-root compression raises the vocal margin only slightly, from `+0.026`
  to `+0.029`;
- none of the modes fixes piano, confirming that its main problem is sparse or
  non-discriminating note content rather than velocity scaling.

Linear velocity remains the default. On this corpus, the Basic Pitch amplitude
estimate contributes more useful relative accent information than noise, while
compression offers no consistent gain.

The complete results are stored in
`experiences/03/midi-velocity-experiment.csv`.
