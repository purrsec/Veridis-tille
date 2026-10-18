# Local melodic sequence alignment experiment

## Scope

The fixed-window vectors require notes to fall into the same 500 ms frames.
This experiment replaces that comparison for bass and vocals with a local
Smith-Waterman alignment of consecutive note transitions. The representation
is invariant to a constant pitch shift and tests six tempo ratios. It tolerates
inserted or deleted notes with a gap penalty and requires eight aligned
transitions for full confidence.

Two pitch comparisons are tested:

- `exact`: signed semitone intervals with exponentially decreasing tolerance;
- `direction`: five coarse descending, repeated, and ascending classes.

Near-simultaneous Basic Pitch notes are reduced to the strongest melodic
candidate. All classifier combinations are selected inside each training fold
and evaluated with the same leave-two-families-out protocol as the fusion
experiment.

## Pairwise results

| Classifier | TP | FP | Precision | Recall | F1 | Rasputin | Brother Louie | Mambo No. 5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-sync-bass-other | 40/56 | 7/56 | 0.851 | 0.714 | 0.777 | 6/7 | 0/7 | 0/7 |
| sequence-exact-bass | 22/56 | 23/56 | 0.489 | 0.393 | 0.436 | 7/7 | 0/7 | 0/7 |
| sequence-exact-vocals | 16/56 | 7/56 | 0.696 | 0.286 | 0.405 | 2/7 | 7/7 | 7/7 |
| sequence-direction-bass | 23/56 | 21/56 | 0.523 | 0.411 | 0.460 | 6/7 | 0/7 | 0/7 |
| sequence-direction-vocals | 34/56 | 30/56 | 0.531 | 0.607 | 0.567 | 7/7 | 5/7 | 7/7 |
| nested-weight-baseline-exact-vocals | 30/56 | 9/56 | 0.769 | 0.536 | 0.632 | 7/7 | 0/7 | 0/7 |
| nested-OR-baseline-exact-vocals | 42/56 | 7/56 | 0.857 | 0.750 | 0.800 | 6/7 | 7/7 | 2/7 |
| nested-AND-baseline-exact-vocals | 33/56 | 6/56 | 0.846 | 0.589 | 0.695 | 6/7 | 0/7 | 0/7 |

## Detection stability by song family

| Song family | Fusion baseline | Nested OR |
| --- | --- | --- |
| big-in-japan-alphaville | 7/7 | 7/7 |
| brother-louie-modern-talking | 0/7 | 7/7 |
| cheri-cheri-lady-modern-talking | 7/7 | 7/7 |
| daddy-cool-boney-m | 6/7 | 0/7 |
| forever-young-alphaville | 7/7 | 7/7 |
| mambo-no-5-lou-bega | 0/7 | 2/7 |
| rasputin-boney-m | 6/7 | 6/7 |
| youre-my-heart-youre-my-soul-modern-talking | 7/7 | 6/7 |

## Conclusion

The nested OR rule is the best pairwise result so far. It accepts a pair when
either synchronized bass/other fusion or exact vocal sequence alignment passes
its threshold selected inside the training fold. F1 rises from `0.777` to
`0.800`: true positives increase from 40/56 to 42/56 while false positives stay
at 7/56. Precision is `0.857` and recall is `0.750`.

The vocal sequence adds genuinely complementary evidence. It recovers Brother
Louie in 7/7 folds and Mambo No. 5 in 2/7 while preserving Rasputin at 6/7.
The gain is not yet stable across songs: Daddy Cool falls from 6/7 to 0/7 and
You're My Heart falls from 7/7 to 6/7. On only eight families, the aggregate
F1 improvement is therefore promising rather than conclusive.

The nested OR rule becomes the new experimental pairwise candidate. The next
experiment should work directly from audio chroma to add evidence which does
not inherit Spleeter and Basic Pitch note errors.

The complete alignment results are stored in
`experiences/07/sequence-alignment-experiment.csv`.
