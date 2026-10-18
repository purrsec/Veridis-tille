# Basic Pitch tuning for bass stems

## Scope

These observations come from the Spleeter `bass` stem of
`big-in-japan-alphaville-original.wav`, processed with Basic Pitch 0.4.0 and
the CoreML model. They are preliminary POC results, not general-purpose
defaults for every instrument or recording.

Tune extraction parameters only against a known original recording. Do not
calibrate them against an unknown or suspected AI/plagiarized sample, because
that would bias the detector toward the content being evaluated.

## Baseline

The Basic Pitch defaults produced 928 note events:

- onset threshold: `0.50`
- frame threshold: `0.30`
- minimum note length: `127.7 ms`
- Melodia post-processing: enabled
- frequency bounds: none

The transcription was globally good, but contained occasional artifact notes.

## Tests

| Test | Onset | Frame | Minimum length | Frequency range | Melodia | Notes | Listening result |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| Default | 0.50 | 0.30 | 127.7 ms | Unbounded | Enabled | 928 | Good, with occasional artifacts |
| Higher onset threshold | 0.60 | 0.30 | 127.7 ms | Unbounded | Enabled | 662 | Count-only test |
| Higher frame threshold (A) | 0.50 | 0.35 | 127.7 ms | Unbounded | Enabled | 893 | Slightly better than default; preferred |
| Longer notes (B) | 0.50 | 0.30 | 160 ms | Unbounded | Enabled | 827 | Less convincing than A |
| Mild combination | 0.55 | 0.33 | 150 ms | Unbounded | Enabled | 723 | Count-only test |
| Bass frequency range | 0.50 | 0.30 | 127.7 ms | 30–400 Hz | Enabled | 927 | Almost no effect |
| Aggressive combination | 0.60 | 0.40 | 180 ms | 30–400 Hz | Disabled | 512 | Strongly degraded; rejected |
| Strict combination | 0.70 | 0.50 | 220 ms | 30–400 Hz | Disabled | 279 | Count-only test; too aggressive |

## Current conclusion

For a separated bass line, raising only the frame threshold from `0.30` to
`0.35` is the best tested compromise. It removes a small number of unstable
detections while preserving the musical line better than the other tested
settings.

Keep the onset threshold, minimum note length, frequency bounds, and Melodia
post-processing at their defaults for now. In particular, disabling Melodia
while tightening several thresholds at once severely damages the result.

Validate this provisional setting on more known original bass stems before
promoting it to a dataset-wide default.
