from __future__ import annotations

import numpy as np
import pytest

from experiences._shared.pitch_features import average_frames, fold_octaves, sample_midi_activity


def test_octave_folding_preserves_pitch_class_with_non_c_origin():
    values = np.zeros((2, 88), dtype=np.float32)
    values[0, 0] = 0.3  # A0, MIDI 21
    values[0, 12] = 0.8  # A1: same class, maximum rather than sum
    values[0, 3] = 0.6  # C1
    folded = fold_octaves(values)
    assert folded[0, 9] == pytest.approx(0.8)
    assert folded[0, 0] == pytest.approx(0.6)
    assert np.count_nonzero(folded[1]) == 0


def test_common_grid_excludes_padded_model_frames_and_respects_boundaries():
    times = np.array([-0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1])
    values = np.array([[99], [2], [4], [6], [8], [99], [99]], dtype=np.float32)
    assert average_frames(values, times, 1.0).ravel().tolist() == [3.0, 7.0]


def test_decoded_note_duration_and_velocity_use_the_same_model_grid():
    times = np.arange(0.0, 1.0, 0.125)
    events = np.array([[0.125, 0.5, 60, 0.5], [0.25, 0.375, 60, 0.9]])
    activity = sample_midi_activity(events, times)
    np.testing.assert_allclose(activity[:, 39], [0, 0.5, 0.9, 0.5, 0, 0, 0, 0])
    assert np.count_nonzero(activity) == 3


def test_invalid_grid_cannot_silently_corrupt_features():
    with pytest.raises(ValueError):
        average_frames(np.zeros((2, 12)), np.array([0.5, 0.1]), 1.0)
    with pytest.raises(ValueError):
        fold_octaves(np.array([[float("nan")]]))
