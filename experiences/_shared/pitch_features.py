"""Shared numerical operations for controlled pitch-representation ablations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fold_octaves(values: NDArray, first_midi: int = 21) -> NDArray[np.float32]:
    """Keep the strongest activation of each pitch class, including silent input."""
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("expected a time-by-pitch matrix with at least one pitch")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("pitch activations must be finite and nonnegative")
    result = np.zeros((len(values), 12), dtype=np.float32)
    for pitch in range(values.shape[1]):
        category = (first_midi + pitch) % 12
        np.maximum(result[:, category], values[:, pitch], out=result[:, category])
    return result


def average_frames(
    values: NDArray, times: NDArray, duration: float, frame_seconds: float = 0.5
) -> NDArray[np.float32]:
    """Average samples on common half-open time bins, without edge extrapolation."""
    if duration <= 0 or frame_seconds <= 0:
        raise ValueError("duration and frame_seconds must be positive")
    if values.ndim != 2 or times.ndim != 1 or len(values) != len(times):
        raise ValueError("values and times must have matching time dimensions")
    if not np.isfinite(values).all() or not np.isfinite(times).all():
        raise ValueError("frames and times must be finite")
    if np.any(np.diff(times) < 0):
        raise ValueError("frame times must be monotonic")
    valid = (times >= 0) & (times < duration)
    indices = np.floor(times[valid] / frame_seconds).astype(np.int64)
    count = int(np.ceil(duration / frame_seconds))
    result = np.zeros((count, values.shape[1]), dtype=np.float32)
    counts = np.zeros(count, dtype=np.int32)
    np.add.at(result, indices, values[valid])
    np.add.at(counts, indices, 1)
    np.divide(result, counts[:, None], out=result, where=counts[:, None] > 0)
    return result


def sample_midi_activity(events: NDArray, times: NDArray) -> NDArray[np.float32]:
    """Sample decoded MIDI [start, end, pitch, velocity/127] on a model's grid."""
    result = np.zeros((len(times), 88), dtype=np.float32)
    for start, end, pitch, velocity in events:
        pitch_index = int(pitch) - 21
        if not 0 <= pitch_index < 88:
            continue
        left, right = np.searchsorted(times, [max(0.0, start), end], side="left")
        if right > left:
            np.maximum(result[left:right, pitch_index], velocity,
                       out=result[left:right, pitch_index])
    return result
