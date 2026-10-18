from __future__ import annotations

import numpy as np
import pytest
from experiences._shared.midi_similarity import (NoteEvent, build_window_vectors, compute_similarity_matrix)

def motif(start: float = 0.0, transpose: int = 0) -> list[NoteEvent]:
    pitches = (36, 40, 43, 38, 41, 45, 40, 43)
    return [
        NoteEvent(
            start=start + index * 1.5,
            end=start + index * 1.5 + 1.1,
            pitch=pitch + transpose,
            velocity=0.9,
        )
        for index, pitch in enumerate(pitches)
    ]

def test_build_window_vectors_supports_velocity_ablation() -> None:
    events = [NoteEvent(start=0.0, end=1.0, pitch=60, velocity=0.25)]

    linear = build_window_vectors(
        events,
        duration=2.0,
        window_seconds=2.0,
        hop_seconds=1.0,
        frame_seconds=0.5,
        velocity_mode="linear",
    )
    binary = build_window_vectors(
        events,
        duration=2.0,
        window_seconds=2.0,
        hop_seconds=1.0,
        frame_seconds=0.5,
        velocity_mode="binary",
    )
    square_root = build_window_vectors(
        events,
        duration=2.0,
        window_seconds=2.0,
        hop_seconds=1.0,
        frame_seconds=0.5,
        velocity_mode="sqrt",
    )

    assert linear.values[0, 0, 1, 0] == pytest.approx(0.25)
    assert binary.values[0, 0, 1, 0] == pytest.approx(1.0)
    assert square_root.values[0, 0, 1, 0] == pytest.approx(0.5)
