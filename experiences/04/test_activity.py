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

def test_similarity_matrix_can_reject_sparse_windows() -> None:
    query = build_window_vectors(
        motif(),
        duration=12.0,
        window_seconds=12.0,
        hop_seconds=2.0,
        frame_seconds=0.5,
    )
    sparse_reference = build_window_vectors(
        motif(transpose=5)[:2],
        duration=12.0,
        window_seconds=12.0,
        hop_seconds=2.0,
        frame_seconds=0.5,
    )

    baseline, _ = compute_similarity_matrix(query, sparse_reference)
    filtered, _ = compute_similarity_matrix(
        query,
        sparse_reference,
        minimum_window_onsets=3,
    )

    assert baseline[0, 0] > 0
    assert filtered[0, 0] == 0

def test_activity_confidence_penalizes_unequal_note_density() -> None:
    query = build_window_vectors(
        motif(),
        duration=12.0,
        window_seconds=12.0,
        hop_seconds=2.0,
        frame_seconds=0.5,
    )
    sparse_reference = build_window_vectors(
        motif(transpose=5)[:4],
        duration=12.0,
        window_seconds=12.0,
        hop_seconds=2.0,
        frame_seconds=0.5,
    )

    baseline, _ = compute_similarity_matrix(query, sparse_reference)
    weighted, _ = compute_similarity_matrix(
        query,
        sparse_reference,
        activity_target_onsets=8,
        activity_weight=0.5,
    )

    assert 0 < weighted[0, 0] < baseline[0, 0]
