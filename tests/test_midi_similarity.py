from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
import pytest

from veridis_tille.midi_similarity import (
    NoteEvent,
    SimilarityConfig,
    build_window_vectors,
    compare_events,
    compute_similarity_matrix,
    load_midi_events,
)


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


def test_load_midi_events_preserves_seconds_pitch_and_velocity(tmp_path: Path) -> None:
    midi_path = tmp_path / "notes.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    track.append(mido.Message("note_on", note=64, velocity=127, time=480))
    track.append(mido.Message("note_on", note=64, velocity=0, time=240))
    midi.save(midi_path)

    events = load_midi_events(midi_path)

    assert events == [
        NoteEvent(start=0.0, end=0.5, pitch=60, velocity=pytest.approx(64 / 127)),
        NoteEvent(start=1.0, end=1.25, pitch=64, velocity=1.0),
    ]


def test_build_window_vectors_contains_sustain_and_onset_channels() -> None:
    events = [
        NoteEvent(start=0.2, end=1.8, pitch=60, velocity=1.0),
        NoteEvent(start=2.2, end=3.2, pitch=64, velocity=0.5),
    ]

    windows = build_window_vectors(
        events,
        duration=6.0,
        window_seconds=4.0,
        hop_seconds=2.0,
        frame_seconds=1.0,
    )

    assert windows.starts_seconds.tolist() == [0.0, 2.0]
    assert windows.values.shape == (2, 4, 2, 12)
    assert windows.values[0, 0, 0, 0] == 1.0  # C sustain
    assert windows.values[0, 0, 1, 0] == 1.0  # C onset
    assert windows.values[0, 2, 0, 4] == 0.5  # E sustain
    assert windows.values[0, 2, 1, 4] == 0.5  # E onset




def test_similarity_matrix_recovers_pitch_transposition() -> None:
    query = build_window_vectors(motif(), duration=12.0, window_seconds=6.0, hop_seconds=2.0, frame_seconds=0.5)
    reference = build_window_vectors(
        motif(transpose=5),
        duration=12.0,
        window_seconds=6.0,
        hop_seconds=2.0,
        frame_seconds=0.5,
    )

    scores, transpositions = compute_similarity_matrix(query, reference)

    assert np.diag(scores).min() > 0.99
    assert set(np.diag(transpositions)) == {5}






def test_compare_events_finds_copied_passage_and_ignores_one_artifact() -> None:
    query_events = motif()
    query_events.append(NoteEvent(start=5.1, end=5.2, pitch=91, velocity=0.08))
    reference_events = [
        NoteEvent(start=2.0, end=3.0, pitch=31, velocity=0.8),
        *motif(start=20.0, transpose=5),
        NoteEvent(start=40.0, end=41.0, pitch=47, velocity=0.8),
    ]
    config = SimilarityConfig(
        window_seconds=6.0,
        hop_seconds=2.0,
        frame_seconds=0.5,
        cell_threshold=0.85,
        match_threshold=0.9,
        minimum_aligned_windows=3,
        minimum_match_seconds=8.0,
    )

    result = compare_events(
        query_events,
        reference_events,
        query_duration=12.0,
        reference_duration=45.0,
        config=config,
    )

    assert result.is_similar is True
    assert result.matches
    best = result.matches[0]
    assert best.query_start == pytest.approx(0.0)
    assert best.query_end == pytest.approx(12.0)
    assert best.reference_start == pytest.approx(20.0)
    assert best.reference_end == pytest.approx(32.0)
    assert best.transposition == 5
    assert best.score > 0.95


def test_compare_events_rejects_unrelated_material() -> None:
    query_events = motif()
    reference_events = [
        NoteEvent(start=index * 1.5, end=index * 1.5 + 0.3, pitch=48, velocity=1.0)
        for index in range(8)
    ]
    config = SimilarityConfig(
        window_seconds=6.0,
        hop_seconds=2.0,
        frame_seconds=0.5,
        cell_threshold=0.85,
        match_threshold=0.9,
        minimum_aligned_windows=3,
        minimum_match_seconds=8.0,
    )

    result = compare_events(
        query_events,
        reference_events,
        query_duration=12.0,
        reference_duration=12.0,
        config=config,
    )

    assert result.is_similar is False
    assert result.matches == []


def test_compare_events_handles_a_tempo_change() -> None:
    query_events = motif()
    reference_events = [
        NoteEvent(
            start=19.2 + event.start * 1.2,
            end=19.2 + event.end * 1.2,
            pitch=event.pitch - 2,
            velocity=event.velocity,
        )
        for event in query_events
    ]
    config = SimilarityConfig(
        window_seconds=6.0,
        hop_seconds=2.0,
        frame_seconds=0.5,
        cell_threshold=0.85,
        match_threshold=0.9,
        minimum_aligned_windows=3,
        minimum_match_seconds=8.0,
        tempo_ratios=(1.0, 1.2),
    )

    result = compare_events(
        query_events,
        reference_events,
        query_duration=12.0,
        reference_duration=40.0,
        config=config,
    )

    assert result.is_similar is True
    best = result.matches[0]
    assert best.reference_start == pytest.approx(19.2)
    assert best.reference_end == pytest.approx(33.6)
    assert best.tempo_ratio == pytest.approx(1.2)
    assert best.transposition == -2
