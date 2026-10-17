from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from veridis_tille.drum_similarity import (
    DrumSimilarityConfig,
    RhythmEvent,
    compare_rhythm_events,
    load_wav_onsets,
)


def rhythm(start: float = 0.0, scale: float = 1.0) -> list[RhythmEvent]:
    times = (0.0, 0.5, 1.0, 1.75, 2.0, 3.0, 3.5, 4.0, 4.75, 5.0, 6.0, 6.5, 7.0, 7.75, 8.0, 9.0, 9.5, 10.0, 10.75, 11.0)
    strengths = (1.0, 0.5, 0.8, 0.4, 1.0) * 4
    return [
        RhythmEvent(time=start + time * scale, strength=strength)
        for time, strength in zip(times, strengths)
    ]


def test_drum_weights_make_tempo_dominant_and_ignore_pitch() -> None:
    config = DrumSimilarityConfig()

    assert config.tempo_weight == 10_000
    assert config.rhythm_weight == 5_000
    assert config.pitch_weight == 0


def test_compare_rhythm_localizes_a_tempo_modified_pattern() -> None:
    config = DrumSimilarityConfig(
        window_seconds=6.0,
        hop_seconds=2.0,
        frame_seconds=0.25,
        tempo_ratios=(1.0, 1.2),
        cell_threshold=0.85,
        match_threshold=0.9,
        minimum_aligned_windows=3,
        minimum_match_seconds=8.0,
    )

    result = compare_rhythm_events(
        rhythm(),
        rhythm(start=24.0, scale=1.2),
        query_duration=12.0,
        reference_duration=45.0,
        config=config,
    )

    assert result.is_similar is True
    best = result.matches[0]
    assert best.query_start == pytest.approx(0.0)
    assert best.reference_start == pytest.approx(24.0)
    assert best.reference_end == pytest.approx(38.4)
    assert best.tempo_ratio == pytest.approx(1.2)
    assert best.tempo_points > best.rhythm_points
    assert best.pitch_points == 0.0
    assert best.score > 0.99


def test_compare_rhythm_rejects_a_different_pattern_at_the_same_tempo() -> None:
    unrelated = [RhythmEvent(time=index * 0.5, strength=1.0) for index in range(24)]
    config = DrumSimilarityConfig(
        window_seconds=6.0,
        hop_seconds=2.0,
        frame_seconds=0.25,
        tempo_ratios=(1.0,),
        cell_threshold=0.85,
        match_threshold=0.9,
        minimum_aligned_windows=3,
        minimum_match_seconds=8.0,
    )

    result = compare_rhythm_events(
        rhythm(),
        unrelated,
        query_duration=12.0,
        reference_duration=12.0,
        config=config,
    )

    assert result.is_similar is False
    assert result.matches == []


def test_load_wav_onsets_detects_impulses_without_pitch(tmp_path: Path) -> None:
    sample_rate = 8_000
    samples = np.zeros(sample_rate * 2, dtype=np.int16)
    expected = (0.25, 0.75, 1.4)
    for onset in expected:
        start = round(onset * sample_rate)
        samples[start : start + 80] = np.linspace(30_000, 0, 80, dtype=np.int16)

    path = tmp_path / "drums.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.tobytes())

    events, duration = load_wav_onsets(path)

    assert duration == pytest.approx(2.0)
    detected = [event.time for event in events]
    for onset in expected:
        assert any(abs(candidate - onset) < 0.05 for candidate in detected)
