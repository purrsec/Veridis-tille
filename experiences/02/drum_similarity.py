"""Pitch-free local similarity for isolated drum tracks."""

from __future__ import annotations

import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RhythmEvent:
    """One percussive attack expressed only by time and strength."""

    time: float
    strength: float = 1.0


@dataclass(frozen=True)
class DrumSimilarityConfig:
    """Weights and analysis parameters for drum-track comparison."""

    tempo_weight: float = 10_000.0
    rhythm_weight: float = 5_000.0
    pitch_weight: float = 0.0
    window_seconds: float = 10.0
    hop_seconds: float = 2.0
    frame_seconds: float = 0.1
    cell_threshold: float = 0.78
    match_threshold: float = 0.82
    minimum_aligned_windows: int = 2
    minimum_match_seconds: float = 10.0
    max_matches: int = 5
    tempo_ratios: tuple[float, ...] = (0.8, 0.9, 1.0, 1.125, 1.2, 1.25)

    def __post_init__(self) -> None:
        positive = {
            "tempo_weight": self.tempo_weight,
            "rhythm_weight": self.rhythm_weight,
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "frame_seconds": self.frame_seconds,
            "minimum_aligned_windows": self.minimum_aligned_windows,
            "minimum_match_seconds": self.minimum_match_seconds,
            "max_matches": self.max_matches,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.pitch_weight != 0:
            raise ValueError("pitch_weight must stay at zero for drum comparison")
        if not self.tempo_ratios or any(ratio <= 0 for ratio in self.tempo_ratios):
            raise ValueError("tempo_ratios must contain positive values")
        if not np.isclose(
            self.window_seconds / self.frame_seconds,
            round(self.window_seconds / self.frame_seconds),
        ):
            raise ValueError("window_seconds must be divisible by frame_seconds")
        for name, value in {
            "cell_threshold": self.cell_threshold,
            "match_threshold": self.match_threshold,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class RhythmWindows:
    starts_seconds: NDArray[np.float64]
    onsets: NDArray[np.float32]
    tempo_signatures: NDArray[np.float32]
    interval_signatures: NDArray[np.float32]


@dataclass(frozen=True)
class DrumMatch:
    query_start: float
    query_end: float
    reference_start: float
    reference_end: float
    score: float
    tempo_similarity: float
    rhythm_similarity: float
    tempo_points: float
    rhythm_points: float
    pitch_points: float
    tempo_ratio: float
    aligned_windows: int


@dataclass(frozen=True)
class DrumComparisonResult:
    is_similar: bool
    score: float
    matches: list[DrumMatch]
    query_onsets: int
    reference_onsets: int

    def to_dict(self) -> dict[str, object]:
        return {
            "is_similar": self.is_similar,
            "score": self.score,
            "weights": {"tempo": 10_000, "rhythm": 5_000, "pitch": 0},
            "query_onsets": self.query_onsets,
            "reference_onsets": self.reference_onsets,
            "matches": [asdict(match) for match in self.matches],
        }


def _read_pcm16_wav(path: str | Path) -> tuple[NDArray[np.float32], int]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        sample_rate = audio.getframerate()
        frames = audio.readframes(audio.getnframes())
    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM WAV, got {sample_width * 8}-bit audio")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def load_wav_onsets(path: str | Path) -> tuple[list[RhythmEvent], float]:
    """Extract percussive attacks from PCM16 WAV without retaining pitch."""

    samples, sample_rate = _read_pcm16_wav(path)
    duration = len(samples) / sample_rate
    frame_length = max(64, round(sample_rate * 0.032))
    hop_length = max(16, round(sample_rate * 0.01))
    if len(samples) < frame_length:
        return [], duration

    starts = np.arange(0, len(samples) - frame_length + 1, hop_length)
    window = np.hanning(frame_length).astype(np.float32)
    novelty = np.zeros(len(starts), dtype=np.float32)
    previous = np.zeros(frame_length // 2 + 1, dtype=np.float32)
    offsets = np.arange(frame_length)

    for batch_start in range(0, len(starts), 256):
        batch_positions = starts[batch_start : batch_start + 256]
        frames = samples[batch_positions[:, None] + offsets] * window
        magnitudes = np.log1p(np.abs(np.fft.rfft(frames, axis=1))).astype(np.float32)
        preceding = np.vstack((previous, magnitudes[:-1]))
        novelty[batch_start : batch_start + len(batch_positions)] = np.maximum(
            magnitudes - preceding,
            0,
        ).sum(axis=1)
        previous = magnitudes[-1]

    if not np.any(novelty > 0):
        return [], duration
    baseline_width = max(3, round(0.3 * sample_rate / hop_length))
    baseline = np.convolve(
        novelty,
        np.ones(baseline_width, dtype=np.float32) / baseline_width,
        mode="same",
    )
    novelty = np.maximum(novelty - baseline, 0)
    positive = novelty[novelty > 0]
    threshold = max(float(np.percentile(positive, 70)) * 0.35, float(novelty.max()) * 0.015)
    candidates = [
        index
        for index in range(1, len(novelty) - 1)
        if novelty[index] >= threshold
        and novelty[index] >= novelty[index - 1]
        and novelty[index] >= novelty[index + 1]
    ]

    minimum_gap = max(1, round(0.06 * sample_rate / hop_length))
    selected: list[int] = []
    for candidate in candidates:
        if selected and candidate - selected[-1] < minimum_gap:
            if novelty[candidate] > novelty[selected[-1]]:
                selected[-1] = candidate
        else:
            selected.append(candidate)
    peak = float(novelty[selected].max()) if selected else 1.0
    events = [
        RhythmEvent(
            time=float((starts[index] + frame_length / 2) / sample_rate),
            strength=float(novelty[index] / peak),
        )
        for index in selected
    ]
    return events, duration


def _window_starts(duration: float, window_seconds: float, hop_seconds: float) -> NDArray[np.float64]:
    if duration <= 0:
        return np.array([], dtype=np.float64)
    if duration <= window_seconds:
        return np.array([0.0], dtype=np.float64)
    count = int(np.ceil((duration - window_seconds) / hop_seconds)) + 1
    return np.arange(count, dtype=np.float64) * hop_seconds


def _unit_rows(values: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def build_rhythm_windows(
    events: Iterable[RhythmEvent],
    *,
    duration: float,
    window_seconds: float,
    hop_seconds: float,
    frame_seconds: float,
) -> RhythmWindows:
    """Represent each window by onset timing and a tempo autocorrelation."""

    frame_count = round(window_seconds / frame_seconds)
    if not np.isclose(window_seconds / frame_seconds, frame_count):
        raise ValueError("window_seconds must be divisible by frame_seconds")
    starts = _window_starts(duration, window_seconds, hop_seconds)
    onsets = np.zeros((len(starts), frame_count), dtype=np.float32)
    for event in events:
        strength = float(np.clip(event.strength, 0, 1))
        for window_index, start in enumerate(starts):
            if start <= event.time < start + window_seconds:
                frame = min(frame_count - 1, int((event.time - start) / frame_seconds + 1e-9))
                onsets[window_index, frame] = max(onsets[window_index, frame], strength)

    # Small temporal tolerance: an attack contributes to neighbouring bins,
    # but no frequency/pitch information enters this representation.
    padded = np.pad(onsets, ((0, 0), (2, 2)))
    smoothed = np.maximum.reduce(
        [
            padded[:, offset : offset + frame_count] * weight
            for offset, weight in enumerate((0.35, 0.7, 1.0, 0.7, 0.35))
        ]
    )

    # Lags are expressed as fractions of the normalized window. This keeps
    # their dimensions identical when the reference is tempo-stretched.
    first_lag = 1
    last_lag = min(frame_count - 1, max(1, frame_count // 4))
    signatures = np.zeros((len(starts), last_lag - first_lag + 1), dtype=np.float32)
    for lag_index, lag in enumerate(range(first_lag, last_lag + 1)):
        left = smoothed[:, :-lag]
        right = smoothed[:, lag:]
        numerator = np.sum(left * right, axis=1)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        signatures[:, lag_index] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
    intervals = np.zeros((len(starts), frame_count - 1), dtype=np.float32)
    for window_index, onset_row in enumerate(onsets):
        positions = np.flatnonzero(onset_row)
        for distance in np.diff(positions):
            if 0 < distance < frame_count:
                intervals[window_index, distance - 1] += 1
    return RhythmWindows(
        starts_seconds=starts,
        onsets=smoothed,
        tempo_signatures=signatures,
        interval_signatures=intervals,
    )


def compute_rhythm_similarity(
    query: RhythmWindows,
    reference: RhythmWindows,
    *,
    config: DrumSimilarityConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Return weighted, tempo, and rhythmic similarity matrices."""

    onset_similarity = _unit_rows(query.onsets) @ _unit_rows(reference.onsets).T
    interval_similarity = (
        _unit_rows(query.interval_signatures)
        @ _unit_rows(reference.interval_signatures).T
    )
    query_density = query.onsets.sum(axis=1, keepdims=True)
    reference_density = reference.onsets.sum(axis=1, keepdims=True).T
    density_similarity = np.divide(
        np.minimum(query_density, reference_density),
        np.maximum(query_density, reference_density),
        out=np.zeros((len(query.onsets), len(reference.onsets)), dtype=np.float32),
        where=np.maximum(query_density, reference_density) > 0,
    )
    rhythm = onset_similarity * interval_similarity * density_similarity
    tempo = _unit_rows(query.tempo_signatures) @ _unit_rows(reference.tempo_signatures).T
    weight_sum = config.tempo_weight + config.rhythm_weight
    combined = (tempo * config.tempo_weight + rhythm * config.rhythm_weight) / weight_sum
    return combined.astype(np.float32), tempo.astype(np.float32), rhythm.astype(np.float32)


def _candidate_runs(scores: NDArray[np.float32], threshold: float) -> Iterable[list[tuple[int, int]]]:
    query_count, reference_count = scores.shape
    for offset in range(-query_count + 1, reference_count):
        run: list[tuple[int, int]] = []
        for query_index in range(max(0, -offset), min(query_count, reference_count - offset)):
            reference_index = query_index + offset
            if scores[query_index, reference_index] >= threshold:
                run.append((query_index, reference_index))
            elif run:
                yield run
                run = []
        if run:
            yield run


def _best_sequence_score(scores: NDArray[np.float32], length: int) -> float:
    best = 0.0
    for offset in range(-scores.shape[0] + 1, scores.shape[1]):
        diagonal = np.diagonal(scores, offset=offset)
        if len(diagonal) < length:
            continue
        means = np.convolve(diagonal, np.ones(length) / length, mode="valid")
        best = max(best, float(means.max(initial=0.0)))
    return best


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    common = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    shorter = min(end_a - start_a, end_b - start_b)
    return common / shorter if shorter > 0 else 0.0


def _deduplicate(matches: list[DrumMatch], limit: int) -> list[DrumMatch]:
    selected: list[DrumMatch] = []
    for candidate in sorted(matches, key=lambda item: item.score, reverse=True):
        if any(
            _overlap(candidate.query_start, candidate.query_end, item.query_start, item.query_end) >= 0.8
            and _overlap(
                candidate.reference_start,
                candidate.reference_end,
                item.reference_start,
                item.reference_end,
            )
            >= 0.8
            for item in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


def compare_rhythm_events(
    query_events: Iterable[RhythmEvent],
    reference_events: Iterable[RhythmEvent],
    *,
    query_duration: float,
    reference_duration: float,
    config: DrumSimilarityConfig | None = None,
) -> DrumComparisonResult:
    """Compare and locate rhythms; pitch is absent from every calculation."""

    config = config or DrumSimilarityConfig()
    query_events = list(query_events)
    reference_events = list(reference_events)
    query = build_rhythm_windows(
        query_events,
        duration=query_duration,
        window_seconds=config.window_seconds,
        hop_seconds=config.hop_seconds,
        frame_seconds=config.frame_seconds,
    )
    candidates: list[DrumMatch] = []
    best_sequence = 0.0
    for tempo_ratio in config.tempo_ratios:
        reference_window_seconds = config.window_seconds * tempo_ratio
        reference = build_rhythm_windows(
            reference_events,
            duration=reference_duration,
            window_seconds=reference_window_seconds,
            hop_seconds=config.hop_seconds * tempo_ratio,
            frame_seconds=config.frame_seconds * tempo_ratio,
        )
        scores, tempo_scores, rhythm_scores = compute_rhythm_similarity(
            query,
            reference,
            config=config,
        )
        best_sequence = max(
            best_sequence,
            _best_sequence_score(scores, config.minimum_aligned_windows),
        )
        for run in _candidate_runs(scores, config.cell_threshold):
            if len(run) < config.minimum_aligned_windows:
                continue
            query_indices, reference_indices = zip(*run)
            mean_score = float(np.mean(scores[query_indices, reference_indices]))
            if mean_score < config.match_threshold:
                continue
            first_query, first_reference = run[0]
            last_query, last_reference = run[-1]
            query_start = float(query.starts_seconds[first_query])
            query_end = min(
                query_duration,
                float(query.starts_seconds[last_query] + config.window_seconds),
            )
            reference_start = float(reference.starts_seconds[first_reference])
            reference_end = min(
                reference_duration,
                float(reference.starts_seconds[last_reference] + reference_window_seconds),
            )
            matched_query_seconds = min(
                query_end - query_start,
                (reference_end - reference_start) / tempo_ratio,
            )
            if matched_query_seconds < config.minimum_match_seconds:
                continue
            tempo_similarity = float(np.mean(tempo_scores[query_indices, reference_indices]))
            rhythm_similarity = float(np.mean(rhythm_scores[query_indices, reference_indices]))
            candidates.append(
                DrumMatch(
                    query_start=query_start,
                    query_end=query_end,
                    reference_start=reference_start,
                    reference_end=reference_end,
                    score=mean_score,
                    tempo_similarity=tempo_similarity,
                    rhythm_similarity=rhythm_similarity,
                    tempo_points=tempo_similarity * config.tempo_weight,
                    rhythm_points=rhythm_similarity * config.rhythm_weight,
                    pitch_points=0.0,
                    tempo_ratio=tempo_ratio,
                    aligned_windows=len(run),
                )
            )
    matches = _deduplicate(candidates, config.max_matches)
    return DrumComparisonResult(
        is_similar=bool(matches),
        score=matches[0].score if matches else best_sequence,
        matches=matches,
        query_onsets=len(query_events),
        reference_onsets=len(reference_events),
    )


def compare_drum_audio(
    query_path: str | Path,
    reference_path: str | Path,
    *,
    config: DrumSimilarityConfig | None = None,
) -> DrumComparisonResult:
    query_events, query_duration = load_wav_onsets(query_path)
    reference_events, reference_duration = load_wav_onsets(reference_path)
    return compare_rhythm_events(
        query_events,
        reference_events,
        query_duration=query_duration,
        reference_duration=reference_duration,
        config=config,
    )
