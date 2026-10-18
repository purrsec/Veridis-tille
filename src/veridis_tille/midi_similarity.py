"""Local similarity detection for MIDI note events.

Each sliding window is a piano-roll-like tensor containing pitch-class
sustain and onset energy. Windows are compared under all twelve pitch
transpositions, then consecutive high-scoring cells along a matrix diagonal
are turned into timed matches.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import mido
import numpy as np
from numpy.typing import NDArray


from .constants import TEMPO_RATIOS


@dataclass(frozen=True)
class NoteEvent:
    """A sounding MIDI note, with times in seconds and velocity in [0, 1]."""

    start: float
    end: float
    pitch: int
    velocity: float


@dataclass(frozen=True)
class WindowVectors:
    """Sliding-window features and their positions in the source recording."""

    starts_seconds: NDArray[np.float64]
    values: NDArray[np.float32]


@dataclass(frozen=True)
class SimilarityConfig:
    """Parameters controlling MIDI vectorization and local-match selection."""

    window_seconds: float = 10.0
    hop_seconds: float = 2.0
    frame_seconds: float = 0.5
    cell_threshold: float = 0.72
    match_threshold: float = 0.78
    minimum_aligned_windows: int = 2
    minimum_match_seconds: float = 10.0
    max_matches: int = 5
    tempo_ratios: tuple[float, ...] = TEMPO_RATIOS

    def __post_init__(self) -> None:
        positive = {
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "frame_seconds": self.frame_seconds,
            "minimum_match_seconds": self.minimum_match_seconds,
            "minimum_aligned_windows": self.minimum_aligned_windows,
            "max_matches": self.max_matches,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.tempo_ratios or any(ratio <= 0 for ratio in self.tempo_ratios):
            raise ValueError("tempo_ratios must contain positive values")
        for name, value in {
            "cell_threshold": self.cell_threshold,
            "match_threshold": self.match_threshold,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        frame_count = self.window_seconds / self.frame_seconds
        if not np.isclose(frame_count, round(frame_count)):
            raise ValueError("window_seconds must be divisible by frame_seconds")


@dataclass(frozen=True)
class Match:
    """One aligned passage shared by the query and reference."""

    query_start: float
    query_end: float
    reference_start: float
    reference_end: float
    score: float
    mean_similarity: float
    transposition: int
    tempo_ratio: float
    aligned_windows: int


@dataclass(frozen=True)
class ComparisonResult:
    """Global decision plus the passages which support it."""

    is_similar: bool
    score: float
    matches: list[Match]

    def to_dict(self) -> dict[str, object]:
        return {
            "is_similar": self.is_similar,
            "score": self.score,
            "matches": [asdict(match) for match in self.matches],
        }


def load_midi_events(path: str | Path) -> list[NoteEvent]:
    """Read all completed MIDI notes and express their timing in seconds."""

    midi = mido.MidiFile(path)
    active: dict[tuple[int, int], deque[tuple[float, int]]] = defaultdict(deque)
    events: list[NoteEvent] = []
    elapsed = 0.0

    # MidiFile iteration merges tracks and converts delta ticks to seconds,
    # including tempo changes.
    for message in midi:
        elapsed += float(message.time)
        if message.type not in {"note_on", "note_off"}:
            continue

        channel = int(getattr(message, "channel", 0))
        key = (channel, int(message.note))
        is_note_on = message.type == "note_on" and message.velocity > 0
        if is_note_on:
            active[key].append((elapsed, int(message.velocity)))
            continue

        if not active[key]:
            continue
        start, velocity = active[key].popleft()
        if elapsed > start:
            events.append(
                NoteEvent(
                    start=start,
                    end=elapsed,
                    pitch=int(message.note),
                    velocity=velocity / 127.0,
                )
            )

    return sorted(events, key=lambda event: (event.start, event.pitch, event.end))


def _window_starts(duration: float, window_seconds: float, hop_seconds: float) -> NDArray[np.float64]:
    if duration <= 0:
        return np.array([], dtype=np.float64)
    if duration <= window_seconds:
        return np.array([0.0], dtype=np.float64)
    count = int(np.ceil((duration - window_seconds) / hop_seconds)) + 1
    return np.arange(count, dtype=np.float64) * hop_seconds


def build_window_vectors(
    events: Iterable[NoteEvent],
    *,
    duration: float,
    window_seconds: float,
    hop_seconds: float,
    frame_seconds: float,
) -> WindowVectors:
    """Build sustain/onset pitch-class tensors for overlapping time windows."""

    if window_seconds <= 0 or hop_seconds <= 0 or frame_seconds <= 0:
        raise ValueError("window, hop, and frame durations must be positive")
    frame_count_float = window_seconds / frame_seconds
    frame_count = round(frame_count_float)
    if not np.isclose(frame_count_float, frame_count):
        raise ValueError("window_seconds must be divisible by frame_seconds")

    starts = _window_starts(duration, window_seconds, hop_seconds)
    values = np.zeros((len(starts), frame_count, 2, 12), dtype=np.float32)

    for event in events:
        if event.end <= event.start or not 0 <= event.pitch <= 127:
            continue
        velocity = float(np.clip(event.velocity, 0.0, 1.0))
        pitch_class = event.pitch % 12
        for window_index, window_start in enumerate(starts):
            window_end = window_start + window_seconds
            if event.end <= window_start or event.start >= window_end:
                continue

            sustain_start = max(event.start, window_start)
            sustain_end = min(event.end, window_end)
            first_frame = max(
                0,
                int(np.floor((sustain_start - window_start) / frame_seconds + 1e-9)),
            )
            last_frame = min(
                frame_count - 1,
                int(np.ceil((sustain_end - window_start) / frame_seconds - 1e-9)) - 1,
            )
            if first_frame <= last_frame:
                values[window_index, first_frame : last_frame + 1, 0, pitch_class] = np.maximum(
                    values[window_index, first_frame : last_frame + 1, 0, pitch_class],
                    velocity,
                )

            if window_start <= event.start < window_end:
                onset_frame = min(
                    frame_count - 1,
                    int((event.start - window_start) / frame_seconds + 1e-9),
                )
                values[window_index, onset_frame, 1, pitch_class] = max(
                    values[window_index, onset_frame, 1, pitch_class],
                    velocity,
                )

    return WindowVectors(starts_seconds=starts, values=values)


def compute_similarity_matrix(
    query: WindowVectors,
    reference: WindowVectors,
) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
    """Compare every window pair, retaining its best pitch transposition."""

    query_count = len(query.starts_seconds)
    reference_count = len(reference.starts_seconds)
    best_scores = np.zeros((query_count, reference_count), dtype=np.float32)
    best_shifts = np.zeros((query_count, reference_count), dtype=np.int8)

    reference_flat = reference.values.reshape(reference_count, -1)
    reference_norms = np.linalg.norm(reference_flat, axis=1, keepdims=True)
    reference_unit = np.divide(
        reference_flat,
        reference_norms,
        out=np.zeros_like(reference_flat),
        where=reference_norms > 0,
    )

    for shift in range(12):
        shifted = np.roll(query.values, shift=shift, axis=-1).reshape(query_count, -1)
        query_norms = np.linalg.norm(shifted, axis=1, keepdims=True)
        query_unit = np.divide(
            shifted,
            query_norms,
            out=np.zeros_like(shifted),
            where=query_norms > 0,
        )
        scores = query_unit @ reference_unit.T
        scores = np.clip(scores, 0.0, 1.0)
        improved = scores > best_scores
        best_scores[improved] = scores[improved]
        signed_shift = shift if shift <= 6 else shift - 12
        best_shifts[improved] = signed_shift

    return best_scores, best_shifts


def _candidate_runs(scores: NDArray[np.float32], threshold: float) -> Iterable[list[tuple[int, int]]]:
    """Yield contiguous high-score runs along each similarity-matrix diagonal."""

    query_count, reference_count = scores.shape
    for offset in range(-query_count + 1, reference_count):
        run: list[tuple[int, int]] = []
        query_start = max(0, -offset)
        query_end = min(query_count, reference_count - offset)
        for query_index in range(query_start, query_end):
            reference_index = query_index + offset
            if scores[query_index, reference_index] >= threshold:
                run.append((query_index, reference_index))
            elif run:
                yield run
                run = []
        if run:
            yield run


def _overlap_ratio(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    shorter = min(end_a - start_a, end_b - start_b)
    return overlap / shorter if shorter > 0 else 0.0


def _deduplicate_matches(matches: list[Match], limit: int) -> list[Match]:
    selected: list[Match] = []
    for candidate in sorted(matches, key=lambda match: match.score, reverse=True):
        duplicate = any(
            _overlap_ratio(
                candidate.query_start,
                candidate.query_end,
                existing.query_start,
                existing.query_end,
            )
            >= 0.8
            and _overlap_ratio(
                candidate.reference_start,
                candidate.reference_end,
                existing.reference_start,
                existing.reference_end,
            )
            >= 0.8
            for existing in selected
        )
        if not duplicate:
            selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def compare_events(
    query_events: Iterable[NoteEvent],
    reference_events: Iterable[NoteEvent],
    *,
    query_duration: float,
    reference_duration: float,
    config: SimilarityConfig | None = None,
) -> ComparisonResult:
    """Detect transposition-invariant, equal-tempo local matches."""

    config = config or SimilarityConfig()
    query = build_window_vectors(
        query_events,
        duration=query_duration,
        window_seconds=config.window_seconds,
        hop_seconds=config.hop_seconds,
        frame_seconds=config.frame_seconds,
    )
    reference_events = list(reference_events)
    candidates: list[Match] = []
    best_cell_score = 0.0
    for tempo_ratio in config.tempo_ratios:
        reference_window = config.window_seconds * tempo_ratio
        reference = build_window_vectors(
            reference_events,
            duration=reference_duration,
            window_seconds=reference_window,
            hop_seconds=config.hop_seconds * tempo_ratio,
            frame_seconds=config.frame_seconds * tempo_ratio,
        )
        scores, transpositions = compute_similarity_matrix(
            query,
            reference,
        )
        best_cell_score = max(best_cell_score, float(np.max(scores, initial=0.0)))

        for run in _candidate_runs(scores, config.cell_threshold):
            if len(run) < config.minimum_aligned_windows:
                continue
            first_query, first_reference = run[0]
            last_query, last_reference = run[-1]
            query_start = float(query.starts_seconds[first_query])
            reference_start = float(reference.starts_seconds[first_reference])
            query_end = min(
                query_duration,
                float(query.starts_seconds[last_query] + config.window_seconds),
            )
            reference_end = min(
                reference_duration,
                float(reference.starts_seconds[last_reference] + reference_window),
            )
            query_span = query_end - query_start
            reference_span = reference_end - reference_start
            matched_query_seconds = min(query_span, reference_span / tempo_ratio)
            if matched_query_seconds < config.minimum_match_seconds:
                continue

            query_indices, reference_indices = zip(*run)
            run_scores = scores[query_indices, reference_indices]
            mean_similarity = float(np.mean(run_scores))
            if mean_similarity < config.match_threshold:
                continue
            shifts = [
                int(transpositions[query_index, reference_index])
                for query_index, reference_index in run
            ]
            transposition = Counter(shifts).most_common(1)[0][0]
            score = mean_similarity * min(
                1.0,
                matched_query_seconds / config.minimum_match_seconds,
            )
            candidates.append(
                Match(
                    query_start=query_start,
                    query_end=query_end,
                    reference_start=reference_start,
                    reference_end=reference_end,
                    score=score,
                    mean_similarity=mean_similarity,
                    transposition=transposition,
                    tempo_ratio=tempo_ratio,
                    aligned_windows=len(run),
                )
            )

    matches = _deduplicate_matches(candidates, config.max_matches)
    global_score = matches[0].score if matches else best_cell_score
    return ComparisonResult(is_similar=bool(matches), score=global_score, matches=matches)


def compare_midi(
    query_path: str | Path,
    reference_path: str | Path,
    *,
    config: SimilarityConfig | None = None,
) -> ComparisonResult:
    """Load two MIDI files, then detect and locate their shared passages."""

    query_events = load_midi_events(query_path)
    reference_events = load_midi_events(reference_path)
    query_duration = max((event.end for event in query_events), default=0.0)
    reference_duration = max((event.end for event in reference_events), default=0.0)
    return compare_events(
        query_events,
        reference_events,
        query_duration=query_duration,
        reference_duration=reference_duration,
        config=config,
    )
