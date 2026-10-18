"""Retained full-mix chroma extraction and local alignment (experiment 08)."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ChromaTrack:
    values: NDArray[np.float32]
    frame_seconds: float
    duration: float


@dataclass(frozen=True)
class ChromaWindows:
    starts: NDArray[np.float64]
    values: NDArray[np.float32]


@dataclass(frozen=True)
class LocalResult:
    score: float
    query_start: float
    reference_start: float
    tempo_ratio: float
    transposition: int
    aligned_windows: int


def read_wav_mono(path: Path) -> tuple[NDArray[np.float32], int]:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        sample_rate = stream.getframerate()
        frame_count = stream.getnframes()
        payload = stream.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"{path.name}: expected 16-bit PCM")
    audio = np.frombuffer(payload, dtype="<i2").reshape(-1, channels)
    mono = audio.astype(np.float32).mean(axis=1) / 32768.0
    return mono, sample_rate


def chroma_projection(sample_rate: int, fft_size: int) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    selected = np.flatnonzero((frequencies >= 65.0) & (frequencies <= 4200.0))
    midi = 69.0 + 12.0 * np.log2(frequencies[selected] / 440.0)
    lower = np.floor(midi).astype(np.int64)
    fraction = (midi - lower).astype(np.float32)
    projection = np.zeros((len(selected), 12), dtype=np.float32)
    indices = np.arange(len(selected))
    projection[indices, lower % 12] += 1.0 - fraction
    projection[indices, (lower + 1) % 12] += fraction
    return selected, projection


def extract_chroma(
    path: Path,
    *,
    frame_seconds: float = 0.5,
    fft_size: int = 8192,
    hop_size: int = 2048,
) -> ChromaTrack:
    audio, sample_rate = read_wav_mono(path)
    return extract_chroma_samples(
        audio, sample_rate, frame_seconds=frame_seconds,
        fft_size=fft_size, hop_size=hop_size,
    )


def extract_chroma_samples(
    audio: NDArray[np.float32],
    sample_rate: int,
    *,
    frame_seconds: float = 0.5,
    fft_size: int = 8192,
    hop_size: int = 2048,
) -> ChromaTrack:
    """Apply the identical extractor to in-memory, potentially floating WAV audio."""
    duration = len(audio) / sample_rate
    if len(audio) < fft_size:
        audio = np.pad(audio, (0, fft_size - len(audio)))
    frame_count = 1 + (len(audio) - fft_size) // hop_size
    starts = np.arange(frame_count, dtype=np.int64) * hop_size
    selected, projection = chroma_projection(sample_rate, fft_size)
    window = np.hanning(fft_size).astype(np.float32)
    time_bins = max(1, int(np.ceil(duration / frame_seconds)))
    accumulated = np.zeros((time_bins, 12), dtype=np.float32)
    counts = np.zeros(time_bins, dtype=np.int32)

    for batch_start in range(0, frame_count, 256):
        batch_starts = starts[batch_start : batch_start + 256]
        frames = np.stack([audio[start : start + fft_size] for start in batch_starts])
        magnitudes = np.abs(np.fft.rfft(frames * window, axis=1))[:, selected]
        # Square-root compression prevents a few loud partials from dominating.
        chroma = np.sqrt(magnitudes, dtype=np.float32) @ projection
        norms = np.linalg.norm(chroma, axis=1, keepdims=True)
        chroma = np.divide(chroma, norms, out=np.zeros_like(chroma), where=norms > 0)
        centers = (batch_starts + fft_size / 2) / sample_rate
        bin_indices = np.minimum((centers / frame_seconds).astype(np.int64), time_bins - 1)
        np.add.at(accumulated, bin_indices, chroma)
        np.add.at(counts, bin_indices, 1)

    accumulated = np.divide(
        accumulated,
        counts[:, None],
        out=np.zeros_like(accumulated),
        where=counts[:, None] > 0,
    )
    # Remove part of the broadband floor before the final frame normalization.
    accumulated = np.maximum(
        accumulated - 0.5 * np.median(accumulated, axis=1, keepdims=True),
        0.0,
    )
    norms = np.linalg.norm(accumulated, axis=1, keepdims=True)
    accumulated = np.divide(
        accumulated,
        norms,
        out=np.zeros_like(accumulated),
        where=norms > 0,
    )
    return ChromaTrack(accumulated, frame_seconds, duration)


def window_starts(duration: float, window_seconds: float, hop_seconds: float) -> NDArray[np.float64]:
    if duration <= window_seconds:
        return np.asarray([0.0], dtype=np.float64)
    count = int(np.ceil((duration - window_seconds) / hop_seconds)) + 1
    return np.arange(count, dtype=np.float64) * hop_seconds


def build_windows(track: ChromaTrack, tempo_ratio: float) -> ChromaWindows:
    window_seconds = 10.0 * tempo_ratio
    hop_seconds = 2.0 * tempo_ratio
    starts = window_starts(track.duration, window_seconds, hop_seconds)
    sample_offsets = (np.arange(20, dtype=np.float64) + 0.5) * 0.5 * tempo_ratio
    source_times = (np.arange(len(track.values), dtype=np.float64) + 0.5) * track.frame_seconds
    windows = np.empty((len(starts), 20, 12), dtype=np.float32)
    for window_index, start in enumerate(starts):
        targets = start + sample_offsets
        for pitch_class in range(12):
            windows[window_index, :, pitch_class] = np.interp(
                targets,
                source_times,
                track.values[:, pitch_class],
                left=0.0,
                right=0.0,
            )
    return ChromaWindows(starts, windows)


def similarity_matrix(
    query: ChromaWindows,
    reference: ChromaWindows,
) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
    reference_flat = reference.values.reshape(len(reference.starts), -1)
    reference_norm = np.linalg.norm(reference_flat, axis=1, keepdims=True)
    reference_unit = np.divide(
        reference_flat,
        reference_norm,
        out=np.zeros_like(reference_flat),
        where=reference_norm > 0,
    )
    best_scores = np.zeros((len(query.starts), len(reference.starts)), dtype=np.float32)
    best_shifts = np.zeros(best_scores.shape, dtype=np.int8)
    for shift in range(12):
        shifted = np.roll(query.values, shift=shift, axis=-1).reshape(len(query.starts), -1)
        query_norm = np.linalg.norm(shifted, axis=1, keepdims=True)
        query_unit = np.divide(
            shifted,
            query_norm,
            out=np.zeros_like(shifted),
            where=query_norm > 0,
        )
        scores = np.clip(query_unit @ reference_unit.T, 0.0, 1.0)
        improved = scores > best_scores
        best_scores[improved] = scores[improved]
        best_shifts[improved] = shift if shift <= 6 else shift - 12
    return best_scores, best_shifts


def best_diagonal_run(
    scores: NDArray[np.float32],
    shifts: NDArray[np.int8],
    length: int,
) -> tuple[float, int, int, int]:
    best = (0.0, 0, 0, 0)
    query_count, reference_count = scores.shape
    for offset in range(-query_count + 1, reference_count):
        query_start = max(0, -offset)
        query_end = min(query_count, reference_count - offset)
        if query_end - query_start < length:
            continue
        diagonal_scores = np.asarray(
            [scores[index, index + offset] for index in range(query_start, query_end)]
        )
        means = np.convolve(diagonal_scores, np.ones(length) / length, mode="valid")
        local_index = int(np.argmax(means))
        score = float(means[local_index])
        query_index = query_start + local_index
        reference_index = query_index + offset
        run_shifts = shifts[
            query_index : query_index + length,
            reference_index : reference_index + length,
        ].diagonal()
        transposition = int(round(float(np.median(run_shifts))))
        if score > best[0]:
            best = (score, query_index, reference_index, transposition)
    return best


def compare_chroma(
    query: ChromaWindows,
    references: dict[float, ChromaWindows],
) -> LocalResult:
    """Retained score: three consecutive aligned windows, over six tempo ratios."""
    best = LocalResult(0.0, 0.0, 0.0, 1.0, 0, 3)
    for ratio, reference in references.items():
        scores, shifts = similarity_matrix(query, reference)
        score, qi, ri, shift = best_diagonal_run(scores, shifts, 3)
        if score > best.score:
            best = LocalResult(score, float(query.starts[qi]), float(reference.starts[ri]),
                               ratio, shift, 3)
    return best
