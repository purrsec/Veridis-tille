"""The retained pairwise detector: synchronized MIDI OR continuous mix chroma."""

from __future__ import annotations

import json
import math
import wave
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np

from .chroma import ChromaWindows, build_windows, compare_chroma, extract_chroma
from .constants import FRAME_SECONDS, HOP_SECONDS, STEMS, TEMPO_RATIOS, WINDOW_SECONDS
from .midi_similarity import WindowVectors, build_window_vectors, compute_similarity_matrix, load_midi_events


@dataclass(frozen=True)
class Calibration:
    midi_threshold: float
    chroma_threshold: float
    name: str = "custom"

    def __post_init__(self) -> None:
        for value in (self.midi_threshold, self.chroma_threshold):
            if not math.isfinite(value) or value < 0:
                raise ValueError("thresholds must be finite and nonnegative")

    @classmethod
    def load(cls, path: str | Path | None = None) -> Calibration:
        """Load explicit thresholds; the bundled profile is exploratory POC data."""
        source = Path(path) if path is not None else files("veridis_tille").joinpath("profiles/poc.json")
        data = json.loads(source.read_text(encoding="utf-8"))
        return cls(float(data["midi_threshold"]), float(data["chroma_threshold"]), data["name"])


@dataclass(frozen=True)
class BranchMatch:
    score: float
    query_start: float | None
    query_end: float | None
    reference_start: float | None
    reference_end: float | None
    tempo_ratio: float
    transpositions: dict[str, int]


@dataclass(frozen=True)
class PairComparison:
    query: str
    reference: str
    is_similar: bool
    matched_by: tuple[str, ...]
    midi: BranchMatch
    chroma: BranchMatch
    calibration: Calibration

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrackFeatures:
    source: Path
    duration: float
    midi: dict[str, dict[float, WindowVectors]]
    chroma: dict[float, ChromaWindows]


def load_features(source: str | Path, *, reference: bool = False) -> TrackFeatures:
    """Read a master and its flat companion MIDI files, without fitting anything."""
    source = Path(source)
    with wave.open(str(source), "rb") as stream:
        seconds = stream.getnframes() / stream.getframerate()
    if seconds <= 0:
        raise ValueError(f"empty audio: {source}")
    ratios = TEMPO_RATIOS if reference else (1.0,)
    midi = {}
    for stem in STEMS:
        path = source.with_suffix("") / f"{stem}_basic_pitch.mid"
        events = load_midi_events(path)
        midi[stem] = {
            ratio: build_window_vectors(
                events, duration=seconds, window_seconds=WINDOW_SECONDS * ratio,
                hop_seconds=HOP_SECONDS * ratio, frame_seconds=FRAME_SECONDS * ratio,
            ) for ratio in ratios
        }
    track = extract_chroma(source)
    return TrackFeatures(source, seconds, midi, {r: build_windows(track, r) for r in ratios})


def compare_midi_stems(query: TrackFeatures, reference: TrackFeatures) -> BranchMatch:
    """Average bass/other at the same times and tempo before selecting a maximum."""
    best = BranchMatch(0.0, None, None, None, None, 1.0, {})
    for ratio in TEMPO_RATIOS:
        matrices, shifts = {}, {}
        for stem in STEMS:
            matrices[stem], shifts[stem] = compute_similarity_matrix(
                query.midi[stem][1.0], reference.midi[stem][ratio])
        fused = np.mean(np.stack([matrices[s] for s in STEMS]), axis=0)
        qi, ri = np.unravel_index(int(np.argmax(fused)), fused.shape)
        score = float(fused[qi, ri])
        if score <= best.score:
            continue
        qs = float(query.midi[STEMS[0]][1.0].starts_seconds[qi])
        rs = float(reference.midi[STEMS[0]][ratio].starts_seconds[ri])
        best = BranchMatch(
            score, qs, min(query.duration, qs + WINDOW_SECONDS),
            rs, min(reference.duration, rs + WINDOW_SECONDS * ratio), ratio,
            {s: int(shifts[s][qi, ri]) for s in STEMS},
        )
    return best


def compare_features(
    query: TrackFeatures, reference: TrackFeatures, *, calibration: Calibration | None = None,
) -> PairComparison:
    """Decide for one proposed pair; no catalog, labels or research files are read."""
    calibration = calibration or Calibration.load()
    midi = compare_midi_stems(query, reference)
    local = compare_chroma(query.chroma[1.0], reference.chroma)
    span = WINDOW_SECONDS + HOP_SECONDS * (local.aligned_windows - 1)
    chroma = BranchMatch(
        local.score, local.query_start, min(query.duration, local.query_start + span),
        local.reference_start, min(reference.duration, local.reference_start + span * local.tempo_ratio),
        local.tempo_ratio, {"mix": local.transposition},
    )
    matched = tuple(name for name, active in (
        ("midi", midi.score >= calibration.midi_threshold),
        ("chroma", chroma.score >= calibration.chroma_threshold),
    ) if active)
    return PairComparison(str(query.source), str(reference.source), bool(matched),
                          matched, midi, chroma, calibration)


def compare_tracks(
    query: str | Path, reference: str | Path, *, calibration: Calibration | None = None,
    prepare: bool = True,
) -> PairComparison:
    """Compare two WAV masters, preparing missing Spleeter/Basic Pitch files if needed."""
    if prepare:
        from .preprocessing import prepare_audio
        prepare_audio([query, reference])
    return compare_features(load_features(query), load_features(reference, reference=True),
                            calibration=calibration)
