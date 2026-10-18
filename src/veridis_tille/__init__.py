"""Tools for detecting and locating musical similarity."""

from .midi_similarity import (
    ComparisonResult,
    Match,
    NoteEvent,
    SimilarityConfig,
    compare_midi,
)
from .pipeline import Calibration, PairComparison, compare_tracks

__all__ = [
    "ComparisonResult",
    "Match",
    "NoteEvent",
    "SimilarityConfig",
    "compare_midi",
    "Calibration",
    "PairComparison",
    "compare_tracks",
]
