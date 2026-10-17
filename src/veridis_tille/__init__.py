"""Tools for detecting and locating musical similarity."""

from .midi_similarity import (
    ComparisonResult,
    Match,
    NoteEvent,
    SimilarityConfig,
    compare_midi,
)
from .drum_similarity import DrumComparisonResult, DrumMatch, DrumSimilarityConfig, compare_drum_audio

__all__ = [
    "ComparisonResult",
    "Match",
    "NoteEvent",
    "SimilarityConfig",
    "compare_midi",
    "DrumComparisonResult",
    "DrumMatch",
    "DrumSimilarityConfig",
    "compare_drum_audio",
]
