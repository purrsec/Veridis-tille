#!/usr/bin/env python3
"""Compare pitch-class vectors with transposition-invariant melodic contours."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
import re
from dataclasses import dataclass
from statistics import mean, median
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from experiences._shared.midi_similarity import NoteEvent, compare_events, load_midi_events


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data" / "dataset"
OUTPUT_DIR = ROOT / "experiences" / "05"
RESULTS_PATH = OUTPUT_DIR / "melodic-contour-experiment.csv"
REPORT_PATH = ROOT / "experiences" / "05" / "rapport.md"
STEMS = ("bass", "piano", "vocals", "other")
TEMPO_RATIOS = (0.8, 0.9, 1.0, 1.125, 1.2, 1.25)
VARIANT_SUFFIX = re.compile(r"-(?:original|AI-[A-Za-z0-9]+)$")
ContourRepresentation = Literal[
    "direction",
    "coarse_interval",
    "exact_interval",
    "direction_exact",
    "relative_first",
    "relative_median_coarse",
]
CONTOUR_REPRESENTATIONS: tuple[ContourRepresentation, ...] = (
    "direction",
    "coarse_interval",
    "exact_interval",
    "direction_exact",
    "relative_first",
    "relative_median_coarse",
)
REPRESENTATIONS = ("pitch_class", *CONTOUR_REPRESENTATIONS)
FIELDS = (
    "representation",
    "query",
    "reference",
    "expected_pair",
    "feature",
    "score",
    "query_start",
    "reference_start",
    "tempo_ratio",
)


@dataclass(frozen=True)
class VectorWindows:
    starts: NDArray[np.float64]
    values: NDArray[np.float32]


def track_key(path: Path | str) -> str:
    name = path.name if isinstance(path, Path) else path
    return VARIANT_SUFFIX.sub("", name)


def directories() -> tuple[list[Path], list[Path]]:
    paths = sorted(path for path in DATASET_DIR.iterdir() if path.is_dir())
    return (
        [path for path in paths if "-AI-" in path.name],
        [path for path in paths if path.name.endswith("-original")],
    )


def window_starts(
    duration: float,
    window_seconds: float,
    hop_seconds: float,
) -> NDArray[np.float64]:
    if duration <= 0:
        return np.array([], dtype=np.float64)
    if duration <= window_seconds:
        return np.array([0.0], dtype=np.float64)
    count = int(np.ceil((duration - window_seconds) / hop_seconds)) + 1
    return np.arange(count, dtype=np.float64) * hop_seconds


def melody_notes(
    events: list[NoteEvent],
    *,
    simultaneous_tolerance: float = 0.08,
) -> list[NoteEvent]:
    """Collapse near-simultaneous Basic Pitch notes to one melodic candidate."""

    ordered = sorted(events, key=lambda event: (event.start, -event.velocity, -event.end))
    groups: list[list[NoteEvent]] = []
    for event in ordered:
        if (
            not groups
            or event.start - groups[-1][0].start > simultaneous_tolerance
        ):
            groups.append([event])
        else:
            groups[-1].append(event)
    return [
        max(group, key=lambda event: (event.velocity, event.end - event.start))
        for group in groups
    ]


def interval_category(interval: float, representation: str) -> tuple[int, int]:
    if representation in {"direction", "direction_exact"}:
        category = (
            0
            if interval <= -3
            else 1
            if interval < 0
            else 2
            if interval == 0
            else 3
            if interval < 3
            else 4
        )
        return category, 5
    if representation == "coarse_interval":
        category = (
            0
            if interval <= -8
            else 1
            if interval <= -4
            else 2
            if interval <= -2
            else 3
            if interval < 0
            else 4
            if interval == 0
            else 5
            if interval < 2
            else 6
            if interval < 4
            else 7
            if interval < 8
            else 8
        )
        return category, 9
    return int(np.clip(round(interval), -12, 12)) + 12, 25


def build_interval_vectors(
    events: list[NoteEvent],
    *,
    duration: float,
    window_seconds: float,
    hop_seconds: float,
    frame_seconds: float,
    representation: ContourRepresentation,
) -> VectorWindows:
    notes = melody_notes(events)
    intervals = [
        (
            following.start,
            following.pitch - previous.pitch,
            float(np.sqrt(previous.velocity * following.velocity)),
        )
        for previous, following in zip(notes, notes[1:])
        if following.start - previous.start <= 3.0
    ]
    starts = window_starts(duration, window_seconds, hop_seconds)
    frame_count = round(window_seconds / frame_seconds)
    dimensions = (
        30
        if representation == "direction_exact"
        else interval_category(0, representation)[1]
    )
    values = np.zeros((len(starts), frame_count, dimensions), dtype=np.float32)

    for window_index, window_start in enumerate(starts):
        window_end = window_start + window_seconds
        for onset, interval, velocity in intervals:
            if not window_start <= onset < window_end:
                continue
            frame = min(frame_count - 1, int((onset - window_start) / frame_seconds))
            category, _ = interval_category(interval, representation)
            values[window_index, frame, category] = max(
                values[window_index, frame, category], velocity
            )
            if representation == "direction_exact":
                exact, _ = interval_category(interval, "exact_interval")
                values[window_index, frame, 5 + exact] = max(
                    values[window_index, frame, 5 + exact], velocity
                )

    return VectorWindows(starts=starts, values=values)


def build_relative_vectors(
    events: list[NoteEvent],
    *,
    duration: float,
    window_seconds: float,
    hop_seconds: float,
    frame_seconds: float,
    representation: ContourRepresentation,
) -> VectorWindows:
    notes = melody_notes(events)
    starts = window_starts(duration, window_seconds, hop_seconds)
    frame_count = round(window_seconds / frame_seconds)
    dimensions = 25 if representation == "relative_first" else 9
    values = np.zeros((len(starts), frame_count, dimensions), dtype=np.float32)

    for window_index, window_start in enumerate(starts):
        window_end = window_start + window_seconds
        selected = [note for note in notes if window_start <= note.start < window_end]
        if not selected:
            continue
        anchor = (
            selected[0].pitch
            if representation == "relative_first"
            else float(np.median([note.pitch for note in selected]))
        )
        category_mode = (
            "exact_interval"
            if representation == "relative_first"
            else "coarse_interval"
        )
        for note in selected:
            frame = min(
                frame_count - 1,
                int((note.start - window_start) / frame_seconds),
            )
            category, _ = interval_category(note.pitch - anchor, category_mode)
            values[window_index, frame, category] = max(
                values[window_index, frame, category], note.velocity
            )

    return VectorWindows(starts=starts, values=values)


def build_contour_vectors(
    events: list[NoteEvent],
    *,
    duration: float,
    tempo_ratio: float,
    representation: ContourRepresentation,
) -> VectorWindows:
    arguments = {
        "duration": duration,
        "window_seconds": 10.0 * tempo_ratio,
        "hop_seconds": 2.0 * tempo_ratio,
        "frame_seconds": 0.5 * tempo_ratio,
        "representation": representation,
    }
    if representation.startswith("relative_"):
        return build_relative_vectors(events, **arguments)
    return build_interval_vectors(events, **arguments)


def normalized_rows(values: NDArray[np.float32]) -> NDArray[np.float32]:
    flattened = values.reshape(len(values), -1)
    norms = np.linalg.norm(flattened, axis=1, keepdims=True)
    return np.divide(
        flattened,
        norms,
        out=np.zeros_like(flattened),
        where=norms > 0,
    )


def best_contour_similarity(
    query: VectorWindows,
    references: dict[float, VectorWindows],
) -> tuple[float, float, float, float]:
    query_values = normalized_rows(query.values)
    best_score = 0.0
    best_query_start = 0.0
    best_reference_start = 0.0
    best_ratio = 1.0
    for ratio, reference in references.items():
        if not len(query.starts) or not len(reference.starts):
            continue
        scores = query_values @ normalized_rows(reference.values).T
        flat_index = int(np.argmax(scores))
        query_index, reference_index = np.unravel_index(flat_index, scores.shape)
        score = float(scores[query_index, reference_index])
        if score > best_score:
            best_score = score
            best_query_start = float(query.starts[query_index])
            best_reference_start = float(reference.starts[reference_index])
            best_ratio = ratio
    return best_score, best_query_start, best_reference_start, best_ratio


def load_cache(
    paths: list[Path],
) -> dict[tuple[str, str], tuple[list[NoteEvent], float]]:
    cache: dict[tuple[str, str], tuple[list[NoteEvent], float]] = {}
    for path in paths:
        for stem in STEMS:
            events = load_midi_events(path / f"{stem}_basic_pitch.mid")
            duration = max((event.end for event in events), default=0.0)
            cache[(path.name, stem)] = (events, duration)
    return cache


def result_row(
    *,
    representation: str,
    query: str,
    reference: str,
    feature: str,
    score: float,
    query_start: float | None = None,
    reference_start: float | None = None,
    tempo_ratio: float | None = None,
) -> dict[str, object]:
    return {
        "representation": representation,
        "query": query,
        "reference": reference,
        "expected_pair": str(track_key(query) == track_key(reference)).lower(),
        "feature": feature,
        "score": f"{score:.9f}",
        "query_start": "" if query_start is None else f"{query_start:.6f}",
        "reference_start": (
            "" if reference_start is None else f"{reference_start:.6f}"
        ),
        "tempo_ratio": "" if tempo_ratio is None else f"{tempo_ratio:.6f}",
    }


def compare_all() -> list[dict[str, object]]:
    queries, references = directories()
    event_cache = load_cache([*queries, *references])
    rows: list[dict[str, object]] = []

    print("[1/7] representation=pitch_class", flush=True)
    for query in queries:
        for reference in references:
            for stem in STEMS:
                query_events, query_duration = event_cache[(query.name, stem)]
                reference_events, reference_duration = event_cache[(reference.name, stem)]
                result = compare_events(
                    query_events,
                    reference_events,
                    query_duration=query_duration,
                    reference_duration=reference_duration,
                )
                best = result.matches[0] if result.matches else None
                rows.append(
                    result_row(
                        representation="pitch_class",
                        query=query.name,
                        reference=reference.name,
                        feature=stem,
                        score=result.score,
                        query_start=None if best is None else best.query_start,
                        reference_start=None if best is None else best.reference_start,
                        tempo_ratio=None if best is None else best.tempo_ratio,
                    )
                )

    for representation_index, representation in enumerate(
        CONTOUR_REPRESENTATIONS, start=2
    ):
        print(
            f"[{representation_index}/7] representation={representation}",
            flush=True,
        )
        vector_cache: dict[tuple[str, str, float], VectorWindows] = {}
        for (name, stem), (events, duration) in event_cache.items():
            vector_cache[(name, stem, 1.0)] = build_contour_vectors(
                events,
                duration=duration,
                tempo_ratio=1.0,
                representation=representation,
            )
            if name.endswith("-original"):
                for ratio in TEMPO_RATIOS:
                    vector_cache[(name, stem, ratio)] = build_contour_vectors(
                        events,
                        duration=duration,
                        tempo_ratio=ratio,
                        representation=representation,
                    )

        for query in queries:
            for reference in references:
                for stem in STEMS:
                    score, query_start, reference_start, ratio = best_contour_similarity(
                        vector_cache[(query.name, stem, 1.0)],
                        {
                            candidate_ratio: vector_cache[
                                (reference.name, stem, candidate_ratio)
                            ]
                            for candidate_ratio in TEMPO_RATIOS
                        },
                    )
                    rows.append(
                        result_row(
                            representation=representation,
                            query=query.name,
                            reference=reference.name,
                            feature=stem,
                            score=score,
                            query_start=query_start,
                            reference_start=reference_start,
                            tempo_ratio=ratio,
                        )
                    )
    return rows


def flag(row: dict[str, object], name: str) -> bool:
    return row[name] == "true"


def score(row: dict[str, object]) -> float:
    return float(row["score"])


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def rank_metrics(
    rows: list[dict[str, object]], representation: str, stem: str
) -> tuple[int, float, float, float]:
    selected = [
        row
        for row in rows
        if row["representation"] == representation and row["feature"] == stem
    ]
    queries = sorted({str(row["query"]) for row in selected})
    margins: list[float] = []
    for query in queries:
        query_rows = [row for row in selected if row["query"] == query]
        positive = next(row for row in query_rows if flag(row, "expected_pair"))
        negative = max(
            score(row) for row in query_rows if not flag(row, "expected_pair")
        )
        margins.append(score(positive) - negative)
    positives = [score(row) for row in selected if flag(row, "expected_pair")]
    negatives = [score(row) for row in selected if not flag(row, "expected_pair")]
    return (
        sum(margin > 1e-9 for margin in margins),
        mean(margins),
        median(positives),
        median(negatives),
    )


def rasputin_metrics(
    rows: list[dict[str, object]], representation: str, stem: str
) -> tuple[int, float]:
    selected = sorted(
        (
            row
            for row in rows
            if row["representation"] == representation
            and row["feature"] == stem
            and row["query"] == "rasputin-boney-m-AI-GEMA"
        ),
        key=score,
        reverse=True,
    )
    rank = next(
        index
        for index, row in enumerate(selected, start=1)
        if flag(row, "expected_pair")
    )
    positive = next(score(row) for row in selected if flag(row, "expected_pair"))
    return rank, positive


def write_results(rows: list[dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, object]]) -> None:
    summary_rows: list[tuple[object, ...]] = []
    rasputin_rows: list[tuple[object, ...]] = []
    for representation in REPRESENTATIONS:
        for stem in STEMS:
            ranks, margin, positive_median, negative_median = rank_metrics(
                rows, representation, stem
            )
            summary_rows.append(
                (
                    representation,
                    stem,
                    f"{ranks}/8",
                    f"{margin:+.3f}",
                    f"{positive_median:.3f}",
                    f"{negative_median:.3f}",
                )
            )
            rank, rasputin_score = rasputin_metrics(rows, representation, stem)
            rasputin_rows.append(
                (representation, stem, f"{rank}/8", f"{rasputin_score:.3f}")
            )

    content = f"""# Melodic interval and contour experiment

## Scope

The pitch-class baseline already searches all twelve transpositions. This
experiment tests whether a more explicit melody representation can tolerate
pitching and Basic Pitch errors better. Notes starting within 80 ms are reduced
to the strongest melodic candidate. Six representations are compared with the
same 10-second windows, 2-second hops, tempo ratios, 64 query/reference pairs,
and four MIDI stems:

- `direction`: five descending, repeated, or ascending movement classes;
- `coarse_interval`: nine signed interval ranges;
- `exact_interval`: signed semitone intervals clipped to one octave;
- `direction_exact`: direction and exact interval channels together;
- `relative_first`: every pitch relative to the first note in its window;
- `relative_median_coarse`: coarse pitch position around the window median.

Contour scores are maximum window cosine similarities. They are evaluated as
retrieval signals only: no detection threshold was fitted on this eight-pair
POC.

## Full-dataset results

{markdown_table(
    (
        "Representation",
        "Stem",
        "Ranks first",
        "Mean margin",
        "Expected median",
        "Other median",
    ),
    summary_rows,
)}

## Rasputin

{markdown_table(
    ("Representation", "Stem", "Expected rank", "Expected score"),
    rasputin_rows,
)}

## Conclusion

The explicit contour hypothesis does not improve the detector globally. The
pitch-class baseline remains strongest with 5/8 rank-one results for bass and
vocals. Directional contour moves the Rasputin vocal pair from rank 8 to rank
2, and `relative_median_coarse` also places it second, but those
representations retrieve only 2/8 and 3/8 expected vocal pairs respectively.
The apparent Rasputin gain therefore does not generalize.

`relative_first` keeps bass at 5/8 and places Rasputin bass first, but its mean
margin is negative. Exact intervals amplify transcription errors and perform
worse than coarse direction for the difficult vocal example.

These results support keeping pitch-class vectors as the melodic baseline.
The useful Rasputin evidence still comes from combining non-vocal stems; the
next experiment should evaluate calibrated multi-stem score fusion rather than
replace the current representation with contours.

The complete results are stored in
`experiences/05/melodic-contour-experiment.csv`.
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> int:
    rows = compare_all()
    write_results(rows)
    write_report(rows)
    print(f"Wrote {len(rows)} comparisons to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote report to {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
