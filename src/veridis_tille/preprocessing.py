"""Run the retained Spleeter/Basic Pitch preparation in its Python 3.11 environment."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import subprocess
import sys

from .constants import STEMS

ALL_STEMS = ("vocals", "drums", "bass", "piano", "other")


def analysis_files(source: str | Path) -> list[Path]:
    directory = Path(source).with_suffix("")
    return [directory / f"{stem}{suffix}" for stem in ALL_STEMS
            for suffix in (".wav", "_basic_pitch.mid", "_basic_pitch.csv")]


def prepare_audio(
    sources: Iterable[str | Path], *, force: bool = False, complete: bool = False,
) -> None:
    """Keep each master intact and write its analysis into one flat companion directory."""
    pending = []
    for source in dict.fromkeys(Path(p).resolve() for p in sources):
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() != ".wav":
            raise ValueError(f"expected a WAV master: {source}")
        required = analysis_files(source) if complete else [
            source.with_suffix("") / f"{stem}_basic_pitch.mid" for stem in STEMS]
        if force or not all(p.is_file() and p.stat().st_size > 0 for p in required):
            pending.append(source)
    if not pending:
        return
    worker = Path(__file__).with_name("_preprocess_worker.py")
    command = ["uv", "run", "--script", str(worker), *map(str, pending)]
    if force:
        command.append("--force")
    # Model progress goes to stderr, preserving machine-readable CLI JSON on stdout.
    subprocess.run(command, check=True, stdout=sys.stderr)
