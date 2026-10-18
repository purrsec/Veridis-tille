from __future__ import annotations

import json
from pathlib import Path
import wave

import mido
import numpy as np
import pytest

from veridis_tille.cli import main
from veridis_tille.pipeline import Calibration, compare_tracks


def write_track(path: Path, *, offset: float, duration: float) -> None:
    sr = 44100
    times = np.arange(int(duration * sr)) / sr
    audio = np.zeros(len(times), dtype=np.float64)
    pitches = (36, 40, 43, 38, 41, 45, 40, 43)
    notes = []
    for i, pitch in enumerate(pitches):
        start = offset + 0.2 + i * 1.5
        end = start + 1.1
        selected = (times >= start) & (times < end)
        audio[selected] = 0.3 * np.sin(2*np.pi*440*2**((pitch-69)/12)*times[selected])
        notes.extend([(round(start*960), "note_on", pitch, 80),
                      (round(end*960), "note_off", pitch, 0)])
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, sr, 0, "NONE", "not compressed"))
        stream.writeframes((audio*32767).astype("<i2").tobytes())
    path.with_suffix("").mkdir()
    for stem in ("bass", "other"):
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=500000))
        previous = 0
        for tick, type_, pitch, velocity in sorted(notes):
            track.append(mido.Message(type_, note=pitch, velocity=velocity, time=tick-previous))
            previous = tick
        midi.save(path.with_suffix("") / f"{stem}_basic_pitch.mid")


def test_retained_pipeline_locates_a_copy_with_arbitrary_filenames(tmp_path):
    query, reference = tmp_path / "unknown.wav", tmp_path / "candidate.wav"
    write_track(query, offset=0.0, duration=16.0)
    write_track(reference, offset=4.0, duration=24.0)
    result = compare_tracks(query, reference, prepare=False)
    assert result.is_similar
    assert result.midi.score == pytest.approx(1.0)
    assert result.midi.query_start == 0.0
    assert result.midi.reference_start == 4.0
    assert result.midi.tempo_ratio == 1.0
    assert "midi" in result.matched_by
    assert result.calibration.name == "poc-eight-families"


def test_cli_outputs_json_for_two_prepared_masters(tmp_path, capsys):
    query, reference = tmp_path / "one.wav", tmp_path / "two.wav"
    write_track(query, offset=0.0, duration=16.0)
    write_track(reference, offset=4.0, duration=24.0)
    assert main(["compare", str(query), str(reference), "--no-prepare"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["is_similar"] is True
    assert result["midi"]["reference_start"] == 4.0
    assert "calibration" in result


def test_uncalibrated_nan_cannot_silently_disable_a_branch():
    with pytest.raises(ValueError):
        Calibration(float("nan"), 0.9)


def test_missing_analysis_is_reported_without_starting_models(tmp_path, capsys):
    query = tmp_path / "missing.wav"
    write_track(query, offset=0.0, duration=16.0)
    (query.with_suffix("") / "bass_basic_pitch.mid").unlink()
    with pytest.raises(SystemExit) as error:
        main(["compare", str(query), str(query), "--no-prepare"])
    assert error.value.code == 2
    assert "bass_basic_pitch.mid" in capsys.readouterr().err
