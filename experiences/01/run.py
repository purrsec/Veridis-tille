# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["basic-pitch==0.4.0", "setuptools<81", "spleeter==2.4.2"]
# ///
"""Recreate the historical bass parameter sweep; listening remains manual."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=ROOT / "data/dataset/big-in-japan-alphaville-original/bass.wav")
    args = parser.parse_args()
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, run_inference
    from basic_pitch.note_creation import model_output_to_notes
    prediction = run_inference(args.source, Model(ICASSP_2022_MODEL_PATH))
    trials = (
        ("default", .5, .3, 127.7, None, None, True),
        ("onset", .6, .3, 127.7, None, None, True),
        ("frame035", .5, .35, 127.7, None, None, True),
        ("longer", .5, .3, 160, None, None, True),
        ("mild", .55, .33, 150, None, None, True),
        ("bass-range", .5, .3, 127.7, 30, 400, True),
        ("aggressive", .6, .4, 180, 30, 400, False),
        ("strict", .7, .5, 220, 30, 400, True),
    )
    rows = []
    for name, onset, frame, length, low, high, melodia in trials:
        _, notes = model_output_to_notes(
            {k: v.copy() for k, v in prediction.items()}, onset_thresh=onset,
            frame_thresh=frame, min_note_len=round(length/1000*22050/256),
            min_freq=low, max_freq=high, melodia_trick=melodia,
        )
        rows.append(dict(trial=name, onset=onset, frame=frame, minimum_ms=length,
                         minimum_hz=low, maximum_hz=high, melodia=melodia, notes=len(notes)))
    with (Path(__file__).parent / "parameter-sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote parameter-sweep.csv. Historical listening results used CoreML; counts can differ by backend.")


if __name__ == "__main__":
    main()
