# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["basic-pitch==0.4.0", "setuptools<81", "spleeter==2.4.2"]
# ///
"""Reproduce experiment 10; cached audio stays in sandbox/."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiences._shared import pitch_ablation as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=('extract', 'evaluate', 'report', 'all'), default="all")
    args = parser.parse_args()
    if args.stage == "all":
        for stage in ('extract', 'evaluate'):
            subprocess.run([sys.executable, __file__, "--stage", stage], check=True)
    else:
        sys.argv = [__file__, "--stage", {'extract': 'extract', 'evaluate': 'evaluate', 'report': 'report'}[args.stage]]
        experiment.main()


if __name__ == "__main__":
    main()
