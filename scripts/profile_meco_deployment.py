#!/usr/bin/env python3
"""Measure synthetic trained-scorer microbenchmarks on CPU and ROCm."""
import argparse
import json
from pathlib import Path

from frc_defense.meco_deployment import profile_report
from frc_defense.meco_tracks import MECO_TRACKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", choices=MECO_TRACKS, default=list(MECO_TRACKS))
    parser.add_argument("--checkpoint-root", type=Path, required=True,
                        help="training root containing TRACK/latest.pt for every requested track")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("evaluations/meco-deployment/profile.json"))
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    report = profile_report(args.tracks, args.checkpoint_root, args.iterations, args.warmup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
