#!/usr/bin/env python3
"""Run the frozen MECO A/B matrix on one shared checkpoint/seed bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense.meco_ablation_matrix import run_matrix
from frc_defense.meco_tracks import MECO_TRACKS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--categories", required=True,
                        help="comma-separated exact curriculum categories")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261024,
                        help="first seed; the matrix uses a contiguous explicit seed list")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "evaluations" / "meco-ablation-matrix")
    parser.add_argument("--inference-disable-tick", type=int,
                        help="requested physics tick for the paired D1 fallback condition")
    parser.add_argument("--qualification-protocol", type=Path,
                        help="frozen protocol whose validation/test seeds the matrix must use")
    for track in MECO_TRACKS:
        parser.add_argument(f"--{track.lower()}-checkpoint", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.episodes < 1 or args.seed < 0 or args.seed + args.episodes > 2**63:
        raise SystemExit("episodes and seed must define a valid signed-63-bit seed list")
    categories = tuple(part for part in args.categories.split(",") if part)
    checkpoints = {track: str(getattr(args, f"{track.lower()}_checkpoint"))
                   for track in MECO_TRACKS}
    report = run_matrix(checkpoints=checkpoints, scenario_id=args.scenario,
        categories=categories,
        seeds=tuple(args.seed + index for index in range(args.episodes)),
        device=args.device, output_dir=args.output_dir,
        inference_disable_tick=args.inference_disable_tick,
        qualification_protocol=args.qualification_protocol)
    print(json.dumps({"matrix_id": report["matrix_id"],
        "conditions": len(report["conditions"]),
        "contrasts": len(report["contrasts"]),
        "output": str(args.output_dir / f"{report['matrix_id']}.matrix.json")},
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
