#!/usr/bin/env python3
"""Run the frozen MECO A/B matrix on one shared checkpoint/seed bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
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
    parser.add_argument("--seed-list", help="comma-separated exact seed list; overrides --episodes/--seed")
    parser.add_argument("--seed-manifest", type=Path,
                        help="exact seed manifest bound to the frozen protocol, split, scenario, categories, and stratum")
    parser.add_argument("--device", default="cuda:0",
                        help="indexed CUDA/HIP accelerator; CPU evaluation is unsupported")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "evaluations" / "meco-ablation-matrix")
    parser.add_argument("--inference-disable-tick", type=int,
                        help="requested physics tick for the paired D1 fallback condition")
    parser.add_argument("--qualification-protocol", type=Path,
                        help="frozen protocol whose validation/test seeds the matrix must use")
    parser.add_argument("--qualification-case-json", type=Path,
                        help="JSON object with layout_id, defender_family, start_condition, sensor_profile")
    parser.add_argument("--phase4-opponent-schedule-json", type=Path,
                        help="protocol-bound validation opponent policy assignment")
    parser.add_argument("--phase4-start-profile",
                        choices=("team_start_set_a", "team_start_set_b", "team_start_set_c"),
                        help="registry-bound full-team start profile for full-match validation")
    parser.add_argument("--conditions",
                        help="comma-separated exact condition names; diagnostic subset mode")
    parser.add_argument("--replay-output-dir", type=Path,
                        help="opt-in dashboard replay directory, typically <run>/.scenario-progress")
    parser.add_argument("--replay-seeds", type=int, nargs="+",
                        help="legacy seed selector; captures only the architecture condition")
    parser.add_argument("--replay-selection-json", type=Path,
                        help="condition-to-policy-to-exact-seeds mapping; controls replay exports")
    for track in MECO_TRACKS:
        parser.add_argument(f"--{track.lower()}-checkpoint", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if re.fullmatch(r"cuda:(?:0|[1-9][0-9]*)", args.device) is None:
        raise SystemExit("--device must name an indexed CUDA/HIP accelerator such as cuda:0; CPU is unsupported")
    if args.seed_manifest and (args.seed_list or args.replay_seeds):
        raise SystemExit("--seed-manifest cannot be combined with --seed-list or legacy --replay-seeds")
    if args.replay_selection_json and args.replay_seeds:
        raise SystemExit("use --replay-selection-json or --replay-seeds, not both")
    if args.seed_list:
        try:
            seeds = tuple(int(value) for value in args.seed_list.split(","))
        except ValueError as exc:
            raise SystemExit("--seed-list must contain comma-separated integers") from exc
        if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**63 for seed in seeds):
            raise SystemExit("--seed-list must contain unique signed-63-bit seeds")
    elif args.seed_manifest:
        seeds = None
    elif args.episodes >= 1 and args.seed >= 0 and args.seed + args.episodes <= 2**63:
        seeds = tuple(args.seed + index for index in range(args.episodes))
    else:
        raise SystemExit("episodes and seed must define a valid signed-63-bit seed list")
    replay_selection = (json.loads(args.replay_selection_json.read_text(encoding="utf-8"))
                        if args.replay_selection_json else None)
    has_replay_selector = args.replay_seeds is not None or replay_selection is not None
    if (args.replay_output_dir is None) != (not has_replay_selector):
        raise SystemExit("--replay-output-dir and explicit --replay-seeds must be supplied together")
    categories = tuple(part for part in args.categories.split(",") if part)
    checkpoints = {track: str(getattr(args, f"{track.lower()}_checkpoint"))
                   for track in MECO_TRACKS}
    report = run_matrix(checkpoints=checkpoints, scenario_id=args.scenario,
        categories=categories,
        seeds=seeds,
        device=args.device, output_dir=args.output_dir,
        inference_disable_tick=args.inference_disable_tick,
        qualification_protocol=args.qualification_protocol,
        qualification_case=(json.loads(args.qualification_case_json.read_text(encoding="utf-8"))
            if args.qualification_case_json else None),
        replay_output_dir=args.replay_output_dir,
        replay_seeds=(None if args.replay_seeds is None else tuple(args.replay_seeds)),
        seed_manifest=args.seed_manifest,
        replay_selection=replay_selection,
        phase4_opponent_schedule=(json.loads(args.phase4_opponent_schedule_json.read_text(
            encoding="utf-8")) if args.phase4_opponent_schedule_json else None),
        start_profile_registry_id=args.phase4_start_profile,
        condition_names=(None if args.conditions is None else tuple(
            part for part in args.conditions.split(",") if part)))
    print(json.dumps({"matrix_id": report["matrix_id"],
        "conditions": len(report["conditions"]),
        "contrasts": len(report["contrasts"]),
        "output": str(args.output_dir / f"{report['matrix_id']}.matrix.json")},
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
