#!/usr/bin/env python3
"""Run/resume the exact 28-job MECO diagnostic validation campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense.curriculum_training import source_identity
from frc_defense.fuel_physics_hip import prepare_native_fuel_extension
from frc_defense.meco_ablation_matrix import _checkpoint_configuration_provenance
from frc_defense.meco_tracks import MECO_TRACKS
from frc_defense.meco_validation_campaign import compile_validation_campaign
from frc_defense.meco_validation_campaign_runner import run_campaign


def _object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True,
                        help="one explicit indexed CUDA/HIP device for all 28 jobs")
    for track in MECO_TRACKS:
        parser.add_argument(f"--{track.lower()}-checkpoint", required=True, type=Path)
    args = parser.parse_args(argv)
    if re.fullmatch(r"cuda:(?:0|[1-9][0-9]*)", args.device) is None:
        parser.error("--device must be an indexed CUDA/HIP accelerator; CPU is unsupported")
    checkpoints = {track: str(getattr(args, f"{track.lower()}_checkpoint").resolve())
                   for track in MECO_TRACKS}
    try:
        # Match matrix execution's provenance boundary: prepare the native
        # extension for this exact device before freezing source identity.
        prepare_native_fuel_extension(args.device)
        configs, protocols, _ = _checkpoint_configuration_provenance(checkpoints)
        if set(protocols.values()) != {protocols[MECO_TRACKS[0]]}:
            raise ValueError("five checkpoints do not share one training protocol")
        plan = compile_validation_campaign(protocol_path=args.protocol,
            checkpoint_paths=checkpoints, source_identity=source_identity(),
            training_configuration_sha256=configs, qualification_claim=False)
        result = run_campaign(plan, output_dir=args.output_dir, device=args.device,
            checkpoint_paths=checkpoints,
            matrix_script=ROOT / "scripts" / "run_meco_ablation_matrix.py",
            protocol_path=args.protocol, current_source_identity=source_identity())
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"campaign_sha256": result["campaign_sha256"],
        "jobs": result["job_count"], "split": result["split"],
        "diagnostic_only": result["diagnostic_only"],
        "output": str((args.output_dir / "campaign.complete.json").resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
