#!/usr/bin/env python3
"""Verify frozen Plan A HIP fixtures offline or collect leased candidate data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from frc_defense.meco_plan_a_hip_parity_harness import (
    run_device, run_offline,
)


def _new_report_path(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return directory / f"plan-a-hip-evidence-{stamp}-{uuid.uuid4().hex[:12]}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "device"), default="offline")
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"))
    parser.add_argument("--lease-handoff", type=Path)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--report-dir", type=Path,
        default=Path("artifacts/plan_a_hip_evidence"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.mode == "offline":
        if args.device or args.lease_handoff:
            parser.error("offline mode does not accept device or lease arguments")
        report = run_offline(root)
    else:
        if not args.device or not args.lease_handoff:
            parser.error("device mode requires --device and --lease-handoff")
        report = run_device(root, args.device, args.lease_handoff,
                            warmups=args.warmups, repeats=args.repeats)
    report_path = _new_report_path(args.report_dir)
    # Exclusive creation makes each report immutable and prevents replacement.
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(report_path)
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
