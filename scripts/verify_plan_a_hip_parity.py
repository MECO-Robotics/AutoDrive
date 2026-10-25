#!/usr/bin/env python3
"""Verify frozen Plan A HIP fixtures offline or collect leased candidate data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
from pathlib import Path
import uuid

from frc_defense.meco_plan_a_hip_parity_harness import (
    FIXTURE_RELATIVE, run_device, run_offline, source_hashes,
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
    if args.mode == "offline" and (args.device or args.lease_handoff):
        parser.error("offline mode does not accept device or lease arguments")
    if args.mode == "device" and (not args.device or not args.lease_handoff):
        parser.error("device mode requires --device and --lease-handoff")
    evidence_class = ("offline_fixture_only" if args.mode == "offline" else
                      "hip_parity_profile_candidate")
    try:
        if args.mode == "offline":
            report = run_offline(root)
        else:
            report = run_device(root, args.device, args.lease_handoff,
                                warmups=args.warmups, repeats=args.repeats)
    except Exception as exc:
        fixture_path = root / FIXTURE_RELATIVE
        try:
            fixture_bytes = fixture_path.read_bytes()
            fixture = json.loads(fixture_bytes)
            fixture_cases = [{"id": case.get("id"),
                "status": "not_reported_due_to_run_failure",
                "graph_sha256": case.get("graph_sha256"),
                "tensor_sha256": case.get("tensor_sha256"),
                "expected_error": case.get("expected_error"),
                "expected_route_output_sha256": case.get("route_output_sha256"),
                "expected_route_output_field_sha256": case.get("route_output_field_sha256"),
                "expected_support_digest_sha256": case.get("support_digest_sha256")}
                for case in fixture.get("cases", [])]
            manifest_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
        except Exception:
            fixture, fixture_cases, manifest_sha256 = {}, [], None
        try:
            pinned_sources = source_hashes(root)
        except Exception:
            pinned_sources = None
        git_commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True).stdout.strip() or None
        git_dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
            check=False, capture_output=True, text=True).stdout)
        report = {"schema_version": 1, "evidence_class": evidence_class,
            "passed": False, "production_gate_open": False,
            "training_configuration_modified": False,
            "failure_stage": args.mode, "error_type": type(exc).__name__,
            "error": str(exc)[:1000], "git_commit": git_commit,
            "git_dirty": git_dirty, "source_sha256": pinned_sources,
            "fixture_manifest_sha256": manifest_sha256,
            "fixture_source_sha256": fixture.get("source_sha256"),
            "fixture_cases_expected": sorted(row["id"] for row in fixture_cases
                                                if row.get("id")),
            "fixture_cases_reported": [], "cases": fixture_cases,
            "requested_device": args.device,
            "device_identity": "not recorded because the candidate invocation failed"}
        if args.lease_handoff and args.lease_handoff.is_file():
            report["lease_handoff_sha256"] = hashlib.sha256(
                args.lease_handoff.read_bytes()).hexdigest()
    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["requested_mode"] = args.mode
    if args.device:
        report["requested_device"] = args.device
    report_path = _new_report_path(args.report_dir)
    # Exclusive creation makes each report immutable and prevents replacement.
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(report_path)
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
