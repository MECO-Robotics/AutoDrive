#!/usr/bin/env python3
"""Measure synthetic diagnostics or strict production-capture scorer profiles."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from frc_defense.meco_deployment import (profile_report,
    production_capture_profile_report)
from frc_defense.meco_tracks import MECO_TRACKS


def _write_report_exclusive(output: Path, payload: str) -> None:
    """Publish a complete report atomically without following/replacing a target."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Hard-link publication is atomic and fails if output is any
            # existing directory entry, including a dangling or live symlink.
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite existing report: {output}") from None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", choices=MECO_TRACKS, default=list(MECO_TRACKS))
    parser.add_argument("--checkpoint-root", type=Path, required=True,
                        help="training root containing TRACK/latest.pt for every requested track")
    parser.add_argument("--production-capture-dir", type=Path,
                        help="enable strict five-track, attested-WX9100 capture profiling")
    parser.add_argument("--device", action="append", dest="devices",
                        help="capture mode only; repeat exactly for cuda:0 and cuda:1")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("evaluations/meco-deployment/profile.json"))
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    if args.production_capture_dir:
        if tuple(args.tracks) != tuple(MECO_TRACKS):
            parser.error("production-capture mode requires the exact complete five-track set")
        if tuple(args.devices or ()) != ("cuda:0", "cuda:1"):
            parser.error("production-capture mode requires --device cuda:0 --device cuda:1")
        try:
            report = production_capture_profile_report(args.checkpoint_root,
                args.production_capture_dir, args.devices, args.iterations, args.warmup)
        except Exception as exc:
            report = {"schema": "meco-production-capture-scorer-profile-v1",
                "status": "failed", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "evidence_class": "production-capture-backed scorer microbenchmark; not deployment qualification",
                "input_mode": "production_capture", "tracks_required": list(MECO_TRACKS),
                "tracks_profiled": [], "devices_required": ["cuda:0", "cuda:1"],
                "artifact_validation": {"status": "failed"},
                "error": f"{type(exc).__name__}: {exc}",
                "profiler_source_identity": None,
                "physical_device_attestation": None,
                "deployment_qualification": False}
            _write_report_exclusive(args.output,
                json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(report, indent=2), file=sys.stderr)
            return 1
    else:
        if args.devices:
            parser.error("--device is only valid with --production-capture-dir")
        report = profile_report(args.tracks, args.checkpoint_root, args.iterations, args.warmup)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    _write_report_exclusive(args.output, payload)
    if args.production_capture_dir:
        print(f"profile_sha256={hashlib.sha256(payload.encode()).hexdigest()}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
