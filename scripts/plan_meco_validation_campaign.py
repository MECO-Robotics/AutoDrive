#!/usr/bin/env python3
"""Compile a frozen validation-only MECO matched-evaluation campaign plan."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense.meco_tracks import MECO_TRACKS
from frc_defense.meco_validation_campaign import compile_validation_campaign


def _write_immutable(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2,
                         allow_nan=False).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"immutable campaign artifact conflicts with existing file: {path}")
        import hashlib
        return hashlib.sha256(encoded).hexdigest()
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise ValueError(f"immutable campaign artifact conflicts with existing file: {path}")
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)
    import hashlib
    return hashlib.sha256(encoded).hexdigest()


def _json_file(path: Path, label: str):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {label} JSON {path}: {exc}") from exc
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True,
                        help="frozen all-scenario qualification protocol JSON")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new or matching immutable plan output directory")
    parser.add_argument("--qualification-claim", action="store_true",
                        help="require a calibrated v2 protocol and frozen safety gates")
    parser.add_argument("--source-identity-json", type=Path,
                        help="optional source_identity() JSON object to bind into the plan")
    parser.add_argument("--training-configuration-hashes-json", type=Path,
                        help="optional track-to-SHA-256 JSON object")
    for track in MECO_TRACKS:
        parser.add_argument(f"--{track.lower()}-checkpoint", type=Path)
    args = parser.parse_args(argv)
    paths = {track: getattr(args, f"{track.lower()}_checkpoint") for track in MECO_TRACKS}
    provided = [path is not None for path in paths.values()]
    if any(provided) and not all(provided):
        parser.error("provide checkpoints for all five tracks or none")
    try:
        plan = compile_validation_campaign(protocol_path=args.protocol,
            checkpoint_paths=(paths if all(provided) else None),
            source_identity=(_json_file(args.source_identity_json, "source identity")
                if args.source_identity_json else None),
            training_configuration_sha256=(
                _json_file(args.training_configuration_hashes_json, "training configuration hashes")
                if args.training_configuration_hashes_json else None),
            qualification_claim=args.qualification_claim)
        root = args.output_dir
        plan_path = root / "plan.json"
        _write_immutable(plan_path, plan)
        for job in plan["jobs"]:
            _write_immutable(root / job["seed_manifest_file"], job["seed_manifest"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"plan_digest": plan["plan_digest"],
        "jobs": plan["job_count"], "split": plan["split"],
        "qualification_eligible": plan["qualification_eligible"],
        "output": str(plan_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
