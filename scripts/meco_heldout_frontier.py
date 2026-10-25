#!/usr/bin/env python3
"""Compile or verify a provisional, validation-only MECO input frontier."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense.meco_heldout_frontier import (
    compile_heldout_frontier,
    validate_heldout_frontier,
)
from frc_defense.meco_tracks import MECO_TRACKS


def _read(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must contain an object")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compile", "verify"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-identity-json", type=Path, required=True)
    parser.add_argument("--training-configuration-hashes-json", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True,
                        help="root frontier JSON to create immutably or verify")
    for track in MECO_TRACKS:
        parser.add_argument(f"--{track.lower()}-checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    checkpoint_paths = {track: getattr(args, f"{track.lower()}_checkpoint")
                        for track in MECO_TRACKS}
    try:
        source = _read(args.source_identity_json, "source identity")
        configs = _read(args.training_configuration_hashes_json, "training configuration hashes")
        if args.mode == "compile":
            frontier = compile_heldout_frontier(protocol_path=args.protocol,
                checkpoint_paths=checkpoint_paths, source_identity=source,
                training_configuration_sha256=configs)
            encoded = (json.dumps(frontier, sort_keys=True, indent=2,
                                  allow_nan=False) + "\n").encode()
            args.frontier.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(prefix=f".{args.frontier.name}.",
                                             suffix=".tmp", dir=args.frontier.parent)
            temp_path = Path(temp_name)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temp_path, args.frontier)
                except FileExistsError as exc:
                    raise ValueError("frontier root already exists; it is immutable") from exc
                directory_fd = os.open(args.frontier.parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                temp_path.unlink(missing_ok=True)
        else:
            frontier = _read(args.frontier, "frontier")
            validate_heldout_frontier(frontier, protocol_path=args.protocol,
                checkpoint_paths=checkpoint_paths, source_identity=source,
                training_configuration_sha256=configs)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({"frontier_sha256": frontier["frontier_sha256"],
                      "status": frontier["version_status"],
                      "split": frontier["split"],
                      "qualification_eligible": frontier["qualification_eligible"],
                      "path": str(args.frontier)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
