#!/usr/bin/env python3
"""Export fixed-shape MECO track scorer graphs to ONNX."""
import argparse
import json
from pathlib import Path

from frc_defense.meco_deployment import export_manifest, export_track
from frc_defense.meco_tracks import MECO_TRACKS
from frc_defense.meco_request_capture import load_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", choices=MECO_TRACKS, default=list(MECO_TRACKS))
    parser.add_argument("--checkpoint-root", type=Path, required=True,
                        help="training root containing TRACK/latest.pt for every requested track")
    parser.add_argument("--output", type=Path, default=Path("artifacts/meco-onnx"))
    parser.add_argument("--manifest", type=Path,
                        help="JSON export manifest path (default: OUTPUT/export-manifest.json)")
    parser.add_argument("--production-capture-dir", type=Path,
                        help="explicit directory containing TRACK-*.npz captures; exports use captured shapes")
    args = parser.parse_args()
    reports = []
    for track in args.tracks:
        tensors, capture = None, None
        if args.production_capture_dir:
            candidates = sorted(args.production_capture_dir.glob(f"{track}-*.npz"))
            if len(candidates) != 1:
                parser.error(f"expected exactly one {track} capture in {args.production_capture_dir}, found {len(candidates)}")
            capture, tensors = load_capture(candidates[0], expected_track=track)
        reports.append(export_track(track, args.checkpoint_root / track / "latest.pt",
                                    args.output / f"{track}.onnx", tensors,
                                    capture_provenance=capture))
    manifest = args.manifest or args.output / "export-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = export_manifest(reports)
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest), "exports": reports,
        "parameter_inventory": payload["parameter_inventory"]}, indent=2))


if __name__ == "__main__":
    main()
