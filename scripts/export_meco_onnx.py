#!/usr/bin/env python3
"""Export fixed-shape MECO track scorer graphs to ONNX."""
import argparse
import json
from pathlib import Path

from frc_defense.meco_deployment import export_manifest, export_track
from frc_defense.meco_tracks import MECO_TRACKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", choices=MECO_TRACKS, default=list(MECO_TRACKS))
    parser.add_argument("--checkpoint-root", type=Path, required=True,
                        help="training root containing TRACK/latest.pt for every requested track")
    parser.add_argument("--output", type=Path, default=Path("artifacts/meco-onnx"))
    parser.add_argument("--manifest", type=Path,
                        help="JSON export manifest path (default: OUTPUT/export-manifest.json)")
    args = parser.parse_args()
    reports = []
    for track in args.tracks:
        reports.append(export_track(track, args.checkpoint_root / track / "latest.pt",
                                    args.output / f"{track}.onnx"))
    manifest = args.manifest or args.output / "export-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = export_manifest(reports)
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest), "exports": reports,
        "parameter_inventory": payload["parameter_inventory"]}, indent=2))


if __name__ == "__main__":
    main()
