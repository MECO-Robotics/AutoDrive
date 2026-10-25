#!/usr/bin/env python3
"""Audit a source-bound Plan C repeated-label train corpus offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense.meco_plan_c_label_audit import audit_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="counterfactual v2 JSONL dataset")
    parser.add_argument("--qualification-protocol", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="write JSON report (default: stdout)")
    args = parser.parse_args()
    report = audit_dataset(args.dataset,
                           qualification_protocol=args.qualification_protocol)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
