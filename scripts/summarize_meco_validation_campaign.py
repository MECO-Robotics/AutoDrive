#!/usr/bin/env python3
"""Verify and summarize a complete 28-job MECO validation campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense.meco_validation_campaign_runner import aggregate_campaign


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--campaign", required=True, type=Path,
                        help="campaign.complete.json produced by the runner")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        result = aggregate_campaign(plan, args.campaign)
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
                raise ValueError("immutable campaign summary conflicts with existing output")
            if not args.output.exists():
                args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
