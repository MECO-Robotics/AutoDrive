#!/usr/bin/env python3
"""Plan, resume, and merge a full registered-scenario D0 calibration campaign.

This is an orchestration layer over the immutable per-scenario shard contract
in frc_defense.meco_qualification. It does not change that contract or run
simulations while planning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense import meco_qualification as q
from frc_defense.training_curriculum import SCENARIOS

CAMPAIGN_TYPE = "meco-d0-calibration-campaign-v1"


def _sha(value):
    return q._canonical_sha256(value)


def _closure():
    """Full registry closure in stable curriculum order, never a target subset."""
    names = tuple(SCENARIOS)
    if len(names) != 12 or len(set(names)) != len(names):
        raise ValueError("D0 campaign requires the complete 12-scenario registry")
    for name, definition in SCENARIOS.items():
        if definition.scenario_id != name:
            raise ValueError("scenario registry key/id mismatch")
        for prerequisite in definition.prerequisites:
            if prerequisite not in SCENARIOS:
                raise ValueError(f"scenario {name} has an unregistered prerequisite")
    return names


def plan_campaign(*, episodes=100, repetitions=3, master_seed=20261024,
                  shard_count=8, device="cuda:0"):
    scenarios = _closure()
    runtime = q._require_calibration_gpu(device)
    scenario_plans = []
    for scenario in scenarios:
        manifests = q._d0_calibration_shard_manifests(
            scenario_id=scenario, episodes=episodes, repetitions=repetitions,
            master_seed=master_seed, shard_count=shard_count, device=device)
        scenario_plans.append({
            "scenario_id": scenario,
            "work_unit_count": manifests[0]["work_unit_count"],
            "manifest_ids": [m["manifest_id"] for m in manifests],
            "manifests": manifests,
        })
    identity = {
        "schema_version": 1,
        "artifact_type": CAMPAIGN_TYPE,
        "config": {"episodes_per_category_stratum": episodes,
                   "repetitions": repetitions, "master_seed": master_seed,
                   "shard_count": shard_count, "device": device},
        "scenario_ids": list(scenarios),
        "source_identity_sha256": q._calibration_source_identity_sha256(),
        "runtime_device": runtime,
        "normalized_runtime_identity": q._normalized_calibration_runtime(runtime),
        "scenario_plans": scenario_plans,
        "total_work_unit_count": sum(x["work_unit_count"] for x in scenario_plans),
        "total_record_count": sum(x["work_unit_count"] for x in scenario_plans),
    }
    identity["campaign_id"] = _sha(identity)
    return identity


def validate_campaign(campaign):
    if not isinstance(campaign, dict) or campaign.get("artifact_type") != CAMPAIGN_TYPE:
        raise ValueError("unsupported D0 calibration campaign manifest")
    unsigned = {k: v for k, v in campaign.items() if k != "campaign_id"}
    if campaign.get("campaign_id") != _sha(unsigned):
        raise ValueError("D0 calibration campaign digest is invalid")
    scenarios = _closure()
    if campaign.get("scenario_ids") != list(scenarios):
        raise ValueError("campaign does not cover the full registered scenario closure")
    config = campaign.get("config")
    if not isinstance(config, dict):
        raise ValueError("campaign config is invalid")
    if (type(config.get("episodes_per_category_stratum")) is not int
            or config["episodes_per_category_stratum"] < q.MIN_EPISODES
            or type(config.get("repetitions")) is not int or config["repetitions"] < 2
            or type(config.get("master_seed")) is not int or config["master_seed"] < 0
            or type(config.get("shard_count")) is not int or config["shard_count"] < 1
            or not isinstance(config.get("device"), str)):
        raise ValueError("campaign config does not satisfy the frozen calibration contract")
    if campaign.get("source_identity_sha256") != q._calibration_source_identity_sha256():
        raise ValueError("campaign source identity differs from current evaluation source")
    q._validate_calibration_runtime_device(campaign.get("runtime_device"))
    if campaign.get("normalized_runtime_identity") != q._normalized_calibration_runtime(
            campaign["runtime_device"]):
        raise ValueError("campaign runtime identity is invalid")
    plans = campaign.get("scenario_plans")
    if not isinstance(plans, list) or [p.get("scenario_id") for p in plans] != list(scenarios):
        raise ValueError("campaign scenario plan list is incomplete or unordered")
    total = 0
    for plan in plans:
        manifests = plan.get("manifests")
        if not isinstance(manifests, list) or len(manifests) != config["shard_count"]:
            raise ValueError("campaign shard manifest set is incomplete")
        if plan.get("manifest_ids") != [m.get("manifest_id") for m in manifests]:
            raise ValueError("campaign manifest index differs from embedded manifests")
        for index, manifest in enumerate(manifests):
            q._validate_d0_shard_manifest(manifest)
            if (manifest["config"]["scenario_id"] != plan["scenario_id"]
                    or manifest["config"]["episodes_per_category_stratum"] != config["episodes_per_category_stratum"]
                    or manifest["config"]["repetitions"] != config["repetitions"]
                    or manifest["config"]["master_seed"] != config["master_seed"]
                    or manifest["shard_index"] != index
                    or manifest["shard_count"] != config["shard_count"]
                    or manifest["runtime_device"].get("device") != config["device"]
                    or manifest["source_identity_sha256"] != campaign["source_identity_sha256"]
                    or manifest["runtime_device"] != campaign["runtime_device"]):
                raise ValueError("campaign shard does not match its source, runtime, or scenario")
        count = manifests[0]["work_unit_count"]
        if plan.get("work_unit_count") != count:
            raise ValueError("campaign scenario work count is invalid")
        total += count
    if total != campaign.get("total_work_unit_count") or total != campaign.get("total_record_count"):
        raise ValueError("campaign total work count is invalid")
    return campaign


def _artifact_path(root: Path, scenario: str, index: int, count: int):
    return root / scenario / f"shard-{index:04d}-of-{count:04d}.artifact.json"


def run_campaign(campaign, output_root: str | Path):
    """Run missing shards; completed immutable artifacts are validated and reused."""
    validate_campaign(campaign)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    device = campaign["config"]["device"]
    completed = 0
    for plan in campaign["scenario_plans"]:
        scenario = plan["scenario_id"]
        for manifest in plan["manifests"]:
            output = _artifact_path(root, scenario, manifest["shard_index"],
                                    manifest["shard_count"])
            q.run_d0_calibration_shard(manifest, device=device, output=output)
            completed += 1
    return {"campaign_id": campaign["campaign_id"], "shards_validated_or_run": completed,
            "shard_count": sum(len(p["manifests"]) for p in campaign["scenario_plans"]),
            "output_root": str(root)}


def merge_campaign(campaign, output_root: str | Path, report_root: str | Path):
    """Require every exact shard, then merge each scenario and the full bundle."""
    validate_campaign(campaign)
    root, reports = Path(output_root), Path(report_root)
    reports.mkdir(parents=True, exist_ok=True)
    scenario_reports = []
    coverage = {}
    completed = {}
    # Validate all expected artifacts before writing any merged report. A
    # missing late scenario must not leave a misleading partial campaign.
    for plan in campaign["scenario_plans"]:
        scenario = plan["scenario_id"]
        manifests, artifacts = [], []
        for manifest in plan["manifests"]:
            path = _artifact_path(root, scenario, manifest["shard_index"],
                                  manifest["shard_count"])
            if not path.is_file():
                raise ValueError(f"campaign merge missing shard artifact: {path}")
            artifact = q._load_json(path)
            q._validate_d0_shard_artifact(artifact, manifest)
            manifests.append(manifest)
            artifacts.append(artifact)
        completed[scenario] = (manifests, artifacts)
    for plan in campaign["scenario_plans"]:
        scenario = plan["scenario_id"]
        manifests, artifacts = completed[scenario]
        report = q._merged_d0_calibration_report(manifests, artifacts)
        report_path = reports / f"{scenario}.calibration.json"
        digest = q.write_immutable_report(report_path, report)
        scenario_reports.append(report_path)
        coverage[scenario] = {
            "records": len(report["records"]),
            "cases": len(report["case_ids"]),
            "strata": len(report["summary"]),
            "summary": report["summary"],
            "calibration_id": report["calibration_id"],
            "report_sha256": digest,
            "report": str(report_path),
        }
    bundle = {"schema_version": 1, "artifact_type": "meco-d0-calibration-campaign-report-v1",
              "campaign_id": campaign["campaign_id"],
              "source_identity_sha256": campaign["source_identity_sha256"],
              "runtime_device": campaign["runtime_device"],
              "scenario_ids": campaign["scenario_ids"],
              "expected_record_count": campaign["total_record_count"],
              "observed_record_count": sum(v["records"] for v in coverage.values()),
              "scenario_reports": coverage}
    if bundle["expected_record_count"] != bundle["observed_record_count"]:
        raise ValueError("campaign merged record count does not match planned closure")
    bundle["campaign_report_id"] = _sha(bundle)
    bundle_path = reports / "campaign.report.json"
    digest = q.write_immutable_report(bundle_path, bundle)
    return {"campaign_id": campaign["campaign_id"],
            "campaign_report_id": bundle["campaign_report_id"],
            "report_sha256": digest, "scenario_count": len(coverage),
            "record_count": bundle["observed_record_count"],
            "output": str(bundle_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--device", required=True, help="explicit visible GPU, e.g. cuda:0")
    plan.add_argument("--episodes", type=int, default=100)
    plan.add_argument("--repetitions", type=int, default=3)
    plan.add_argument("--master-seed", type=int, default=20261024)
    plan.add_argument("--shard-count", type=int, default=8)
    plan.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("--campaign", required=True)
    run.add_argument("--output-root", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--campaign", required=True)
    merge.add_argument("--output-root", required=True)
    merge.add_argument("--report-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        campaign = plan_campaign(episodes=args.episodes, repetitions=args.repetitions,
            master_seed=args.master_seed, shard_count=args.shard_count, device=args.device)
        q._write_immutable_or_verify(args.output, campaign)
        print(json.dumps({"campaign_id": campaign["campaign_id"],
            "scenarios": len(campaign["scenario_ids"]),
            "shards": sum(len(p["manifests"]) for p in campaign["scenario_plans"]),
            "records": campaign["total_record_count"], "output": args.output}, indent=2))
    elif args.command == "run":
        print(json.dumps(run_campaign(q._load_json(args.campaign), args.output_root), indent=2))
    else:
        print(json.dumps(merge_campaign(q._load_json(args.campaign), args.output_root,
                                        args.report_root), indent=2))


if __name__ == "__main__":
    main()
