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
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense import meco_qualification as q
from frc_defense.training_curriculum import SCENARIOS

CAMPAIGN_TYPE = "meco-d0-calibration-campaign-v1"
OFFLINE_SCHEDULE_TYPE = "meco-d0-calibration-offline-schedule-v1"


def _sha(value):
    return q._canonical_sha256(value)


def _write_and_sync_temp(path, encoded):
    with open(path, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _write_immutable_or_verify_stable(path, payload):
    """Atomically publish once or accept identical content, with a stable digest."""
    destination = Path(path)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if destination.exists():
        if q._load_json(destination) != payload:
            raise ValueError(f"immutable artifact already exists with different content: {destination}")
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.",
        suffix=".tmp", dir=destination.parent)
    temp_path = Path(temp_name)
    try:
        os.close(descriptor)
        _write_and_sync_temp(temp_path, encoded)
        try:
            os.link(temp_path, destination)
        except FileExistsError:
            if q._load_json(destination) != payload:
                raise ValueError(
                    f"immutable artifact already exists with different content: {destination}")
            return digest
        directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)
    return digest


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


def describe_campaign(*, episodes=100, repetitions=3, shard_count=8):
    """Describe exact D0 calibration work counts without runtime or seed data.

    The returned schedule is deliberately not an execution manifest: it has no
    source/runtime identity, work-unit assignments, or serialized seed values.
    `run_campaign` and `merge_campaign` continue to accept only the separate,
    source- and GPU-bound campaign manifest.
    """
    scenarios = _closure()
    if type(episodes) is not int or episodes < q.MIN_EPISODES:
        raise ValueError(f"D0 calibration requires at least {q.MIN_EPISODES} episodes per stratum")
    if type(repetitions) is not int or repetitions < 2:
        raise ValueError("D0 calibration requires at least two repetitions")
    if type(shard_count) is not int or shard_count < 1:
        raise ValueError("calibration shard count must be a positive integer")

    scenario_plans = []
    for scenario in scenarios:
        cases = q.protocol_cases(scenario, "calibration", episodes=episodes)
        case_rows = [{
            "category": case.category,
            "layout_id": case.layout_id,
            "defender_family": case.defender_family,
            "start_condition": case.start_condition,
            "sensor_profile": case.sensor_profile,
            "record_count": episodes * repetitions,
        } for case in cases]
        record_count = len(case_rows) * episodes * repetitions
        base, remainder = divmod(record_count, shard_count)
        shards = [{"shard_index": index, "shard_count": shard_count,
                   "record_count": base + (index < remainder)}
                  for index in range(shard_count)]
        scenario_plans.append({
            "scenario_id": scenario,
            "case_count": len(case_rows),
            "record_count": record_count,
            "shard_count": shard_count,
            "case_counts": case_rows,
            "shards": shards,
        })

    schedule = {
        "schema_version": 1,
        "artifact_type": OFFLINE_SCHEDULE_TYPE,
        "execution_ready": False,
        "config": {"split": "calibration",
                   "episodes_per_category_stratum": episodes,
                   "repetitions": repetitions,
                   "shard_count": shard_count},
        "scenario_ids": list(scenarios),
        "scenario_count": len(scenario_plans),
        "case_count": sum(plan["case_count"] for plan in scenario_plans),
        "shard_count": sum(plan["shard_count"] for plan in scenario_plans),
        "total_record_count": sum(plan["record_count"] for plan in scenario_plans),
        "scenario_plans": scenario_plans,
        "seed_disclosure": "none; seed pools and work-unit assignments are not serialized",
    }
    schedule["schedule_id"] = _sha(schedule)
    return schedule


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
        digest = _write_immutable_or_verify_stable(report_path, report)
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
    digest = _write_immutable_or_verify_stable(bundle_path, bundle)
    return {"campaign_id": campaign["campaign_id"],
            "campaign_report_id": bundle["campaign_report_id"],
            "report_sha256": digest, "scenario_count": len(coverage),
            "record_count": bundle["observed_record_count"],
            "output": str(bundle_path)}


def _load_config_artifact_bytes(index_path):
    """Read the caller's logical-ID to source-file map without storing paths."""
    index = q._load_json(index_path)
    from frc_defense.meco_baseline_freeze import REQUIRED_CONFIGURATION_ARTIFACT_IDS
    required = REQUIRED_CONFIGURATION_ARTIFACT_IDS
    if not isinstance(index, dict) or set(index) != required:
        raise ValueError("configuration artifact index must cover the exact required IDs")
    artifacts = {}
    for artifact_id, source_path in index.items():
        if not isinstance(source_path, str) or not source_path:
            raise ValueError("configuration artifact source path is invalid")
        path = Path(source_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"configuration artifact is missing or symlinked: {artifact_id}")
        artifacts[artifact_id] = path.read_bytes()
    return artifacts


def plan_paired_campaign(*, repo_root, freeze_path, partition_path,
        approval_path, configuration_artifact_index, worker_assignments_path,
        episodes, repetitions, master_seed, shard_count, output):
    """Compile a paired manifest only from the real Phase 0 gate artifacts."""
    config_bytes = _load_config_artifact_bytes(configuration_artifact_index)
    bindings = q.prepare_paired_calibration_gate_bindings(repo_root=repo_root,
        baseline_freeze_path=freeze_path, partition_path=partition_path,
        estimand_approval_path=approval_path,
        configuration_artifacts=config_bytes)
    assignments = q._load_json(worker_assignments_path)
    campaign = q.build_paired_calibration_campaign(episodes=episodes,
        repetitions=repetitions, master_seed=master_seed,
        shard_count=shard_count, worker_assignments=assignments, **bindings)
    q.validate_paired_calibration_gate_artifacts(campaign, repo_root=repo_root,
        configuration_artifacts=config_bytes)
    digest = q.write_paired_calibration_artifact_immutable(output, campaign)
    return {"campaign_id": campaign["campaign_id"], "artifact_sha256": digest,
        "work_unit_count": campaign["work_unit_count"],
        "shard_count": len(campaign["shards"]), "output": str(output),
        "execution_blockers": q.paired_calibration_execution_blockers(campaign)}


def merge_paired_campaign(*, repo_root, campaign_path, final_shard_root,
        replay_root, approval_path, configuration_artifact_index, output):
    """Validate exact gate bytes and canonical final shards before strict merge."""
    campaign = q._load_json(campaign_path)
    if Path(approval_path).as_posix() != campaign.get("estimand_approval", {}).get("path"):
        raise ValueError("requested estimand approval path differs from campaign binding")
    config_bytes = _load_config_artifact_bytes(configuration_artifact_index)
    gate = q.validate_paired_calibration_gate_artifacts(campaign,
        repo_root=repo_root, configuration_artifacts=config_bytes)
    final_root = Path(final_shard_root)
    raws = []
    for shard in campaign["shards"]:
        path = final_root / f"shard-{shard['shard_index']:04d}.final.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"paired calibration final shard is missing or symlinked: {path}")
        raws.append(path.read_bytes())
    report = q.build_paired_calibration_merged_report(campaign, raws,
        replay_root=replay_root, approval=gate["approval"])
    bundle_outputs = _publish_paired_control_bundles(output, report)
    digest = q.write_paired_calibration_artifact_immutable(output, report)
    return {"campaign_id": campaign["campaign_id"], "report_id": report["report_id"],
        "report_sha256": digest, "output": str(output),
        "control_bundles": bundle_outputs,
        "official_match_points": None, "qualification_eligible": False}


def _publish_paired_control_bundles(output, report):
    control_artifacts = report.pop("_control_bundle_artifacts")
    if len(control_artifacts) != len(report["control_bundles"]):
        raise ValueError("paired calibration control bundle references are incomplete")
    output_path = Path(output)
    published = []
    for artifact, reference in zip(control_artifacts, report["control_bundles"]):
        control_id = artifact["control_id"]
        if reference["control_id"] != control_id or reference["bundle_id"] != artifact["bundle_id"]:
            raise ValueError("paired calibration control bundle reference mismatch")
        bundle_path = output_path.with_name(
            f"{output_path.stem}.control-{control_id.replace('/', '-')}.json")
        digest = q.write_paired_calibration_artifact_immutable(bundle_path, artifact)
        if digest != reference["artifact_sha256"]:
            raise ValueError("paired calibration control bundle digest mismatch")
        published.append({"control_id": control_id, "bundle_id": artifact["bundle_id"],
            "artifact_sha256": digest, "output": str(bundle_path)})
    return published


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
    describe = sub.add_parser("describe",
        help="compile a count-only schedule without GPU discovery or seed disclosure")
    describe.add_argument("--episodes", type=int, default=100)
    describe.add_argument("--repetitions", type=int, default=3)
    describe.add_argument("--shard-count", type=int, default=8)
    describe.add_argument("--output",
        help="optionally write the immutable count-only schedule JSON")
    run = sub.add_parser("run")
    run.add_argument("--campaign", required=True)
    run.add_argument("--output-root", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--campaign", required=True)
    merge.add_argument("--output-root", required=True)
    merge.add_argument("--report-root", required=True)
    paired_plan = sub.add_parser("paired-plan",
        help="compile a paired calibration manifest from actual Pass 1 gate artifacts")
    paired_plan.add_argument("--repo-root", default=str(ROOT))
    paired_plan.add_argument("--baseline-freeze", required=True)
    paired_plan.add_argument("--partition", required=True)
    paired_plan.add_argument("--estimand-approval", required=True)
    paired_plan.add_argument("--configuration-artifact-index", required=True,
        help="JSON mapping every required logical config ID to its source file")
    paired_plan.add_argument("--worker-assignments", required=True,
        help="JSON array with the two exact indexed WX9100 worker identities")
    paired_plan.add_argument("--episodes", type=int, default=100)
    paired_plan.add_argument("--repetitions", type=int, default=3)
    paired_plan.add_argument("--master-seed", type=int, default=20261024)
    paired_plan.add_argument("--shard-count", type=int, default=2)
    paired_plan.add_argument("--output", required=True)
    paired_merge = sub.add_parser("paired-merge",
        help="validate frozen gates and strictly merge paired final shard artifacts")
    paired_merge.add_argument("--repo-root", default=str(ROOT))
    paired_merge.add_argument("--campaign", required=True)
    paired_merge.add_argument("--final-shard-root", required=True)
    paired_merge.add_argument("--replay-root", required=True)
    paired_merge.add_argument("--estimand-approval", required=True)
    paired_merge.add_argument("--configuration-artifact-index", required=True)
    paired_merge.add_argument("--output", required=True)
    paired_run = sub.add_parser("paired-run",
        help="run or resume one paired shard after exact gate/device checks")
    paired_run.add_argument("--repo-root", default=str(ROOT))
    paired_run.add_argument("--campaign", required=True)
    paired_run.add_argument("--shard-index", type=int, required=True)
    paired_run.add_argument("--replay-root", required=True)
    paired_run.add_argument("--output-root", required=True)
    paired_run.add_argument("--configuration-artifact-index", required=True)
    args = parser.parse_args(argv)
    if args.command == "describe":
        schedule = describe_campaign(episodes=args.episodes,
            repetitions=args.repetitions, shard_count=args.shard_count)
        if args.output:
            _write_immutable_or_verify_stable(args.output, schedule)
        print(json.dumps({"schedule_id": schedule["schedule_id"],
            "scenarios": schedule["scenario_count"],
            "cases": schedule["case_count"],
            "shards": schedule["shard_count"],
            "records": schedule["total_record_count"],
            "output": args.output}, indent=2))
    elif args.command == "plan":
        campaign = plan_campaign(episodes=args.episodes, repetitions=args.repetitions,
            master_seed=args.master_seed, shard_count=args.shard_count, device=args.device)
        q._write_immutable_or_verify(args.output, campaign)
        print(json.dumps({"campaign_id": campaign["campaign_id"],
            "scenarios": len(campaign["scenario_ids"]),
            "shards": sum(len(p["manifests"]) for p in campaign["scenario_plans"]),
            "records": campaign["total_record_count"], "output": args.output}, indent=2))
    elif args.command == "run":
        print(json.dumps(run_campaign(q._load_json(args.campaign), args.output_root), indent=2))
    elif args.command == "merge":
        print(json.dumps(merge_campaign(q._load_json(args.campaign), args.output_root,
                                        args.report_root), indent=2))
    elif args.command == "paired-plan":
        print(json.dumps(plan_paired_campaign(repo_root=args.repo_root,
            freeze_path=args.baseline_freeze, partition_path=args.partition,
            approval_path=args.estimand_approval,
            configuration_artifact_index=args.configuration_artifact_index,
            worker_assignments_path=args.worker_assignments,
            episodes=args.episodes, repetitions=args.repetitions,
            master_seed=args.master_seed, shard_count=args.shard_count,
            output=args.output), indent=2))
    elif args.command == "paired-merge":
        print(json.dumps(merge_paired_campaign(repo_root=args.repo_root,
            campaign_path=args.campaign, final_shard_root=args.final_shard_root,
            replay_root=args.replay_root, approval_path=args.estimand_approval,
            configuration_artifact_index=args.configuration_artifact_index,
            output=args.output), indent=2))
    else:
        configuration_artifacts = _load_config_artifact_bytes(
            args.configuration_artifact_index)
        campaign = q._load_json(args.campaign)
        print(json.dumps(q.run_paired_calibration_shard(campaign,
            args.shard_index, repo_root=args.repo_root,
            replay_root=args.replay_root, output_root=args.output_root,
            configuration_artifacts=configuration_artifacts), indent=2))


if __name__ == "__main__":
    main()
