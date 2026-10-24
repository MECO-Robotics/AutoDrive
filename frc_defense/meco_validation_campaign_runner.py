"""Resumable validation-only execution and aggregation for MECO campaigns."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable

from .meco_ablation_matrix import _records_by_key
from .meco_tracks import MECO_TRACKS
from .training_curriculum import SCENARIOS
from .meco_validation_campaign import _artifact_bytes, _canonical_bytes, _sha
from .meco_validation_campaign import compile_validation_campaign


CAMPAIGN_SCHEMA = "meco-validation-campaign-complete-v1"
JOB_INPUT_SCHEMA = "meco-validation-campaign-job-input-v1"
JOB_COMPLETE_SCHEMA = "meco-validation-campaign-job-complete-v1"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_immutable(path: Path, value: dict) -> str:
    data = _artifact_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable campaign artifact conflicts: {path}")
        return hashlib.sha256(data).hexdigest()
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"immutable campaign artifact conflicts: {path}")
        dfd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        temp.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def _validate_plan(plan: dict) -> None:
    if plan.get("schema") != "meco-validation-campaign-plan-v1":
        raise ValueError("unsupported validation plan schema")
    unhashed = {key: value for key, value in plan.items() if key != "plan_digest"}
    if plan.get("plan_digest") != _sha(unhashed):
        raise ValueError("validation plan digest mismatch")
    if (plan.get("split") != "validation" or plan.get("qualification_claim") is not False
            or plan.get("qualification_eligible") is not False
            or plan.get("final_test_seed_handling") != "not_exported"):
        raise ValueError("campaign execution is restricted to diagnostic validation plans")
    if plan.get("protocol", {}).get("gate_calibration_status") != "provisional_pending_baseline_variance_review":
        raise ValueError("campaign runner requires the currently provisional diagnostic protocol")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 28 or plan.get("job_count") != 28:
        raise ValueError("campaign requires the exact 28-job all-scenario validation matrix")
    if plan.get("scenario_ids") != list(SCENARIOS):
        raise ValueError("campaign plan scenario registry differs from the current curriculum")
    seen = set()
    seen_case_ids = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("campaign plan contains a non-validation job")
        jid = job.get("job_id")
        if not isinstance(jid, str) or jid != _sha({k: v for k, v in job.items() if k != "job_id"}) or jid in seen:
            raise ValueError("job ID is invalid or duplicated")
        seen.add(jid)
        case_identity = (job.get("scenario_id"), _canonical_bytes(job.get("qualification_case")))
        if (job.get("scenario_id") not in SCENARIOS or case_identity in seen_case_ids
                or not set(job.get("categories", ())).issubset(
                    set(SCENARIOS[job["scenario_id"]].required_categories))):
            raise ValueError("job scenario/category/qualification-case identity is invalid or duplicated")
        seen_case_ids.add(case_identity)
        if (not isinstance(job.get("checkpoint_sha256"), dict)
                or set(job["checkpoint_sha256"]) != set(MECO_TRACKS)
                or not isinstance(job.get("source_identity"), dict)
                or not isinstance(job.get("training_configuration_sha256"), dict)
                or set(job["training_configuration_sha256"]) != set(MECO_TRACKS)):
            raise ValueError("each job requires all five checkpoint/config hashes and a bound source identity")
        manifest = job.get("seed_manifest")
        if (not isinstance(manifest, dict) or manifest.get("split") != "validation"
                or manifest.get("scenario_id") != job.get("scenario_id")
                or manifest.get("seed_list") != job.get("seed_list")
                or hashlib.sha256(_artifact_bytes(manifest)).hexdigest() != job.get("seed_manifest_sha256")):
            raise ValueError("job seed manifest binding is invalid")
        if not job.get("conditions"):
            raise ValueError("job has no matrix conditions")
        for track in MECO_TRACKS:
            for binding in (job["checkpoint_sha256"][track],
                            job["training_configuration_sha256"][track]):
                if (not isinstance(binding, str) or len(binding) != 64
                        or any(c not in "0123456789abcdef" for c in binding)):
                    raise ValueError(f"{track} job binding is not a SHA-256 digest")


def _expected_keys(condition: dict) -> set[tuple[str, str, int]]:
    coverage = condition.get("expected_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("condition lacks expected coverage contract")
    expected = {(policy, category, seed) for policy in coverage["policies"]
                for category in coverage["categories"] for seed in coverage["seed_list"]}
    if len(expected) != coverage.get("record_count"):
        raise ValueError("condition expected coverage count is inconsistent")
    return expected


def _validate_replay_artifact(replay_record: dict, replay_ref: dict, *,
                              report: dict, report_path: Path, row: dict,
                              report_index: int, job: dict) -> None:
    """Bind a producer-shaped replay bundle back to its exact evaluation row."""
    provenance = replay_record.get("replay_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("replay bundle lacks producer replay_provenance")
    expected_record_digest = _sha({key: value for key, value in row.items()
                                   if key != "dashboard_replay"})
    expected_policy = row.get("policy")
    expected_checkpoint = row.get("checkpoint_sha256")
    expected_case = job["qualification_case"]
    if (replay_record.get("evaluation_id") != report.get("evaluation_id")
            or replay_record.get("condition_id") != report.get("experiment_matrix", {}).get("condition_id")
            or replay_ref.get("simulation_id") != replay_record.get("simulation_id")
            or replay_record.get("scenario_id") != job["scenario_id"]
            or replay_record.get("category") != row.get("category")
            or replay_record.get("seed") != row.get("seed")
            or replay_record.get("policy") != expected_policy
            or replay_record.get("split") != "validation"
            or replay_record.get("qualification_case") != expected_case):
        raise ValueError("replay root identity differs from its selected evaluation row")
    if (replay_ref.get("evaluation_report_path") != str(report_path.resolve())
            or replay_record.get("evaluation_report_path") != str(report_path.resolve())
            or replay_ref.get("evaluation_record_index") != report_index
            or replay_record.get("evaluation_record_index") != report_index
            or replay_ref.get("evaluation_record_digest") != expected_record_digest
            or replay_record.get("evaluation_record_digest") != expected_record_digest):
        raise ValueError("replay pointer/digest does not bind the exact evaluation row")
    source_sha = job["source_identity"]["sha256"]
    protocol_id = job["seed_manifest"]["protocol_id"]
    if (replay_record.get("source_identity_sha256") != source_sha
            or replay_record.get("replay_source_sha256") != source_sha
            or provenance.get("replay_source_sha256") != source_sha
            or replay_record.get("qualification_protocol_id") != protocol_id
            or provenance.get("qualification_protocol_id") != protocol_id
            or replay_record.get("protocol_id") != row.get("protocol_id")
            or provenance.get("protocol_id") != row.get("protocol_id")
            or replay_record.get("checkpoint_sha256") != expected_checkpoint
            or provenance.get("checkpoint_sha256") != expected_checkpoint
            or provenance.get("evaluation_id") != report.get("evaluation_id")
            or provenance.get("condition_id") != replay_record.get("condition_id")
            or provenance.get("scenario_id") != job["scenario_id"]
            or provenance.get("category") != row.get("category")
            or provenance.get("seed") != row.get("seed")
            or provenance.get("policy") != expected_policy
            or provenance.get("split") != "validation"
            or provenance.get("qualification_case") != expected_case
            or provenance.get("evaluation_record_index") != report_index
            or provenance.get("evaluation_record_digest") != expected_record_digest):
        raise ValueError("replay provenance differs from campaign source/protocol/row binding")
    safety_metrics = row.get("meco_metric_ledger", {}).get("safety_metrics")
    route_digest = _sha({"route_trace": row.get("route_trace"),
        "executed_planner_route_trace": row.get("executed_planner_route_trace")})
    outcome_digest = _sha({"metrics": row.get("metrics"),
        "safety_metrics": safety_metrics,
        "execution_failures": row.get("execution_failures")})
    safety_digest = _sha(safety_metrics)
    for field, expected in (("route_digest", route_digest),
                            ("outcome_digest", outcome_digest),
                            ("safety_digest", safety_digest)):
        if replay_record.get(field) != expected or provenance.get(field) != expected:
            raise ValueError(f"replay {field} differs from its source evaluation record")
    if expected_policy in MECO_TRACKS and expected_checkpoint != job["checkpoint_sha256"][expected_policy]:
        raise ValueError("replay checkpoint differs from the bound learned-track checkpoint")


def validate_job_artifacts(job: dict, job_dir: Path, *, device: str) -> dict:
    """Validate a complete matrix and every condition/replay before permitting a seal."""
    matrix_dir = job_dir / "matrix"
    matrices = sorted(matrix_dir.glob("*.matrix.json"))
    if len(matrices) != 1:
        raise ValueError("job does not contain exactly one completed matrix report")
    matrix_path = matrices[0]
    # This performs condition report digest/provenance, paired-contrast and replay
    # reference checks. It is diagnostic-only and does not execute simulations.
    from scripts.summarize_meco_ablation_matrix import summarize_matrix
    summarize_matrix(matrix_path, replicates=100)
    matrix = _read_json(matrix_path)
    protocol = matrix.get("protocol", {})
    if (matrix.get("evaluation_split") != "validation"
            or matrix.get("evaluation_design") != "frozen_protocol_held_out"
            or matrix.get("qualification_protocol_id") != job["seed_manifest"]["protocol_id"]
            or matrix.get("scenario_id") != job["scenario_id"]
            or matrix.get("qualification_case") != job["qualification_case"]
            or matrix.get("protocol", {}).get("qualification_protocol", {}).get("protocol_sha256")
                != job["seed_manifest"]["protocol_sha256"]):
        raise ValueError("matrix protocol/split/scenario differs from immutable job input")
    if (protocol.get("seed_list") != job["seed_list"]
            or protocol.get("categories") != job["categories"]
            or protocol.get("qualification_case") != job["qualification_case"]
            or protocol.get("checkpoint_sha256") != job["checkpoint_sha256"]
            or protocol.get("training_configuration_sha256") != job["training_configuration_sha256"]
            or protocol.get("source_identity_sha256") != job["source_identity"]["sha256"]
            or protocol.get("execution_device", {}).get("device") != device):
        raise ValueError("matrix inputs differ from immutable campaign job binding")
    expected_replay_selector = {condition["condition_name"]: condition["replay_selection"]
        for condition in job["conditions"] if condition.get("replay_selection")}
    if protocol.get("replay_selection") != expected_replay_selector:
        raise ValueError("matrix replay selector differs from immutable campaign inputs")
    condition_map = matrix.get("conditions")
    if not isinstance(condition_map, dict) or set(condition_map) != {c["condition_name"] for c in job["conditions"]}:
        raise ValueError("matrix condition table differs from exact planned condition set")
    report_digests = {}
    replay_digests = {}
    referenced_replays = set()
    for condition in job["conditions"]:
        condition_fields = ("condition_name", "prediction_mode", "control_prediction_mode",
            "candidate_k", "inference_disable_tick", "phase4_factor", "categories",
            "qualification_case")
        expected_spec = {key: condition.get(key) for key in condition_fields}
        observed_specs = {item.get("condition_name"): item for item in protocol.get("conditions", [])}
        if _canonical_bytes(observed_specs.get(condition["condition_name"])) != _canonical_bytes(expected_spec):
            raise ValueError("matrix condition specification differs from the planned condition")
        entry = condition_map[condition["condition_name"]]
        if (entry.get("execution_device", {}).get("device") != device
                or entry.get("source_identity_sha256") != job["source_identity"]["sha256"]
                or entry.get("checkpoint_sha256") != job["checkpoint_sha256"]
                or entry.get("training_configuration_sha256") != job["training_configuration_sha256"]):
            raise ValueError("condition provenance differs from campaign input")
        report_path = Path(entry["report_path"]).resolve()
        if not report_path.is_file():
            raise ValueError("condition evaluation report is missing")
        report = _read_json(report_path)
        records = _records_by_key(report)
        if set(records) != _expected_keys(condition):
            raise ValueError(f"condition record coverage differs: {condition['condition_name']}")
        if report.get("split") != "validation" or report.get("scenario_id") != job["scenario_id"]:
            raise ValueError("condition report split/scenario mismatch")
        report_digests[str(report_path)] = _file_sha(report_path)
        selector = condition.get("replay_selection", {})
        selected = {(policy, category, seed) for policy, seeds in selector.items()
                    for seed in seeds for category in job["categories"]}
        exports = report.get("dashboard_replays", [])
        exported = set()
        for replay in exports:
            index = replay.get("evaluation_record_index")
            if type(index) is not int or not 0 <= index < len(report.get("records", [])):
                raise ValueError("replay export record index is invalid")
            row = report["records"][index]
            key = (row.get("policy"), row.get("category"), row.get("seed"))
            exported.add(key)
            replay_path = Path(replay.get("result_path", "")).resolve()
            if not replay_path.is_file():
                raise ValueError("selected replay artifact is missing")
            replay_record = _read_json(replay_path)
            _validate_replay_artifact(replay_record, replay, report=report,
                report_path=report_path, row=row, report_index=index, job=job)
            replay_digests[str(replay_path)] = _file_sha(replay_path)
            referenced_replays.add(replay_path)
        if exported != selected:
            raise ValueError(f"replay export coverage differs: {condition['condition_name']}")
    replay_root = job_dir / "replays"
    if replay_root.exists():
        actual_replays = {path.resolve() for path in replay_root.glob("*.result.json")}
        if actual_replays != referenced_replays:
            raise ValueError("replay directory has missing or unreferenced partial artifacts")
    seed_path = job_dir / job["seed_manifest_file"]
    case_path = job_dir / "qualification-case.json"
    selector_path = job_dir / "replay-selection.json"
    if not seed_path.is_file() or _read_json(seed_path) != job["seed_manifest"]:
        raise ValueError("immutable per-job seed manifest is missing or changed")
    if not case_path.is_file() or _read_json(case_path) != job["qualification_case"]:
        raise ValueError("immutable per-job qualification case is missing or changed")
    expected_selector = {condition["condition_name"]: condition["replay_selection"]
        for condition in job["conditions"] if condition.get("replay_selection")}
    if expected_selector:
        if not selector_path.is_file() or _read_json(selector_path) != expected_selector:
            raise ValueError("immutable replay selector is missing or changed")
        selector_digest = _file_sha(selector_path)
    else:
        selector_digest = None
    return {"matrix_path": str(matrix_path), "matrix_sha256": _file_sha(matrix_path),
            "condition_report_sha256": report_digests,
            "replay_sha256": replay_digests,
            "seed_manifest_sha256": _file_sha(seed_path),
            "qualification_case_sha256": _file_sha(case_path),
            "replay_selection_sha256": selector_digest,
            "matrix_report_sha256": matrix.get("report_sha256"),
            "matrix_id": matrix.get("matrix_id")}


def _completion_valid(job: dict, job_dir: Path, *, device: str,
                     expected_plan_digest: str) -> dict | None:
    seal_path = job_dir / "job.complete.json"
    if not seal_path.is_file():
        return None
    seal = _read_json(seal_path)
    payload = {key: value for key, value in seal.items() if key != "completion_sha256"}
    if (seal.get("schema") != JOB_COMPLETE_SCHEMA or seal.get("job_id") != job["job_id"]
            or seal.get("plan_digest") != expected_plan_digest
            or seal.get("completion_sha256") != _sha(payload)):
        raise ValueError("job completion seal is invalid")
    if seal.get("device") != device or seal.get("diagnostic_only") is not True:
        raise ValueError("job completion seal has invalid device or diagnostic gate")
    input_path = job_dir / "job.input.json"
    job_input = _read_json(input_path) if input_path.is_file() else {}
    if (not input_path.is_file() or _file_sha(input_path) != seal.get("job_input_sha256")
            or _canonical_bytes(job_input.get("job")) != _canonical_bytes(job)
            or job_input.get("plan_digest") != seal.get("plan_digest")):
        raise ValueError("job completion seal is not bound to its immutable job input")
    validated = validate_job_artifacts(job, job_dir, device=device)
    if seal.get("artifacts") != validated:
        raise ValueError("sealed job artifacts changed after completion")
    return seal


def run_campaign(plan: dict, *, output_dir: str | Path, device: str,
                 checkpoint_paths: dict[str, str], matrix_script: str | Path,
                 protocol_path: str | Path,
                 current_source_identity: dict,
                 runner: Callable[[list[str]], int] | None = None) -> dict:
    """Execute/resume a diagnostic validation campaign; no test split is accepted."""
    _validate_plan(plan)
    if re.fullmatch(r"cuda:(?:0|[1-9][0-9]*)", device) is None:
        raise ValueError("device must be an indexed CUDA/HIP accelerator")
    if set(checkpoint_paths) != set(MECO_TRACKS):
        raise ValueError("checkpoint paths must cover exactly all five tracks")
    for track, path in checkpoint_paths.items():
        if _file_sha(Path(path)) != plan["jobs"][0]["checkpoint_sha256"][track]:
            raise ValueError(f"{track} checkpoint differs from plan")
    protocol_path = Path(protocol_path).resolve()
    if _file_sha(protocol_path) != plan["protocol"]["protocol_sha256"]:
        raise ValueError("raw qualification protocol SHA-256 differs from immutable plan")
    planned_job = plan["jobs"][0]
    if planned_job["source_identity"]["identity"] != current_source_identity:
        raise ValueError("current prepared source identity differs from frozen plan")
    rebuilt = compile_validation_campaign(protocol_path=protocol_path,
        checkpoint_paths=checkpoint_paths, source_identity=current_source_identity,
        training_configuration_sha256=planned_job["training_configuration_sha256"],
        qualification_claim=False)
    if _canonical_bytes(rebuilt) != _canonical_bytes(plan):
        raise ValueError("plan differs from deterministic all-scenario validation compilation")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    device_lock_root = Path(tempfile.gettempdir()) / "autodrive-meco-validation-locks"
    device_lock_root.mkdir(parents=True, exist_ok=True)
    device_lock = device_lock_root / f"{device.replace(':', '-')}.lock"
    campaign_lock = root / ".campaign.lock"
    locks = (device_lock, campaign_lock)
    acquired = []
    success = False
    try:
        for lock in locks:
            try:
                fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as exc:
                raise ValueError(f"campaign/device lock exists and must be inspected manually: {lock}") from exc
            acquired.append(lock)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps({"device": device, "plan_digest": plan["plan_digest"],
                    "pid": os.getpid()}, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    except BaseException:
        # A newly acquired lock is removed only when the acquisition itself
        # failed before work began; existing locks are never cleared.
        for lock in acquired:
            lock.unlink(missing_ok=True)
        raise
    try:
        plan_path = root / "plan.json"
        _write_immutable(plan_path, plan)
        for job in plan["jobs"]:
            job_dir = root / "jobs" / job["job_id"]
            job_dir.mkdir(parents=True, exist_ok=True)
            input_payload = {"schema": JOB_INPUT_SCHEMA, "plan_digest": plan["plan_digest"],
                "job": job, "device": device, "diagnostic_only": True,
                "qualification_claim": False}
            input_path = job_dir / "job.input.json"
            input_digest = _write_immutable(input_path, input_payload)
            if (job_dir / "job.complete.json").exists():
                seal = _completion_valid(job, job_dir, device=device,
                    expected_plan_digest=plan["plan_digest"])
                if seal["job_input_sha256"] != input_digest:
                    raise ValueError("completed job input digest differs")
                continue
            matrix_dir = job_dir / "matrix"
            existing = list(job_dir.iterdir())
            unexpected = [p for p in existing if p.name not in
                          ("job.input.json", "matrix")]
            if unexpected:
                raise ValueError("unsealed job directory contains unexpected/partial artifacts")
            if matrix_dir.exists() and any(matrix_dir.iterdir()):
                # An unsealed directory may be sealed only if the matrix and all
                # linked condition/replay artifacts validate in full.
                artifacts = validate_job_artifacts(job, job_dir, device=device)
            else:
                if matrix_dir.exists():
                    matrix_dir.rmdir()
                matrix_dir.mkdir(parents=True, exist_ok=True)
                seed_path = job_dir / job["seed_manifest_file"]
                seed_digest = _write_immutable(seed_path, job["seed_manifest"])
                selector = {condition["condition_name"]: condition["replay_selection"]
                    for condition in job["conditions"] if condition.get("replay_selection")}
                argv = [sys.executable, str(Path(matrix_script).resolve()),
                    "--scenario", job["scenario_id"], "--categories", ",".join(job["categories"]),
                    "--seed-manifest", str(seed_path.resolve()), "--qualification-protocol",
                    str(protocol_path), "--qualification-case-json", "",
                    "--device", device, "--output-dir", str(matrix_dir)]
                # The matrix reads only the named case object, so use a dedicated
                # immutable case file instead of the broader job input payload.
                case_path = job_dir / "qualification-case.json"
                case_digest = _write_immutable(case_path, job["qualification_case"])
                argv[argv.index("")] = str(case_path.resolve())
                if selector:
                    selector_path = job_dir / "replay-selection.json"
                    selector_digest = _write_immutable(selector_path, selector)
                    argv += ["--replay-selection-json", str(selector_path.resolve()),
                             "--replay-output-dir", str((job_dir / "replays").resolve())]
                else:
                    selector_digest = None
                for track in MECO_TRACKS:
                    argv += [f"--{track.lower()}-checkpoint", str(Path(checkpoint_paths[track]).resolve())]
                execute = runner or (lambda command: subprocess.run(command, check=False).returncode)
                result = execute(argv)
                if result != 0:
                    raise RuntimeError(f"matrix runner failed for job {job['job_id']}: exit {result}")
                artifacts = validate_job_artifacts(job, job_dir, device=device)
                if (artifacts["seed_manifest_sha256"] != seed_digest
                        or artifacts["qualification_case_sha256"] != case_digest
                        or artifacts["replay_selection_sha256"] != selector_digest):
                    raise ValueError("validated job input artifact digests changed")
            seal = {"schema": JOB_COMPLETE_SCHEMA, "job_id": job["job_id"],
                "plan_digest": plan["plan_digest"], "job_input_sha256": input_digest,
                "device": device, "diagnostic_only": True, "artifacts": artifacts}
            seal["completion_sha256"] = _sha(seal)
            _write_immutable(job_dir / "job.complete.json", seal)
        completions = []
        for job in plan["jobs"]:
            seal = _completion_valid(job, root / "jobs" / job["job_id"], device=device,
                expected_plan_digest=plan["plan_digest"])
            completions.append({"job_id": job["job_id"],
                "completion_sha256": seal["completion_sha256"]})
        campaign = {"schema": CAMPAIGN_SCHEMA, "plan_digest": plan["plan_digest"],
            "diagnostic_only": True, "qualification_claim": False,
            "qualification_eligible": False, "split": "validation", "device": device,
            "job_count": 28, "jobs": completions}
        campaign["campaign_sha256"] = _sha(campaign)
        _write_immutable(root / "campaign.complete.json", campaign)
        success = True
        return campaign
    finally:
        # Keep lock on failure as a manual inspection signal. Remove only after
        # successful sealing, to serialize simultaneous campaigns safely.
        if success:
            for lock in acquired:
                lock.unlink(missing_ok=True)


def aggregate_campaign(plan: dict, campaign_path: str | Path) -> dict:
    """Verify an exact complete 28-job set and preserve per-scenario summaries."""
    _validate_plan(plan)
    path = Path(campaign_path).resolve()
    campaign = _read_json(path)
    unsigned = {key: value for key, value in campaign.items() if key != "campaign_sha256"}
    if (campaign.get("schema") != CAMPAIGN_SCHEMA or campaign.get("campaign_sha256") != _sha(unsigned)
            or campaign.get("plan_digest") != plan["plan_digest"] or campaign.get("split") != "validation"
            or campaign.get("diagnostic_only") is not True or campaign.get("qualification_claim") is not False
            or campaign.get("qualification_eligible") is not False):
        raise ValueError("campaign completion manifest is invalid or not diagnostic validation")
    by_id = {job["job_id"]: job for job in plan["jobs"]}
    rows = campaign.get("jobs")
    if not isinstance(rows, list) or len(rows) != 28 or {row.get("job_id") for row in rows} != set(by_id):
        raise ValueError("campaign manifest does not cover the exact 28 planned jobs")
    scenario_rows: dict[str, list[dict]] = {}
    sample_identities = set()
    for row in rows:
        job = by_id[row["job_id"]]
        case = job["qualification_case"]
        for category in job["categories"]:
            for seed in job["seed_list"]:
                sample = (job["scenario_id"], _canonical_bytes(case), category, seed)
                if sample in sample_identities:
                    raise ValueError("campaign contains duplicate scenario-aware sample identity")
                sample_identities.add(sample)
        # The aggregate's sibling job directory is resolved from the campaign root.
        job_dir = path.parent / "jobs" / job["job_id"]
        seal = _completion_valid(job, job_dir, device=campaign["device"],
            expected_plan_digest=plan["plan_digest"])
        if seal["completion_sha256"] != row.get("completion_sha256"):
            raise ValueError("campaign links to a different job completion seal")
        matrix_path = Path(seal["artifacts"]["matrix_path"])
        from scripts.summarize_meco_ablation_matrix import summarize_matrix
        summary = summarize_matrix(matrix_path, replicates=100)
        scenario_rows.setdefault(job["scenario_id"], []).append({
            "job_id": job["job_id"], "matrix_id": summary["matrix"]["matrix_id"],
            "matrix_sha256": summary["matrix"]["file_sha256"],
            "summary": [{**item, "scenario_id": job["scenario_id"],
                         "job_id": job["job_id"]} for item in summary["summaries"]]})
    scenarios = [{"scenario_id": scenario, "jobs": scenario_rows[scenario]}
                 for scenario in sorted(scenario_rows)]
    if len(scenarios) != len({job["scenario_id"] for job in plan["jobs"]}):
        raise ValueError("campaign scenario coverage is incomplete")
    result = {"schema": "meco-validation-campaign-diagnostic-summary-v1",
        "campaign_sha256": campaign["campaign_sha256"], "plan_digest": plan["plan_digest"],
        "diagnostic_only": True, "promotion_inference": False,
        "pooled_estimate": None,
        "pooling_note": "No cross-scenario pooled estimate is computed; operational weighting is unspecified.",
        "scenario_count": len(scenarios), "job_count": len(rows), "scenarios": scenarios}
    result["summary_sha256"] = _sha(result)
    return result
