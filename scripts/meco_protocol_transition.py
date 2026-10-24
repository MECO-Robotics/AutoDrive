#!/usr/bin/env python3
"""Audit historical MECO exposure and export explicit actor-only transitions.

This tool is diagnostic and initialization-only. It never marks historical
training as qualification evidence and never resumes optimizer/RNG state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense.meco_candidates import CANDIDATE_BASE_FEATURE_COUNT
from frc_defense.meco_supervised import _load_protocol
from frc_defense.meco_tracks import MECO_TRACKS, make_meco_track
from frc_defense.meco_track_training import ValueHead


ARTIFACT_SCHEMA = "meco-weights-only-protocol-transition-v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _json(path: Path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"actor state {name} is not a tensor")
        cpu = tensor.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(_canonical(list(cpu.shape)))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _source_digest(sources):
    return _sha(json.dumps(sources, sort_keys=True, allow_nan=False).encode())


def _assignment_train_check(protocol, assignment):
    if not isinstance(assignment, dict):
        raise ValueError("scheduler history contains a malformed assignment")
    scenario, category, seed = (assignment.get("scenario_id"),
                                assignment.get("category"),
                                assignment.get("seed"))
    matches = [case for case in protocol.get("cases", [])
        if case.get("scenario_id") == scenario and case.get("category") == category
        and case.get("split") == "train"]
    if not matches or type(seed) is not int or not any(seed in row.get("seeds", [])
                                                       for row in matches):
        raise ValueError("scheduler assignment seed is not in the exact train split")
    if any(seed in row.get("seeds", []) for row in protocol.get("cases", [])
           if row.get("split") in ("validation", "test")):
        raise ValueError("scheduler assignment seed overlaps validation/test")


def audit_segment(segment_dir: Path, qualification_protocol: Path,
                  evaluations: list[Path] | None = None) -> dict:
    """Reconcile scheduler history, all five lane frontiers, and report hashes."""
    root = Path(segment_dir).resolve()
    protocol, protocol_id, protocol_sha, _ = _load_protocol(Path(qualification_protocol))
    scheduler = _json(root / "task-scheduler-state.json")
    if scheduler.get("pending_assignment") is not None:
        raise ValueError("segment has a pending assignment; frontier is not quiescent")
    completed = scheduler.get("completed_cycle")
    history = scheduler.get("history")
    if type(completed) is not int or completed < 1 or not isinstance(history, list) or len(history) != completed:
        raise ValueError("scheduler completed-cycle/history frontier is inconsistent")

    previous_target = 0
    assignment_frontiers = []
    for index, row in enumerate(history, 1):
        if not isinstance(row, dict):
            raise ValueError("scheduler history row is malformed")
        assignment = row.get("assignment")
        if assignment.get("cycle") != index:
            raise ValueError("scheduler cycles are missing, duplicated, or out of order")
        _assignment_train_check(protocol, assignment)
        target = assignment.get("target_update")
        if type(target) is not int or target <= previous_target:
            raise ValueError("scheduler target-update history is invalid")
        previous_target = target
        assignment_frontiers.append((target, assignment))
        results = row.get("lane_results")
        if not isinstance(results, dict) or set(results) != set(MECO_TRACKS):
            raise ValueError("scheduler cycle does not contain exactly five lane results")
        for track in MECO_TRACKS:
            result = results[track]
            if (not isinstance(result, dict)
                    or result.get("assignment_id") != assignment.get("assignment_id")
                    or result.get("completed_updates") != target):
                raise ValueError(f"{track} scheduler outcome disagrees with assignment frontier")

    source_value = None
    shared_config = None
    common_protocol = None
    checkpoints = {}
    latest_assignment = history[-1]["assignment"]
    for track in MECO_TRACKS:
        lane = root / track
        checkpoint_path, run_path = lane / "latest.pt", lane / "run.json"
        if not checkpoint_path.is_file() or not run_path.is_file():
            raise ValueError(f"missing {track} checkpoint or run manifest")
        manifest = _json(run_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"{track} checkpoint is malformed")
        if (manifest.get("track") != track or manifest.get("status") != "complete"
                or manifest.get("completed_updates") != previous_target
                or checkpoint.get("update") != previous_target):
            raise ValueError(f"{track} run/checkpoint update frontier is not the common completed frontier")
        if (manifest.get("run_id") != checkpoint.get("run_id")
                or manifest.get("protocol_id") != checkpoint.get("protocol_id")
                or manifest.get("task_assignment") != latest_assignment
                or checkpoint.get("task_assignment") != latest_assignment):
            raise ValueError(f"{track} run/checkpoint identity disagrees with scheduler frontier")
        progress = checkpoint.get("task_progress")
        if (not isinstance(progress, dict)
                or progress.get("assignment_id") != latest_assignment.get("assignment_id")
                or progress.get("cycle") != completed):
            raise ValueError(f"{track} checkpoint task progress disagrees with scheduler")
        if manifest.get("sources") != checkpoint.get("sources"):
            raise ValueError(f"{track} run/checkpoint source manifests disagree")
        sources = checkpoint.get("sources")
        if not isinstance(sources, dict):
            raise ValueError(f"{track} source identity is missing")
        digest = _source_digest(sources)
        if manifest.get("source_identity_sha256") != digest:
            raise ValueError(f"{track} source identity digest is invalid")
        if source_value is None:
            source_value = sources
        elif source_value != sources:
            raise ValueError("matched lane checkpoints have different full source identities")
        config = checkpoint.get("config")
        if not isinstance(config, dict) or config.get("track") != track:
            raise ValueError(f"{track} checkpoint config is malformed")
        if manifest.get("config") != config:
            raise ValueError(f"{track} run/checkpoint configurations disagree")
        config_digest = _sha(json.dumps({key: value for key, value in config.items()
            if key != "updates"}, sort_keys=True, allow_nan=False).encode())
        if manifest.get("configuration_sha256") != config_digest:
            raise ValueError(f"{track} immutable configuration digest is invalid")
        if common_protocol is None:
            common_protocol = checkpoint.get("protocol_id")
        elif checkpoint.get("protocol_id") != common_protocol:
            raise ValueError("matched lane checkpoints have different supervisor protocol identities")
        normalized = {key: value for key, value in config.items()
                      if key not in ("track", "device", "output", "updates")}
        if shared_config is None:
            shared_config = normalized
        elif shared_config != normalized:
            raise ValueError("matched lane checkpoints have different shared configurations")
        lane_history = checkpoint.get("history")
        if (not isinstance(lane_history, list) or len(lane_history) != previous_target
                or [item.get("update") for item in lane_history] != list(range(1, previous_target + 1))):
            raise ValueError(f"{track} checkpoint history does not cover the common update frontier")
        for update_index, item in enumerate(lane_history, 1):
            expected_assignment = next(assignment for frontier, assignment in assignment_frontiers
                                       if update_index <= frontier)
            if item.get("task_assignment_id") != expected_assignment.get("assignment_id"):
                raise ValueError(f"{track} update history disagrees with scheduler assignments")
        expected_outcome = history[-1]["lane_results"][track]
        if manifest.get("task_outcomes") != expected_outcome:
            raise ValueError(f"{track} run outcome disagrees with scheduler history")
        for key in ("assignment_id", "cycle", "scenario_id", "category", "episodes",
                    "successes", "failures", "aborted_no_legal_support"):
            if key in expected_outcome and progress.get(key) != expected_outcome[key]:
                raise ValueError(f"{track} checkpoint task progress disagrees with scheduler outcome")
        model_state = checkpoint.get("model")
        if not isinstance(model_state, dict) or not model_state:
            raise ValueError(f"{track} checkpoint has no policy actor weights")
        checkpoint_digest = _sha(checkpoint_path.read_bytes())
        checkpoints[track] = {"path": str(checkpoint_path),
            "sha256": checkpoint_digest, "run_id": checkpoint["run_id"],
            "update": previous_target, "protocol_id": checkpoint.get("protocol_id"),
            "configuration_sha256": config_digest,
            "actor_state_sha256": _state_digest(model_state),
            "actor_keys": sorted(model_state)}

    eval_index = []
    for report_path in evaluations or []:
        report_path = Path(report_path).resolve()
        report = _json(report_path)
        if report.get("source_identity_sha256") != _source_digest(source_value):
            raise ValueError(f"evaluation source identity mismatch: {report_path}")
        policy_manifest = report.get("policy_manifest")
        if not isinstance(policy_manifest, dict):
            raise ValueError(f"evaluation lacks policy manifest: {report_path}")
        for track in MECO_TRACKS:
            item = policy_manifest.get(track)
            if not isinstance(item, dict) or item.get("checkpoint_sha256") != checkpoints[track]["sha256"]:
                raise ValueError(f"evaluation does not bind exact {track} checkpoint: {report_path}")
        eval_index.append({"path": str(report_path), "sha256": _sha(report_path.read_bytes()),
            "evaluation_id": report.get("evaluation_id"),
            "protocol_id": report.get("protocol_id"),
            "source_identity_sha256": report["source_identity_sha256"]})

    return {"schema_version": 1, "audit_status": "consistent_historical_segment",
        "qualification_status": "unbound_nonqualifying_historical_segment",
        "qualification_eligible": False,
        "qualification_note": "Retrospective train-split membership does not bind this historical segment to a prospective qualification protocol.",
        "segment_path": str(root), "completed_cycle": completed,
        "completed_update_frontier": previous_target,
        "pending_assignment": None,
        "scheduler_history_sha256": _sha((root / "task-scheduler-state.json").read_bytes()),
        "qualification_protocol_id_checked": protocol_id,
        "qualification_protocol_sha256_checked": protocol_sha,
        "historical_protocol_binding": "unbound",
        "source_identity": source_value,
        "source_identity_sha256": _source_digest(source_value),
        "shared_training_configuration": shared_config,
        "checkpoints": checkpoints, "evaluations": eval_index}


def export_transition(audit_path: Path, output_path: Path) -> dict:
    """Export matched policy actors only; evidence remains explicitly nonqualifying."""
    audit_path, output_path = Path(audit_path).resolve(), Path(output_path).resolve()
    audit = _json(audit_path)
    if audit.get("qualification_eligible") is not False or audit.get("audit_status") != "consistent_historical_segment":
        raise ValueError("transition export requires a consistent, explicitly nonqualifying audit")
    actors = {}
    for track in MECO_TRACKS:
        item = audit.get("checkpoints", {}).get(track)
        if not isinstance(item, dict):
            raise ValueError(f"audit is missing {track} checkpoint identity")
        checkpoint_path = Path(item["path"])
        if not checkpoint_path.is_file() or _sha(checkpoint_path.read_bytes()) != item.get("sha256"):
            raise ValueError(f"{track} checkpoint changed since audit")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actor = checkpoint.get("model")
        if _state_digest(actor) != item.get("actor_state_sha256"):
            raise ValueError(f"{track} actor weights changed since audit")
        model = make_meco_track(track, candidate_features=CANDIDATE_BASE_FEATURE_COUNT)
        model.load_state_dict(actor, strict=True)
        actors[track] = {key: value.detach().cpu().clone() for key, value in actor.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() or sidecar_path.exists():
        raise FileExistsError("refusing to replace an existing transition artifact or sidecar")
    artifact = {"schema": ARTIFACT_SCHEMA, "transition_id": uuid.uuid4().hex,
        "audit_sha256": _sha(audit_path.read_bytes()),
        "historical_qualification_status": audit["qualification_status"],
        "qualification_eligible": False,
        "historical_source_identity": audit["source_identity"],
        "historical_source_identity_sha256": audit["source_identity_sha256"],
        "checked_qualification_protocol_id": audit["qualification_protocol_id_checked"],
        "checked_qualification_protocol_sha256": audit["qualification_protocol_sha256_checked"],
        "checkpoint_identities": audit["checkpoints"],
        "evaluation_identities": audit["evaluations"],
        "shared_training_configuration": audit["shared_training_configuration"],
        "model_contract": {track: {"state_sha256": _state_digest(actors[track]),
            "keys": sorted(actors[track]),
            "shapes": {key: list(value.shape) for key, value in actors[track].items()},
            "dtypes": {key: str(value.dtype) for key, value in actors[track].items()}}
            for track in MECO_TRACKS},
        "transition_contract": {"mode": "actor_weights_only_fresh_initialization",
            "load": "strict actor state_dict only",
            "critic": "fresh initialization required",
            "optimizer": "fresh initialization required; historical optimizer forbidden",
            "rng": "new explicit seed required; historical RNG forbidden",
            "counters": {"update": 0, "episode": 0, "history": []},
            "run_id": "new UUID required",
            "resume": "not a resume artifact; existing resume identity checks unchanged"},
        "actors": actors}
    fd, payload_tmp = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp",
                                       dir=output_path.parent)
    os.close(fd)
    sidecar_tmp = None
    try:
        torch.save(artifact, payload_tmp)
        artifact_sha = _sha(Path(payload_tmp).read_bytes())
        sidecar = {key: value for key, value in artifact.items() if key != "actors"}
        sidecar.update(artifact_sha256=artifact_sha,
            actor_state_sha256={track: _state_digest(actors[track]) for track in MECO_TRACKS})
        fd, sidecar_tmp = tempfile.mkstemp(prefix=f".{sidecar_path.name}.", suffix=".tmp",
                                           dir=output_path.parent)
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(sidecar, sort_keys=True, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(payload_tmp, output_path)
        os.replace(sidecar_tmp, sidecar_path)
    except BaseException:
        for candidate in (payload_tmp, sidecar_tmp):
            if candidate and os.path.exists(candidate):
                os.unlink(candidate)
        for candidate in (output_path, sidecar_path):
            if candidate.exists():
                candidate.unlink()
        raise
    return sidecar


def consume_transition(path: Path, *, seed: int, models: dict | None = None):
    """Strictly load only actors into fresh models and return reset-state metadata.

    Callers create critic/optimizer/RNG state from scratch using the returned
    initialization seed. No historical trainer state is present in the artifact.
    """
    path = Path(path).resolve()
    sidecar = _json(path.with_suffix(path.suffix + ".json"))
    if _sha(path.read_bytes()) != sidecar.get("artifact_sha256"):
        raise ValueError("transition artifact SHA-256 does not match sidecar")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("schema") != ARTIFACT_SCHEMA or artifact.get("qualification_eligible") is not False:
        raise ValueError("unsupported or qualification-claiming transition artifact")
    if models is None:
        models = {track: make_meco_track(track,
            candidate_features=CANDIDATE_BASE_FEATURE_COUNT) for track in MECO_TRACKS}
    if set(models) != set(MECO_TRACKS):
        raise ValueError("consumer requires exactly the five MECO track models")
    if type(seed) is not int or seed < 0:
        raise ValueError("transition consumer requires an explicit nonnegative fresh RNG seed")
    torch.manual_seed(seed)
    random.seed(seed)
    loaded = {}
    for track in MECO_TRACKS:
        actor = artifact.get("actors", {}).get(track)
        contract = artifact.get("model_contract", {}).get(track, {})
        if not isinstance(actor, dict) or _state_digest(actor) != contract.get("state_sha256"):
            raise ValueError(f"{track} actor weights fail transition contract")
        models[track].load_state_dict(actor, strict=True)
        loaded[track] = models[track]
    learning_rate = artifact.get("shared_training_configuration", {}).get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("transition artifact lacks a valid fresh optimizer learning rate")
    critics = {track: ValueHead() for track in MECO_TRACKS}
    optimizers = {track: torch.optim.Adam(
        list(loaded[track].parameters()) + list(critics[track].parameters()),
        lr=float(learning_rate)) for track in MECO_TRACKS}
    return {"models": loaded, "critics": critics, "optimizers": optimizers,
        "initialization": "actor_weights_only_fresh_initialization", "rng_seed": seed,
        "run_id": uuid.uuid4().hex, "update": 0, "episode": 0, "history": [],
        "rng": "new Python/PyTorch seeds initialized; historical RNG not loaded",
        "qualification_eligible": False,
        "lineage": {key: artifact.get(key) for key in (
            "transition_id", "audit_sha256", "historical_source_identity",
            "historical_source_identity_sha256", "checked_qualification_protocol_id",
            "checked_qualification_protocol_sha256", "checkpoint_identities")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--segment-dir", type=Path, required=True)
    audit_parser.add_argument("--qualification-protocol", type=Path, required=True)
    audit_parser.add_argument("--evaluation", type=Path, action="append", default=[])
    audit_parser.add_argument("--output", type=Path, required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--audit", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "audit":
        report = audit_segment(args.segment_dir, args.qualification_protocol, args.evaluation)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True, indent=2,
                                          allow_nan=False) + "\n")
    else:
        report = export_transition(args.audit, args.output)
    print(json.dumps({key: report.get(key) for key in (
        "audit_status", "qualification_status", "qualification_eligible",
        "completed_cycle", "completed_update_frontier", "artifact_sha256")
        if key in report}, sort_keys=True))


if __name__ == "__main__":
    main()
