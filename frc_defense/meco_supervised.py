"""Versioned, observation-only MECO demonstrations and warm-start weights.

The bundled teacher is diagnostic. Records supervise its observed behavior or
candidate action only; they do not claim optimal goals or counterfactual value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from .curriculum_scenarios import make_curriculum_env, teacher_actions
from .learning_contract import CandidateTable
from .meco_candidates import (ROUTE_COST_POINTS_PER_SECOND,
    build_route_feasible_candidates)
from .meco_candidates import CANDIDATE_BOUNDARY, EXTRA_FEATURES
from .meco_pose_graph import build_legal_pose_graph
from .meco_plan_a_policy import SUPPORT_ORDER
from .meco_tracks import (MECO_TRACKS, MECO_CANDIDATE_RAW_FEATURE_COUNT,
    MECO_CANDIDATE_MODEL_INPUT_WIDTH,
    SAFE_ACTION_FEATURES, make_meco_track, score_candidates,
    PREDICTION_MODES, prediction_contract)
from .training_curriculum import get_scenario

DATASET_SCHEMA = "meco-supervised-trajectory-v1"
TARGET_SCHEMA = "meco-observed-action-target-v1"
WEIGHTS_SCHEMA = "meco-warm-start-weights-v1"
COUNTERFACTUAL_WEIGHTS_SCHEMA = "meco-counterfactual-warm-start-weights-v2"
COUNTERFACTUAL_PLAN_C_REPEATED_WEIGHTS_SCHEMA = "meco-counterfactual-plan-c-repeated-warm-start-weights-v1"
COUNTERFACTUAL_PLAN_A_VALUE_WEIGHTS_SCHEMA = "meco-counterfactual-plan-a-value-warm-start-weights-v3"
COUNTERFACTUAL_PLAN_A_P_WEIGHTS_SCHEMA = "meco-counterfactual-plan-a-p-route-warm-start-weights-v1"
PLAN_A_LEGAL_NODE_WEIGHTS_SCHEMA = "meco-plan-a-legal-node-u-b-weights-v1"
PLAN_A_JOINT_UBP_WEIGHTS_SCHEMA = "meco-plan-a-joint-u-b-p-weights-v1"
PLAN_A_EXPERT_GOAL_DATASET_SCHEMA = "meco-plan-a-expert-goal-annotations-v1"
PLAN_A_EXPERT_GOAL_TARGET_SCHEMA = "meco-plan-a-explicit-goal-u-b-targets-v1"
PLAN_A_EXPERT_GOAL_WEIGHTS_SCHEMA = "meco-plan-a-expert-goals-weights-v1"
PLAN_A_EXPERT_GOAL_SEMANTICS = (
    "explicit_annotated_goal_value_and_behavior_preference; teacher selection is not globally optimal")
COUNTERFACTUAL_LABEL_PROVENANCE = "within-prefix matched simulator gross_horizon_return pairwise comparisons"
PLAN_A_VALUE_LABEL_PROVENANCE = (
    "within-prefix matched simulator gross_horizon_return regression and pairwise comparisons"
)
COUNTERFACTUAL_DISCOUNT_SEMANTICS = (
    "one discount transition per macro decision, independent of macro duration; "
    "first branch reward exponent 0 and each continuation decision increments "
    "exponent once, including a final partial macro")
LABEL_PROVENANCE = "curriculum_scenarios.teacher_actions; sensed observations only"
ACTION_TO_BEHAVIOR = {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 6: 3, 5: 4}


def _prepare_native_gpu_runtime(device):
    from .meco_device import require_indexed_cuda_device
    selected = require_indexed_cuda_device(device, purpose="MECO data collection or fitting")
    from .fuel_physics_hip import prepare_native_fuel_extension
    prepare_native_fuel_extension(str(selected))


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode()).hexdigest()


def validate_record(record):
    if not isinstance(record, dict) or record.get("schema") != DATASET_SCHEMA:
        raise ValueError("unsupported MECO supervised dataset record schema")
    if record.get("target_schema") != TARGET_SCHEMA:
        raise ValueError("unsupported MECO supervised target schema")
    if record.get("label_provenance") != LABEL_PROVENANCE:
        raise ValueError("unsupported demonstration label provenance")
    if record.get("track_scope") not in ("all", *MECO_TRACKS):
        raise ValueError("invalid track scope")
    source = record.get("sources")
    if not isinstance(source, dict) or not source:
        raise ValueError("dataset record must bind source identity")
    if record.get("source_identity_sha256") != _canonical_hash(source):
        raise ValueError("dataset source identity digest mismatch")
    if type(record.get("protocol_id")) is not str or not record["protocol_id"]:
        raise ValueError("dataset must bind a nonempty frozen protocol ID")
    if type(record.get("action")) is not int or record["action"] not in range(8):
        raise ValueError("invalid observed teacher action")
    if record.get("behavior_target") != ACTION_TO_BEHAVIOR.get(record["action"]):
        raise ValueError("behavior label must follow the explicit action mapping")
    raster = record.get("raster")
    shape = record.get("raster_shape")
    globals_ = record.get("globals")
    if (not isinstance(shape, list) or len(shape) != 3 or
            any(type(v) is not int or v < 1 for v in shape) or
            not isinstance(raster, list) or len(raster) != math.prod(shape) or
            not isinstance(globals_, list) or len(globals_) != 7):
        raise ValueError("invalid observation tensor dimensions")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in raster + globals_):
        raise ValueError("observation contains nonfinite values")
    candidates = record.get("candidates")
    if record["track_scope"] == "all":
        if not isinstance(candidates, dict):
            raise ValueError("all-track records require a frozen candidate table")
        if candidates.get("snapshot_id") != record.get("snapshot_id"):
            raise ValueError("candidate table and observation snapshot differ")
        _validate_candidates(candidates)
        graph = record.get("graph_inputs")
        if (not isinstance(graph, dict) or not isinstance(graph.get("node_features"), list)
                or not isinstance(graph.get("static_digest"), str)):
            raise ValueError("all-track record lacks dynamic/static graph inputs")
    elif candidates is not None:
        _validate_candidates(candidates)
    return record


def _validate_candidates(table):
    if not isinstance(table, dict):
        raise ValueError("candidate table must be an object")
    ids, rows = table.get("candidate_ids"), table.get("features")
    names, mask = table.get("feature_names"), table.get("valid_mask")
    if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or
            not isinstance(rows, list) or len(rows) != len(ids) or
            not isinstance(names, list) or not names or
            not isinstance(mask, list) or len(mask) != len(ids) or
            any(type(v) is not bool for v in mask)):
        raise ValueError("malformed candidate table")
    if any(not isinstance(row, list) or len(row) != len(names) for row in rows):
        raise ValueError("candidate feature dimensions do not match schema")
    if "action_index" not in names:
        raise ValueError("candidate table has no action_index target grouping")
    for row in rows:
        if any(v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)) for v in row):
            raise ValueError("candidate features contain invalid values")
    _candidate_table(record_id=table.get("snapshot_id"), table=table)


def _candidate_table(record_id, table):
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("candidate snapshot ID is missing")
    return CandidateTable(record_id, tuple(table["candidate_ids"]),
        tuple(tuple(row) for row in table["features"]), tuple(table["valid_mask"]),
        tuple(table["feature_names"]), route_costs=(None if table.get("route_costs") is None
            else tuple(table["route_costs"])),
        schema_version=int(table.get("schema_version", 1)),
        context_digest=table.get("context_digest"))


def _load_protocol(path: Path):
    path = Path(path).resolve()
    payload_bytes = path.read_bytes()
    protocol = json.loads(payload_bytes)
    if not isinstance(protocol, dict):
        raise ValueError("qualification protocol must be a JSON object")
    protocol_id = protocol.get("protocol_id")
    body = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if (not isinstance(protocol_id, str) or
            hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode()).hexdigest() != protocol_id):
        raise ValueError("qualification protocol ID does not match its contents")
    if protocol.get("protocol_version") != "meco-qualification-v1":
        raise ValueError("unsupported qualification protocol version")
    split_seeds = {"train": set(), "validation": set(), "test": set()}
    for case in protocol.get("cases", []):
        key = (case.get("scenario_id"), case.get("category"))
        split = case.get("split")
        seeds = case.get("seeds")
        if (not all(isinstance(value, str) and value for value in key) or
                split not in ("train", "validation", "test") or
                not isinstance(seeds, list) or any(type(seed) is not int or seed < 0
                                                    for seed in seeds)):
            raise ValueError("qualification protocol contains malformed split cases")
        if len(set(seeds)) != len(seeds):
            raise ValueError("qualification protocol contains duplicate seeds within a case")
        split_seeds[split].update(seeds)
    names = tuple(split_seeds)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            if split_seeds[first] & split_seeds[second]:
                raise ValueError("qualification train/validation/test seed partitions overlap")
    return protocol, protocol_id, hashlib.sha256(payload_bytes).hexdigest(), path


def _protocol_train_seeds(protocol, scenario_id, category):
    matches = [case for case in protocol.get("cases", [])
        if case.get("scenario_id") == scenario_id and case.get("category") == category
        and case.get("split") == "train"]
    if not matches:
        raise ValueError("protocol has no matching train cases")
    seeds = []
    for case in matches:
        part = case.get("seeds")
        if (not isinstance(part, list) or not part or
                any(type(seed) is not int or seed < 0 for seed in part) or
                len(set(part)) != len(part)):
            raise ValueError("qualification protocol has an invalid train seed list")
        seeds.extend(part)
    return set(seeds)


def validate_training_assignment(protocol: dict, assignment: dict):
    scenario_id, category = assignment.get("scenario_id"), assignment.get("category")
    seed = assignment.get("seed")
    if type(seed) is not int or seed not in _protocol_train_seeds(protocol, scenario_id, category):
        raise ValueError("training assignment is outside the frozen qualification train split")


def collect_dataset(output: Path, *, scenario_id: str, category: str,
                    seeds: list[int], qualification_protocol: Path, device: str | None = None,
                    horizon: int = 160, action_ticks: int = 12,
                    candidate_k: int = 16, prediction_mode: str = "cv"):
    """Collect deterministic teacher observations from train-partition seeds."""
    protocol, protocol_id, protocol_sha256, protocol_path = _load_protocol(qualification_protocol)
    if prediction_mode not in PREDICTION_MODES:
        raise ValueError(f"prediction_mode must be one of {PREDICTION_MODES}")
    definition = get_scenario(scenario_id)
    if category not in definition.required_categories:
        raise ValueError("category is not registered for scenario")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("provide nonnegative training seeds")
    allowed_seeds = _protocol_train_seeds(protocol, scenario_id, category)
    if len(set(seeds)) != len(seeds) or not set(seeds) <= allowed_seeds:
        raise ValueError("dataset seeds must be unique members of the frozen train split")
    if candidate_k not in (8, 16, 32) or horizon < 1 or action_ticks < 1:
        raise ValueError("invalid collection horizon or candidate K")
    from .curriculum_training import source_identity
    from .meco_track_training import (_atomic_json, _graph_inputs, _reset_episode,
        _run_macro_step)
    _prepare_native_gpu_runtime(device)
    sources = source_identity()
    manifest = {"schema": DATASET_SCHEMA, "target_schema": TARGET_SCHEMA,
        "label_provenance": LABEL_PROVENANCE, "track_scope": "all",
        "scenario_id": scenario_id, "category": category,
        "seeds": list(seeds), "split": "train", "protocol_id": protocol_id,
        "qualification_protocol_sha256": protocol_sha256,
        "qualification_protocol_file": protocol_path.name,
        "prediction_mode": prediction_mode,
        "prediction_contract": prediction_contract(prediction_mode),
        "sources": sources, "source_identity_sha256": _canonical_hash(sources),
        "candidate_boundary": CANDIDATE_BOUNDARY, "candidate_k": candidate_k,
        "action_ticks": action_ticks, "horizon_physics_ticks": horizon,
        "target_semantics": {
            "A1_A2_A3": "observed_teacher_action_to_behavior_index_only; no goal-value labels",
            "C1_C2": "probability_mass over feasible candidates sharing observed teacher action",
            "counterfactual_rewards": "not_collected"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_path = output
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    count = 0
    graph_static = None
    env = make_curriculum_env(scenario_id, category, 1, device, seeds[0])
    try:
        graph = build_legal_pose_graph(env, slot=3)
        import os
        with os.fdopen(fd, "w", encoding="utf-8", closefd=True) as stream:
            for episode_seed in seeds:
                obs = _reset_episode(env, episode_seed)
                generation = f"demo-{scenario_id}-{category}-{episode_seed}"
                for sequence in range(max(1, horizon // action_ticks)):
                    safe = __import__("frc_defense.meco_tracks", fromlist=["build_safe_action_candidates"]).build_safe_action_candidates(
                        env, obs, world=0, slot=3, episode_id=generation)
                    raster = torch.tensor(safe.observation.raster, dtype=torch.float32,
                        device=device).reshape(1, *safe.observation.raster_shape)
                    action = int(teacher_actions(env, obs)[0, 3].item())
                    graph_inputs = _graph_inputs("A2", graph, env, raster)
                    if graph_inputs and graph_static is None:
                        graph_static = {key: value.detach().cpu()
                            for key, value in graph_inputs.items() if key != "node_features"}
                    candidates = build_route_feasible_candidates(env, obs, graph,
                        episode_id=generation, action_ticks=action_ticks,
                        candidate_k=candidate_k, prediction_mode=prediction_mode)
                    table = candidates.table
                    record = {**manifest, "snapshot_id": safe.observation.snapshot_id,
                        "sequence": sequence, "episode_seed": episode_seed,
                        "raster_shape": list(safe.observation.raster_shape),
                        "raster": list(safe.observation.raster),
                        "globals": list(safe.observation.globals), "action": action,
                        "behavior_target": ACTION_TO_BEHAVIOR.get(action),
                        "graph_inputs": ({"node_features": graph_inputs["node_features"].detach().cpu().tolist(),
                            "static_digest": _canonical_hash({key: value.tolist()
                                for key, value in graph_static.items()})}
                            if graph_inputs else None),
                        "candidates": {"snapshot_id": table.snapshot_id,
                            "candidate_ids": list(table.candidate_ids),
                            "features": [list(row) for row in table.features],
                            "valid_mask": list(table.valid_mask),
                            "feature_names": list(table.feature_names),
                            "route_costs": list(table.route_costs),
                            "schema_version": table.schema_version,
                            "context_digest": table.context_digest},
                        "route_cost_points": candidates.route_cost_points.detach().cpu().reshape(-1).tolist(),
                        "risk_cost_points": candidates.risk_cost_points.detach().cpu().reshape(-1).tolist()}
                    validate_record(record)
                    stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                    count += 1
                    _, done, truncated, _, *_ = _run_macro_step(env, action, action_ticks)
                    obs = env._last_obs
                    if done or truncated:
                        break
            stream.flush()
            import os
            os.fsync(stream.fileno())
        if source_identity() != sources:
            raise RuntimeError("source/runtime changed during supervised collection")
        graph_path = rows_path.with_suffix(rows_path.suffix + ".graph.pt")
        graph_digest = None
        if graph_static is not None:
            torch.save({"schema": "meco-supervised-graph-static-v1",
                        "tensors": graph_static}, graph_path)
            graph_digest = hashlib.sha256(graph_path.read_bytes()).hexdigest()
        os.replace(temp_name, rows_path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    digest = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    _atomic_json(rows_path.with_suffix(rows_path.suffix + ".manifest.json"),
        {**manifest, "record_count": count, "dataset_sha256": digest,
         "graph_static_file": (graph_path.name if graph_digest else None),
         "graph_static_sha256": graph_digest,
         "candidate_commitment": "none; trajectory observations are independent labels"})
    return {"path": str(rows_path), "record_count": count, "dataset_sha256": digest}


def _read_dataset(path, *, qualification_protocol, expected_sources):
    path = Path(path)
    protocol, protocol_id, protocol_sha256, protocol_path = _load_protocol(qualification_protocol)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != DATASET_SCHEMA or manifest.get("target_schema") != TARGET_SCHEMA:
        raise ValueError("unsupported dataset manifest schema")
    if (manifest.get("protocol_id") != protocol_id or
            manifest.get("qualification_protocol_sha256") != protocol_sha256 or
            manifest.get("qualification_protocol_file") != protocol_path.name):
        raise ValueError("dataset frozen qualification protocol differs from supplied artifact")
    if manifest.get("sources") != expected_sources:
        raise ValueError("dataset source identity differs from current training source")
    mode = manifest.get("prediction_mode")
    if (mode not in PREDICTION_MODES or
            manifest.get("prediction_contract") != prediction_contract(mode)):
        raise ValueError("dataset prediction mode/provenance is missing or invalid")
    if manifest.get("candidate_boundary") != CANDIDATE_BOUNDARY:
        raise ValueError("dataset candidate boundary differs from the current adapter")
    if manifest.get("source_identity_sha256") != _canonical_hash(manifest["sources"]):
        raise ValueError("dataset source identity digest mismatch")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest.get("dataset_sha256"):
        raise ValueError("dataset content digest mismatch")
    with path.open(encoding="utf-8") as stream:
        records = [validate_record(json.loads(line)) for line in stream if line.strip()]
    if len(records) != manifest.get("record_count") or not records:
        raise ValueError("dataset record count mismatch or empty dataset")
    if any(row["protocol_id"] != protocol_id or
           row.get("qualification_protocol_sha256") != protocol_sha256 or
           row["sources"] != expected_sources or
           row.get("source_identity_sha256") != _canonical_hash(row["sources"])
           for row in records):
        raise ValueError("mixed source or protocol identities in dataset")
    if any(row.get("scenario_id") != manifest.get("scenario_id") or
           row.get("category") != manifest.get("category") or
           row.get("split") != manifest.get("split") or
           row.get("prediction_mode") != mode
           for row in records):
        raise ValueError("dataset record task/split differs from its manifest")
    allowed_seeds = _protocol_train_seeds(protocol, manifest.get("scenario_id"),
                                          manifest.get("category"))
    if (manifest.get("split") != "train" or
            not set(manifest.get("seeds", [])) <= allowed_seeds or
            any(row.get("episode_seed") not in allowed_seeds or
                row.get("episode_seed") not in manifest.get("seeds", [])
                for row in records)):
        raise ValueError("dataset records do not belong to the frozen train seed split")
    graph_file = manifest.get("graph_static_file")
    if not isinstance(graph_file, str) or Path(graph_file).name != graph_file:
        raise ValueError("dataset is missing its static graph artifact")
    if graph_file:
        graph_path = manifest_path.parent / graph_file
        graph_sha = manifest.get("graph_static_sha256")
        if (not isinstance(graph_sha, str) or len(graph_sha) != 64 or
                hashlib.sha256(graph_path.read_bytes()).hexdigest() != graph_sha):
            raise ValueError("static graph input digest mismatch")
        graph_payload = torch.load(graph_path, map_location="cpu", weights_only=True)
        if graph_payload.get("schema") != "meco-supervised-graph-static-v1":
            raise ValueError("unsupported static graph input schema")
        graph_static = graph_payload["tensors"]
        required_graph = {"edges", "edge_mask", "region_ids", "coarse_edges", "coarse_edge_mask"}
        if not isinstance(graph_static, dict) or set(graph_static) != required_graph:
            raise ValueError("static graph artifact is incomplete")
        graph_digest = _canonical_hash({key: value.tolist()
            for key, value in graph_static.items()})
        for row in records:
            if row.get("graph_inputs") is not None:
                if row["graph_inputs"].pop("static_digest", None) != graph_digest:
                    raise ValueError("record static graph identity mismatch")
                row["graph_inputs"].update({key: value.tolist()
                    for key, value in graph_static.items()})
    return records, digest, protocol_id, protocol_sha256, manifest


def warm_start(track: str, dataset: Path, output: Path, *, qualification_protocol: Path,
               updates: int = 10, learning_rate: float = 1e-4, device: str | None = None):
    """Fit observed action preferences and save model weights only."""
    if track not in MECO_TRACKS or type(updates) is not int or updates < 1:
        raise ValueError("invalid track or update count")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be positive and finite")
    from .curriculum_training import source_identity
    _prepare_native_gpu_runtime(device)
    sources = source_identity()
    records, dataset_hash, protocol_id, protocol_sha256, dataset_manifest = _read_dataset(dataset,
        qualification_protocol=qualification_protocol, expected_sources=sources)
    records = [row for row in records if row["track_scope"] in ("all", track)]
    if not records:
        raise ValueError(f"dataset has no examples for {track}")
    expected_feature_names = tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)
    if any(tuple(row["candidates"]["feature_names"]) != expected_feature_names
           for row in records):
        raise ValueError("candidate feature schema differs from the MECO model contract")
    model = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    model.train()
    for _ in range(updates):
        epoch_losses = []
        for row in records:
            raster = torch.tensor(row["raster"], dtype=torch.float32, device=device).reshape(1, *row["raster_shape"])
            globals_ = torch.tensor(row["globals"], dtype=torch.float32, device=device).reshape(1, 7)
            if track in ("A1", "A2", "A3"):
                if row["behavior_target"] is None:
                    continue
                kwargs = {}
                if track in ("A2", "A3"):
                    graph_values = row.get("graph_inputs")
                    if not isinstance(graph_values, dict):
                        raise ValueError(f"dataset record has no graph input for {track}")
                    kwargs = {key: torch.tensor(value, device=device)
                              for key, value in graph_values.items()}
                    for name in ("edges", "region_ids", "coarse_edges"):
                        if name in kwargs:
                            kwargs[name] = kwargs[name].long()
                    for name in ("edge_mask", "coarse_edge_mask"):
                        if name in kwargs:
                            kwargs[name] = kwargs[name].bool()
                output_values = model(raster=raster, globals_=globals_, **kwargs)
                target = torch.tensor([row["behavior_target"]], dtype=torch.long, device=device)
                loss = F.cross_entropy(output_values["behavior_logits"], target)
            else:
                table_data = row["candidates"]
                table = _candidate_table(row["snapshot_id"], table_data)
                route = torch.tensor(row["route_cost_points"], dtype=torch.float32, device=device).reshape(1, -1)
                risk = torch.tensor(row["risk_cost_points"], dtype=torch.float32, device=device).reshape(1, -1)
                result = score_candidates(track, model, raster=raster, globals_=globals_,
                    table=table, route_cost=route, risk_cost=risk)
                logits = result["logits"].reshape(-1)
                action_col = table.feature_names.index("action_index")
                target_mask = torch.tensor([valid and row_values[action_col] == row["action"]
                    for row_values, valid in zip(table.features, table.valid_mask)],
                    dtype=torch.bool, device=device)
                valid_mask = torch.tensor(table.valid_mask, dtype=torch.bool, device=device)
                if not bool(target_mask.any()):
                    continue
                loss = torch.logsumexp(logits[valid_mask], 0) - torch.logsumexp(logits[target_mask], 0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if not epoch_losses:
            raise ValueError("no records contain a feasible observed-action candidate")
        losses.append(sum(epoch_losses) / len(epoch_losses))
    payload = {"schema": WEIGHTS_SCHEMA, "schema_version": 1, "track": track,
        "model_schema_version": 1, "model": model.state_dict(),
        "dataset_sha256": dataset_hash, "dataset_schema": DATASET_SCHEMA,
        "target_schema": TARGET_SCHEMA, "label_provenance": LABEL_PROVENANCE,
        "target_semantics": ("observed_teacher_behavior_only" if track.startswith("A")
                             else "observed_teacher_action_probability_mass"),
        "candidate_raw_feature_count": MECO_CANDIDATE_RAW_FEATURE_COUNT,
        "candidate_model_input_width": MECO_CANDIDATE_MODEL_INPUT_WIDTH,
        "prediction_mode": dataset_manifest["prediction_mode"],
        "prediction_contract": dataset_manifest["prediction_contract"],
        "candidate_feature_names": list(expected_feature_names),
        "source_identity_sha256": _canonical_hash(sources), "sources": sources,
        "protocol_id": protocol_id,
        "qualification_protocol_sha256": protocol_sha256,
        "qualification_protocol_file": Path(qualification_protocol).name,
        "optimizer_state_included": False,
        "resume_compatible": False, "updates": updates, "learning_rate": learning_rate,
        "mean_loss_by_epoch": losses}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {"track": track, "checkpoint": str(output), "dataset_sha256": dataset_hash,
            "mean_loss_by_epoch": losses}


def _finite_number(value):
    return (type(value) in (int, float) and math.isfinite(value))


def _validate_plan_c_repeated_labels(row, *, repeat_protocol, protocol_id,
        scenario_id, category, seed, decision_index):
    from .meco_counterfactual import (CONTINUATION_POLICY,
        COUNTERFACTUAL_DISCOUNT_SEMANTICS, PLAN_C_MATCHED_REPEAT_COUNT,
        PLAN_C_REPEAT_TARGET_SCHEMA, _plan_c_repeated_outcome,
        _plan_c_pairwise_labels, plan_c_repeat_seed)
    if (row.get("plan_c_target_schema") != PLAN_C_REPEAT_TARGET_SCHEMA or
            row.get("plan_c_repeat_protocol_id") != repeat_protocol.get("repeat_protocol_id")):
        raise ValueError("row Plan C repeat identity differs from its manifest")
    outcomes = row.get("plan_c_counterfactuals")
    labels = row.get("plan_c_pairwise_labels")
    if not isinstance(outcomes, list) or len(outcomes) < 2 or not isinstance(labels, list):
        raise ValueError("Plan C repeated outcomes or paired labels are missing")
    expected_seeds = [plan_c_repeat_seed(protocol_id=protocol_id,
        scenario_id=scenario_id, category=category, seed=seed,
        decision_index=decision_index, repeat_index=repeat_index)
        for repeat_index in range(PLAN_C_MATCHED_REPEAT_COUNT)]
    generator_provenance_by_repeat = {}
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise ValueError("Plan C candidate outcome is malformed")
        repeats = outcome.get("repeat_returns")
        if (outcome.get("repeat_count") != PLAN_C_MATCHED_REPEAT_COUNT or
                not isinstance(repeats, list) or len(repeats) != PLAN_C_MATCHED_REPEAT_COUNT or
                [item.get("repeat_index") if isinstance(item, dict) else None
                 for item in repeats] != list(range(PLAN_C_MATCHED_REPEAT_COUNT)) or
                [item.get("repeat_seed") if isinstance(item, dict) else None
                 for item in repeats] != expected_seeds):
            raise ValueError("Plan C candidate is missing its complete, matched repeat set")
        for repeat_index, repeat in enumerate(repeats):
            if (any(not _finite_number(repeat.get(key)) for key in
                    ("gross_macro_return", "gross_horizon_return", "discounted_horizon_return")) or
                    repeat.get("continuation_policy") != CONTINUATION_POLICY or
                    repeat.get("discount_semantics") != COUNTERFACTUAL_DISCOUNT_SEMANTICS or
                    type(repeat.get("continuation_macro_horizon")) is not int or
                    repeat["continuation_macro_horizon"] < 0 or
                    type(repeat.get("terminal")) is not bool):
                raise ValueError("Plan C repeat return/provenance is malformed")
            rng_provenance = repeat.get("repeat_rng_provenance")
            if (not isinstance(rng_provenance, dict) or
                    type(rng_provenance.get("generator_count")) is not int or
                    rng_provenance["generator_count"] < 1 or
                    not isinstance(rng_provenance.get("seed_manifest_sha256"), str) or
                    len(rng_provenance["seed_manifest_sha256"]) != 64 or
                    any(char not in "0123456789abcdef"
                        for char in rng_provenance["seed_manifest_sha256"])):
                raise ValueError("Plan C repeat is missing environment RNG provenance")
            prior = generator_provenance_by_repeat.setdefault(repeat_index, rng_provenance)
            if prior != rng_provenance:
                raise ValueError("Plan C candidates do not share the repeat RNG generator manifest")
        try:
            canonical = _plan_c_repeated_outcome(outcome.get("descriptor"), repeats)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("Plan C repeat outcome cannot be summarized") from exc
        if canonical != outcome:
            raise ValueError("Plan C repeat means/variance differ from per-repeat returns")
    try:
        expected_labels = _plan_c_pairwise_labels(outcomes)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("Plan C paired labels cannot be derived from repeats") from exc
    if any(not isinstance(label, dict) or type(label.get("stable_preference")) is not bool
           for label in labels):
        raise ValueError("Plan C repeated label stability must use exact boolean values")
    if labels != expected_labels:
        raise ValueError("Plan C paired stability labels/variance are not derived from repeat returns")
    return True


def _plan_c_stable_pair_indices(outcomes, labels):
    candidate_index_by_id = {}
    for outcome in outcomes:
        descriptor = outcome["descriptor"]
        candidate_id, candidate_index = descriptor.get("candidate_id"), descriptor.get("candidate_index")
        if (not isinstance(candidate_id, str) or type(candidate_index) is not int or
                candidate_index < 0 or candidate_id in candidate_index_by_id):
            raise ValueError("Plan C candidate-to-table index mapping is invalid")
        candidate_index_by_id[candidate_id] = candidate_index
    stable = []
    for label in labels:
        if not label["stable_preference"]:
            continue
        left, right = label["left_id"], label["right_id"]
        if left not in candidate_index_by_id or right not in candidate_index_by_id:
            raise ValueError("stable Plan C preference references an unknown candidate")
        preferred = label.get("preferred_id")
        if preferred not in (left, right):
            raise ValueError("stable Plan C preference has no preferred candidate")
        stable.append((candidate_index_by_id[left], candidate_index_by_id[right],
                       preferred == left))
    return stable


def load_plan_a_expert_goal_annotations(dataset: Path, legal_node_dataset: Path,
        source_dataset: Path, *, qualification_protocol: Path,
        expected_sources: dict, annotation_source_path: Path, device: str):
    """Load only explicit, source-bound expert goal/behavior annotations.

    This is a strict data contract, not a label synthesizer or fitter. In
    particular, an observed action alone cannot identify an expert goal or a
    goal value and is rejected. U is available only when the annotation gives
    a finite value and declares complete uncensored support; B is an explicit
    selected feasible behavior preference, never an optimality claim.
    """
    import hashlib
    dataset, legal_node_dataset, source_dataset = map(
        lambda path: Path(path).resolve(), (dataset, legal_node_dataset, source_dataset))
    manifest_path = dataset.with_suffix(dataset.suffix + ".manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    data_bytes = dataset.read_bytes()
    data_sha = hashlib.sha256(data_bytes).hexdigest()
    legal_path = legal_node_dataset.with_suffix(legal_node_dataset.suffix + ".manifest.json")
    legal_manifest_bytes = legal_path.read_bytes()
    legal_manifest = json.loads(legal_manifest_bytes)
    legal_bytes = legal_node_dataset.read_bytes()
    legal_sha = hashlib.sha256(legal_bytes).hexdigest()
    annotation_source_sha = hashlib.sha256(Path(annotation_source_path).resolve().read_bytes()).hexdigest()
    from .meco_counterfactual import (PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
        PLAN_A_LEGAL_NODE_TARGET_SCHEMA,
        PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA, _replay_plan_a_expert_goal_source_row)
    expert_source_manifest_path = source_dataset.with_suffix(source_dataset.suffix + ".manifest.json")
    expert_source_manifest = json.loads(expert_source_manifest_path.read_text())
    expert_source_sha = hashlib.sha256(source_dataset.read_bytes()).hexdigest()
    legal_source_dataset = source_dataset
    if expert_source_manifest.get("schema") == PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA:
        base_name = expert_source_manifest.get("base_dataset")
        if (not isinstance(base_name, str) or not base_name or
                Path(base_name).is_absolute()):
            raise ValueError("expert-goal source has invalid base dataset reference")
        base_dataset = (source_dataset.parent / base_name).resolve()
        legal_source_dataset = base_dataset
        source_rows, source_sha, source_manifest_sha, protocol, protocol_id, protocol_sha, source_manifest = \
            _read_counterfactual_dataset(base_dataset,
                qualification_protocol=qualification_protocol, expected_sources=expected_sources)
        if (expert_source_manifest.get("target_schema") !=
                "meco-plan-a-explicit-selected-goal-trace-v1" or
                expert_source_manifest.get("base_dataset_schema") !=
                    "meco-counterfactual-trajectory-v3" or
                expert_source_manifest.get("base_dataset_sha256") != source_sha or
                expert_source_manifest.get("base_manifest_sha256") != source_manifest_sha or
                expert_source_manifest.get("sources") != expected_sources or
                expert_source_manifest.get("source_identity_sha256") != _canonical_hash(expected_sources) or
                expert_source_manifest.get("protocol_id") != protocol_id or
                expert_source_manifest.get("qualification_protocol_sha256") != protocol_sha or
                expert_source_manifest.get("dataset_sha256") != expert_source_sha or
                expert_source_manifest.get("support_order") != SUPPORT_ORDER or
                expert_source_manifest.get("split") != "train" or
                expert_source_manifest.get("records_have_replay_bound_selected_goal") is not True):
            raise ValueError("expert-goal source does not bind exact base data/protocol/runtime")
        goal_rows = [json.loads(line) for line in source_dataset.read_text().splitlines() if line]
        if len(goal_rows) != expert_source_manifest.get("record_count"):
            raise ValueError("expert-goal source record count mismatch")
        source_by_key = {(r["scenario_id"], r["category"], r["seed"], r["decision_index"]): r
            for r in source_rows}
        seen_goal_keys = set()
        for goal_row in goal_rows:
            key = (goal_row.get("scenario_id"), goal_row.get("category"),
                   goal_row.get("seed"), goal_row.get("decision_index"))
            base = source_by_key.get(key)
            if (base is None or key in seen_goal_keys or goal_row.get("base_record") != base or
                    goal_row.get("schema") != PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA or
                    goal_row.get("target_schema") != "meco-plan-a-explicit-selected-goal-trace-v1" or
                    goal_row.get("snapshot_id") != base.get("branch_state", {}).get("snapshot_id") or
                    goal_row.get("state_sha256") != base.get("predecision_simulator_state_sha256") or
                    goal_row.get("support_digest") != base.get("support_digest")):
                raise ValueError("expert-goal source row does not map to exact frozen v3 record")
            seen_goal_keys.add(key)
            verified = _replay_plan_a_expert_goal_source_row(goal_row, source_manifest)
            source_by_key[key] = {**base, **verified}
        if seen_goal_keys != set(source_by_key):
            raise ValueError("expert-goal source does not cover the exact v3 source row set")
        source_rows = list(source_by_key.values())
    else:
        source_rows, source_sha, source_manifest_sha, protocol, protocol_id, protocol_sha, source_manifest = \
            _read_counterfactual_dataset(source_dataset,
                qualification_protocol=qualification_protocol, expected_sources=expected_sources)
    annotation_source = manifest.get("annotation_source")
    if (manifest.get("schema") != PLAN_A_EXPERT_GOAL_DATASET_SCHEMA or
            manifest.get("target_schema") != PLAN_A_EXPERT_GOAL_TARGET_SCHEMA or
            manifest.get("target_semantics") != PLAN_A_EXPERT_GOAL_SEMANTICS or
            manifest.get("split") != "train" or
            manifest.get("source_dataset_sha256") != source_sha or
            manifest.get("source_manifest_sha256") != source_manifest_sha or
            (expert_source_manifest.get("schema") == PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA and
             manifest.get("expert_goal_source_sha256") != expert_source_sha) or
            manifest.get("legal_node_dataset_sha256") != legal_sha or
            manifest.get("legal_node_manifest_sha256") != hashlib.sha256(legal_manifest_bytes).hexdigest() or
            manifest.get("protocol_id") != protocol_id or
            manifest.get("qualification_protocol_sha256") != protocol_sha or
            manifest.get("sources") != expected_sources or
            manifest.get("source_identity_sha256") != _canonical_hash(expected_sources) or
            manifest.get("dataset_sha256") != data_sha or
            not isinstance(annotation_source, dict) or
            not isinstance(annotation_source.get("source_id"), str) or not annotation_source["source_id"] or
            not isinstance(annotation_source.get("sha256"), str) or len(annotation_source["sha256"]) != 64 or
            annotation_source.get("sha256") != annotation_source_sha or
            any(char not in "0123456789abcdef" for char in annotation_source["sha256"]) or
            manifest.get("support_order") != SUPPORT_ORDER):
        raise ValueError("expert-goal annotation manifest/source/protocol binding mismatch")
    if (legal_manifest.get("schema") != PLAN_A_LEGAL_NODE_DATASET_SCHEMA or
            legal_manifest.get("target_schema") != PLAN_A_LEGAL_NODE_TARGET_SCHEMA or
            legal_manifest.get("split") != "train" or
            legal_manifest.get("source_dataset_sha256") != source_sha or
            legal_manifest.get("source_manifest_sha256") != source_manifest_sha or
            legal_manifest.get("protocol_id") != protocol_id or
            legal_manifest.get("qualification_protocol_sha256") != protocol_sha or
            legal_manifest.get("support_order") != SUPPORT_ORDER or
            legal_manifest.get("sources") != expected_sources or
            legal_manifest.get("source_identity_sha256") != _canonical_hash(expected_sources) or
            legal_manifest.get("dataset_sha256") != legal_sha):
        raise ValueError("expert-goal annotations do not bind the frozen legal-node source")
    if (expert_source_manifest.get("schema") == PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA and
            (legal_manifest.get("expert_goal_source_sha256") != expert_source_sha or
             legal_manifest.get("expert_goal_source_manifest_sha256") !=
                hashlib.sha256(expert_source_manifest_path.read_bytes()).hexdigest())):
        raise ValueError("legal-node coverage was not collected with this exact expert-goal source")
    replayed_legal, _, replayed_legal_sha, _ = _read_and_replay_plan_a_legal_node_dataset(
        legal_node_dataset, legal_source_dataset, qualification_protocol, device, track=None)
    if replayed_legal_sha != legal_sha:
        raise ValueError("bound legal-node dataset changed during replay validation")
    legal_rows = [sample["row"] for sample in replayed_legal]
    if len(legal_rows) != legal_manifest.get("record_count") or not legal_rows:
        raise ValueError("bound legal-node dataset row count mismatch")
    legal_by_key = {}
    for legal in legal_rows:
        key = (legal.get("scenario_id"), legal.get("category"),
               legal.get("seed"), legal.get("decision_index"))
        if key in legal_by_key:
            raise ValueError("bound legal-node dataset contains duplicate branch identities")
        legal_by_key[key] = legal
    source_by_key = {(r.get("scenario_id"), r.get("category"), r.get("seed"), r.get("decision_index")): r
                     for r in source_rows}
    rows = [json.loads(line) for line in data_bytes.decode("utf-8").splitlines() if line.strip()]
    if not rows or len(rows) != manifest.get("record_count"):
        raise ValueError("expert-goal annotation row count mismatch or empty data")
    allowed = _protocol_train_seeds(protocol, source_manifest["scenario_id"], source_manifest["category"])
    seen = set()
    for row in rows:
        key = (row.get("scenario_id"), row.get("category"), row.get("seed"), row.get("decision_index"))
        if (row.get("schema") != PLAN_A_EXPERT_GOAL_DATASET_SCHEMA or
                row.get("target_schema") != PLAN_A_EXPERT_GOAL_TARGET_SCHEMA or
                row.get("annotation_source_id") != annotation_source["source_id"] or
                key in seen or key not in legal_by_key or key not in source_by_key or
                type(key[2]) is not int or key[2] not in allowed):
            raise ValueError("expert-goal row is malformed, duplicated, or outside bound train support")
        seen.add(key)
        legal, source = legal_by_key[key], source_by_key[key]
        source_state_sha = source.get("simulator_state_sha256")
        if (not isinstance(source.get("snapshot_id"), str) or not source["snapshot_id"] or
                not isinstance(source_state_sha, str) or len(source_state_sha) != 64 or
                any(char not in "0123456789abcdef" for char in source_state_sha) or
                not isinstance(row.get("snapshot_id"), str) or not row["snapshot_id"] or
                not isinstance(row.get("state_sha256"), str) or len(row["state_sha256"]) != 64 or
                row.get("support_digest") != legal.get("support_digest") or
                row.get("snapshot_id") != source.get("snapshot_id") or
                row.get("state_sha256") != source_state_sha or
                legal.get("support_digest") is None):
            raise ValueError("expert annotation does not identify the exact replay support/execution state")
        if (not isinstance(source.get("teacher_selected_goal_descriptor_sha256"), str) or
                len(source["teacher_selected_goal_descriptor_sha256"]) != 64 or
                row.get("executed_goal_descriptor_sha256") !=
                    source["teacher_selected_goal_descriptor_sha256"]):
            raise ValueError("source row has no replay-bound executed-goal identity; recollect with explicit goal provenance")
        executed_action = row.get("executed_action")
        expected_action = source.get("teacher_selected_action", source.get("teacher_action"))
        expected_behavior = source.get("teacher_selected_behavior",
            ACTION_TO_BEHAVIOR.get(executed_action))
        if (type(executed_action) is not int or executed_action != expected_action or
                type(expected_behavior) is not int or
                type(row.get("selected_behavior_id")) is not int or
                row["selected_behavior_id"] != expected_behavior):
            raise ValueError("annotation behavior/action does not match the replayed teacher execution")
        goal = row.get("selected_goal")
        if not isinstance(goal, dict) or type(goal.get("node_id")) is not int:
            raise ValueError("action-only row rejected: explicit selected legal goal annotation is required")
        node = goal["node_id"]
        pose = next((p for p in legal.get("canonical_poses", []) if p.get("node_id") == node), None)
        if pose is None or node not in legal.get("feasible_node_ids", []):
            raise ValueError("annotated goal is not reachable and feasible in exact replay support")
        if (goal.get("x_m"), goal.get("y_m"), goal.get("heading_bin")) != (
                pose.get("x_m"), pose.get("y_m"), pose.get("heading_bin")):
            raise ValueError("annotated goal pose differs from replayed canonical legal goal")
        if goal.get("descriptor_sha256") != row.get("executed_goal_descriptor_sha256"):
            raise ValueError("annotated legal goal differs from the replay-bound executed goal")
        behavior = row.get("selected_behavior_id")
        if type(behavior) is not int:
            raise ValueError("explicit selected feasible behavior annotation is required")
        matching = [p for p in legal.get("pairs", []) if p.get("node_id") == node and
                    p.get("behavior_id") == behavior]
        if not matching or not any(p.get("status") == "available" for p in matching):
            raise ValueError("annotated behavior is not feasible and available in exact support")
        node_target = next((item for item in legal.get("node_u_targets", [])
                            if item.get("node_id") == node), None)
        if node_target is None:
            raise ValueError("annotated goal has no legal-node target support record")
        censored = row.get("censored_feasible_behavior_ids")
        if (not isinstance(censored, list) or any(type(x) is not int for x in censored) or
                len(set(censored)) != len(censored) or
                set(censored) != set(node_target.get("censored_feasible_behavior_ids", []))):
            raise ValueError("annotation censoring inventory differs from replayed feasible support")
        u = row.get("u_target")
        u_status = row.get("u_status")
        if u_status == "available":
            if (not _finite_number(u) or censored or
                    node_target.get("status") != "available" or
                    set(node_target.get("feasible_behavior_ids", [])) !=
                    set(node_target.get("available_behavior_ids", []))):
                raise ValueError("U target must be finite and uncensored across feasible behavior support")
        elif u_status in ("unavailable", "censored"):
            if u is not None or (u_status == "censored" and not censored):
                raise ValueError("unavailable/censored U target has inconsistent value or support")
        else:
            raise ValueError("U target status must be explicitly available, unavailable, or censored")
        row["u_mask"] = u_status == "available"
        row["b_mask"] = True
        row["u_target"] = u if row["u_mask"] else None
        row["b_target"] = behavior
        row["target_semantics"] = PLAN_A_EXPERT_GOAL_SEMANTICS
    return rows, manifest, data_sha, hashlib.sha256(manifest_bytes).hexdigest()


def warm_start_plan_a_expert_goals_v1(track: str, dataset: Path,
        legal_node_dataset: Path, source_dataset: Path, annotation_source_path: Path,
        output: Path, *, qualification_protocol: Path, updates=10,
        learning_rate=1e-4, device=None):
    """Fit Plan A U/B from explicitly annotated replay-bound feasible goals.

    U is trained only when the annotation loader proves an available uncensored
    value. B is a preference among legal behaviors at that annotated goal.
    The selected goal is not treated as a globally optimal goal, and P stays at
    p0 because this artifact contains no route-cost oracle target.
    """
    if (track not in ("A1", "A2", "A3") or type(updates) is not int or updates < 1 or
            not _finite_number(learning_rate) or learning_rate <= 0):
        raise ValueError("expert-goal warm start requires A1/A2/A3 and positive finite fit settings")
    _prepare_native_gpu_runtime(device)
    from .meco_device import require_indexed_cuda_device
    from .curriculum_training import source_identity
    from .meco_plan_a_policy import gather_pose_behavior_logits
    device = str(require_indexed_cuda_device(device,
        purpose="Plan A expert-goal fitting"))
    sources = source_identity()
    annotations, manifest, data_sha, annotation_manifest_sha = (
        load_plan_a_expert_goal_annotations(dataset, legal_node_dataset, source_dataset,
            qualification_protocol=qualification_protocol, expected_sources=sources,
            annotation_source_path=annotation_source_path, device=device))
    # The legal-node records are replayed again here to obtain model inputs. The
    # strict annotation loader above independently replays the same bytes.
    base_manifest_path = Path(source_dataset).resolve().with_suffix(
        Path(source_dataset).resolve().suffix + ".manifest.json")
    source_manifest = json.loads(base_manifest_path.read_text())
    base_name = source_manifest.get("base_dataset")
    if not isinstance(base_name, str) or not base_name or Path(base_name).is_absolute():
        raise ValueError("expert-goal source has invalid base trajectory reference")
    base_dataset = (Path(source_dataset).resolve().parent / base_name).resolve()
    samples, legal_manifest, legal_sha, legal_manifest_sha = (
        _read_and_replay_plan_a_legal_node_dataset(legal_node_dataset, base_dataset,
            qualification_protocol, device, track))
    if legal_manifest.get("device", {}).get("device") != device:
        raise ValueError("expert-goal legal support and fitter devices differ")
    expert_source_device = source_manifest.get("device", {}).get("device")
    if expert_source_device != device:
        raise ValueError("expert-goal source and fitter devices differ")
    sample_by_key = {(s["row"]["scenario_id"], s["row"]["category"],
        s["row"]["seed"], s["row"]["decision_index"]): s for s in samples}
    annotated = []
    for row in annotations:
        key = (row["scenario_id"], row["category"], row["seed"], row["decision_index"])
        sample = sample_by_key.get(key)
        if sample is None or sample["row"].get("support_digest") != row.get("support_digest"):
            raise ValueError("expert annotation has no exact replayed legal support sample")
        goal_node = row["selected_goal"]["node_id"]
        node_position = goal_node if 0 <= goal_node < sample["graph"].node_count else None
        if node_position is None:
            raise ValueError("annotated goal node is absent from replayed graph")
        b_pairs = [(pair["behavior_id"], pair["pair_index"])
            for pair in sample["row"]["pairs"] if pair.get("node_id") == goal_node and
            pair.get("status") == "available"]
        if row["b_target"] not in {behavior for behavior, _ in b_pairs}:
            raise ValueError("annotated behavior is absent from exact legal goal support")
        annotated.append((row, sample, node_position, b_pairs))
    if not annotated:
        raise ValueError("expert-goal annotation dataset has no replayed rows")
    u_count = sum(row["u_mask"] for row, _, _, _ in annotated)
    b_count = sum(len(pairs) > 1 for _, _, _, pairs in annotated)
    if u_count < 1 and b_count < 1:
        raise ValueError("expert-goal annotations provide no trainable uncensored U or B comparisons")

    model = make_meco_track(track,
        candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    value_losses, preference_losses = [], []
    model.train()
    for _ in range(updates):
        epoch_u, epoch_b = [], []
        for row, sample, node_position, behavior_pairs in annotated:
            raw = model(raster=sample["raster"], globals_=sample["globals"],
                **sample["graph_inputs"])
            logits = gather_pose_behavior_logits(track, raw, sample["graph"],
                sample["raster"], sample["support"],
                field_length=sample["field_length"], field_width=sample["field_width"])
            terms = []
            if row["u_mask"]:
                target_u = torch.tensor(float(row["u_target"]), device=device,
                    dtype=raw["utility"].dtype)
                if track == "A1":
                    h, w = sample["raster"].shape[-2:]
                    pose_node = sample["graph"].xy[node_position]
                    col = (pose_node[0] / sample["field_length"] * w).floor().long().clamp(0, w-1)
                    rr = (pose_node[1] / sample["field_width"] * h).floor().long().clamp(0, h-1)
                    prediction_u = raw["utility"][0, sample["graph"].heading_bin[node_position], rr, col]
                else:
                    prediction_u = raw["utility"][0, node_position]
                loss_u = F.smooth_l1_loss(prediction_u, target_u, beta=1.)
                terms.append(loss_u)
                epoch_u.append(float(loss_u.detach().cpu()))
            if len(behavior_pairs) > 1:
                pair_indices = [index for _, index in behavior_pairs]
                behavior_ids = [behavior for behavior, _ in behavior_pairs]
                local_logits = logits[pair_indices]
                target_index = behavior_ids.index(row["b_target"])
                target = torch.tensor([target_index], dtype=torch.long, device=device)
                loss_b = F.cross_entropy(local_logits.reshape(1, -1), target)
                terms.append(loss_b)
                epoch_b.append(float(loss_b.detach().cpu()))
            if terms:
                loss = torch.stack(terms).sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
        if not epoch_u and not epoch_b:
            raise ValueError("expert-goal epoch contains no trainable target")
        value_losses.append(sum(epoch_u) / len(epoch_u) if epoch_u else 0.)
        preference_losses.append(sum(epoch_b) / len(epoch_b) if epoch_b else 0.)
    if source_identity() != sources:
        raise RuntimeError("source/runtime changed during Plan A expert-goal fitting")
    _initialize_unlabelled_plan_a_penalty_head(model, track)
    from .meco_counterfactual import _device_identity
    protocol, protocol_id, protocol_sha, protocol_path = _load_protocol(qualification_protocol)
    payload = {"schema": PLAN_A_EXPERT_GOAL_WEIGHTS_SCHEMA, "schema_version": 1,
        "artifact_kind": "weights_only_plan_a_explicit_expert_goal_u_b_v1",
        "track": track, "model_schema_version": 1, "model": model.state_dict(),
        "dataset_path": str(Path(dataset).resolve()), "dataset_sha256": data_sha,
        "dataset_manifest_sha256": annotation_manifest_sha,
        "annotation_source_manifest_sha256": hashlib.sha256(
            Path(dataset).resolve().with_suffix(Path(dataset).resolve().suffix + ".manifest.json").read_bytes()).hexdigest(),
        "legal_node_dataset_path": str(Path(legal_node_dataset).resolve()),
        "legal_node_dataset_sha256": legal_sha,
        "legal_node_manifest_sha256": legal_manifest_sha,
        "source_dataset_path": str(Path(source_dataset).resolve()),
        "source_dataset_sha256": hashlib.sha256(Path(source_dataset).read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(Path(source_dataset).resolve().with_suffix(
            Path(source_dataset).resolve().suffix + ".manifest.json").read_bytes()).hexdigest(),
        "base_dataset_sha256": legal_manifest["source_dataset_sha256"],
        "base_manifest_sha256": legal_manifest["source_manifest_sha256"],
        "annotation_source_path": str(Path(annotation_source_path).resolve()),
        "annotation_source_sha256": hashlib.sha256(Path(annotation_source_path).read_bytes()).hexdigest(),
        "dataset_schema": PLAN_A_EXPERT_GOAL_DATASET_SCHEMA,
        "target_schema": PLAN_A_EXPERT_GOAL_TARGET_SCHEMA,
        "label_provenance": "explicit replay-bound selected feasible goal and behavior annotation",
        "target_semantics": PLAN_A_EXPERT_GOAL_SEMANTICS,
        "u_target_semantics": "uncensored_explicit_expert_goal_value_only; absent_values_masked",
        "b_target_semantics": "selected_behavior_preference_within_annotated_legal_goal_support",
        "p_target": None, "p_mode": "p0", "p_prior": 0.0,
        "candidate_raw_feature_count": MECO_CANDIDATE_RAW_FEATURE_COUNT,
        "candidate_model_input_width": MECO_CANDIDATE_MODEL_INPUT_WIDTH,
        "candidate_feature_names": list(tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)),
        "prediction_mode": legal_manifest["prediction_mode"],
        "prediction_contract": legal_manifest["prediction_contract"],
        "protocol_id": protocol_id, "protocol_sha256": protocol_sha,
        "qualification_protocol_sha256": protocol_sha,
        "qualification_protocol_file": protocol_path.name,
        "split": "train", "sources": sources,
        "source_identity_sha256": _canonical_hash(sources),
        "collection_device": legal_manifest["device"],
        "training_device": _device_identity(device),
        "optimizer_state_included": False, "resume_compatible": False,
        "updates": updates, "learning_rate": learning_rate,
        "annotated_record_count": len(annotated), "available_u_target_count": u_count,
        "comparable_b_record_count": b_count,
        "mean_u_loss_by_epoch": value_losses,
        "mean_b_loss_by_epoch": preference_losses}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    validate_plan_a_expert_goal_artifact(track, output,
        qualification_protocol=qualification_protocol, expected_sources=sources)
    return {"track": track, "checkpoint": str(output),
        "schema": PLAN_A_EXPERT_GOAL_WEIGHTS_SCHEMA,
        "annotated_records": len(annotated), "available_u_targets": u_count,
        "comparable_b_records": b_count}


PLAN_A_HUMAN_PREFERENCES_V2_WEIGHTS_SCHEMA = "meco-plan-a-human-preferences-weights-v2"


def warm_start_plan_a_preferences_v2(track: str, dataset: Path,
        legal_node_dataset: Path, source_dataset: Path, authority_json: Path,
        output: Path, *, qualification_protocol: Path, updates=10,
        learning_rate=1e-4, device=None):
    """Fit U and B only from pairwise human preferences over replayed support."""
    if (track not in ("A1", "A2", "A3") or type(updates) is not int or updates < 1 or
            not _finite_number(learning_rate) or learning_rate <= 0):
        raise ValueError("Plan A preference v2 fitter requires A1/A2/A3 and positive finite settings")
    output = Path(output).resolve()
    inputs = [Path(dataset).resolve(), Path(legal_node_dataset).resolve(),
        Path(source_dataset).resolve(), Path(authority_json).resolve(),
        Path(qualification_protocol).resolve()]
    inputs += [path.with_suffix(path.suffix + ".manifest.json")
        for path in inputs[:3]]
    if output in inputs:
        raise ValueError("Plan A preference artifact output collides with a bound input")
    _prepare_native_gpu_runtime(device)
    from .meco_device import require_indexed_cuda_device
    from .curriculum_training import source_identity
    from .meco_plan_a_policy import gather_pose_behavior_logits
    from . import meco_plan_a_preferences_v2 as preferences
    device = str(require_indexed_cuda_device(device, purpose="Plan A human preference fitting"))
    sources = source_identity()
    labelled, manifest, data_sha, data_manifest_sha, legal_manifest = \
        preferences.load_plan_a_preferences_v2(dataset, legal_node_dataset, source_dataset,
            qualification_protocol=qualification_protocol, expected_sources=sources,
            authority_json=authority_json, device=device, track=track)
    if legal_manifest.get("device", {}).get("device") != device:
        raise ValueError("preference collection and fitter devices differ")
    from .meco_counterfactual import _device_identity
    model = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    goal_pair_count = sum(len(row["goal_preferences"]) for row, _ in labelled)
    behavior_pair_count = sum(len(row["behavior_preferences"]) for row, _ in labelled)
    action_pair_count = sum(len(row.get("action_pair_preferences", []))
        for row, _ in labelled)
    if goal_pair_count + action_pair_count < 1:
        raise ValueError("preference dataset has no pairwise action or U ranking targets")
    goal_losses, behavior_losses, action_losses = [], [], []
    model.train()
    for _ in range(updates):
        epoch_goal, epoch_behavior, epoch_action = [], [], []
        for row, sample in labelled:
            graph = sample["graph"]
            raw = model(raster=sample["raster"], globals_=sample["globals"],
                **sample["graph_inputs"])
            logits = gather_pose_behavior_logits(track, raw, graph, sample["raster"],
                sample["support"], field_length=sample["field_length"],
                field_width=sample["field_width"])
            node_u = raw["utility"]
            if track == "A1":
                h, w = sample["raster"].shape[-2:]
                col = (graph.xy[:, 0] / sample["field_length"] * w).floor().long().clamp(0, w-1)
                rr = (graph.xy[:, 1] / sample["field_width"] * h).floor().long().clamp(0, h-1)
                node_values = node_u[0, graph.heading_bin, rr, col]
            else:
                node_values = node_u[0]
            goal_terms = [F.softplus(-(node_values[p["preferred_node_id"]] -
                node_values[p["right_node_id"] if p["preferred_node_id"] == p["left_node_id"]
                    else p["left_node_id"]])) for p in row["goal_preferences"]]
            goal_loss = (torch.stack(goal_terms).mean() if goal_terms
                else logits.new_zeros(()))
            behavior_terms = []
            available = {}
            for pair in sample["row"]["pairs"]:
                if pair.get("status") == "available":
                    available[(pair["node_id"], pair["behavior_id"])] = pair["pair_index"]
            centered = {}
            for node in {pref["node_id"] for pref in row["behavior_preferences"]}:
                node_pairs = [(behavior, index) for (pair_node, behavior), index in available.items()
                    if pair_node == node]
                values = torch.stack([logits[index] for _, index in node_pairs])
                values = values - values.mean()
                centered.update({(node, behavior): values[index]
                    for index, (behavior, _) in enumerate(node_pairs)})
            for pref in row["behavior_preferences"]:
                left = centered[(pref["node_id"], pref["left_behavior_id"])]
                right = centered[(pref["node_id"], pref["right_behavior_id"])]
                if pref["preferred_behavior_id"] == pref["left_behavior_id"]:
                    preferred, other = left, right
                else:
                    preferred, other = right, left
                behavior_terms.append(F.softplus(-(preferred - other)))
            behavior_loss = (torch.stack(behavior_terms).mean() if behavior_terms
                else goal_loss.new_zeros(()))
            pair_index = {(pair["node_id"], pair["behavior_id"]): pair["pair_index"]
                for pair in sample["row"]["pairs"] if pair.get("status") == "available"}
            action_terms = []
            for pref in row.get("action_pair_preferences", []):
                left = logits[pair_index[(pref["left_node_id"], pref["left_behavior_id"])]]
                right = logits[pair_index[(pref["right_node_id"], pref["right_behavior_id"])]]
                preferred, other = ((left, right) if (pref["preferred_node_id"],
                    pref["preferred_behavior_id"]) == (pref["left_node_id"],
                    pref["left_behavior_id"]) else (right, left))
                action_terms.append(F.softplus(-(preferred - other)))
            action_loss = (torch.stack(action_terms).mean() if action_terms
                else goal_loss.new_zeros(()))
            loss = goal_loss + behavior_loss + action_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            epoch_goal.append(float(goal_loss.detach().cpu()))
            if behavior_terms:
                epoch_behavior.append(float(behavior_loss.detach().cpu()))
            if action_terms:
                epoch_action.append(float(action_loss.detach().cpu()))
        goal_losses.append(sum(epoch_goal) / len(epoch_goal))
        behavior_losses.append(sum(epoch_behavior) / len(epoch_behavior) if epoch_behavior else 0.)
        action_losses.append(sum(epoch_action) / len(epoch_action) if epoch_action else 0.)
    if source_identity() != sources:
        raise RuntimeError("source/runtime changed during Plan A preference fitting")
    _initialize_unlabelled_plan_a_penalty_head(model, track)
    protocol, protocol_id, protocol_sha, protocol_path = _load_protocol(qualification_protocol)
    dataset_path = Path(dataset).resolve()
    legal_path, source_path = Path(legal_node_dataset).resolve(), Path(source_dataset).resolve()
    authority_path = Path(authority_json).resolve()
    payload = {"schema": PLAN_A_HUMAN_PREFERENCES_V2_WEIGHTS_SCHEMA, "schema_version": 2,
        "artifact_kind": "weights_only_plan_a_pairwise_human_preferences_v2",
        "track": track, "model_schema_version": 1, "model": model.state_dict(),
        "dataset_path": str(dataset_path), "dataset_sha256": data_sha,
        "dataset_manifest_sha256": data_manifest_sha,
        "legal_node_dataset_path": str(legal_path),
        "legal_node_dataset_sha256": hashlib.sha256(legal_path.read_bytes()).hexdigest(),
        "legal_node_manifest_sha256": hashlib.sha256(legal_path.with_suffix(
            legal_path.suffix + ".manifest.json").read_bytes()).hexdigest(),
        "source_dataset_path": str(source_path),
        "source_dataset_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(source_path.with_suffix(
            source_path.suffix + ".manifest.json").read_bytes()).hexdigest(),
        "authority_path": str(authority_path),
        "authority_sha256": hashlib.sha256(authority_path.read_bytes()).hexdigest(),
        "dataset_schema": "meco-plan-a-human-preferences-v2",
        "target_schema": "meco-plan-a-pairwise-goal-and-behavior-preferences-v2",
        "target_semantics": "pairwise_human_goal_and_complete_action_preferences_for_U_plus_B",
        "goal_loss": "pairwise_softplus_U(preferred_goal, less_preferred_goal)",
        "behavior_loss": "within_goal_centered_pairwise_softplus_B(preferred_behavior, alternative)",
        "action_pair_loss": "pairwise_softplus_(U_plus_B)(preferred_complete_action, alternative)",
        "u_numeric_target": None,
        "u_numeric_unavailable_reason": "no independent complete uncensored oracle values supplied",
        "p_target": None, "p_mode": "p0", "p_prior": 0.0,
        "candidate_raw_feature_count": MECO_CANDIDATE_RAW_FEATURE_COUNT,
        "candidate_model_input_width": MECO_CANDIDATE_MODEL_INPUT_WIDTH,
        "candidate_feature_names": list(tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)),
        "prediction_mode": legal_manifest["prediction_mode"],
        "prediction_contract": legal_manifest["prediction_contract"],
        "protocol_id": protocol_id, "protocol_sha256": protocol_sha,
        "qualification_protocol_sha256": protocol_sha,
        "qualification_protocol_file": protocol_path.name,
        "split": "train", "sources": sources,
        "source_identity_sha256": _canonical_hash(sources),
        "collection_device": legal_manifest["device"],
        "training_device": _device_identity(device),
        "optimizer_state_included": False, "resume_compatible": False,
        "updates": updates, "learning_rate": learning_rate,
        "annotated_record_count": len(labelled), "goal_pair_count": goal_pair_count,
        "behavior_pair_count": behavior_pair_count,
        "mean_goal_preference_loss_by_epoch": goal_losses,
        "mean_behavior_preference_loss_by_epoch": behavior_losses,
        "mean_action_pair_preference_loss_by_epoch": action_losses,
        "action_pair_count": action_pair_count}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    validate_plan_a_preferences_v2_artifact(track, output,
        qualification_protocol=qualification_protocol, expected_sources=sources)
    return {"track": track, "checkpoint": str(output), "schema": payload["schema"],
        "annotated_records": len(labelled), "goal_pairs": goal_pair_count,
        "behavior_pairs": behavior_pair_count, "action_pairs": action_pair_count}


def validate_plan_a_preferences_v2_artifact(track: str, checkpoint: Path, *,
        qualification_protocol: Path, expected_sources: dict,
        expected_prediction_mode: str | None = None):
    """Validate preference weights, immutable dataset bytes, and p0 provenance."""
    import hashlib
    from .curriculum_training import source_identity
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    try:
        paths = {key: Path(payload[key]).resolve() for key in (
            "dataset_path", "legal_node_dataset_path", "source_dataset_path", "authority_path")}
        dataset_bytes, data_manifest_bytes = paths["dataset_path"].read_bytes(), paths["dataset_path"].with_suffix(
            paths["dataset_path"].suffix + ".manifest.json").read_bytes()
        legal_bytes, legal_manifest_bytes = paths["legal_node_dataset_path"].read_bytes(), paths[
            "legal_node_dataset_path"].with_suffix(paths["legal_node_dataset_path"].suffix + ".manifest.json").read_bytes()
        source_bytes, source_manifest_bytes = paths["source_dataset_path"].read_bytes(), paths[
            "source_dataset_path"].with_suffix(paths["source_dataset_path"].suffix + ".manifest.json").read_bytes()
        authority_bytes = paths["authority_path"].read_bytes()
        data_manifest, legal_manifest = json.loads(data_manifest_bytes), json.loads(legal_manifest_bytes)
        source_manifest = json.loads(source_manifest_bytes)
        annotation_source = data_manifest.get("annotation_source")
        protocol, protocol_id, protocol_sha, protocol_path = _load_protocol(qualification_protocol)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Plan A preference artifact provenance files are missing or malformed") from exc
    sources = source_identity()
    if (track not in ("A1", "A2", "A3") or payload.get("schema") != PLAN_A_HUMAN_PREFERENCES_V2_WEIGHTS_SCHEMA or
            payload.get("schema_version") != 2 or
            payload.get("artifact_kind") != "weights_only_plan_a_pairwise_human_preferences_v2" or
            payload.get("track") != track or payload.get("model_schema_version") != 1 or
            payload.get("sources") != expected_sources or expected_sources != sources or
            payload.get("source_identity_sha256") != _canonical_hash(sources) or
            payload.get("protocol_id") != protocol_id or payload.get("protocol_sha256") != protocol_sha or
            payload.get("qualification_protocol_sha256") != protocol_sha or
            payload.get("qualification_protocol_file") != protocol_path.name or
            payload.get("split") != "train" or payload.get("dataset_schema") != "meco-plan-a-human-preferences-v2" or
            payload.get("target_schema") != "meco-plan-a-pairwise-goal-and-behavior-preferences-v2" or
            payload.get("target_semantics") !=
                "pairwise_human_goal_and_complete_action_preferences_for_U_plus_B" or
            payload.get("action_pair_loss") !=
                "pairwise_softplus_(U_plus_B)(preferred_complete_action, alternative)" or
            payload.get("optimizer_state_included") is not False or payload.get("resume_compatible") is not False or
            payload.get("u_numeric_target") is not None or payload.get("p_target") is not None or
            payload.get("p_mode") != "p0" or
            (expected_prediction_mode is not None and payload.get("prediction_mode") != expected_prediction_mode)):
        raise ValueError("Plan A preference weights metadata contract mismatch")
    digests = {
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "dataset_manifest_sha256": hashlib.sha256(data_manifest_bytes).hexdigest(),
        "legal_node_dataset_sha256": hashlib.sha256(legal_bytes).hexdigest(),
        "legal_node_manifest_sha256": hashlib.sha256(legal_manifest_bytes).hexdigest(),
        "source_dataset_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
        "authority_sha256": hashlib.sha256(authority_bytes).hexdigest()}
    if any(payload.get(key) != value for key, value in digests.items()):
        raise ValueError("Plan A preference artifact dataset/source digest mismatch")
    if (data_manifest.get("schema") != "meco-plan-a-human-preferences-v2" or
            data_manifest.get("target_schema") != "meco-plan-a-pairwise-goal-and-behavior-preferences-v2" or
            data_manifest.get("dataset_sha256") != digests["dataset_sha256"] or
            data_manifest.get("source_dataset_sha256") != digests["source_dataset_sha256"] or
            data_manifest.get("source_manifest_sha256") != digests["source_manifest_sha256"] or
            data_manifest.get("legal_node_dataset_sha256") != digests["legal_node_dataset_sha256"] or
            data_manifest.get("legal_node_manifest_sha256") != digests["legal_node_manifest_sha256"] or
            data_manifest.get("protocol_id") != protocol_id or
            data_manifest.get("qualification_protocol_sha256") != protocol_sha or
            data_manifest.get("sources") != sources or data_manifest.get("split") != "train" or
            data_manifest.get("u_numeric_target") is not None or
            data_manifest.get("authority_declared_only") is not
                (data_manifest.get("signed_human_choice_authority") is None) or
            data_manifest.get("authority_is_authenticated_identity") is not
                (data_manifest.get("signed_human_choice_authority") is not None) or
            data_manifest.get("authority_proves_expertise_or_optimality") is not False or
            not isinstance(annotation_source, dict) or
            legal_manifest.get("dataset_sha256") != digests["legal_node_dataset_sha256"] or
            legal_manifest.get("source_dataset_sha256") != digests["source_dataset_sha256"] or
            legal_manifest.get("protocol_id") != protocol_id or
            legal_manifest.get("qualification_protocol_sha256") != protocol_sha or
            legal_manifest.get("split") != "train" or legal_manifest.get("sources") != sources or
            source_manifest.get("dataset_sha256") != digests["source_dataset_sha256"] or
            source_manifest.get("protocol_id") != protocol_id or
            source_manifest.get("qualification_protocol_sha256") != protocol_sha or
            source_manifest.get("split") != "train" or source_manifest.get("sources") != sources or
            annotation_source.get("sha256") != digests["authority_sha256"]):
        raise ValueError("Plan A preference artifact manifests are not identically bound")
    for key in ("updates", "annotated_record_count"):
        if type(payload.get(key)) is not int or payload[key] < 1:
            raise ValueError("Plan A preference artifact has invalid update or label counts")
    for key in ("goal_pair_count", "behavior_pair_count", "action_pair_count"):
        if type(payload.get(key)) is not int or payload[key] < 0:
            raise ValueError("Plan A preference artifact has invalid preference pair counts")
    if payload["goal_pair_count"] + payload["action_pair_count"] < 1:
        raise ValueError("Plan A preference artifact has no fitted goal/action labels")
    for key in ("mean_goal_preference_loss_by_epoch", "mean_behavior_preference_loss_by_epoch",
                "mean_action_pair_preference_loss_by_epoch"):
        values = payload.get(key)
        if (not isinstance(values, list) or len(values) != payload["updates"] or
                any(not _finite_number(value) or value < 0 for value in values)):
            raise ValueError("Plan A preference artifact has invalid loss history")
    signed_authority = data_manifest.get("signed_human_choice_authority")
    if signed_authority is not None:
        from . import meco_plan_a_expert_choice as choice_authority
        from .meco_plan_a_preferences_v2 import _protocol_trusted_keys
        trusted_keys = _protocol_trusted_keys(protocol)
        authority_payload = choice_authority.verify_authority(signed_authority,
            trusted_keys=trusted_keys)
        rows = [json.loads(line) for line in dataset_bytes.decode().splitlines() if line.strip()]
        choice_rows = [row.get("signed_human_choice") for row in rows]
        sidecar = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"),
            allow_nan=False) + "\n").encode() for row in choice_rows)
        expected_signed_hashes = {
            "sidecar_sha256": hashlib.sha256(sidecar).hexdigest(),
            "source_sha256": digests["source_dataset_sha256"],
            "source_manifest_sha256": digests["source_manifest_sha256"],
            "replay_sha256": digests["legal_node_dataset_sha256"],
            "replay_manifest_sha256": digests["legal_node_manifest_sha256"]}
        if (any(not isinstance(row, dict) for row in choice_rows) or
                data_manifest.get("signed_human_choice_sidecar_sha256") !=
                    expected_signed_hashes["sidecar_sha256"] or
                any(authority_payload.get(key) != value
                    for key, value in expected_signed_hashes.items()) or
                authority_payload.get("author_id") != annotation_source.get("source_id")):
            raise ValueError("signed human preference authority signature/provenance mismatch")
    model = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (KeyError, RuntimeError, TypeError) as exc:
        raise ValueError("Plan A preference weights do not match the requested model") from exc
    layer, mask = _plan_a_p_output_parameters(model, track)
    weights = layer.weight.detach() * mask.reshape(-1, *([1] * (layer.weight.ndim - 1)))
    biases = layer.bias.detach()[1::2] if track == "A1" else layer.bias.detach()[1:2]
    if (not bool(torch.isfinite(weights).all()) or not bool(torch.isfinite(biases).all()) or
            bool((weights != 0).any()) or bool((biases != -20.).any())):
        raise ValueError("Plan A preference artifact must leave the unlabelled P head at p0")
    return payload


def validate_plan_a_expert_goal_artifact(track: str, checkpoint: Path, *,
        qualification_protocol: Path, expected_sources: dict,
        expected_prediction_mode: str | None = None):
    """Validate replay-bound expert-goal lineage and weights-only model state."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    from .curriculum_training import source_identity
    import hashlib
    _, protocol_id, protocol_sha, protocol_path = _load_protocol(qualification_protocol)
    paths = {key: Path(payload.get(key, "")).resolve() for key in (
        "dataset_path", "legal_node_dataset_path", "source_dataset_path",
        "annotation_source_path")}
    try:
        data_bytes = paths["dataset_path"].read_bytes()
        data_manifest_bytes = paths["dataset_path"].with_suffix(
            paths["dataset_path"].suffix + ".manifest.json").read_bytes()
        legal_bytes = paths["legal_node_dataset_path"].read_bytes()
        legal_manifest_bytes = paths["legal_node_dataset_path"].with_suffix(
            paths["legal_node_dataset_path"].suffix + ".manifest.json").read_bytes()
        source_bytes = paths["source_dataset_path"].read_bytes()
        source_manifest_bytes = paths["source_dataset_path"].with_suffix(
            paths["source_dataset_path"].suffix + ".manifest.json").read_bytes()
        annotation_source_bytes = paths["annotation_source_path"].read_bytes()
        data_manifest = json.loads(data_manifest_bytes)
        legal_manifest = json.loads(legal_manifest_bytes)
        source_manifest = json.loads(source_manifest_bytes)
        if not isinstance(source_manifest, dict):
            raise ValueError("expert-goal source manifest is not an object")
        base_name = source_manifest.get("base_dataset")
        if not isinstance(base_name, str) or not base_name or Path(base_name).is_absolute():
            raise ValueError("expert-goal source manifest has invalid base dataset reference")
        base_path = (paths["source_dataset_path"].parent / base_name).resolve()
        base_bytes = base_path.read_bytes()
        base_manifest_bytes = base_path.with_suffix(base_path.suffix + ".manifest.json").read_bytes()
    except (OSError, ValueError) as exc:
        raise ValueError("expert-goal artifact's bound source files are unavailable or malformed") from exc
    sources = source_identity()
    mode = payload.get("prediction_mode")
    annotation_authority = data_manifest.get("annotation_source")
    collection_device = payload.get("collection_device")
    training_device = payload.get("training_device")
    losses_u, losses_b = payload.get("mean_u_loss_by_epoch"), payload.get("mean_b_loss_by_epoch")
    if (track not in ("A1", "A2", "A3") or
            payload.get("schema") != PLAN_A_EXPERT_GOAL_WEIGHTS_SCHEMA or
            payload.get("schema_version") != 1 or
            payload.get("artifact_kind") != "weights_only_plan_a_explicit_expert_goal_u_b_v1" or
            payload.get("track") != track or payload.get("model_schema_version") != 1 or
            payload.get("dataset_schema") != PLAN_A_EXPERT_GOAL_DATASET_SCHEMA or
            payload.get("target_schema") != PLAN_A_EXPERT_GOAL_TARGET_SCHEMA or
            payload.get("target_semantics") != PLAN_A_EXPERT_GOAL_SEMANTICS or
            payload.get("sources") != expected_sources or expected_sources != sources or
            payload.get("source_identity_sha256") != _canonical_hash(sources) or
            payload.get("protocol_id") != protocol_id or
            payload.get("protocol_sha256") != protocol_sha or
            payload.get("qualification_protocol_sha256") != protocol_sha or
            payload.get("qualification_protocol_file") != protocol_path.name or
            payload.get("split") != "train" or mode not in PREDICTION_MODES or
            payload.get("prediction_contract") != prediction_contract(mode) or
            (expected_prediction_mode is not None and mode != expected_prediction_mode) or
            data_manifest.get("schema") != PLAN_A_EXPERT_GOAL_DATASET_SCHEMA or
            data_manifest.get("target_schema") != PLAN_A_EXPERT_GOAL_TARGET_SCHEMA or
            data_manifest.get("split") != "train" or
            data_manifest.get("sources") != sources or
            not isinstance(annotation_authority, dict) or
            data_manifest.get("protocol_id") != protocol_id or
            data_manifest.get("qualification_protocol_sha256") != protocol_sha or
            hashlib.sha256(data_bytes).hexdigest() != payload.get("dataset_sha256") or
            data_manifest.get("dataset_sha256") != payload.get("dataset_sha256") or
            hashlib.sha256(data_manifest_bytes).hexdigest() != payload.get("dataset_manifest_sha256") or
            hashlib.sha256(data_manifest_bytes).hexdigest() != payload.get("annotation_source_manifest_sha256") or
            hashlib.sha256(legal_bytes).hexdigest() != payload.get("legal_node_dataset_sha256") or
            hashlib.sha256(legal_manifest_bytes).hexdigest() != payload.get("legal_node_manifest_sha256") or
            hashlib.sha256(source_bytes).hexdigest() != payload.get("source_dataset_sha256") or
            hashlib.sha256(source_manifest_bytes).hexdigest() != payload.get("source_manifest_sha256") or
            hashlib.sha256(annotation_source_bytes).hexdigest() != payload.get("annotation_source_sha256") or
            annotation_authority.get("sha256") != hashlib.sha256(annotation_source_bytes).hexdigest() or
            data_manifest.get("expert_goal_source_sha256") != hashlib.sha256(source_bytes).hexdigest() or
            source_manifest.get("schema") != "meco-plan-a-expert-goal-source-v1" or
            source_manifest.get("target_schema") != "meco-plan-a-explicit-selected-goal-trace-v1" or
            source_manifest.get("dataset_sha256") != hashlib.sha256(source_bytes).hexdigest() or
            source_manifest.get("protocol_id") != protocol_id or
            source_manifest.get("qualification_protocol_sha256") != protocol_sha or
            source_manifest.get("sources") != sources or
            source_manifest.get("split") != "train" or
            source_manifest.get("device") != payload.get("collection_device") or
            source_manifest.get("base_dataset_sha256") != hashlib.sha256(base_bytes).hexdigest() or
            source_manifest.get("base_manifest_sha256") != hashlib.sha256(base_manifest_bytes).hexdigest() or
            data_manifest.get("source_dataset_sha256") != payload.get("base_dataset_sha256") or
            data_manifest.get("source_manifest_sha256") != payload.get("base_manifest_sha256") or
            payload.get("base_dataset_sha256") != legal_manifest.get("source_dataset_sha256") or
            payload.get("base_manifest_sha256") != legal_manifest.get("source_manifest_sha256") or
            legal_manifest.get("schema") != "meco-counterfactual-plan-a-legal-node-v1" or
            legal_manifest.get("protocol_id") != protocol_id or
            legal_manifest.get("qualification_protocol_sha256") != protocol_sha or
            legal_manifest.get("sources") != sources or
            payload.get("prediction_mode") != legal_manifest.get("prediction_mode") or
            payload.get("collection_device") != legal_manifest.get("device") or
            not isinstance(collection_device, dict) or not isinstance(training_device, dict) or
            collection_device.get("device") != training_device.get("device") or
            payload.get("p_target", "missing") is not None or
            payload.get("p_mode") != "p0" or payload.get("p_prior") != 0.0 or
            payload.get("optimizer_state_included") is not False or
            payload.get("resume_compatible") is not False or
            type(payload.get("updates")) is not int or payload["updates"] < 1 or
            not _finite_number(payload.get("learning_rate")) or payload["learning_rate"] <= 0 or
            not isinstance(losses_u, list) or len(losses_u) != payload["updates"] or
            not isinstance(losses_b, list) or len(losses_b) != payload["updates"] or
            any(not _finite_number(x) or x < 0 for x in losses_u + losses_b) or
            type(payload.get("annotated_record_count")) is not int or payload["annotated_record_count"] < 1 or
            type(payload.get("available_u_target_count")) is not int or payload["available_u_target_count"] < 0 or
            type(payload.get("comparable_b_record_count")) is not int or payload["comparable_b_record_count"] < 0 or
            payload.get("candidate_raw_feature_count") != MECO_CANDIDATE_RAW_FEATURE_COUNT or
            payload.get("candidate_model_input_width") != MECO_CANDIDATE_MODEL_INPUT_WIDTH or
            payload.get("candidate_feature_names") != list(tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)) or
            not isinstance(payload.get("model"), dict)):
        raise ValueError("Plan A expert-goal weights/source/protocol/model contract mismatch")
    model = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("expert-goal model weights do not match the Plan A model schema") from exc
    expected_p = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT)
    _initialize_unlabelled_plan_a_penalty_head(expected_p, track)
    actual_layer, mask = _plan_a_p_output_parameters(model, track)
    expected_layer, expected_mask = _plan_a_p_output_parameters(expected_p, track)
    if not torch.equal(mask, expected_mask):
        raise ValueError("Plan A P head mask changed unexpectedly")
    output_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    actual, expected = actual_layer.weight[output_indices], expected_layer.weight[output_indices]
    actual_bias, expected_bias = actual_layer.bias[output_indices], expected_layer.bias[output_indices]
    if not torch.equal(actual, expected) or not torch.equal(actual_bias, expected_bias):
        raise ValueError("expert-goal artifact modified unlabelled Plan A P head from p0")
    return payload


def _read_counterfactual_dataset(path: Path, *, qualification_protocol: Path,
                                 expected_sources: dict):
    """Validate v2 bytes and frozen identities before any replay is attempted."""
    from .meco_counterfactual import (DATASET_SCHEMA as cf_dataset_schema,
        TARGET_SCHEMA as cf_target_schema, CONTINUATION_POLICY,
        PLAN_C_REPEAT_TARGET_SCHEMA, plan_c_repeat_protocol,
        MIN_ELIGIBLE_BRANCHES_PER_SEED,
        _artifact_identity_sha256, _device_identity,
        _require_native_extension_manifest_identity,
        _verify_native_fuel_extension)

    path = Path(path).resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    protocol, protocol_id, protocol_sha, protocol_path = _load_protocol(qualification_protocol)
    device_metadata = manifest.get("device")
    if not isinstance(device_metadata, dict) or not isinstance(device_metadata.get("device"), str):
        raise ValueError("counterfactual dataset has no frozen device identity")
    if (device_metadata["device"].startswith("cuda") and
            not isinstance(manifest.get("native_extension_identity"), dict)):
        raise ValueError("legacy counterfactual v2 dataset lacks native HIP identity; recollect it")
    expected_device = _device_identity(device_metadata["device"])
    if type(manifest.get("record_count")) is not int or manifest["record_count"] < 1:
        raise ValueError("counterfactual manifest has an invalid record count")
    if (manifest.get("schema") != cf_dataset_schema or
            manifest.get("target_schema") != cf_target_schema or
            manifest.get("records_have_prefix_replay_parity") is not True or
            manifest.get("split") != "train" or
            manifest.get("protocol_id") != protocol_id or
            manifest.get("qualification_protocol_sha256") != protocol_sha or
            manifest.get("qualification_protocol_file") != protocol_path.name or
            Path(manifest.get("qualification_protocol_file", "")).name !=
                manifest.get("qualification_protocol_file") or
            manifest.get("sources") != expected_sources or
            manifest.get("source_identity_sha256") != _canonical_hash(expected_sources) or
            manifest.get("device") != expected_device or
            manifest.get("state_digest_schema") != "meco-full-environment-state-v2" or
            manifest.get("prediction_mode") not in PREDICTION_MODES or
            manifest.get("prediction_contract") != prediction_contract(manifest.get("prediction_mode")) or
            manifest.get("candidate_boundary") != CANDIDATE_BOUNDARY or
            manifest.get("plan_a_support_order") != SUPPORT_ORDER or
            manifest.get("route_cost_points_per_second") != ROUTE_COST_POINTS_PER_SECOND or
            manifest.get("discount_semantics") != COUNTERFACTUAL_DISCOUNT_SEMANTICS or
            manifest.get("continuation_policy") != CONTINUATION_POLICY or
            manifest.get("plan_a_p_target", "missing") is not None or
            manifest.get("plan_a_p_prior") != 0.0 or
            manifest.get("source_runtime_device_bound") is not True):
        raise ValueError("counterfactual v2 manifest identity or semantics mismatch")
    if (manifest.get("minimum_eligible_branches_per_seed") != MIN_ELIGIBLE_BRANCHES_PER_SEED or
            manifest.get("eligibility_coverage_policy") !=
                "prefix through final eligible branch point; later teacher-only suffix excluded"):
        raise ValueError("counterfactual manifest lacks the required branch eligibility policy")
    for key, minimum in (("horizon_physics_ticks", 1), ("max_macro_ticks", 1),
                         ("candidate_k", 1),
                         ("max_plan_c_descriptors_per_prefix", 2),
                         ("max_plan_a_pairs_per_prefix", 2)):
        if type(manifest.get(key)) is not int or manifest[key] < minimum:
            raise ValueError(f"counterfactual manifest has invalid {key}")
    if manifest["candidate_k"] not in (8, 16, 32):
        raise ValueError("counterfactual candidate K is outside the supported boundary")
    gamma = manifest.get("gamma_per_macro")
    if not _finite_number(gamma) or not 0 < gamma <= 1:
        raise ValueError("counterfactual manifest has invalid macro discount")
    if (manifest.get("plan_c_target_semantics") !=
            "gross simulator reward; exact archived candidate; same deterministic teacher continuation" or
            manifest.get("plan_a_target_semantics") !=
            "gross simulator reward per sampled valid node_behavior_descriptor; no P route-cost label" or
            manifest.get("plan_a_p_unlabelled_reason") !=
            "simulator executor does not execute P-modified routes"):
        raise ValueError("counterfactual target semantics differ from v2 pairwise contract")
    plan_c_repeated = manifest.get("plan_c_target_schema") == PLAN_C_REPEAT_TARGET_SCHEMA
    if plan_c_repeated:
        if manifest.get("plan_c_repeat_protocol") != plan_c_repeat_protocol(protocol_id):
            raise ValueError("Plan C repeat protocol is not bound to the supplied qualification protocol")
    elif "plan_c_repeat_protocol" in manifest or "plan_c_target_schema" in manifest:
        raise ValueError("counterfactual manifest has a partial or unsupported Plan C repeat protocol")
    native_extension_identity = manifest.get("native_extension_identity")
    if manifest.get("native_extension_identity_schema") != 1:
        raise ValueError("legacy counterfactual v2 dataset lacks native identity schema v1; recollect it")
    if (device_metadata["device"].startswith("cuda") and
            (not isinstance(native_extension_identity, dict) or
             native_extension_identity.get("status") != "present")):
        raise ValueError("GPU counterfactual artifact has no native HIP binary identity")
    if manifest.get("artifact_identity_sha256") != _artifact_identity_sha256(
            manifest.get("source_identity_sha256"), native_extension_identity):
        raise ValueError("counterfactual artifact identity digest mismatch")
    try:
        _require_native_extension_manifest_identity(manifest)
        _verify_native_fuel_extension(native_extension_identity)
    except RuntimeError as exc:
        raise ValueError(f"counterfactual native HIP artifact identity mismatch: {exc}") from exc
    data_bytes = path.read_bytes()
    data_sha = hashlib.sha256(data_bytes).hexdigest()
    if data_sha != manifest.get("dataset_sha256"):
        raise ValueError("counterfactual v2 dataset byte digest mismatch")
    rows = [json.loads(line) for line in data_bytes.decode("utf-8").splitlines()
            if line.strip()]
    if len(rows) != manifest["record_count"] or not rows:
        raise ValueError("counterfactual v2 dataset row count mismatch or empty data")
    scenario_id, category = manifest.get("scenario_id"), manifest.get("category")
    allowed_seeds = _protocol_train_seeds(protocol, scenario_id, category)
    seeds = manifest.get("seeds")
    if (not isinstance(seeds, list) or not seeds or
            any(type(seed) is not int for seed in seeds) or
            len(set(seeds)) != len(seeds) or not set(seeds) <= allowed_seeds):
        raise ValueError("counterfactual dataset seeds are outside the frozen train split")
    scenario_definition = get_scenario(scenario_id)
    if category not in scenario_definition.required_categories:
        raise ValueError("counterfactual manifest category is not registered for its scenario")
    keys, by_seed = set(), {}
    seed_summaries = manifest.get("seed_branch_summaries")
    if not isinstance(seed_summaries, dict) or set(seed_summaries) != {str(s) for s in seeds}:
        raise ValueError("counterfactual manifest has incomplete per-seed eligibility summaries")
    for seed_key, summary in seed_summaries.items():
        if not isinstance(summary, dict):
            raise ValueError("counterfactual seed eligibility summary is malformed")
        summary_body = {k: v for k, v in summary.items() if k != "summary_sha256"}
        eligible_indices = summary.get("eligible_decision_indices")
        reason_counts = summary.get("skipped_reason_counts")
        if (summary.get("minimum_eligible_branches_per_seed") != MIN_ELIGIBLE_BRANCHES_PER_SEED or
                not isinstance(eligible_indices, list) or
                any(type(i) is not int or i < 0 for i in eligible_indices) or
                eligible_indices != sorted(set(eligible_indices)) or
                len(eligible_indices) < MIN_ELIGIBLE_BRANCHES_PER_SEED or
                type(summary.get("episode_decision_count")) is not int or
                summary.get("episode_decision_count") < len(eligible_indices) or
                summary.get("coverage_through_decision_index") != eligible_indices[-1] or
                summary.get("covered_prefix_count") != summary.get("episode_decision_count") or
                any(i >= summary.get("episode_decision_count", 0) for i in eligible_indices) or
                summary.get("skipped_count") != summary.get("episode_decision_count", 0) - len(eligible_indices) or
                summary.get("eligibility_digest") is None or
                not isinstance(reason_counts, dict) or
                any(key not in ("fewer_than_two_plan_c_descriptors",
                                "fewer_than_two_plan_a_descriptors") or
                    type(value) is not int or value < 0 or value > summary.get("skipped_count", -1)
                    for key, value in reason_counts.items()) or
                summary.get("summary_sha256") != _canonical_hash(summary_body)):
            raise ValueError(f"counterfactual seed {seed_key} eligibility summary is invalid")
    for row in rows:
        if (not isinstance(row, dict) or row.get("schema") != cf_dataset_schema or
                row.get("target_schema") != cf_target_schema or
                row.get("scenario_id") != scenario_id or row.get("category") != category or
                row.get("protocol_id") != protocol_id or
                row.get("qualification_protocol_sha256") != protocol_sha or
                row.get("source_identity_sha256") != manifest["source_identity_sha256"] or
                row.get("device") != expected_device):
            raise ValueError("mixed schema, source, protocol, task, or device in v2 data")
        if plan_c_repeated and (row.get("plan_c_target_schema") != PLAN_C_REPEAT_TARGET_SCHEMA or
                row.get("plan_c_repeat_protocol_id") !=
                    manifest["plan_c_repeat_protocol"]["repeat_protocol_id"]):
            raise ValueError("row Plan C repeat identity differs from its manifest")
        seed, index = row.get("seed"), row.get("decision_index")
        if type(seed) is not int or seed not in seeds or seed not in allowed_seeds or type(index) is not int or index < 0:
            raise ValueError("counterfactual row is outside the frozen train assignment")
        key = (scenario_id, category, seed, index)
        if key in keys:
            raise ValueError("duplicate counterfactual branch identity")
        keys.add(key)
        by_seed.setdefault(seed, []).append(row)
        summary = seed_summaries[str(seed)]
        if (row.get("seed_branch_summary") != summary or
                index not in summary["eligible_decision_indices"] or
                not isinstance(row.get("branch_eligibility"), dict) or
                row["branch_eligibility"].get("eligible") is not True or
                row["branch_eligibility"].get("skip_reasons") != [] or
                row["branch_eligibility"].get("metadata_sha256") != _canonical_hash({
                    k: v for k, v in row["branch_eligibility"].items()
                    if k != "metadata_sha256"}) or
                row["branch_eligibility"].get("plan_c_descriptor_count") !=
                    len(row.get("plan_c_counterfactuals", [])) or
                row["branch_eligibility"].get("plan_a_descriptor_count") !=
                    len(row.get("plan_a_sampled_node_behavior_labels", []))):
            raise ValueError("counterfactual row eligibility metadata differs from manifest")
        if (row.get("gamma_per_macro") != gamma or
                row.get("discount_semantics") != COUNTERFACTUAL_DISCOUNT_SEMANTICS or
                row.get("max_macro_ticks") != manifest["max_macro_ticks"] or
                row.get("horizon_physics_ticks") != manifest["horizon_physics_ticks"] or
                row.get("teacher_prefix_provenance") != CONTINUATION_POLICY):
            raise ValueError("counterfactual row discount/horizon/provenance differs from manifest")
        transcript = row.get("prefix_transcript")
        if (not isinstance(transcript, list) or len(transcript) != index or
                row.get("prefix_transcript_sha256") != _canonical_hash(transcript) or
                row.get("prefix_sha256") != _canonical_hash(transcript) or
                type(row.get("macro_start_tick")) is not int or row["macro_start_tick"] < 0):
            raise ValueError("counterfactual transcript structure or digest is invalid")
        if (not isinstance(row.get("branch_state"), dict) or
                not isinstance(row.get("plan_c_counterfactuals"), list) or
                not isinstance(row.get("plan_a_sampled_node_behavior_labels"), list) or
                not isinstance(row.get("plan_c_pairwise_labels"), list) or
                not isinstance(row.get("plan_a_pairwise_preferences"), list)):
            raise ValueError("counterfactual branch lacks outcomes or pairwise labels")
        for kind, outcomes, pairs, id_key in (
                ("C", row["plan_c_counterfactuals"], row["plan_c_pairwise_labels"], "candidate_id"),
                ("A", row["plan_a_sampled_node_behavior_labels"], row["plan_a_pairwise_preferences"], "descriptor_sha256")):
            if len(outcomes) < 2:
                raise ValueError(f"Plan {kind} branch outcomes are empty or duplicated")
            outcome_ids = []
            for outcome in outcomes:
                descriptor = outcome.get("descriptor") if isinstance(outcome, dict) else None
                if (not isinstance(descriptor, dict) or
                        not isinstance(descriptor.get("descriptor_sha256"), str) or
                        not isinstance(descriptor.get(id_key), str)):
                    raise ValueError(f"Plan {kind} descriptor is malformed")
                outcome_ids.append(descriptor[id_key])
                if (not _finite_number(outcome.get("gross_macro_return")) or
                        not _finite_number(outcome.get("gross_horizon_return")) or
                        not _finite_number(outcome.get("discounted_horizon_return")) or
                        outcome.get("discount_semantics") != COUNTERFACTUAL_DISCOUNT_SEMANTICS or
                        outcome.get("continuation_policy") != CONTINUATION_POLICY):
                    raise ValueError(f"Plan {kind} outcome has invalid gross/discounted returns")
                if kind == "A" and (outcome.get("p_target", "missing") is not None or
                        outcome.get("p_prior") != 0.0 or
                        outcome.get("p_unlabelled_reason") !=
                            manifest.get("plan_a_p_unlabelled_reason")):
                    raise ValueError("Plan A P must remain explicitly unlabelled with zero prior")
            if len(set(outcome_ids)) != len(outcome_ids):
                raise ValueError(f"Plan {kind} branch outcomes are duplicated")
            if kind == "C" and plan_c_repeated:
                _validate_plan_c_repeated_labels(row,
                    repeat_protocol=manifest["plan_c_repeat_protocol"],
                    protocol_id=protocol_id, scenario_id=scenario_id,
                    category=category, seed=seed, decision_index=index)
                continue
            expected_count = len(outcomes) * (len(outcomes) - 1) // 2
            if len(pairs) != expected_count:
                raise ValueError(f"Plan {kind} pair labels do not cover all within-prefix pairs")
            pair_seen = set()
            for pair in pairs:
                if not isinstance(pair, dict):
                    raise ValueError(f"Plan {kind} pair label is malformed")
                left_key = ("left_id" if kind == "C" else "left_descriptor_sha256")
                right_key = ("right_id" if kind == "C" else "right_descriptor_sha256")
                preferred_key = ("preferred_id" if kind == "C" else "preferred_descriptor_sha256")
                left, right = pair.get(left_key), pair.get(right_key)
                if not isinstance(left, str) or not isinstance(right, str):
                    raise ValueError(f"Plan {kind} pair label IDs must be strings")
                pair_key = frozenset((left, right))
                values = {outcome["descriptor"][id_key]: outcome["gross_horizon_return"]
                          for outcome in outcomes}
                if (left not in values or right not in values or left == right or
                        pair_key in pair_seen or
                        not _finite_number(pair.get("gross_horizon_return_difference"))):
                    raise ValueError(f"Plan {kind} pair label references invalid outcomes")
                pair_seen.add(pair_key)
                difference = values[left] - values[right]
                preferred = left if difference > 0 else right if difference < 0 else None
                if (pair.get("gross_horizon_return_difference") != difference or
                        pair.get(preferred_key) != preferred):
                    raise ValueError(f"Plan {kind} pair label is not derived from gross horizon returns")
    if any(sorted(row.get("decision_index") for row in seed_rows) !=
           seed_summaries[str(seed)].get("eligible_decision_indices")
           for seed, seed_rows in by_seed.items()):
        raise ValueError("counterfactual branch indices differ from declared eligible points")
    for seed, seed_rows in by_seed.items():
        summary = seed_summaries[str(seed)]
        terminal_rows = [row for row in seed_rows if row.get("decision_index") ==
                         summary["coverage_through_decision_index"]]
        episode = terminal_rows[0].get("episode_transcript") if len(terminal_rows) == 1 else None
        eligibility = ([item.get("branch_eligibility") for item in episode]
                       if isinstance(episode, list) and all(isinstance(item, dict) for item in episode)
                       else None)
        if (eligibility is None or len(episode) != summary["episode_decision_count"] or
                any(not isinstance(item, dict) or
                    not isinstance(item.get("skip_reasons"), list) or
                    any(reason not in ("fewer_than_two_plan_c_descriptors",
                                       "fewer_than_two_plan_a_descriptors")
                        for reason in item.get("skip_reasons", [])) or
                    item.get("eligible") is not (not item.get("skip_reasons")) or
                    item.get("metadata_sha256") != _canonical_hash({
                        k: v for k, v in item.items() if k != "metadata_sha256"})
                    for item in eligibility) or
                [i for i, item in enumerate(eligibility) if item["eligible"]] !=
                    summary["eligible_decision_indices"] or
                _canonical_hash(eligibility) != summary["eligibility_digest"] or
                terminal_rows[0].get("episode_transcript_sha256") != _canonical_hash(episode)):
            raise ValueError("counterfactual final eligible row lacks full episode eligibility evidence")
    if set(by_seed) != set(seeds):
        raise ValueError("counterfactual dataset omitted one or more declared train seeds")
    return rows, data_sha, manifest_sha, protocol, protocol_id, protocol_sha, manifest


def _read_plan_a_p_route_dataset(path: Path, *, qualification_protocol: Path,
                                 expected_sources: dict):
    """Read the separate executed-route P schema without weakening v2 checks."""
    from .meco_counterfactual import (PLAN_A_P_ROUTE_DATASET_SCHEMA,
        PLAN_A_P_ROUTE_TARGET_SCHEMA, PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA,
        PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA, PLAN_A_P_ROUTE_ORACLE_VERSION,
        CONTINUATION_POLICY,
        _artifact_identity_sha256, _device_identity,
        _require_native_extension_manifest_identity,
        _verify_native_fuel_extension)
    path = Path(path).resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    protocol, protocol_id, protocol_sha, protocol_path = _load_protocol(qualification_protocol)
    device = manifest.get("device")
    if not isinstance(device, dict) or not isinstance(device.get("device"), str):
        raise ValueError("Plan A P dataset has no frozen device identity")
    if (device["device"].startswith("cuda") and
            (not isinstance(manifest.get("native_extension_identity"), dict) or
             manifest["native_extension_identity"].get("status") != "present")):
        raise ValueError("legacy Plan A P dataset lacks native HIP identity; recollect it")
    if manifest.get("native_extension_identity_schema") != 1:
        raise ValueError("legacy Plan A P dataset lacks native identity schema v1; recollect it")
    oracle_v3 = manifest.get("schema") == PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA
    expected_schema = (PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA if oracle_v3
                       else PLAN_A_P_ROUTE_DATASET_SCHEMA)
    expected_target = (PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA if oracle_v3
                       else PLAN_A_P_ROUTE_TARGET_SCHEMA)
    if (manifest.get("schema") != expected_schema or
            manifest.get("target_schema") != expected_target or
            manifest.get("source_counterfactual_schema") != "meco-counterfactual-trajectory-v3" or
            manifest.get("p_intervention_generator") != "seeded_smooth_positive_xy_maps_v1" or
            (not oracle_v3 and manifest.get("plan_a_p_oracle_target") !=
                "exact_simulator_gross_horizon_return_for_executed_native_adstar_route") or
            (not oracle_v3 and manifest.get("plan_a_p_supervision_semantics") !=
                "within_prefix_route_pairwise_preference; no per-cell oracle target") or
            (not oracle_v3 and manifest.get("plan_a_p_target") !=
                "within-prefix executed-route preference from simulator gross horizon return") or
            (oracle_v3 and (manifest.get("p_oracle_version") != PLAN_A_P_ROUTE_ORACLE_VERSION or
                manifest.get("plan_a_p_target") != manifest.get("plan_a_p_oracle_target") or
                manifest.get("plan_a_p_supervision_semantics") !=
                    "non_tied sampled executed-path cost preference; lower independent oracle cost is preferred")) or
            manifest.get("plan_a_p_prior") != 0.0 or
            manifest.get("plan_a_p_unlabelled_reason") is not None or
            manifest.get("records_have_exact_weighted_route_replay") is not True or
            manifest.get("source_runtime_device_bound") is not True or
            manifest.get("split") != "train" or
            manifest.get("protocol_id") != protocol_id or
            manifest.get("qualification_protocol_sha256") != protocol_sha or
            manifest.get("qualification_protocol_file") != protocol_path.name or
            manifest.get("sources") != expected_sources or
            manifest.get("source_identity_sha256") != _canonical_hash(expected_sources) or
            manifest.get("device") != _device_identity(device["device"]) or
            manifest.get("continuation_policy") != CONTINUATION_POLICY or
            type(manifest.get("p_intervention_maps_per_descriptor")) is not int or
            not 1 <= manifest["p_intervention_maps_per_descriptor"] <= 8 or
            not _finite_number(manifest.get("p_intervention_scale")) or
            manifest["p_intervention_scale"] <= 0 or
            not isinstance(manifest.get("source_counterfactual_dataset_sha256"), str) or
            len(manifest["source_counterfactual_dataset_sha256"]) != 64 or
            not isinstance(manifest.get("source_counterfactual_manifest_sha256"), str) or
            len(manifest["source_counterfactual_manifest_sha256"]) != 64):
        raise ValueError("Plan A P route dataset manifest identity or semantics mismatch")
    native_identity = manifest.get("native_extension_identity")
    if manifest.get("artifact_identity_sha256") != _artifact_identity_sha256(
            manifest.get("source_identity_sha256"), native_identity):
        raise ValueError("Plan A P route artifact identity digest mismatch")
    try:
        _require_native_extension_manifest_identity(manifest)
        _verify_native_fuel_extension(native_identity)
    except RuntimeError as exc:
        raise ValueError(f"Plan A P route native HIP identity mismatch: {exc}") from exc
    payload = path.read_bytes()
    data_sha = hashlib.sha256(payload).hexdigest()
    if data_sha != manifest.get("dataset_sha256"):
        raise ValueError("Plan A P route dataset digest mismatch")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if not rows or len(rows) != manifest.get("record_count"):
        raise ValueError("Plan A P route dataset row count mismatch or empty data")
    seeds = manifest.get("seeds")
    scenario, category = manifest.get("scenario_id"), manifest.get("category")
    allowed = _protocol_train_seeds(protocol, scenario, category)
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or
            any(type(seed) is not int or seed not in allowed for seed in seeds)):
        raise ValueError("Plan A P route dataset seeds are outside the frozen train split")
    unique, by_seed = set(), {}
    route_count = 0
    for row in rows:
        if (row.get("schema") != expected_schema or
                row.get("target_schema") != expected_target or
                row.get("scenario_id") != scenario or row.get("category") != category or
                row.get("protocol_id") != protocol_id or
                row.get("qualification_protocol_sha256") != protocol_sha or
                row.get("source_identity_sha256") != manifest["source_identity_sha256"] or
                row.get("device") != manifest["device"]):
            raise ValueError("mixed schema, source, protocol, task, or device in Plan A P data")
        if (row.get("gamma_per_macro") != manifest.get("gamma_per_macro") or
                row.get("discount_semantics") != manifest.get("discount_semantics") or
                row.get("max_macro_ticks") != manifest.get("max_macro_ticks") or
                row.get("horizon_physics_ticks") != manifest.get("horizon_physics_ticks")):
            raise ValueError("Plan A P row discount/horizon differs from its manifest")
        seed, decision = row.get("seed"), row.get("decision_index")
        if type(seed) is not int or seed not in seeds or type(decision) is not int or decision < 0:
            raise ValueError("Plan A P route row is outside the frozen train assignment")
        key = (scenario, category, seed, decision)
        if key in unique:
            raise ValueError("duplicate Plan A P route branch identity")
        unique.add(key)
        by_seed.setdefault(seed, []).append(decision)
        transcript = row.get("prefix_transcript")
        if (not isinstance(transcript, list) or len(transcript) != decision or
                row.get("prefix_transcript_sha256") != _canonical_hash(transcript) or
                row.get("prefix_sha256") != _canonical_hash(transcript) or
                not isinstance(row.get("branch_state"), dict)):
            raise ValueError("Plan A P route prefix or state provenance is invalid")
        groups = row.get("plan_a_p_route_groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError("Plan A P route row has no route groups")
        for group_index, group in enumerate(groups):
            descriptor, outcomes = group.get("descriptor"), group.get("outcomes")
            if (not isinstance(descriptor, dict) or
                    not isinstance(descriptor.get("descriptor_sha256"), str) or
                    not isinstance(outcomes, list) or len(outcomes) < 2):
                raise ValueError("malformed Plan A P route group")
            expected_maps = manifest["p_intervention_maps_per_descriptor"]
            if (len(outcomes) != expected_maps + 1 or
                    len({item.get("intervention_id") for item in outcomes}) != len(outcomes)):
                raise ValueError("duplicate Plan A P intervention identity")
            if outcomes[0].get("intervention_id") != "P0" or outcomes[0].get("intervention_seed") is not None:
                raise ValueError("Plan A P route group must begin with its P0 control")
            paths = set()
            seed_base = int.from_bytes(hashlib.sha256(
                f"{seed}:{decision}:{group_index}:p-route-v1".encode()).digest()[:8],
                "little") & ((1 << 63) - 1)
            for outcome_index, outcome in enumerate(outcomes):
                route = outcome.get("native_route")
                expected_id = "P0" if outcome_index == 0 else f"probe-{outcome_index}"
                expected_seed = (None if outcome_index == 0 else
                    (seed_base + outcome_index) & ((1 << 63) - 1))
                if (not _finite_number(outcome.get("gross_horizon_return")) or
                        outcome.get("intervention_id") != expected_id or
                        outcome.get("intervention_seed") != expected_seed or
                        outcome.get("continuation_policy") != CONTINUATION_POLICY or
                        outcome.get("discount_semantics") != COUNTERFACTUAL_DISCOUNT_SEMANTICS or
                        not isinstance(route, dict) or
                        not isinstance(route.get("route_points_m"), list) or
                        not route["route_points_m"] or
                        route.get("route_sha256") != _canonical_hash(route["route_points_m"]) or
                        not isinstance(route.get("route_trace"), list) or
                        not route["route_trace"] or
                        route.get("route_trace_sha256") != _canonical_hash(
                            [item.get("route_sha256") for item in route["route_trace"]]) or
                        type(route.get("map_applied")) is not bool):
                    raise ValueError("invalid Plan A P executed-route outcome")
                if route["map_applied"] != (outcome.get("intervention_id") != "P0"):
                    raise ValueError("Plan A P intervention was not applied as declared")
                if (route.get("selected_action") != descriptor.get("action") or
                        route.get("selected_goal_xy_m") != descriptor.get("goal_xy_m") or
                        route.get("selected_goal_descriptor_sha256") !=
                            descriptor.get("descriptor_sha256") or
                        route.get("selected_goal_refresh_forced") is not True):
                    raise ValueError("Plan A P branch selected-goal provenance mismatch")
                if route.get("map_scale") != (None if expected_seed is None else
                                                manifest["p_intervention_scale"]):
                    raise ValueError("Plan A P route scale differs from its manifest")
                if oracle_v3:
                    oracle = route.get("independent_cost_oracle")
                    base_grid = oracle.get("base_bump_cost_grid") if isinstance(oracle, dict) else None
                    effective_grid = oracle.get("effective_cost_grid") if isinstance(oracle, dict) else None
                    snapshot_costs = oracle.get("route_snapshot_costs_grid_edges") if isinstance(oracle, dict) else None
                    valid_base = (isinstance(base_grid, list) and bool(base_grid) and
                        all(isinstance(column, list) and len(column) == len(base_grid[0])
                            and all(_finite_number(value) and value >= 0 for value in column)
                            for column in base_grid))
                    valid_effective = (isinstance(effective_grid, list) and valid_base and
                        len(effective_grid) == len(base_grid) and
                        all(isinstance(column, list) and len(column) == len(base_grid[0])
                            and all(_finite_number(value) and value >= 0 for value in column)
                            for column in effective_grid))
                    if (not isinstance(oracle, dict) or
                            oracle.get("version") != PLAN_A_P_ROUTE_ORACLE_VERSION or
                            oracle.get("route_trace_sha256") != route["route_trace_sha256"] or
                            oracle.get("selected_goal_descriptor_sha256") !=
                                descriptor["descriptor_sha256"] or
                            oracle.get("intervention_map_sha256") != route.get("map_sha256") or
                            not _finite_number(oracle.get("route_cost_grid_edges")) or
                            not _finite_number(oracle.get("resolution_m")) or oracle["resolution_m"] <= 0 or
                            not valid_base or not valid_effective or
                            oracle.get("grid_shape") != [len(base_grid), len(base_grid[0]) if valid_base else 0] or
                            not isinstance(snapshot_costs, list) or
                            len(snapshot_costs) != len(route["route_trace"]) or
                            any(not _finite_number(value) or value < 0 for value in snapshot_costs) or
                            oracle["route_cost_grid_edges"] != sum(snapshot_costs) / len(snapshot_costs) or
                            oracle.get("intervention_scale") !=
                                (manifest["p_intervention_scale"] if expected_seed is not None else 0.) or
                            oracle.get("grid_inputs_sha256") != _canonical_hash({
                                "traversability": oracle.get("base_traversability_cost_grid"),
                                "base": oracle.get("base_bump_cost_grid"),
                                "effective": oracle.get("effective_cost_grid"),
                                "resolution_m": oracle.get("resolution_m")})):
                        raise ValueError("invalid independent Plan A P route-cost oracle provenance")
                previous_tick = -1
                for snapshot in route["route_trace"]:
                    points = snapshot.get("route_points_m")
                    if (type(snapshot.get("tick_index")) is not int or
                            not previous_tick < snapshot["tick_index"] <
                                manifest["max_macro_ticks"] or
                            not isinstance(points, list) or not points or
                            snapshot.get("route_sha256") != _canonical_hash(points) or
                            snapshot.get("p_map_applied") != route["map_applied"]):
                        raise ValueError("invalid P native route trace snapshot")
                    previous_tick = snapshot["tick_index"]
                if route["route_trace"][-1]["route_points_m"] != route["route_points_m"]:
                    raise ValueError("P native route trace does not end at its final route")
                paths.add(route["route_trace_sha256"])
            route_count += 1
            # Identical route variants carry no independent route-shaping signal.
            if len(paths) < 2:
                continue
    if set(row.get("seed") for row in rows) != set(seeds):
        raise ValueError("Plan A P route dataset omitted a declared train seed")
    if any(sorted(indices) != list(range(len(indices)))
           for indices in by_seed.values()):
        raise ValueError("Plan A P decision indices are not contiguous per train seed")
    if route_count != manifest.get("route_group_count"):
        raise ValueError("Plan A P route-group count mismatch")
    return rows, data_sha, manifest_sha, protocol, protocol_id, protocol_sha, manifest


MULTICASE_COUNTERFACTUAL_DATASET_SCHEMA = "meco-counterfactual-multicase-composite-v1"
MULTICASE_COUNTERFACTUAL_WEIGHTS_SCHEMA = "meco-counterfactual-multicase-composite-weights-v1"


def _read_counterfactual_composite_dataset(shards, *, expected_cases,
        qualification_protocol: Path, expected_sources: dict):
    """Validate each legacy one-case shard, then pool rows without scalar case metadata."""
    if not isinstance(shards, (list, tuple)) or not shards or len(shards) != len(expected_cases):
        raise ValueError("composite counterfactual input requires one shard per declared case")
    rows, bindings, coverage, reference, manifests = [], [], {}, None, []
    seen_cases, identities = set(), set()
    common_fields = ("schema", "target_schema", "records_have_prefix_replay_parity",
        "source_runtime_device_bound", "state_digest_schema", "sources",
        "source_identity_sha256", "device", "native_extension_identity_schema",
        "native_extension_identity", "artifact_identity_sha256", "prediction_mode",
        "prediction_contract", "candidate_boundary", "candidate_k",
        "plan_a_support_order", "route_cost_points_per_second",
        "discount_semantics", "continuation_policy", "plan_c_target_schema",
        "plan_c_repeat_protocol", "plan_a_p_target", "plan_a_p_prior",
        "plan_c_repeat_variance_note", "max_macro_ticks", "horizon_physics_ticks",
        "gamma_per_macro", "max_plan_c_descriptors_per_prefix",
        "max_plan_a_pairs_per_prefix")
    protocol_id = protocol_sha = None
    for case, entry in zip(expected_cases, shards):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), (str, Path)):
            raise ValueError("composite counterfactual shard entry is malformed")
        key = (case.get("scenario_id"), case.get("category"))
        if key in seen_cases:
            raise ValueError("composite counterfactual case list contains a duplicate")
        seen_cases.add(key)
        path = Path(entry["path"]).resolve(strict=True)
        local_rows, data_sha, manifest_sha, protocol, local_protocol_id, local_protocol_sha, manifest = (
            _read_counterfactual_dataset(path, qualification_protocol=qualification_protocol,
                expected_sources=expected_sources))
        if ((manifest.get("scenario_id"), manifest.get("category")) != key or
                manifest.get("seeds") != case.get("seeds")):
            raise ValueError("counterfactual shard case/seeds differ from the frozen composite case")
        if protocol_id is not None and (local_protocol_id != protocol_id or local_protocol_sha != protocol_sha):
            raise ValueError("counterfactual shards use different frozen protocols")
        protocol_id, protocol_sha = local_protocol_id, local_protocol_sha
        if reference is None:
            reference = manifest
        elif any(manifest.get(field) != reference.get(field) for field in common_fields):
            raise ValueError("counterfactual shards have incompatible target or model contracts")
        ids = set()
        for row in local_rows:
            identity = (key[0], key[1], row["seed"], row["decision_index"])
            if identity in identities or identity in ids:
                raise ValueError("composite counterfactual shards duplicate a case-seed-decision identity")
            ids.add(identity)
        identities.update(ids)
        observed_seeds = sorted({row["seed"] for row in local_rows})
        if observed_seeds != sorted(case["seeds"]):
            raise ValueError("counterfactual shard is missing one or more declared seeds")
        coverage[f"{key[0]}::{key[1]}"] = {"expected_seeds": sorted(case["seeds"]),
            "observed_seeds": observed_seeds, "row_count": len(local_rows),
            "seed_branch_summaries": {str(seed): manifest["seed_branch_summaries"][str(seed)]
                for seed in sorted(case["seeds"])}}
        rows.extend(local_rows)
        manifests.append(manifest)
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        bindings.append({"case_key": [key[0], key[1]], "seeds": list(case["seeds"]),
            "dataset_path": str(path), "dataset_sha256": data_sha,
            "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
            "record_count": len(local_rows)})
    case_order = {(case["scenario_id"], case["category"]): index
        for index, case in enumerate(expected_cases)}
    rows.sort(key=lambda row: (case_order[(row["scenario_id"], row["category"])],
        row["seed"], row["decision_index"]))
    case_set_sha256 = _canonical_hash([[case["scenario_id"], case["category"], case["seeds"]]
        for case in expected_cases])
    aggregate = {"schema": MULTICASE_COUNTERFACTUAL_DATASET_SCHEMA,
        "case_set_sha256": case_set_sha256, "cases": bindings,
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "sources": expected_sources, "source_identity_sha256": _canonical_hash(expected_sources),
        "prediction_mode": reference["prediction_mode"],
        "prediction_contract": reference["prediction_contract"],
        "target_schema": reference["target_schema"],
        "case_coverage": coverage,
        "dataset_sha256": _canonical_hash([item["dataset_sha256"] for item in bindings]),
        "manifest_sha256": _canonical_hash([item["manifest_sha256"] for item in bindings])}
    return rows, aggregate, reference, manifests


def _materialize_counterfactual_rows(rows, manifest, *, track, train_device):
    from .meco_counterfactual import replay_counterfactual_branch, _hash
    from .meco_track_training import _graph_inputs

    device = torch.device(train_device)
    result = []
    for row in rows:
        replay = replay_counterfactual_branch(row, manifest,
            require_hip_plan_a=True)
        try:
            env, graph, snapshot = replay["env"], replay["graph"], replay["snapshot"]
            safe = replay["observation"]
            if row.get("branch_teacher_action_reference") != snapshot["teacher_action"]:
                raise ValueError("branch teacher reference differs from replay")
            raster = torch.tensor(safe.raster, dtype=torch.float32, device=device).reshape(
                1, *safe.raster_shape)
            globals_ = torch.tensor(safe.globals, dtype=torch.float32, device=device).reshape(1, 7)
            field_length = float(env.sim.field_length)
            field_width = float(env.sim.field_width)
            graph_values = _graph_inputs(track, graph, env, raster)
            graph_values = {} if graph_values is None else graph_values
            graph_values = {name: value.to(device) for name, value in graph_values.items()}
            if track in ("A1", "A2", "A3"):
                support = replay["support"]
                expected = snapshot["plan_a_descriptors"]
                outcomes = row["plan_a_sampled_node_behavior_labels"]
                regenerated = {item["descriptor_sha256"]: item for item in expected}
                if len(outcomes) != len(expected):
                    raise ValueError("Plan A outcome count differs from replayed valid support sample")
                pairs = []
                for outcome in outcomes:
                    descriptor = outcome["descriptor"]
                    exact = regenerated.get(descriptor["descriptor_sha256"])
                    if exact != descriptor:
                        raise ValueError("Plan A descriptor differs from exact replay support")
                    if _hash({key: value for key, value in descriptor.items()
                              if key != "descriptor_sha256"}) != descriptor["descriptor_sha256"]:
                        raise ValueError("Plan A descriptor digest mismatch")
                    pair_index = descriptor.get("pair_index")
                    if type(pair_index) is not int or not 0 <= pair_index < support.pair_count:
                        raise ValueError("Plan A descriptor pair index is out of support bounds")
                    pair_values = (int(support.pair_node_ids[pair_index].item()),
                        int(support.behavior_ids[pair_index].item()),
                        int(support.descriptor_ids[pair_index].item()),
                        int(support.action_ids[pair_index].item()))
                    if pair_values != (descriptor.get("node_id"), descriptor.get("behavior_id"),
                                       descriptor.get("descriptor_id"), descriptor.get("action")):
                        raise ValueError("Plan A descriptor does not map to its exact support pair")
                    pairs.append((pair_index, outcome["gross_horizon_return"], descriptor["descriptor_sha256"]))
                result.append({"kind": "A", "raster": raster, "globals": globals_,
                    "graph": graph_values, "support": support, "graph_object": graph,
                    "field_length": field_length, "field_width": field_width,
                    "pairs": pairs})
            else:
                if manifest.get("plan_c_target_schema") != "meco-plan-c-matched-repeated-gross-return-v1":
                    raise ValueError("Plan C fitting requires the repeated matched-rollout label schema")
                candidate_set = replay["candidate_set"]
                table = candidate_set.table
                outcomes = row["plan_c_counterfactuals"]
                regenerated = {item["descriptor_sha256"]: item
                               for item in snapshot["candidate_descriptors"]}
                if len(outcomes) != len(regenerated):
                    raise ValueError("Plan C outcome count differs from replayed valid candidate sample")
                route = replay["route_cost"].to(device)
                risk = replay["risk_cost"].to(device)
                pairs = []
                candidate_index_by_id = {}
                for outcome in outcomes:
                    descriptor = outcome["descriptor"]
                    exact = regenerated.get(descriptor["descriptor_sha256"])
                    if exact != descriptor:
                        raise ValueError("Plan C descriptor differs from exact replay candidate")
                    candidate_index = descriptor.get("candidate_index")
                    if (type(candidate_index) is not int or
                            not 0 <= candidate_index < len(table.candidate_ids) or
                            table.candidate_ids[candidate_index] != descriptor.get("candidate_id") or
                            not table.valid_mask[candidate_index] or
                            int(candidate_set.action_indices[candidate_index]) != descriptor.get("action")):
                        raise ValueError("Plan C descriptor does not map to its exact valid candidate")
                    pairs.append((candidate_index, outcome["gross_horizon_return"],
                                  descriptor["candidate_id"]))
                    candidate_index_by_id[descriptor["candidate_id"]] = candidate_index
                stable_pairs = _plan_c_stable_pair_indices(
                    outcomes, row["plan_c_pairwise_labels"])
                result.append({"kind": "C", "raster": raster, "globals": globals_,
                    "table": table, "route": route, "risk": risk,
                    "pairs": pairs, "stable_pairs": stable_pairs})
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    return result


def warm_start_counterfactual_v2(track: str, dataset: Path, output: Path, *,
        qualification_protocol: Path, updates: int = 10,
        learning_rate: float = 1e-4, device: str | None = None):
    """Fit only within-prefix gross-return pairwise preferences from v2 data."""
    if track not in MECO_TRACKS or type(updates) is not int or updates < 1:
        raise ValueError("invalid track or update count")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be positive and finite")
    from .curriculum_training import source_identity
    from .meco_counterfactual import _device_identity
    from .meco_plan_a_policy import gather_pose_behavior_logits

    _prepare_native_gpu_runtime(device)
    sources = source_identity()
    rows, data_hash, manifest_hash, _, protocol_id, protocol_sha, manifest = _read_counterfactual_dataset(
        dataset, qualification_protocol=qualification_protocol,
        expected_sources=sources)
    if _device_identity(device) != manifest["device"]:
        raise ValueError("counterfactual replay/training device must match the frozen dataset runtime")
    if (track.startswith("A") and track not in ("A1", "A2", "A3") or
            track.startswith("C") and track not in ("C1", "C2")):
        raise ValueError("v2 counterfactual fitting supports only Plan A1-A3 or Plan C1-C2")
    if track.startswith("C") and manifest.get("plan_c_target_schema") != \
            "meco-plan-c-matched-repeated-gross-return-v1":
        raise ValueError("C1/C2 fitting rejects legacy single-rollout labels")
    rows = [row for row in rows if row["scenario_id"] == manifest["scenario_id"]]
    replayed = _materialize_counterfactual_rows(rows, manifest, track=track, train_device=device)
    tie_count = 0
    pair_count = 0
    for sample in replayed:
        values = sample["pairs"]
        if sample["kind"] == "C":
            total = len(values) * (len(values) - 1) // 2
            pair_count += len(sample["stable_pairs"])
            tie_count += total - len(sample["stable_pairs"])
            continue
        for index, (left_index, left_return, _) in enumerate(values):
            for right_index, right_return, _ in values[index + 1:]:
                if left_return == right_return:
                    tie_count += 1
                else:
                    pair_count += 1
    if pair_count < 1:
        raise ValueError("v2 dataset has no non-tied feasible within-prefix return pair")
    expected_kind = "A" if track.startswith("A") else "C"
    if any(sample["kind"] != expected_kind for sample in replayed):
        raise ValueError("counterfactual data kind does not match requested model track")

    expected_names = tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)
    model = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    model.train()
    for _ in range(updates):
        epoch_losses = []
        for sample in replayed:
            if sample["kind"] == "A":
                raw = model(raster=sample["raster"], globals_=sample["globals"],
                            **sample["graph"])
                logits = gather_pose_behavior_logits(track, raw, sample["graph_object"],
                    sample["raster"], sample["support"],
                    field_length=sample["field_length"],
                    field_width=sample["field_width"])
                scores = [(logits[index], ret) for index, ret, _ in sample["pairs"]]
            else:
                result = score_candidates(track, model, raster=sample["raster"],
                    globals_=sample["globals"], table=sample["table"],
                    route_cost=sample["route"], risk_cost=sample["risk"])
                scores = [(result["utility"][index], ret)
                          for index, ret, _ in sample["pairs"]]
            pair_losses = []
            if sample["kind"] == "C":
                for left_index, right_index, left_preferred in sample["stable_pairs"]:
                    sign = 1. if left_preferred else -1.
                    pair_losses.append(F.softplus(-sign *
                        (result["utility"][left_index] - result["utility"][right_index])))
            else:
                for index, (left_logit, left_return) in enumerate(scores):
                    for right_logit, right_return in scores[index + 1:]:
                        if left_return == right_return:
                            continue
                        sign = 1. if left_return > right_return else -1.
                        pair_losses.append(F.softplus(-sign * (left_logit - right_logit)))
            if not pair_losses:
                continue
            loss = torch.stack(pair_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if not epoch_losses:
            raise ValueError("no replayed branch produced a trainable pairwise loss")
        losses.append(sum(epoch_losses) / len(epoch_losses))
    repeated_plan_c = track.startswith("C")
    payload = {"schema": (COUNTERFACTUAL_PLAN_C_REPEATED_WEIGHTS_SCHEMA if repeated_plan_c
                           else COUNTERFACTUAL_WEIGHTS_SCHEMA),
        "schema_version": 1 if repeated_plan_c else 2,
        "artifact_kind": ("weights_only_plan_c_repeated_pairwise_v1" if repeated_plan_c
                          else "weights_only_counterfactual_pairwise_v2"), "track": track,
        "model_schema_version": 1, "model": model.state_dict(),
        "dataset_sha256": data_hash, "dataset_schema": manifest["schema"],
        "dataset_manifest_sha256": manifest_hash,
        "records_have_prefix_replay_parity": manifest["records_have_prefix_replay_parity"],
        "source_runtime_device_bound": manifest["source_runtime_device_bound"],
        "state_digest_schema": manifest["state_digest_schema"],
        "target_schema": manifest["target_schema"],
        "label_provenance": ("within-prefix matched repeated simulator gross-horizon returns; stable paired sign only"
            if repeated_plan_c else COUNTERFACTUAL_LABEL_PROVENANCE),
        "target_semantics": ("within_prefix_repeated_matched_stable_pairwise_preference_from_gross_horizon_return"
            if repeated_plan_c else "within_prefix_pairwise_preference_from_gross_horizon_return"),
        "scoring_semantics": ("raw_U_node_plus_B_behavior; no P gradient; p_mode=p0"
            if track.startswith("A") else
            "raw_candidate_utility; route_and_risk_costs_excluded_for_gross_return_target"),
        "plan_a_p_target": None, "plan_a_p_prior": 0.0,
        "plan_a_p_mode": ("p0" if track.startswith("A") else None),
        "candidate_raw_feature_count": MECO_CANDIDATE_RAW_FEATURE_COUNT,
        "candidate_model_input_width": MECO_CANDIDATE_MODEL_INPUT_WIDTH,
        "candidate_feature_names": list(expected_names),
        "prediction_mode": manifest["prediction_mode"],
        "prediction_contract": manifest["prediction_contract"],
        "candidate_boundary": manifest["candidate_boundary"],
        "candidate_k": manifest["candidate_k"],
        "max_plan_c_descriptors_per_prefix": manifest["max_plan_c_descriptors_per_prefix"],
        "max_plan_a_pairs_per_prefix": manifest["max_plan_a_pairs_per_prefix"],
        "route_cost_points_per_second": manifest["route_cost_points_per_second"],
        "plan_a_support_order": manifest["plan_a_support_order"],
        "plan_a_p_unlabelled_reason": manifest["plan_a_p_unlabelled_reason"],
        "horizon_physics_ticks": manifest["horizon_physics_ticks"],
        "max_macro_ticks": manifest["max_macro_ticks"],
        "gamma_per_macro": manifest["gamma_per_macro"],
        "discount_semantics": manifest["discount_semantics"],
        "continuation_policy": manifest["continuation_policy"],
        "plan_c_target_schema": (manifest.get("plan_c_target_schema") if repeated_plan_c else None),
        "plan_c_repeat_protocol": (manifest.get("plan_c_repeat_protocol") if repeated_plan_c else None),
        "plan_c_label_policy": ("stable_same_strict_sign_all_paired_repeats_else_abstain"
            if repeated_plan_c else None),
        "plan_c_repeat_variance_note": ("R=2 is a bounded replication policy, not precise variance calibration"
            if repeated_plan_c else None),
        "plan_c_stable_pair_count": pair_count if repeated_plan_c else None,
        "plan_c_abstained_pair_count": tie_count if repeated_plan_c else None,
        "pair_count_semantics": ("non_tied_pair_count counts stable Plan C labels; tied_pair_count counts all abstentions"
            if repeated_plan_c else None),
        "source_identity_sha256": _canonical_hash(sources), "sources": sources,
        "collection_device": manifest["device"], "training_device": _device_identity(device),
        "protocol_id": protocol_id,
        "qualification_protocol_sha256": protocol_sha,
        "qualification_protocol_file": Path(qualification_protocol).name,
        "split": "train", "scenario_id": manifest["scenario_id"],
        "category": manifest["category"], "seeds": list(manifest["seeds"]),
        "optimizer_state_included": False, "resume_compatible": False,
        "updates": updates, "learning_rate": learning_rate,
        "non_tied_pair_count": pair_count, "tied_pair_count": tie_count,
        "mean_loss_by_epoch": losses}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {"track": track, "checkpoint": str(output), "dataset_sha256": data_hash,
            "non_tied_pair_count": pair_count, "tied_pair_count": tie_count,
            "stable_pair_count": pair_count if repeated_plan_c else None,
            "abstained_pair_count": tie_count if repeated_plan_c else None,
            "mean_loss_by_epoch": losses}




def warm_start_counterfactual_composite_v1(track: str, shards, output: Path, *,
        expected_cases, qualification_protocol: Path, updates=10,
        learning_rate=1e-4, device=None):
    """Fit Plan C repeated labels over independently validated case shards."""
    if track not in ("C1", "C2") or type(updates) is not int or updates < 1 or not _finite_number(learning_rate) or learning_rate <= 0:
        raise ValueError("composite repeated-label fitter requires C1/C2 and valid optimizer settings")
    from .curriculum_training import source_identity
    from .meco_counterfactual import _device_identity, plan_c_repeat_protocol
    from .meco_track_training import score_candidates
    from .meco_tracks import prediction_contract
    _prepare_native_gpu_runtime(device)
    sources = source_identity()
    rows, composite, reference, manifests = _read_counterfactual_composite_dataset(
        shards, expected_cases=expected_cases, qualification_protocol=qualification_protocol,
        expected_sources=sources)
    if reference.get("plan_c_target_schema") != "meco-plan-c-matched-repeated-gross-return-v1":
        raise ValueError("composite C fitter requires repeated matched-rollout labels in every shard")
    if any(manifest["device"] != reference["device"] for manifest in manifests):
        raise ValueError("composite C shards were collected on different devices")
    if _device_identity(device) != reference["device"]:
        raise ValueError("composite C training device differs from frozen collection device")
    replayed = []
    for case, manifest in zip(expected_cases, manifests):
        local = [row for row in rows if
            (row["scenario_id"], row["category"]) == (case["scenario_id"], case["category"])]
        replayed.extend(_materialize_counterfactual_rows(local, manifest,
            track=track, train_device=device))
    if not replayed or any(sample["kind"] != "C" for sample in replayed):
        raise ValueError("composite C fit requires replayed candidate branches for every case")
    pair_count = tie_count = 0
    for sample in replayed:
        total = len(sample["pairs"]) * (len(sample["pairs"]) - 1) // 2
        pair_count += len(sample["stable_pairs"])
        tie_count += total - len(sample["stable_pairs"])
    if pair_count < 1:
        raise ValueError("composite C dataset has no stable non-tied paired labels")
    model = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    model.train()
    for _ in range(updates):
        epoch_losses = []
        for sample in replayed:
            result = score_candidates(track, model, raster=sample["raster"],
                globals_=sample["globals"], table=sample["table"],
                route_cost=sample["route"], risk_cost=sample["risk"])
            pair_losses = []
            for left_index, right_index, left_preferred in sample["stable_pairs"]:
                sign = 1. if left_preferred else -1.
                pair_losses.append(F.softplus(-sign *
                    (result["utility"][left_index] - result["utility"][right_index])))
            if not pair_losses:
                continue
            loss = torch.stack(pair_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if not epoch_losses:
            raise ValueError("composite C epoch has no stable within-prefix pair")
        losses.append(sum(epoch_losses) / len(epoch_losses))
    if source_identity() != sources:
        raise RuntimeError("source/runtime changed during composite C fitting")
    protocol, protocol_id, protocol_sha, protocol_path = _load_protocol(qualification_protocol)
    payload = {"schema": MULTICASE_COUNTERFACTUAL_WEIGHTS_SCHEMA, "schema_version": 1,
        "artifact_kind": "weights_only_multicase_plan_c_repeated_pairwise_v1",
        "track": track, "model_schema_version": 1, "model": model.state_dict(),
        "dataset_schema": MULTICASE_COUNTERFACTUAL_DATASET_SCHEMA,
        "dataset_sha256": composite["dataset_sha256"],
        "dataset_manifest_sha256": composite["manifest_sha256"],
        "case_set_sha256": composite["case_set_sha256"], "cases": composite["cases"],
        "case_coverage": composite["case_coverage"],
        "records_have_prefix_replay_parity": True, "source_runtime_device_bound": True,
        "state_digest_schema": "meco-full-environment-state-v2",
        "target_schema": "meco-matched-prefix-gross-return-v3",
        "label_provenance": "within-prefix matched repeated simulator gross-horizon returns; stable paired sign only",
        "target_semantics": "within_prefix_repeated_matched_stable_pairwise_preference_from_gross_horizon_return",
        "scoring_semantics": "raw_candidate_utility; route_and_risk_costs_excluded_for_gross_return_target",
        "plan_a_p_target": None, "plan_a_p_prior": 0.0, "plan_a_p_mode": None,
        "candidate_raw_feature_count": MECO_CANDIDATE_RAW_FEATURE_COUNT,
        "candidate_model_input_width": MECO_CANDIDATE_MODEL_INPUT_WIDTH,
        "candidate_feature_names": list(tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)),
        "prediction_mode": reference["prediction_mode"],
        "prediction_contract": prediction_contract(reference["prediction_mode"]),
        "candidate_boundary": reference["candidate_boundary"], "candidate_k": reference["candidate_k"],
        "max_plan_c_descriptors_per_prefix": reference["max_plan_c_descriptors_per_prefix"],
        "max_plan_a_pairs_per_prefix": reference["max_plan_a_pairs_per_prefix"],
        "route_cost_points_per_second": reference["route_cost_points_per_second"],
        "plan_a_support_order": reference["plan_a_support_order"],
        "plan_a_p_unlabelled_reason": reference["plan_a_p_unlabelled_reason"],
        "horizon_physics_ticks": reference["horizon_physics_ticks"],
        "max_macro_ticks": reference["max_macro_ticks"], "gamma_per_macro": reference["gamma_per_macro"],
        "discount_semantics": reference["discount_semantics"],
        "continuation_policy": reference["continuation_policy"],
        "plan_c_target_schema": reference["plan_c_target_schema"],
        "plan_c_repeat_protocol": reference["plan_c_repeat_protocol"],
        "plan_c_label_policy": "stable_same_strict_sign_all_paired_repeats_else_abstain",
        "plan_c_repeat_variance_note": reference["plan_c_repeat_variance_note"],
        "plan_c_stable_pair_count": pair_count, "plan_c_abstained_pair_count": tie_count,
        "pair_count_semantics": "non_tied_pair_count counts stable Plan C labels; tied_pair_count counts all abstentions",
        "source_identity_sha256": _canonical_hash(sources), "sources": sources,
        "collection_device": reference["device"], "training_device": _device_identity(device),
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "qualification_protocol_file": protocol_path.name, "split": "train",
        "optimizer_state_included": False, "resume_compatible": False,
        "updates": updates, "learning_rate": learning_rate,
        "non_tied_pair_count": pair_count, "tied_pair_count": tie_count,
        "mean_loss_by_epoch": losses}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    validate_counterfactual_composite_artifact(track, output,
        qualification_protocol=qualification_protocol, expected_sources=sources,
        expected_prediction_mode=reference["prediction_mode"])
    return {"track": track, "checkpoint": str(output), "case_set_sha256": composite["case_set_sha256"],
        "non_tied_pair_count": pair_count, "tied_pair_count": tie_count,
        "mean_loss_by_epoch": losses}


def validate_counterfactual_composite_artifact(track: str, checkpoint: Path, *,
        qualification_protocol: Path, expected_sources: dict,
        expected_prediction_mode: str | None = None):
    """Validate a composite C artifact by reopening every original strict shard."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (track not in ("C1", "C2") or payload.get("schema") != MULTICASE_COUNTERFACTUAL_WEIGHTS_SCHEMA or
            payload.get("schema_version") != 1 or payload.get("artifact_kind") !=
            "weights_only_multicase_plan_c_repeated_pairwise_v1" or payload.get("track") != track):
        raise ValueError("unsupported composite counterfactual artifact")
    cases = [{"scenario_id": item["case_key"][0], "category": item["case_key"][1],
        "seeds": item["seeds"]} for item in payload.get("cases", [])]
    shard_inputs = [{"path": item["dataset_path"]} for item in payload.get("cases", [])]
    rows, composite, reference, _ = _read_counterfactual_composite_dataset(
        shard_inputs, expected_cases=cases, qualification_protocol=qualification_protocol,
        expected_sources=expected_sources)
    protocol, protocol_id, protocol_sha, protocol_path = _load_protocol(qualification_protocol)
    stable_count = sum(len(row["plan_c_pairwise_labels"]) for row in rows)
    all_pair_count = sum(len(row["plan_c_counterfactuals"]) *
        (len(row["plan_c_counterfactuals"]) - 1) // 2 for row in rows)
    expected_fields = {"dataset_schema": MULTICASE_COUNTERFACTUAL_DATASET_SCHEMA,
        "dataset_sha256": composite["dataset_sha256"],
        "dataset_manifest_sha256": composite["manifest_sha256"],
        "case_set_sha256": composite["case_set_sha256"], "cases": composite["cases"],
        "case_coverage": composite["case_coverage"], "protocol_id": protocol_id,
        "qualification_protocol_sha256": protocol_sha,
        "qualification_protocol_file": protocol_path.name, "sources": expected_sources,
        "source_identity_sha256": _canonical_hash(expected_sources),
        "prediction_mode": reference["prediction_mode"],
        "prediction_contract": prediction_contract(reference["prediction_mode"]),
        "collection_device": reference["device"], "split": "train"}
    if (any(payload.get(key) != value for key, value in expected_fields.items()) or
            (expected_prediction_mode is not None and payload.get("prediction_mode") != expected_prediction_mode) or
            payload.get("target_schema") != "meco-matched-prefix-gross-return-v3" or
            payload.get("label_provenance") != "within-prefix matched repeated simulator gross-horizon returns; stable paired sign only" or
            payload.get("target_semantics") != "within_prefix_repeated_matched_stable_pairwise_preference_from_gross_horizon_return" or
            payload.get("scoring_semantics") != "raw_candidate_utility; route_and_risk_costs_excluded_for_gross_return_target" or
            payload.get("plan_c_target_schema") != "meco-plan-c-matched-repeated-gross-return-v1" or
            payload.get("plan_c_repeat_protocol") != reference.get("plan_c_repeat_protocol") or
            payload.get("plan_c_label_policy") != "stable_same_strict_sign_all_paired_repeats_else_abstain" or
            payload.get("plan_c_repeat_variance_note") != reference.get("plan_c_repeat_variance_note") or
            payload.get("plan_c_repeat_protocol") != plan_c_repeat_protocol(protocol_id) or
            payload.get("records_have_prefix_replay_parity") is not True or
            payload.get("source_runtime_device_bound") is not True or
            payload.get("state_digest_schema") != "meco-full-environment-state-v2" or
            payload.get("candidate_boundary") != reference.get("candidate_boundary") or
            payload.get("candidate_k") != reference.get("candidate_k") or
            payload.get("candidate_raw_feature_count") != MECO_CANDIDATE_RAW_FEATURE_COUNT or
            payload.get("candidate_model_input_width") != MECO_CANDIDATE_MODEL_INPUT_WIDTH or
            payload.get("candidate_feature_names") != list(tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)) or
            payload.get("max_plan_c_descriptors_per_prefix") != reference.get("max_plan_c_descriptors_per_prefix") or
            payload.get("max_plan_a_pairs_per_prefix") != reference.get("max_plan_a_pairs_per_prefix") or
            payload.get("horizon_physics_ticks") != reference.get("horizon_physics_ticks") or
            payload.get("max_macro_ticks") != reference.get("max_macro_ticks") or
            payload.get("gamma_per_macro") != reference.get("gamma_per_macro") or
            payload.get("discount_semantics") != reference.get("discount_semantics") or
            payload.get("continuation_policy") != reference.get("continuation_policy") or
            payload.get("plan_c_stable_pair_count") != stable_count or
            payload.get("non_tied_pair_count") != stable_count or
            payload.get("tied_pair_count") != all_pair_count - stable_count or
            payload.get("pair_count_semantics") != "non_tied_pair_count counts stable Plan C labels; tied_pair_count counts all abstentions" or
            type(payload.get("updates")) is not int or payload["updates"] < 1 or
            not _finite_number(payload.get("learning_rate")) or payload["learning_rate"] <= 0 or
            payload.get("training_device") != _device_identity(reference["device"]["device"]) or
            payload.get("optimizer_state_included") is not False or payload.get("resume_compatible") is not False or
            not isinstance(payload.get("model"), dict) or not isinstance(payload.get("mean_loss_by_epoch"), list) or
            len(payload["mean_loss_by_epoch"]) != payload.get("updates") or
            any(not _finite_number(loss) or loss < 0 for loss in payload["mean_loss_by_epoch"])):
        raise ValueError("composite C artifact provenance, target or case-set metadata mismatch")
    model = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("composite C weights do not match the requested track") from exc
    return payload


def _initialize_unlabelled_plan_a_penalty_head(model, track):
    """Keep an unlabelled P output at zero until RL or an oracle trains it."""
    with torch.no_grad():
        if track == "A1":
            weight = model.pose_head.weight.reshape(8, 2, *model.pose_head.weight.shape[1:])
            bias = model.pose_head.bias.reshape(8, 2)
            weight[:, 1].zero_()
            bias[:, 1].fill_(-20.)
        elif track == "A2":
            model.decoder[-1].weight[1].zero_()
            model.decoder[-1].bias[1].fill_(-20.)
        elif track == "A3":
            model.node_head.weight[1].zero_()
            model.node_head.bias[1].fill_(-20.)
        else:
            raise ValueError("Plan A value warm start only supports A1/A2/A3")


def warm_start_counterfactual_plan_a_v3(track: str, dataset: Path, output: Path, *,
        qualification_protocol: Path, updates: int = 10,
        learning_rate: float = 1e-4, device: str | None = None):
    """Regress sampled feasible Plan A U+B values to replayed gross returns.

    The v2 replay-bound dataset is reused unchanged. This v3 artifact adds a
    direct value loss; it does not label or train the P head because the replay
    executor does not execute P-modified routes.
    """
    if track not in ("A1", "A2", "A3") or type(updates) is not int or updates < 1:
        raise ValueError("Plan A v3 requires A1/A2/A3 and a positive update count")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be positive and finite")
    from .curriculum_training import source_identity
    from .meco_counterfactual import _device_identity
    from .meco_plan_a_policy import gather_pose_behavior_logits

    _prepare_native_gpu_runtime(device)
    sources = source_identity()
    rows, data_hash, manifest_hash, _, protocol_id, protocol_sha, manifest = _read_counterfactual_dataset(
        dataset, qualification_protocol=qualification_protocol,
        expected_sources=sources)
    if _device_identity(device) != manifest["device"]:
        raise ValueError("counterfactual replay/training device must match the frozen dataset runtime")
    rows = [row for row in rows if row["scenario_id"] == manifest["scenario_id"]]
    replayed = _materialize_counterfactual_rows(rows, manifest, track=track, train_device=device)
    if not replayed or any(sample["kind"] != "A" for sample in replayed):
        raise ValueError("Plan A v3 requires replayed Plan A support rows")
    non_tied_pair_count = sum(
        1 for sample in replayed for i, (_, left, _) in enumerate(sample["pairs"])
        for _, right, _ in sample["pairs"][i + 1:] if left != right)
    tied_pair_count = sum(
        1 for sample in replayed for i, (_, left, _) in enumerate(sample["pairs"])
        for _, right, _ in sample["pairs"][i + 1:] if left == right)
    if non_tied_pair_count < 1:
        raise ValueError("Plan A v3 dataset has no non-tied feasible gross-return pair")

    model = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    value_losses, preference_losses = [], []
    model.train()
    value_count = sum(len(sample["pairs"]) for sample in replayed)
    for _ in range(updates):
        epoch_value, epoch_preference = [], []
        for sample in replayed:
            raw = model(raster=sample["raster"], globals_=sample["globals"],
                        **sample["graph"])
            logits = gather_pose_behavior_logits(track, raw, sample["graph_object"],
                sample["raster"], sample["support"],
                field_length=sample["field_length"],
                field_width=sample["field_width"])
            pairs = sample["pairs"]
            indices = [index for index, _, _ in pairs]
            targets = torch.tensor([value for _, value, _ in pairs],
                device=device, dtype=logits.dtype)
            predictions = logits[indices]
            if not bool(torch.isfinite(targets).all()) or not bool(torch.isfinite(predictions).all()):
                raise ValueError("Plan A v3 value sample contains nonfinite values")
            value_loss = F.smooth_l1_loss(predictions, targets, beta=1.)
            preference_terms = []
            for left_index, (_, left_return, _) in enumerate(pairs):
                for right_index in range(left_index + 1, len(pairs)):
                    right_return = pairs[right_index][1]
                    if left_return == right_return:
                        continue
                    sign = 1. if left_return > right_return else -1.
                    preference_terms.append(F.softplus(-sign *
                        (predictions[left_index] - predictions[right_index])))
            preference_loss = (torch.stack(preference_terms).mean()
                if preference_terms else value_loss.new_zeros(()))
            loss = value_loss + preference_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            epoch_value.append(float(value_loss.detach().cpu()))
            if preference_terms:
                epoch_preference.append(float(preference_loss.detach().cpu()))
        if not epoch_value:
            raise ValueError("no replayed Plan A branch produced a trainable value loss")
        value_losses.append(sum(epoch_value) / len(epoch_value))
        preference_losses.append(sum(epoch_preference) / len(epoch_preference)
            if epoch_preference else 0.)
    if source_identity() != sources:
        raise RuntimeError("source/runtime changed during Plan A v3 fitting")
    _initialize_unlabelled_plan_a_penalty_head(model, track)
    p_unlabelled_reason = manifest["plan_a_p_unlabelled_reason"]
    payload = {"schema": COUNTERFACTUAL_PLAN_A_VALUE_WEIGHTS_SCHEMA,
        "schema_version": 3,
        "artifact_kind": "weights_only_counterfactual_plan_a_goal_value_v3",
        "track": track, "model_schema_version": 1, "model": model.state_dict(),
        "dataset_sha256": data_hash, "dataset_schema": manifest["schema"],
        "dataset_manifest_sha256": manifest_hash,
        "target_schema": manifest["target_schema"],
        "label_provenance": PLAN_A_VALUE_LABEL_PROVENANCE,
        "target_semantics": "sampled_feasible_node_behavior_gross_horizon_return_regression",
        "scoring_semantics": "raw_U_node_plus_B_behavior; direct gross_horizon_return value and pairwise losses",
        "value_loss_semantics": "smooth_l1_raw_U_plus_B_to_gross_horizon_return_beta_1",
        "preference_loss_semantics": "within_prefix_non_tied_gross_horizon_return_pairwise_softplus",
        "plan_a_support_order": manifest["plan_a_support_order"],
        "plan_a_support_coverage": "sampled_valid_node_behavior_descriptors_from_exact_replayed_full_support",
        "plan_a_value_target_count": value_count,
        "plan_a_p_target": None, "plan_a_p_prior": 0.0,
        "plan_a_p_mode": "p0",
        "plan_a_p_unlabelled_reason": p_unlabelled_reason,
        "candidate_raw_feature_count": MECO_CANDIDATE_RAW_FEATURE_COUNT,
        "candidate_model_input_width": MECO_CANDIDATE_MODEL_INPUT_WIDTH,
        "candidate_feature_names": list(tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)),
        "prediction_mode": manifest["prediction_mode"],
        "prediction_contract": manifest["prediction_contract"],
        "candidate_boundary": manifest["candidate_boundary"],
        "candidate_k": manifest["candidate_k"],
        "max_plan_c_descriptors_per_prefix": manifest["max_plan_c_descriptors_per_prefix"],
        "max_plan_a_pairs_per_prefix": manifest["max_plan_a_pairs_per_prefix"],
        "route_cost_points_per_second": manifest["route_cost_points_per_second"],
        "horizon_physics_ticks": manifest["horizon_physics_ticks"],
        "max_macro_ticks": manifest["max_macro_ticks"],
        "gamma_per_macro": manifest["gamma_per_macro"],
        "discount_semantics": manifest["discount_semantics"],
        "continuation_policy": manifest["continuation_policy"],
        "source_identity_sha256": _canonical_hash(sources), "sources": sources,
        "collection_device": manifest["device"], "training_device": _device_identity(device),
        "protocol_id": protocol_id,
        "qualification_protocol_sha256": protocol_sha,
        "qualification_protocol_file": Path(qualification_protocol).name,
        "split": "train", "scenario_id": manifest["scenario_id"],
        "category": manifest["category"], "seeds": list(manifest["seeds"]),
        "optimizer_state_included": False, "resume_compatible": False,
        "updates": updates, "learning_rate": learning_rate,
        "non_tied_pair_count": non_tied_pair_count,
        "tied_pair_count": tied_pair_count,
        "mean_value_loss_by_epoch": value_losses,
        "mean_preference_loss_by_epoch": preference_losses}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {"track": track, "checkpoint": str(output), "dataset_sha256": data_hash,
        "value_target_count": value_count,
        "non_tied_pair_count": payload["non_tied_pair_count"],
        "mean_value_loss_by_epoch": value_losses,
        "mean_preference_loss_by_epoch": preference_losses}


def _materialize_plan_a_p_route_rows(rows, manifest, *, track, train_device):
    from .meco_counterfactual import (replay_counterfactual_branch,
        _execute_plan_a_p_route_branch)
    from .meco_track_training import _graph_inputs
    device = torch.device(train_device)
    materialized = []
    for row in rows:
        replay = replay_counterfactual_branch(row, manifest,
            require_hip_plan_a=True)
        env, graph = replay["env"], replay["graph"]
        try:
            safe = replay["observation"]
            raster = torch.tensor(safe.raster, dtype=torch.float32,
                device=device).reshape(1, *safe.raster_shape)
            globals_ = torch.tensor(safe.globals, dtype=torch.float32,
                device=device).reshape(1, 7)
            graph_values = _graph_inputs(track, graph, env, raster)
            graph_values = {} if graph_values is None else {
                name: value.to(device) for name, value in graph_values.items()}
            route_groups = []
            for group in row["plan_a_p_route_groups"]:
                descriptor = group["descriptor"]
                verified = []
                for archived in group["outcomes"]:
                    replayed = _execute_plan_a_p_route_branch(row, manifest,
                        descriptor, archived.get("intervention_seed"),
                        p_cost_scale=manifest["p_intervention_scale"])
                    for key in ("gross_macro_return", "gross_horizon_return",
                                "discounted_horizon_return", "continuation_policy",
                                "continuation_macro_horizon", "discount_semantics",
                                "native_route", "executed_action_trace"):
                        if replayed.get(key) != archived.get(key):
                            raise RuntimeError(
                                f"Plan A P strict replay mismatch at {key}")
                    from .meco_counterfactual import (PLAN_A_P_ROUTE_ORACLE_VERSION,
                        _plan_a_p_intervention_map, _plan_a_p_oracle_grid,
                        _plan_a_p_oracle_integral, _hash)
                    if manifest.get("schema") == "meco-counterfactual-plan-a-p-route-oracle-v3":
                        route = archived["native_route"]
                        seed = archived.get("intervention_seed")
                        p_map = None if seed is None else _plan_a_p_intervention_map(env, seed)
                        env._meco_oracle_p_scale = float(manifest["p_intervention_scale"])
                        traversability_grid, base_grid, effective_grid = _plan_a_p_oracle_grid(env, p_map)
                        resolution = float(env.planner.resolution)
                        costs = [_plan_a_p_oracle_integral(effective_grid,
                            snapshot["route_points_m"], resolution)
                            for snapshot in route["route_trace"]]
                        base_costs = [_plan_a_p_oracle_integral(base_grid,
                            snapshot["route_points_m"], resolution)
                            for snapshot in route["route_trace"]]
                        oracle = route.get("independent_cost_oracle", {})
                        regenerated = {"version": PLAN_A_P_ROUTE_ORACLE_VERSION,
                            "implementation": "frc_defense.meco_counterfactual._plan_a_p_oracle_integral",
                            "grid_shape": [len(base_grid), len(base_grid[0])],
                            "resolution_m": resolution,
                            "base_traversability_cost_grid": traversability_grid,
                            "base_bump_cost_grid": base_grid,
                            "effective_cost_grid": effective_grid,
                            "grid_inputs_sha256": _hash({"traversability": traversability_grid,
                                "base": base_grid,
                                "effective": effective_grid, "resolution_m": resolution}),
                            "route_snapshot_costs_grid_edges": costs,
                            "route_cost_grid_edges": sum(costs) / len(costs),
                            "route_trace_sha256": route["route_trace_sha256"],
                            "selected_goal_descriptor_sha256": descriptor["descriptor_sha256"],
                            "intervention_map_sha256": route["map_sha256"],
                            "intervention_scale": (float(manifest["p_intervention_scale"])
                                if seed is not None else 0.)}
                        if regenerated != oracle:
                            raise RuntimeError("Plan A P independent oracle label differs after strict replay")
                        replayed_outcome = dict(archived)
                        # Keep the fixed base-grid contribution beside the label so
                        # training predicts total cost = replayed base + learned P.
                        replayed_outcome["_replayed_base_route_cost_grid_edges"] = (
                            sum(base_costs) / len(base_costs))
                    else:
                        replayed_outcome = archived
                    verified.append(replayed_outcome)
                route_groups.append(verified)
            materialized.append({"raster": raster, "globals": globals_,
                "graph": graph_values, "graph_object": graph,
                "planner": env.planner,
                "field_length": float(env.sim.field_length),
                "field_width": float(env.sim.field_width),
                "resolution_m": float(env.planner.resolution),
                "legal_xy_mask": (graph.lookup >= 0).any(0).detach().cpu(),
                "legal_xy_cell_count": int((graph.lookup >= 0).any(0).sum().item()),
                "p_cost_scale": float(manifest["p_intervention_scale"]),
                "route_groups": route_groups})
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    return materialized


def _plan_a_p_output_parameters(model, track):
    if track == "A1":
        layer = model.pose_head
        mask = torch.zeros(layer.weight.shape[0], device=layer.weight.device)
        mask[1::2] = 1.
    elif track == "A2":
        layer = model.decoder[-1]
        mask = torch.tensor([0., 1.], device=layer.weight.device)
    else:
        layer = model.node_head
        mask = torch.tensor([0., 1.], device=layer.weight.device)
    if layer.weight.shape[0] != mask.numel():
        raise ValueError("Plan A P output channel schema mismatch")
    return layer, mask


def warm_start_counterfactual_plan_a_p_v1(track: str, dataset: Path,
        value_checkpoint: Path, output: Path, *, qualification_protocol: Path,
        updates: int = 10, learning_rate: float = 1e-4, device: str | None = None,
        route_margin: float = 0.02, map_regularization: float = 0.001,
        minimum_coverage_fraction: float = 0.01, oracle_labels: bool = False):
    """Fit the P head from strict simulator outcomes for executed weighted routes."""
    if (track not in ("A1", "A2", "A3") or type(updates) is not int or updates < 1 or
            not math.isfinite(learning_rate) or learning_rate <= 0 or
            not math.isfinite(route_margin) or route_margin <= 0 or
            not math.isfinite(map_regularization) or map_regularization < 0 or
            not math.isfinite(minimum_coverage_fraction) or
            not 0 < minimum_coverage_fraction <= 1):
        raise ValueError("invalid Plan A P route warm-start configuration")
    from .meco_device import require_indexed_cuda_device
    device = str(require_indexed_cuda_device(device,
        purpose="Plan A P warm-start"))
    from .curriculum_training import source_identity
    from .meco_path_penalty import project_pose_penalty, integrate_frozen_routes
    from .meco_counterfactual import (PLAN_A_P_ROUTE_DATASET_SCHEMA,
        PLAN_A_P_ROUTE_TARGET_SCHEMA, PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA,
        PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA, PLAN_A_P_ROUTE_ORACLE_VERSION,
        CONTINUATION_POLICY, _device_identity)
    _prepare_native_gpu_runtime(device)
    sources = source_identity()
    value_payload = validate_counterfactual_plan_a_value_artifact(track,
        Path(value_checkpoint), qualification_protocol=qualification_protocol,
        expected_sources=sources)
    rows, data_hash, manifest_hash, _, protocol_id, protocol_sha, manifest = (
        _read_plan_a_p_route_dataset(dataset,
            qualification_protocol=qualification_protocol,
            expected_sources=sources))
    if oracle_labels != (manifest["schema"] == PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA):
        raise ValueError("Plan A P fitter and dataset label schema differ")
    if manifest["device"] != _device_identity(device):
        raise ValueError("P route replay/training device differs from frozen dataset")
    if value_payload.get("protocol_id") != protocol_id:
        raise ValueError("Plan A U+B value checkpoint uses another qualification protocol")
    if value_payload.get("prediction_mode") != manifest.get("prediction_mode"):
        raise ValueError("Plan A U+B checkpoint prediction mode differs from P route data")
    replayed = _materialize_plan_a_p_route_rows(rows, manifest,
        track=track, train_device=device)
    model = make_meco_track(track,
        candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT).to(device)
    model.load_state_dict(value_payload["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer, channel_mask = _plan_a_p_output_parameters(model, track)
    layer.weight.requires_grad_(True)
    layer.bias.requires_grad_(True)
    weight_mask = channel_mask.reshape(-1, *([1] * (layer.weight.ndim - 1)))
    layer.weight.register_hook(lambda grad: grad * weight_mask)
    layer.bias.register_hook(lambda grad: grad * channel_mask)
    with torch.no_grad():
        layer.weight.mul_(weight_mask)
        # softplus(-20) has effectively zero gradient; seed P at a small
        # positive cost so route-ranking supervision can move it.
        initial_cost = torch.tensor(0.02, device=device, dtype=layer.bias.dtype)
        initial_logit = torch.log(torch.expm1(initial_cost))
        if track == "A1":
            layer.bias[1::2] = initial_logit
        else:
            layer.bias[1] = initial_logit
    optimizer = torch.optim.Adam((layer.weight, layer.bias), lr=learning_rate)
    losses, pair_count = [], 0
    visited_cells = set()
    legal_mask = replayed[0]["legal_xy_mask"]
    if any(not torch.equal(sample["legal_xy_mask"], legal_mask) for sample in replayed):
        raise ValueError("Plan A P route dataset mixes legal XY grid geometries")
    legal_cell_count = int(legal_mask.sum().item())
    model.eval()
    for _ in range(updates):
        epoch_losses = []
        for sample in replayed:
            raw = model(raster=sample["raster"], globals_=sample["globals"],
                        **sample["graph"])
            graph = sample["graph_object"]
            penalty_map = project_pose_penalty(track, raw["penalty"],
                graph.lookup, sample["planner"],
                sample["field_length"], sample["field_width"])
            route_losses = []
            for outcomes in sample["route_groups"]:
                unique = {}
                for outcome in outcomes:
                    route = outcome["native_route"]
                    unique.setdefault(route["route_trace_sha256"], outcome)
                    for snapshot in route["route_trace"]:
                        points = snapshot["route_points_m"]
                        for point in points:
                            ix = min(legal_mask.shape[0] - 1, max(0,
                                int(float(point[0]) / sample["resolution_m"])))
                            iy = min(legal_mask.shape[1] - 1, max(0,
                                int(float(point[1]) / sample["resolution_m"])))
                            if bool(legal_mask[ix, iy]):
                                visited_cells.add((ix, iy))
                        for first, second in zip(points, points[1:]):
                            dx = float(second[0]) - float(first[0])
                            dy = float(second[1]) - float(first[1])
                            count = max(1, math.ceil(math.hypot(dx, dy) /
                                (sample["resolution_m"] * .5)))
                            for index in range(count + 1):
                                fraction = index / count
                                x = float(first[0]) + fraction * dx
                                y = float(first[1]) + fraction * dy
                                ix = min(legal_mask.shape[0] - 1,
                                    max(0, int(x / sample["resolution_m"])))
                                iy = min(legal_mask.shape[1] - 1,
                                    max(0, int(y / sample["resolution_m"])))
                                if bool(legal_mask[ix, iy]):
                                    visited_cells.add((ix, iy))
                variants = list(unique.values())
                for index, left in enumerate(variants):
                    for right in variants[index + 1:]:
                        if oracle_labels:
                            left_cost = left["native_route"]["independent_cost_oracle"]["route_cost_grid_edges"]
                            right_cost = right["native_route"]["independent_cost_oracle"]["route_cost_grid_edges"]
                            delta = right_cost - left_cost
                        else:
                            delta = left["gross_horizon_return"] - right["gross_horizon_return"]
                        if delta == 0.:
                            continue
                        better, worse = (left, right) if delta > 0 else (right, left)
                        route_costs = []
                        for outcome in (better, worse):
                            records = [{"path_points_m": snapshot["route_points_m"]}
                                for snapshot in outcome["native_route"]["route_trace"]]
                            predicted_p_cost = integrate_frozen_routes(penalty_map,
                                records, sample["resolution_m"],
                                map_scale=sample["p_cost_scale"],
                                points_per_grid_edge=1.).mean()
                            if oracle_labels:
                                base_cost = outcome.get(
                                    "_replayed_base_route_cost_grid_edges")
                                if (not _finite_number(base_cost) or base_cost < 0.):
                                    raise ValueError(
                                        "Plan A P oracle outcome lacks replayed base-grid cost")
                                predicted_p_cost = predicted_p_cost + float(base_cost)
                            route_costs.append(predicted_p_cost)
                        costs = torch.stack(route_costs)
                        route_losses.append(F.softplus(
                            (costs[0] - costs[1] + route_margin)))
            if not route_losses:
                continue
            pair_loss = torch.stack(route_losses).mean()
            loss = pair_loss + map_regularization * penalty_map.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((layer.weight, layer.bias), 1.)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            if not losses:
                pair_count += len(route_losses)
        if not epoch_losses:
            raise ValueError("Plan A P data has no non-tied distinct executed-route pair")
        losses.append(sum(epoch_losses) / len(epoch_losses))
    coverage_fraction = len(visited_cells) / max(1, legal_cell_count)
    if coverage_fraction < minimum_coverage_fraction:
        raise ValueError("Plan A P executed paths do not meet the configured legal-grid coverage floor")
    if source_identity() != sources:
        raise RuntimeError("source/runtime changed during Plan A P fitting")
    payload = {"schema": ("meco-counterfactual-plan-a-p-oracle-warm-start-weights-v1"
                           if oracle_labels else COUNTERFACTUAL_PLAN_A_P_WEIGHTS_SCHEMA),
        "schema_version": 2 if oracle_labels else 1,
        "artifact_kind": ("weights_only_counterfactual_plan_a_p_oracle_v1"
                          if oracle_labels else "weights_only_counterfactual_plan_a_p_route_v1"),
        "track": track, "model_schema_version": 1, "model": model.state_dict(),
        "dataset_sha256": data_hash, "dataset_manifest_sha256": manifest_hash,
        "dataset_schema": (PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA if oracle_labels
                            else PLAN_A_P_ROUTE_DATASET_SCHEMA),
        "target_schema": (PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA if oracle_labels
                          else PLAN_A_P_ROUTE_TARGET_SCHEMA),
        "source_counterfactual_dataset_sha256": manifest["source_counterfactual_dataset_sha256"],
        "value_checkpoint_sha256": hashlib.sha256(Path(value_checkpoint).read_bytes()).hexdigest(),
        "label_provenance": ("strict replay and independent recomputation of base bump plus effective P grid path integrals; sampled executed paths only"
                             if oracle_labels else "strict replay of simulator returns for model-independent P0 and seeded weighted AD* route interventions"),
        "target_semantics": ("within_prefix_non_tied_sampled_executed_path_cost_preference_lower_is_better"
                             if oracle_labels else "within_prefix_executed_route_pairwise_preference_from_gross_horizon_return"),
        "loss_semantics": ("softplus_predicted_total_route_cost_preference; total=independently_replayed_base_bump_grid_integral+learned_scaled_P_integral; independent oracle total-cost labels; plus mean_P_regularization"
                           if oracle_labels else "softplus_predicted_integrated_P_cost_preference_plus_mean_P_regularization"),
        "p_oracle_version": PLAN_A_P_ROUTE_ORACLE_VERSION if oracle_labels else None,
        "p_map_initialization": "softplus_inverse_of_0.02_on_penalty_output_channel",
        "p_route_pair_count": pair_count,
        "p_route_group_count": sum(len(sample["route_groups"]) for sample in replayed),
        "p_route_visited_xy_cell_count": len(visited_cells),
        "p_route_legal_xy_cell_count": legal_cell_count,
        "p_route_legal_xy_coverage_fraction": coverage_fraction,
        "p_route_minimum_coverage_fraction": minimum_coverage_fraction,
        "p_route_coverage_scope": "sampled_native_execution_path_cells; not exhaustive legal-grid coverage",
        "p_mode": "learned", "p_cost_scale": manifest["p_intervention_scale"],
        "p_projection": "mean_over_legal_heading_bins_xy_grid_v1",
        "p_map_units": "dimensionless_additive_penalty_per_cardinal_grid_step",
        "p_path_units": "normalized_grid_edge_units",
        "route_margin": route_margin, "map_regularization": map_regularization,
        "prediction_mode": manifest["prediction_mode"],
        "prediction_contract": manifest["prediction_contract"],
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "qualification_protocol_file": Path(qualification_protocol).name,
        "split": "train", "scenario_id": manifest["scenario_id"],
        "category": manifest["category"], "seeds": list(manifest["seeds"]),
        "sources": sources, "source_identity_sha256": _canonical_hash(sources),
        "collection_device": manifest["device"],
        "training_device": _device_identity(device),
        "continuation_policy": CONTINUATION_POLICY,
        "optimizer_state_included": False, "resume_compatible": False,
        "updates": updates, "learning_rate": learning_rate,
        "mean_route_loss_by_epoch": losses}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    validate_counterfactual_plan_a_p_artifact(track, output,
        qualification_protocol=qualification_protocol,
        expected_sources=sources,
        expected_prediction_mode=manifest["prediction_mode"])
    return {"track": track, "checkpoint": str(output),
        "dataset_sha256": data_hash, "p_route_pair_count": pair_count,
        "p_route_visited_xy_cell_count": len(visited_cells),
        "mean_route_loss_by_epoch": losses}


def validate_warm_start_artifact(track: str, checkpoint: Path, *,
                                 qualification_protocol: Path,
                                 expected_sources: dict,
                                 expected_prediction_mode: str | None = None):
    """Validate immutable metadata without restoring any training state."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") == PLAN_A_HUMAN_PREFERENCES_V2_WEIGHTS_SCHEMA:
        return validate_plan_a_preferences_v2_artifact(track, checkpoint,
            qualification_protocol=qualification_protocol,
            expected_sources=expected_sources,
            expected_prediction_mode=expected_prediction_mode)
    if payload.get("schema") == PLAN_A_EXPERT_GOAL_WEIGHTS_SCHEMA:
        return validate_plan_a_expert_goal_artifact(track, checkpoint,
            qualification_protocol=qualification_protocol,
            expected_sources=expected_sources,
            expected_prediction_mode=expected_prediction_mode)
    if payload.get("schema") == PLAN_A_JOINT_UBP_WEIGHTS_SCHEMA:
        return validate_plan_a_joint_ubp_artifact(track, checkpoint,
            qualification_protocol=qualification_protocol,
            expected_sources=expected_sources,
            expected_prediction_mode=expected_prediction_mode)
    if payload.get("schema") == PLAN_A_LEGAL_NODE_WEIGHTS_SCHEMA:
        return validate_plan_a_legal_node_artifact(track, checkpoint,
            qualification_protocol=qualification_protocol,
            expected_sources=expected_sources,
            expected_prediction_mode=expected_prediction_mode)
    if payload.get("schema") == MULTICASE_COUNTERFACTUAL_WEIGHTS_SCHEMA:
        return validate_counterfactual_composite_artifact(track, checkpoint,
            qualification_protocol=qualification_protocol,
            expected_sources=expected_sources,
            expected_prediction_mode=expected_prediction_mode)
    if payload.get("schema") in (COUNTERFACTUAL_WEIGHTS_SCHEMA,
                                  COUNTERFACTUAL_PLAN_C_REPEATED_WEIGHTS_SCHEMA):
        return validate_counterfactual_warm_start_artifact(track, checkpoint,
            qualification_protocol=qualification_protocol,
            expected_sources=expected_sources,
            expected_prediction_mode=expected_prediction_mode)
    if payload.get("schema") == COUNTERFACTUAL_PLAN_A_VALUE_WEIGHTS_SCHEMA:
        return validate_counterfactual_plan_a_value_artifact(track, checkpoint,
            qualification_protocol=qualification_protocol,
            expected_sources=expected_sources,
            expected_prediction_mode=expected_prediction_mode)
    if payload.get("schema") == COUNTERFACTUAL_PLAN_A_P_WEIGHTS_SCHEMA:
        return validate_counterfactual_plan_a_p_artifact(track, checkpoint,
            qualification_protocol=qualification_protocol,
            expected_sources=expected_sources,
            expected_prediction_mode=expected_prediction_mode)
    if payload.get("schema") == "meco-counterfactual-plan-a-p-oracle-warm-start-weights-v1":
        return validate_counterfactual_plan_a_p_artifact(track, checkpoint,
            qualification_protocol=qualification_protocol,
            expected_sources=expected_sources,
            expected_prediction_mode=expected_prediction_mode)
    mode = payload.get("prediction_mode")
    if (mode not in PREDICTION_MODES or
            payload.get("prediction_contract") != prediction_contract(mode)):
        raise ValueError("warm-start prediction mode/provenance is missing or invalid")
    if expected_prediction_mode is not None and mode != expected_prediction_mode:
        raise ValueError("warm-start prediction mode differs from the training configuration")
    _, protocol_id, protocol_sha256, protocol_path = _load_protocol(qualification_protocol)
    source_digest = _canonical_hash(expected_sources)
    expected_semantics = ("observed_teacher_behavior_only" if track.startswith("A")
                          else "observed_teacher_action_probability_mass")
    dataset_digest = payload.get("dataset_sha256")
    losses = payload.get("mean_loss_by_epoch")
    if (payload.get("schema") != WEIGHTS_SCHEMA or payload.get("schema_version") != 1 or
            payload.get("model_schema_version") != 1 or payload.get("track") != track or
            payload.get("dataset_schema") != DATASET_SCHEMA or
            payload.get("target_schema") != TARGET_SCHEMA or
            payload.get("label_provenance") != LABEL_PROVENANCE or
            payload.get("target_semantics") != expected_semantics or
            payload.get("candidate_raw_feature_count") != MECO_CANDIDATE_RAW_FEATURE_COUNT or
            payload.get("candidate_model_input_width") != MECO_CANDIDATE_MODEL_INPUT_WIDTH or
            payload.get("candidate_feature_names") != list(tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)) or
            payload.get("protocol_id") != protocol_id or
            payload.get("qualification_protocol_sha256") != protocol_sha256 or
            payload.get("qualification_protocol_file") != protocol_path.name or
            payload.get("sources") != expected_sources or
            payload.get("source_identity_sha256") != source_digest or
            not isinstance(dataset_digest, str) or len(dataset_digest) != 64 or
            any(char not in "0123456789abcdef" for char in dataset_digest) or
            not isinstance(losses, list) or not losses or
            any(not isinstance(loss, (int, float)) or not math.isfinite(loss) for loss in losses) or
            type(payload.get("updates")) is not int or payload["updates"] < 1 or
            type(payload.get("steps_per_epoch")) is not int or payload["steps_per_epoch"] < 1 or
            payload.get("batch_schedule") != "each epoch visits every legal-node and P dataset row at least once; shorter dataset cycles deterministically" or
            not isinstance(payload.get("learning_rate"), (int, float)) or
            not math.isfinite(payload["learning_rate"]) or payload["learning_rate"] <= 0 or
            payload.get("optimizer_state_included") is not False or
            payload.get("resume_compatible") is not False or
            not isinstance(payload.get("model"), dict)):
        raise ValueError("warm-start artifact identity/schema mismatch")
    model = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("warm-start model weights do not match the model schema") from exc
    return payload


def validate_plan_a_joint_ubp_artifact(track: str, checkpoint: Path, *,
        qualification_protocol: Path, expected_sources: dict,
        expected_prediction_mode: str | None = None):
    """Validate joint U+B+P provenance, exact source bytes and full model state."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    from .meco_counterfactual import (PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
        PLAN_A_LEGAL_NODE_TARGET_SCHEMA, PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA,
        PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA, PLAN_A_P_ROUTE_ORACLE_VERSION,
        DATASET_SCHEMA as TRAJECTORY_SCHEMA, TARGET_SCHEMA as TRAJECTORY_TARGET_SCHEMA)
    from .curriculum_training import source_identity
    import hashlib
    _, protocol_id, protocol_sha, protocol_path = _load_protocol(qualification_protocol)
    paths = {key: Path(payload.get(key, "")).resolve() for key in
        ("legal_dataset_path", "source_dataset_path", "p_dataset_path")}
    try:
        legal_bytes = paths["legal_dataset_path"].read_bytes()
        legal_manifest_bytes = paths["legal_dataset_path"].with_suffix(
            paths["legal_dataset_path"].suffix + ".manifest.json").read_bytes()
        source_bytes = paths["source_dataset_path"].read_bytes()
        source_manifest_bytes = paths["source_dataset_path"].with_suffix(
            paths["source_dataset_path"].suffix + ".manifest.json").read_bytes()
        p_bytes = paths["p_dataset_path"].read_bytes()
        p_manifest_bytes = paths["p_dataset_path"].with_suffix(
            paths["p_dataset_path"].suffix + ".manifest.json").read_bytes()
        legal_manifest = json.loads(legal_manifest_bytes)
        source_manifest = json.loads(source_manifest_bytes)
        p_manifest = json.loads(p_manifest_bytes)
    except (OSError, ValueError) as exc:
        raise ValueError("joint Plan A artifact bound datasets are missing or malformed") from exc
    sources = source_identity()
    mode = payload.get("prediction_mode")
    if (track not in ("A1", "A2", "A3") or
            payload.get("schema") != PLAN_A_JOINT_UBP_WEIGHTS_SCHEMA or
            payload.get("schema_version") != 1 or
            payload.get("artifact_kind") != "weights_only_plan_a_joint_u_b_p_v1" or
            payload.get("track") != track or payload.get("model_schema_version") != 1 or
            payload.get("sources") != expected_sources or expected_sources != sources or
            payload.get("source_identity_sha256") != _canonical_hash(sources) or
            payload.get("protocol_id") != protocol_id or
            payload.get("qualification_protocol_sha256") != protocol_sha or
            payload.get("qualification_protocol_file") != protocol_path.name or
            payload.get("split") != "train" or mode not in PREDICTION_MODES or
            payload.get("prediction_contract") != prediction_contract(mode) or
            (expected_prediction_mode is not None and mode != expected_prediction_mode) or
            payload.get("legal_dataset_schema") != PLAN_A_LEGAL_NODE_DATASET_SCHEMA or
            payload.get("legal_target_schema") != PLAN_A_LEGAL_NODE_TARGET_SCHEMA or
            payload.get("p_dataset_schema") != PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA or
            payload.get("p_target_schema") != PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA or
            payload.get("p_oracle_version") != PLAN_A_P_ROUTE_ORACLE_VERSION or
            payload.get("p_mode") != "learned" or
            not _finite_number(payload.get("p_cost_scale")) or payload["p_cost_scale"] <= 0 or
            payload.get("candidate_raw_feature_count") != MECO_CANDIDATE_RAW_FEATURE_COUNT or
            payload.get("candidate_model_input_width") != MECO_CANDIDATE_MODEL_INPUT_WIDTH or
            payload.get("candidate_feature_names") != list(tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)) or
            payload.get("optimizer_state_included") is not False or
            payload.get("resume_compatible") is not False or
            type(payload.get("updates")) is not int or payload["updates"] < 1 or
            not _finite_number(payload.get("learning_rate")) or payload["learning_rate"] <= 0 or
            not _finite_number(payload.get("route_margin")) or payload["route_margin"] <= 0 or
            not _finite_number(payload.get("map_regularization")) or payload["map_regularization"] < 0 or
            not _finite_number(payload.get("p_route_minimum_coverage_fraction")) or
            not 0 < payload["p_route_minimum_coverage_fraction"] <= 1 or
            not _finite_number(payload.get("p_route_legal_xy_coverage_fraction")) or
            payload["p_route_legal_xy_coverage_fraction"] < payload["p_route_minimum_coverage_fraction"] or
            type(payload.get("p_route_pair_count")) is not int or payload["p_route_pair_count"] < 1 or
            payload.get("task_loss_weights") != {"u": 1.0, "b": 1.0, "p": 1.0} or
            payload.get("target_masks") != {"u": "status=available and finite u_target",
                "b": "status=comparable; both behavior logits available and centered within feasible node support",
                "p": "distinct route trace pairs with non-tied independent oracle-v3 costs"} or
            not isinstance(payload.get("joint_loss_history"), list) or
            len(payload["joint_loss_history"]) != payload.get("updates") or
            any(not isinstance(entry, dict) or set(entry) != {"u", "b", "p", "joint"} or
                any(value is not None and not _finite_number(value)
                    for key, value in entry.items() if key != "joint") or
                not _finite_number(entry.get("joint"))
                for entry in payload["joint_loss_history"]) or
            not isinstance(payload.get("model"), dict)):
        raise ValueError("joint Plan A U+B+P artifact metadata contract mismatch")
    if (hashlib.sha256(legal_bytes).hexdigest() != payload.get("legal_dataset_sha256") or
            hashlib.sha256(legal_manifest_bytes).hexdigest() != payload.get("legal_manifest_sha256") or
            hashlib.sha256(source_bytes).hexdigest() != payload.get("source_dataset_sha256") or
            hashlib.sha256(source_manifest_bytes).hexdigest() != payload.get("source_manifest_sha256") or
            hashlib.sha256(p_bytes).hexdigest() != payload.get("p_dataset_sha256") or
            hashlib.sha256(p_manifest_bytes).hexdigest() != payload.get("p_manifest_sha256") or
            legal_manifest.get("schema") != PLAN_A_LEGAL_NODE_DATASET_SCHEMA or
            legal_manifest.get("target_schema") != PLAN_A_LEGAL_NODE_TARGET_SCHEMA or
            legal_manifest.get("dataset_sha256") != payload.get("legal_dataset_sha256") or
            source_manifest.get("schema") != TRAJECTORY_SCHEMA or
            source_manifest.get("target_schema") != TRAJECTORY_TARGET_SCHEMA or
            p_manifest.get("schema") != PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA or
            p_manifest.get("target_schema") != PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA or
            p_manifest.get("dataset_sha256") != payload.get("p_dataset_sha256") or
            p_manifest.get("p_oracle_version") != PLAN_A_P_ROUTE_ORACLE_VERSION or
            legal_manifest.get("split") != "train" or source_manifest.get("split") != "train" or
            p_manifest.get("split") != "train" or
            legal_manifest.get("sources") != sources or source_manifest.get("sources") != sources or
            p_manifest.get("sources") != sources or
            legal_manifest.get("protocol_id") != protocol_id or
            source_manifest.get("protocol_id") != protocol_id or
            p_manifest.get("protocol_id") != protocol_id or
            legal_manifest.get("qualification_protocol_sha256") != protocol_sha or
            source_manifest.get("qualification_protocol_sha256") != protocol_sha or
            p_manifest.get("qualification_protocol_sha256") != protocol_sha or
            legal_manifest.get("prediction_mode") != mode or
            source_manifest.get("prediction_mode") != mode or p_manifest.get("prediction_mode") != mode or
            legal_manifest.get("prediction_contract") != prediction_contract(mode) or
            source_manifest.get("prediction_contract") != prediction_contract(mode) or
            p_manifest.get("prediction_contract") != prediction_contract(mode) or
            legal_manifest.get("source_dataset_sha256") != payload.get("source_dataset_sha256") or
            legal_manifest.get("source_manifest_sha256") != payload.get("source_manifest_sha256") or
            p_manifest.get("source_counterfactual_dataset_sha256") != payload.get("source_dataset_sha256") or
            p_manifest.get("source_counterfactual_manifest_sha256") != payload.get("source_manifest_sha256") or
            p_manifest.get("p_intervention_scale") != payload.get("p_cost_scale") or
            legal_manifest.get("device") != p_manifest.get("device") or
            payload.get("collection_device") != legal_manifest.get("device") or
            payload.get("training_device") != legal_manifest.get("device")):
        raise ValueError("joint Plan A datasets/source bytes/protocol/device are not identically bound")
    template = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT)
    try:
        template.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("joint Plan A model weights do not match the requested model schema") from exc
    layer, mask = _plan_a_p_output_parameters(template, track)
    selected = (layer.weight.detach() * mask.reshape(-1,
        *([1] * (layer.weight.ndim - 1)))).abs().sum()
    biases = layer.bias.detach()[1::2] if track == "A1" else layer.bias.detach()[1:2]
    if (not bool(torch.isfinite(selected)) or not bool(torch.isfinite(biases).all()) or
            not bool((selected > 0) or (biases > -20).any())):
        raise ValueError("joint Plan A artifact contains no trained P output channel")
    return payload


def validate_plan_a_legal_node_artifact(track: str, checkpoint: Path, *,
        qualification_protocol: Path, expected_sources: dict,
        expected_prediction_mode: str | None = None):
    """Validate legal-node weights, source bytes, split, protocol and model shape."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    import hashlib
    from .meco_counterfactual import (PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
        PLAN_A_LEGAL_NODE_TARGET_SCHEMA, DATASET_SCHEMA as CF_SCHEMA,
        TARGET_SCHEMA as CF_TARGET_SCHEMA)
    from .curriculum_training import source_identity
    _, protocol_id, protocol_sha, protocol_path = _load_protocol(Path(qualification_protocol))
    dataset_path = Path(payload.get("dataset_path", "")).resolve()
    source_path = Path(payload.get("source_dataset_path", "")).resolve()
    try:
        dataset_bytes = dataset_path.read_bytes()
        dataset_manifest_bytes = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json").read_bytes()
        source_bytes = source_path.read_bytes()
        source_manifest_bytes = source_path.with_suffix(source_path.suffix + ".manifest.json").read_bytes()
        dataset_manifest = json.loads(dataset_manifest_bytes)
        source_manifest = json.loads(source_manifest_bytes)
    except (OSError, ValueError) as exc:
        raise ValueError("legal-node artifact's bound dataset files are unavailable or malformed") from exc
    source_digest = _canonical_hash(expected_sources)
    model_template = make_meco_track(track,
        candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT) if track in ("A1", "A2", "A3") else None
    expected_state = model_template.state_dict() if model_template is not None else {}
    model_state = payload.get("model")
    if (track not in ("A1", "A2", "A3") or
            payload.get("schema") != PLAN_A_LEGAL_NODE_WEIGHTS_SCHEMA or
            payload.get("schema_version") != 1 or
            payload.get("artifact_kind") != "weights_only_plan_a_legal_node_u_b_v1" or
            payload.get("model_schema_version") != 1 or payload.get("track") != track or
            payload.get("dataset_schema") != PLAN_A_LEGAL_NODE_DATASET_SCHEMA or
            payload.get("source_dataset_schema") != CF_SCHEMA or
            payload.get("target_schema") != PLAN_A_LEGAL_NODE_TARGET_SCHEMA or
            payload.get("split") != "train" or
            payload.get("sources") != expected_sources or
            payload.get("source_identity_sha256") != source_digest or
            payload.get("protocol_id") != protocol_id or
            payload.get("protocol_sha256") != protocol_sha or
            dataset_manifest.get("schema") != PLAN_A_LEGAL_NODE_DATASET_SCHEMA or
            dataset_manifest.get("target_schema") != PLAN_A_LEGAL_NODE_TARGET_SCHEMA or
            dataset_manifest.get("split") != "train" or
            dataset_manifest.get("sources") != expected_sources or
            dataset_manifest.get("protocol_id") != protocol_id or
            dataset_manifest.get("qualification_protocol_sha256") != protocol_sha or
            dataset_manifest.get("qualification_protocol_file") != protocol_path.name or
            hashlib.sha256(dataset_bytes).hexdigest() != payload.get("dataset_sha256") or
            hashlib.sha256(dataset_manifest_bytes).hexdigest() != payload.get("dataset_manifest_sha256") or
            dataset_manifest.get("dataset_sha256") != payload.get("dataset_sha256") or
            hashlib.sha256(source_bytes).hexdigest() != payload.get("source_dataset_sha256") or
            hashlib.sha256(source_manifest_bytes).hexdigest() != payload.get("source_manifest_sha256") or
            source_manifest.get("schema") != CF_SCHEMA or
            source_manifest.get("target_schema") != CF_TARGET_SCHEMA or
            source_manifest.get("split") != "train" or
            source_manifest.get("sources") != expected_sources or
            payload.get("source_dataset_sha256") != dataset_manifest.get("source_dataset_sha256") or
            payload.get("source_manifest_sha256") != dataset_manifest.get("source_manifest_sha256") or
            payload.get("prediction_mode") != source_manifest.get("prediction_mode") or
            payload.get("prediction_mode") not in PREDICTION_MODES or
            source_manifest.get("prediction_contract") != prediction_contract(payload.get("prediction_mode")) or
            dataset_manifest.get("prediction_mode") != payload.get("prediction_mode") or
            dataset_manifest.get("prediction_contract") != source_manifest.get("prediction_contract") or
            payload.get("candidate_raw_feature_count") != MECO_CANDIDATE_RAW_FEATURE_COUNT or
            payload.get("candidate_model_input_width") != MECO_CANDIDATE_MODEL_INPUT_WIDTH or
            payload.get("candidate_feature_names") != list(tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)) or
            payload.get("device") != dataset_manifest.get("device") or
            payload.get("p_mode") != "p0" or
            payload.get("optimizer_state_included") is not False or
            payload.get("resume_compatible") is not False or
            payload.get("p_target", "missing") is not None or payload.get("p_prior") != 0.0 or
            not isinstance(model_state, dict) or set(model_state) != set(expected_state) or
            any(tuple(model_state[k].shape) != tuple(expected_state[k].shape) for k in expected_state)):
        raise ValueError("Plan A legal-node weights/source/protocol/model contract mismatch")
    if (expected_prediction_mode is not None and
            payload.get("prediction_mode") != expected_prediction_mode):
        raise ValueError("legal-node prediction mode differs from training configuration")
    return payload


def load_warm_start(track: str, model, checkpoint: Path, *, qualification_protocol: Path,
                    expected_sources: dict,
                    expected_prediction_mode: str | None = None):
    """Load only compatible model weights; never restore optimizer, RNG, or cursor."""
    payload = validate_warm_start_artifact(track, checkpoint,
        qualification_protocol=qualification_protocol, expected_sources=expected_sources,
        expected_prediction_mode=expected_prediction_mode)
    if payload.get("schema") == PLAN_A_HUMAN_PREFERENCES_V2_WEIGHTS_SCHEMA:
        from .meco_device import require_indexed_cuda_device
        try:
            model_device = next(model.parameters()).device
        except StopIteration as exc:
            raise ValueError("Plan A preference consumer model has no parameters") from exc
        selected = str(require_indexed_cuda_device(str(model_device),
            purpose="Plan A preference warm-start consumption"))
        if payload.get("training_device", {}).get("device") != selected:
            raise ValueError("Plan A preference weights were fitted on a different indexed CUDA device")
    if payload.get("schema") == PLAN_A_LEGAL_NODE_WEIGHTS_SCHEMA:
        from .meco_device import require_indexed_cuda_device
        try:
            model_device = next(model.parameters()).device
        except StopIteration as exc:
            raise ValueError("legal-node consumer model has no parameters") from exc
        require_indexed_cuda_device(str(model_device),
            purpose="Plan A legal-node warm-start consumption")
    if payload.get("schema") == PLAN_A_EXPERT_GOAL_WEIGHTS_SCHEMA:
        from .meco_device import require_indexed_cuda_device
        try:
            model_device = next(model.parameters()).device
        except StopIteration as exc:
            raise ValueError("expert-goal consumer model has no parameters") from exc
        selected = str(require_indexed_cuda_device(str(model_device),
            purpose="Plan A expert-goal warm-start consumption"))
        if payload.get("training_device", {}).get("device") != selected:
            raise ValueError("expert-goal weights were fitted on a different indexed CUDA device")
    if payload.get("schema") == PLAN_A_JOINT_UBP_WEIGHTS_SCHEMA:
        from .meco_device import require_indexed_cuda_device
        try:
            model_device = next(model.parameters()).device
        except StopIteration as exc:
            raise ValueError("joint Plan A consumer model has no parameters") from exc
        selected = str(require_indexed_cuda_device(str(model_device),
            purpose="joint Plan A U+B+P warm-start consumption"))
        if payload.get("training_device", {}).get("device") != selected:
            raise ValueError("joint Plan A weights were fitted on a different indexed CUDA device")
    model.load_state_dict(payload["model"], strict=True)
    return payload


def validate_counterfactual_warm_start_artifact(track: str, checkpoint: Path, *,
        qualification_protocol: Path, expected_sources: dict,
        expected_prediction_mode: str | None = None):
    """Validate legacy one-case or versioned composite pairwise weights."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") == MULTICASE_COUNTERFACTUAL_WEIGHTS_SCHEMA:
        return validate_counterfactual_composite_artifact(track, checkpoint,
            qualification_protocol=qualification_protocol,
            expected_sources=expected_sources,
            expected_prediction_mode=expected_prediction_mode)
    mode = payload.get("prediction_mode")
    protocol, protocol_id, protocol_sha, protocol_path = _load_protocol(qualification_protocol)
    from .meco_counterfactual import (CONTINUATION_POLICY,
        PLAN_C_REPEAT_TARGET_SCHEMA, plan_c_repeat_protocol)
    valid_track = track in MECO_TRACKS and (
        track in ("A1", "A2", "A3") or track in ("C1", "C2"))
    repeated_plan_c = track in ("C1", "C2")
    expected_target = ("within_prefix_repeated_matched_stable_pairwise_preference_from_gross_horizon_return"
        if repeated_plan_c else "within_prefix_pairwise_preference_from_gross_horizon_return")
    expected_scoring = ("raw_U_node_plus_B_behavior; no P gradient; p_mode=p0"
        if track.startswith("A") else
        "raw_candidate_utility; route_and_risk_costs_excluded_for_gross_return_target")
    losses = payload.get("mean_loss_by_epoch")
    pairs = payload.get("non_tied_pair_count")
    ties = payload.get("tied_pair_count")
    if (not valid_track or payload.get("schema") != (
                COUNTERFACTUAL_PLAN_C_REPEATED_WEIGHTS_SCHEMA if repeated_plan_c else
                COUNTERFACTUAL_WEIGHTS_SCHEMA) or
            payload.get("schema_version") != (1 if repeated_plan_c else 2) or
            payload.get("artifact_kind") != ("weights_only_plan_c_repeated_pairwise_v1"
                if repeated_plan_c else "weights_only_counterfactual_pairwise_v2") or
            payload.get("track") != track or payload.get("model_schema_version") != 1 or
            payload.get("dataset_schema") != "meco-counterfactual-trajectory-v3" or
            payload.get("target_schema") != "meco-matched-prefix-gross-return-v3" or
            payload.get("records_have_prefix_replay_parity") is not True or
            payload.get("source_runtime_device_bound") is not True or
            payload.get("state_digest_schema") != "meco-full-environment-state-v2" or
            payload.get("label_provenance") != ("within-prefix matched repeated simulator gross-horizon returns; stable paired sign only"
                if repeated_plan_c else COUNTERFACTUAL_LABEL_PROVENANCE) or
            payload.get("target_semantics") != expected_target or
            payload.get("scoring_semantics") != expected_scoring or
            payload.get("plan_a_p_target", "missing") is not None or
            payload.get("plan_a_p_prior") != 0.0 or
            payload.get("plan_a_p_mode") != ("p0" if track.startswith("A") else None) or
            payload.get("candidate_raw_feature_count") != MECO_CANDIDATE_RAW_FEATURE_COUNT or
            payload.get("candidate_model_input_width") != MECO_CANDIDATE_MODEL_INPUT_WIDTH or
            payload.get("candidate_feature_names") != list(tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)) or
            not isinstance(payload.get("dataset_manifest_sha256"), str) or
            len(payload["dataset_manifest_sha256"]) != 64 or
            any(c not in "0123456789abcdef" for c in payload["dataset_manifest_sha256"]) or
            mode not in PREDICTION_MODES or
            payload.get("prediction_contract") != prediction_contract(mode) or
            (expected_prediction_mode is not None and mode != expected_prediction_mode) or
            payload.get("protocol_id") != protocol_id or
            payload.get("qualification_protocol_sha256") != protocol_sha or
            payload.get("qualification_protocol_file") != protocol_path.name or
            Path(payload.get("qualification_protocol_file", "")).name !=
                payload.get("qualification_protocol_file") or
            payload.get("split") != "train" or
            not isinstance(payload.get("scenario_id"), str) or
            not isinstance(payload.get("category"), str) or
            not isinstance(payload.get("seeds"), list) or
            not payload["seeds"] or
            any(type(seed) is not int or seed < 0 for seed in payload["seeds"]) or
            len(set(payload["seeds"])) != len(payload["seeds"]) or
            not set(payload["seeds"]) <= _protocol_train_seeds(
                protocol, payload["scenario_id"], payload["category"]) or
            payload.get("sources") != expected_sources or
            payload.get("source_identity_sha256") != _canonical_hash(expected_sources) or
            not isinstance(payload.get("dataset_sha256"), str) or
            len(payload["dataset_sha256"]) != 64 or
            any(c not in "0123456789abcdef" for c in payload["dataset_sha256"]) or
            payload.get("candidate_boundary") != CANDIDATE_BOUNDARY or
            payload.get("candidate_k") not in (8, 16, 32) or
            type(payload.get("max_plan_c_descriptors_per_prefix")) is not int or
            payload["max_plan_c_descriptors_per_prefix"] < 2 or
            type(payload.get("max_plan_a_pairs_per_prefix")) is not int or
            payload["max_plan_a_pairs_per_prefix"] < 2 or
            payload.get("route_cost_points_per_second") != ROUTE_COST_POINTS_PER_SECOND or
            payload.get("plan_a_support_order") != SUPPORT_ORDER or
            payload.get("plan_a_p_unlabelled_reason") !=
                "simulator executor does not execute P-modified routes" or
            type(payload.get("horizon_physics_ticks")) is not int or
            payload["horizon_physics_ticks"] < 1 or
            type(payload.get("max_macro_ticks")) is not int or
            payload["max_macro_ticks"] < 1 or
            not _finite_number(payload.get("gamma_per_macro")) or
            not 0 < payload["gamma_per_macro"] <= 1 or
            payload.get("discount_semantics") != COUNTERFACTUAL_DISCOUNT_SEMANTICS or
            payload.get("continuation_policy") != CONTINUATION_POLICY or
            (repeated_plan_c and (payload.get("plan_c_target_schema") !=
                PLAN_C_REPEAT_TARGET_SCHEMA or payload.get("plan_c_repeat_protocol") !=
                plan_c_repeat_protocol(protocol_id) or payload.get("plan_c_label_policy") !=
                "stable_same_strict_sign_all_paired_repeats_else_abstain" or
                payload.get("plan_c_repeat_variance_note") !=
                "R=2 is a bounded replication policy, not precise variance calibration" or
                payload.get("plan_c_stable_pair_count") != pairs or
                payload.get("plan_c_abstained_pair_count") != ties or
                payload.get("pair_count_semantics") !=
                    "non_tied_pair_count counts stable Plan C labels; tied_pair_count counts all abstentions")) or
            not isinstance(payload.get("collection_device"), dict) or
            payload.get("training_device") != payload.get("collection_device") or
            type(pairs) is not int or pairs < 1 or type(ties) is not int or ties < 0 or
            not isinstance(losses, list) or not losses or
            any(not _finite_number(loss) or loss < 0 for loss in losses) or
            type(payload.get("updates")) is not int or payload["updates"] < 1 or
            not _finite_number(payload.get("learning_rate")) or payload["learning_rate"] <= 0 or
            payload.get("optimizer_state_included") is not False or
            payload.get("resume_compatible") is not False or
            not isinstance(payload.get("model"), dict)):
        raise ValueError("v2 counterfactual warm-start artifact identity/schema mismatch")
    model = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("v2 counterfactual weights do not match the requested track") from exc
    return payload


def validate_counterfactual_plan_a_value_artifact(track: str, checkpoint: Path, *,
        qualification_protocol: Path, expected_sources: dict,
        expected_prediction_mode: str | None = None):
    """Validate v3 direct Plan A value weights and their explicit P gap."""
    from .meco_counterfactual import CONTINUATION_POLICY

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mode = payload.get("prediction_mode")
    protocol, protocol_id, protocol_sha, protocol_path = _load_protocol(qualification_protocol)
    losses = payload.get("mean_value_loss_by_epoch")
    preferences = payload.get("mean_preference_loss_by_epoch")
    digest, manifest_digest = (payload.get("dataset_sha256"),
                               payload.get("dataset_manifest_sha256"))
    if (track not in ("A1", "A2", "A3") or
            payload.get("schema") != COUNTERFACTUAL_PLAN_A_VALUE_WEIGHTS_SCHEMA or
            payload.get("schema_version") != 3 or
            payload.get("artifact_kind") != "weights_only_counterfactual_plan_a_goal_value_v3" or
            payload.get("track") != track or payload.get("model_schema_version") != 1 or
            payload.get("dataset_schema") != "meco-counterfactual-trajectory-v3" or
            payload.get("target_schema") != "meco-matched-prefix-gross-return-v3" or
            payload.get("label_provenance") != PLAN_A_VALUE_LABEL_PROVENANCE or
            payload.get("target_semantics") !=
                "sampled_feasible_node_behavior_gross_horizon_return_regression" or
            payload.get("scoring_semantics") !=
                "raw_U_node_plus_B_behavior; direct gross_horizon_return value and pairwise losses" or
            payload.get("value_loss_semantics") !=
                "smooth_l1_raw_U_plus_B_to_gross_horizon_return_beta_1" or
            payload.get("preference_loss_semantics") !=
                "within_prefix_non_tied_gross_horizon_return_pairwise_softplus" or
            payload.get("plan_a_support_order") != SUPPORT_ORDER or
            payload.get("plan_a_support_coverage") !=
                "sampled_valid_node_behavior_descriptors_from_exact_replayed_full_support" or
            type(payload.get("plan_a_value_target_count")) is not int or
            payload["plan_a_value_target_count"] < 1 or
            payload.get("plan_a_p_target", "missing") is not None or
            payload.get("plan_a_p_prior") != 0.0 or
            payload.get("plan_a_p_mode") != "p0" or
            payload.get("plan_a_p_unlabelled_reason") !=
                "simulator executor does not execute P-modified routes" or
            payload.get("candidate_raw_feature_count") != MECO_CANDIDATE_RAW_FEATURE_COUNT or
            payload.get("candidate_model_input_width") != MECO_CANDIDATE_MODEL_INPUT_WIDTH or
            payload.get("candidate_feature_names") !=
                list(tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)) or
            mode not in PREDICTION_MODES or
            payload.get("prediction_contract") != prediction_contract(mode) or
            (expected_prediction_mode is not None and mode != expected_prediction_mode) or
            payload.get("protocol_id") != protocol_id or
            payload.get("qualification_protocol_sha256") != protocol_sha or
            payload.get("qualification_protocol_file") != protocol_path.name or
            Path(payload.get("qualification_protocol_file", "")).name !=
                payload.get("qualification_protocol_file") or
            payload.get("split") != "train" or
            not isinstance(payload.get("scenario_id"), str) or
            not isinstance(payload.get("category"), str) or
            not isinstance(payload.get("seeds"), list) or not payload["seeds"] or
            any(type(seed) is not int or seed < 0 for seed in payload["seeds"]) or
            len(set(payload["seeds"])) != len(payload["seeds"]) or
            not set(payload["seeds"]) <= _protocol_train_seeds(
                protocol, payload["scenario_id"], payload["category"]) or
            payload.get("sources") != expected_sources or
            payload.get("source_identity_sha256") != _canonical_hash(expected_sources) or
            not isinstance(digest, str) or len(digest) != 64 or
            any(c not in "0123456789abcdef" for c in digest) or
            not isinstance(manifest_digest, str) or len(manifest_digest) != 64 or
            any(c not in "0123456789abcdef" for c in manifest_digest) or
            payload.get("candidate_boundary") != CANDIDATE_BOUNDARY or
            payload.get("candidate_k") not in (8, 16, 32) or
            type(payload.get("max_plan_c_descriptors_per_prefix")) is not int or
            payload["max_plan_c_descriptors_per_prefix"] < 2 or
            type(payload.get("max_plan_a_pairs_per_prefix")) is not int or
            payload["max_plan_a_pairs_per_prefix"] < 2 or
            payload.get("route_cost_points_per_second") != ROUTE_COST_POINTS_PER_SECOND or
            payload.get("plan_a_support_order") != SUPPORT_ORDER or
            type(payload.get("horizon_physics_ticks")) is not int or
            payload["horizon_physics_ticks"] < 1 or
            type(payload.get("max_macro_ticks")) is not int or
            payload["max_macro_ticks"] < 1 or
            not _finite_number(payload.get("gamma_per_macro")) or
            not 0 < payload["gamma_per_macro"] <= 1 or
            payload.get("discount_semantics") != COUNTERFACTUAL_DISCOUNT_SEMANTICS or
            payload.get("continuation_policy") != CONTINUATION_POLICY or
            not isinstance(payload.get("collection_device"), dict) or
            payload.get("training_device") != payload.get("collection_device") or
            type(payload.get("non_tied_pair_count")) is not int or
            payload["non_tied_pair_count"] < 1 or
            type(payload.get("tied_pair_count")) is not int or
            payload["tied_pair_count"] < 0 or
            not isinstance(losses, list) or not losses or
            any(not _finite_number(loss) or loss < 0 for loss in losses) or
            not isinstance(preferences, list) or len(preferences) != len(losses) or
            any(not _finite_number(loss) or loss < 0 for loss in preferences) or
            type(payload.get("updates")) is not int or payload["updates"] < 1 or
            not _finite_number(payload.get("learning_rate")) or payload["learning_rate"] <= 0 or
            payload.get("optimizer_state_included") is not False or
            payload.get("resume_compatible") is not False or
            any(key in payload for key in ("optimizer", "optimizer_state", "rng_state", "cursor")) or
            not isinstance(payload.get("model"), dict)):
        raise ValueError("v3 Plan A counterfactual warm-start artifact identity/schema mismatch")
    model = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("v3 Plan A weights do not match the requested track") from exc
    if track == "A1":
        penalty_weight = model.pose_head.weight.reshape(8, 2, *model.pose_head.weight.shape[1:])[:, 1]
        penalty_bias = model.pose_head.bias.reshape(8, 2)[:, 1]
    elif track == "A2":
        penalty_weight, penalty_bias = model.decoder[-1].weight[1], model.decoder[-1].bias[1]
    else:
        penalty_weight, penalty_bias = model.node_head.weight[1], model.node_head.bias[1]
    if bool(penalty_weight.any()) or not bool((penalty_bias == -20.).all()):
        raise ValueError("v3 Plan A unlabelled penalty head must be initialized to zero")
    return payload


def validate_counterfactual_plan_a_p_artifact(track: str, checkpoint: Path, *,
        qualification_protocol: Path, expected_sources: dict,
        expected_prediction_mode: str | None = None):
    """Validate an executed-route P-only warm-start artifact."""
    from .meco_counterfactual import (PLAN_A_P_ROUTE_DATASET_SCHEMA,
        PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA, PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA,
        PLAN_A_P_ROUTE_ORACLE_VERSION)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mode = payload.get("prediction_mode")
    _, protocol_id, protocol_sha, protocol_path = _load_protocol(qualification_protocol)
    for key in ("dataset_sha256", "dataset_manifest_sha256",
                "source_counterfactual_dataset_sha256", "value_checkpoint_sha256"):
        value = payload.get(key)
        if (not isinstance(value, str) or len(value) != 64 or
                any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"Plan A P artifact has invalid {key}")
    losses = payload.get("mean_route_loss_by_epoch")
    oracle_v3 = payload.get("schema") == "meco-counterfactual-plan-a-p-oracle-warm-start-weights-v1"
    if (track not in ("A1", "A2", "A3") or
            payload.get("schema") != ("meco-counterfactual-plan-a-p-oracle-warm-start-weights-v1"
                if oracle_v3 else COUNTERFACTUAL_PLAN_A_P_WEIGHTS_SCHEMA) or
            payload.get("schema_version") != (2 if oracle_v3 else 1) or
            payload.get("artifact_kind") != ("weights_only_counterfactual_plan_a_p_oracle_v1"
                if oracle_v3 else "weights_only_counterfactual_plan_a_p_route_v1") or
            payload.get("track") != track or payload.get("model_schema_version") != 1 or
            payload.get("dataset_schema") != (PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA if oracle_v3 else PLAN_A_P_ROUTE_DATASET_SCHEMA) or
            payload.get("target_schema") != (PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA if oracle_v3 else "meco-executed-route-gross-return-ranking-v1") or
            payload.get("label_provenance") != ("strict replay and independent recomputation of base bump plus effective P grid path integrals; sampled executed paths only" if oracle_v3 else
                "strict replay of simulator returns for model-independent P0 and seeded weighted AD* route interventions") or
            payload.get("target_semantics") != ("within_prefix_non_tied_sampled_executed_path_cost_preference_lower_is_better" if oracle_v3 else
                "within_prefix_executed_route_pairwise_preference_from_gross_horizon_return") or
            payload.get("loss_semantics") != ("softplus_predicted_total_route_cost_preference; total=independently_replayed_base_bump_grid_integral+learned_scaled_P_integral; independent oracle total-cost labels; plus mean_P_regularization" if oracle_v3 else
                "softplus_predicted_integrated_P_cost_preference_plus_mean_P_regularization") or
            (oracle_v3 and payload.get("p_oracle_version") != PLAN_A_P_ROUTE_ORACLE_VERSION) or
            payload.get("p_map_initialization") !=
                "softplus_inverse_of_0.02_on_penalty_output_channel" or
            payload.get("p_mode") != "learned" or
            payload.get("p_projection") != "mean_over_legal_heading_bins_xy_grid_v1" or
            payload.get("p_map_units") !=
                "dimensionless_additive_penalty_per_cardinal_grid_step" or
            payload.get("p_path_units") != "normalized_grid_edge_units" or
            not _finite_number(payload.get("p_cost_scale")) or
            payload["p_cost_scale"] <= 0 or
            payload.get("p_route_coverage_scope") !=
                "sampled_native_execution_path_cells; not exhaustive legal-grid coverage" or
            type(payload.get("p_route_pair_count")) is not int or
            payload["p_route_pair_count"] < 1 or
            type(payload.get("p_route_group_count")) is not int or
            payload["p_route_group_count"] < 1 or
            type(payload.get("p_route_visited_xy_cell_count")) is not int or
            payload["p_route_visited_xy_cell_count"] < 1 or
            type(payload.get("p_route_legal_xy_cell_count")) is not int or
            payload["p_route_legal_xy_cell_count"] < payload["p_route_visited_xy_cell_count"] or
            not _finite_number(payload.get("p_route_legal_xy_coverage_fraction")) or
            payload["p_route_legal_xy_coverage_fraction"] != (
                payload["p_route_visited_xy_cell_count"] /
                payload["p_route_legal_xy_cell_count"]) or
            not _finite_number(payload.get("p_route_minimum_coverage_fraction")) or
            not 0 < payload["p_route_minimum_coverage_fraction"] <= 1 or
            payload["p_route_legal_xy_coverage_fraction"] <
                payload["p_route_minimum_coverage_fraction"] or
            mode not in PREDICTION_MODES or
            payload.get("prediction_contract") != prediction_contract(mode) or
            (expected_prediction_mode is not None and mode != expected_prediction_mode) or
            payload.get("protocol_id") != protocol_id or
            payload.get("qualification_protocol_sha256") != protocol_sha or
            payload.get("qualification_protocol_file") != protocol_path.name or
            payload.get("split") != "train" or
            not isinstance(payload.get("scenario_id"), str) or
            not isinstance(payload.get("category"), str) or
            not isinstance(payload.get("seeds"), list) or not payload["seeds"] or
            any(type(seed) is not int for seed in payload["seeds"]) or
            len(set(payload["seeds"])) != len(payload["seeds"]) or
            not set(payload["seeds"]) <= _protocol_train_seeds(
                _load_protocol(qualification_protocol)[0],
                payload["scenario_id"], payload["category"]) or
            payload.get("sources") != expected_sources or
            payload.get("source_identity_sha256") != _canonical_hash(expected_sources) or
            payload.get("collection_device") != payload.get("training_device") or
            payload.get("continuation_policy") !=
                "curriculum_scenarios.teacher_actions; deterministic sensed-state policy" or
            payload.get("optimizer_state_included") is not False or
            payload.get("resume_compatible") is not False or
            type(payload.get("updates")) is not int or payload["updates"] < 1 or
            not _finite_number(payload.get("learning_rate")) or payload["learning_rate"] <= 0 or
            not _finite_number(payload.get("route_margin")) or payload["route_margin"] <= 0 or
            not _finite_number(payload.get("map_regularization")) or payload["map_regularization"] < 0 or
            not isinstance(losses, list) or len(losses) != payload.get("updates") or
            any(not _finite_number(value) or value < 0 for value in losses) or
            not isinstance(payload.get("model"), dict)):
        raise ValueError("Plan A P warm-start artifact identity/schema mismatch")
    model = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("Plan A P weights do not match the requested track") from exc
    layer, mask = _plan_a_p_output_parameters(model, track)
    selected = (layer.weight.detach() * mask.reshape(-1,
        *([1] * (layer.weight.ndim - 1)))).abs().sum()
    if track == "A1":
        biases = layer.bias.detach()[1::2]
    else:
        biases = layer.bias.detach()[1:2]
    if not bool(torch.isfinite(selected)) or not bool(torch.isfinite(biases).all()) or not bool(
            (selected > 0) or (biases > -20).any()):
        raise ValueError("Plan A P artifact contains no trained penalty channel")
    return payload


def load_counterfactual_warm_start(track: str, model, checkpoint: Path, *,
        qualification_protocol: Path, expected_sources: dict,
        expected_prediction_mode: str | None = None):
    """Load validated pairwise weights only; optimizer/RNG/cursor stay fresh."""
    payload = validate_counterfactual_warm_start_artifact(track, checkpoint,
        qualification_protocol=qualification_protocol, expected_sources=expected_sources,
        expected_prediction_mode=expected_prediction_mode)
    model.load_state_dict(payload["model"], strict=True)
    return payload


def _read_and_replay_plan_a_legal_node_dataset(dataset, source_dataset, qualification_protocol, device, track):
    """Fail-closed validator: regenerate all labels from strict source replays."""
    from .meco_counterfactual import (collect_plan_a_legal_nodes_v1,
        PLAN_A_LEGAL_NODE_DATASET_SCHEMA, PLAN_A_LEGAL_NODE_TARGET_SCHEMA,
        PLAN_A_U_ORACLE_VERSION, PLAN_A_U_ORACLE_SEMANTICS, CONTINUATION_POLICY,
        _hash, replay_counterfactual_branch)
    from .curriculum_training import source_identity
    import hashlib, tempfile

    dataset, source_dataset = Path(dataset).resolve(), Path(source_dataset).resolve()
    manifest_path = dataset.with_suffix(dataset.suffix + ".manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    data_bytes = dataset.read_bytes()
    digest = hashlib.sha256(data_bytes).hexdigest()
    sources = source_identity()
    if (manifest.get("schema") != PLAN_A_LEGAL_NODE_DATASET_SCHEMA or
            manifest.get("target_schema") != PLAN_A_LEGAL_NODE_TARGET_SCHEMA or
            manifest.get("u_oracle_version") != PLAN_A_U_ORACLE_VERSION or
            manifest.get("u_oracle_semantics") != PLAN_A_U_ORACLE_SEMANTICS or
            manifest.get("continuation_policy") != CONTINUATION_POLICY or
            manifest.get("split") != "train" or manifest.get("sources") != sources or
            manifest.get("source_identity_sha256") != _canonical_hash(sources) or
            manifest.get("dataset_sha256") != digest):
        raise ValueError("legal-node dataset schema, source identity, or bytes mismatch")
    source_rows, source_sha, source_manifest_sha, _, protocol_id, protocol_sha, source_manifest = \
        _read_counterfactual_dataset(source_dataset,
            qualification_protocol=Path(qualification_protocol), expected_sources=sources)
    if (manifest.get("source_dataset_sha256") != source_sha or
            manifest.get("source_manifest_sha256") != source_manifest_sha or
            manifest.get("protocol_id") != protocol_id or
            manifest.get("qualification_protocol_sha256") != protocol_sha or
            manifest.get("device") != source_manifest.get("device")):
        raise ValueError("legal-node source dataset/protocol/device binding mismatch")
    expert_goal_source = manifest.get("expert_goal_source")
    expert_goal_path = None
    if expert_goal_source is not None:
        if not isinstance(expert_goal_source, str) or Path(expert_goal_source).is_absolute():
            raise ValueError("legal-node expert-goal source path is invalid")
        expert_goal_path = (dataset.parent / expert_goal_source).resolve()
        expert_goal_bytes = expert_goal_path.read_bytes()
        expert_goal_manifest_bytes = expert_goal_path.with_suffix(
            expert_goal_path.suffix + ".manifest.json").read_bytes()
        if (hashlib.sha256(expert_goal_bytes).hexdigest() != manifest.get("expert_goal_source_sha256") or
                hashlib.sha256(expert_goal_manifest_bytes).hexdigest() !=
                    manifest.get("expert_goal_source_manifest_sha256")):
            raise ValueError("legal-node expert-goal source bytes changed")
    elif (manifest.get("expert_goal_source_sha256") is not None or
            manifest.get("expert_goal_source_manifest_sha256") is not None):
        raise ValueError("legal-node expert-goal source provenance is incomplete")
    with tempfile.TemporaryDirectory(prefix="meco-legal-node-replay-") as tmp:
        replayed_path = Path(tmp) / "replayed.jsonl"
        collect_plan_a_legal_nodes_v1(replayed_path, dataset=source_dataset,
            qualification_protocol=Path(qualification_protocol), device=device,
            expert_goal_source=expert_goal_path)
        if replayed_path.read_bytes() != data_bytes:
            raise ValueError("legal-node labels differ from fresh deterministic replay")
    legal_rows = [json.loads(line) for line in data_bytes.decode().splitlines() if line]
    if len(legal_rows) != manifest.get("record_count") or not legal_rows:
        raise ValueError("legal-node row count mismatch")
    source_by_key = {(r["scenario_id"], r["category"], r["seed"], r["decision_index"]): r
        for r in source_rows}
    samples = []
    from .meco_track_training import _graph_inputs
    for row in legal_rows:
        key = (row["scenario_id"], row["category"], row["seed"], row["decision_index"])
        record = source_by_key.get(key)
        if record is None:
            raise ValueError("legal-node row does not map to a frozen source branch")
        replay = replay_counterfactual_branch(record, source_manifest,
            require_hip_plan_a=True)
        try:
            env, graph, support = replay["env"], replay["graph"], replay["support"]
            if row.get("support_digest") != support.support_digest:
                raise ValueError("legal-node support digest differs on consumer replay")
            observation = replay["observation"]
            raster = torch.tensor(observation.raster, dtype=torch.float32, device=device).reshape(
                1, *observation.raster_shape)
            globals_ = torch.tensor(observation.globals, dtype=torch.float32, device=device).reshape(1, 7)
            if track is None:
                samples.append({"row": row, "support": support, "graph": graph})
            else:
                graph_inputs = _graph_inputs(track, graph, env, raster) or {}
                graph_inputs = {k: v.to(device) for k, v in graph_inputs.items()}
                samples.append({"row": row, "support": support, "graph": graph,
                    "graph_inputs": graph_inputs, "raster": raster, "globals": globals_,
                    "field_length": float(env.sim.field_length), "field_width": float(env.sim.field_width)})
        finally:
            close = getattr(replay["env"], "close", None)
            if callable(close): close()
    return samples, manifest, digest, hashlib.sha256(manifest_bytes).hexdigest()


def warm_start_plan_a_legal_nodes_v1(track: str, dataset: Path, source_dataset: Path,
        output: Path, *, qualification_protocol: Path, updates=10,
        learning_rate=1e-4, device=None):
    """Fit U regression plus centered feasible B contrasts from legal-node labels."""
    if track not in ("A1", "A2", "A3") or type(updates) is not int or updates < 1:
        raise ValueError("legal-node warm start requires A1/A2/A3 and positive updates")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be positive and finite")
    _prepare_native_gpu_runtime(device)
    from .meco_device import require_indexed_cuda_device
    device = str(require_indexed_cuda_device(device, purpose="Plan A legal-node fitting"))
    samples, manifest, data_sha, manifest_sha = _read_and_replay_plan_a_legal_node_dataset(
        dataset, source_dataset, qualification_protocol, device, track)
    if manifest["device"]["device"] != device:
        raise ValueError("legal-node fitter device must match frozen collection device")
    model = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    from .meco_plan_a_policy import gather_pose_behavior_logits
    value_losses, preference_losses = [], []
    node_count = sum(sum(item.get("status") == "available" for item in sample["row"]["node_u_targets"])
        for sample in samples)
    preference_count = sum(sum(p["status"] == "comparable" for p in sample["row"]["behavior_preferences"])
        for sample in samples)
    if node_count < 1:
        raise ValueError("legal-node dataset has no U targets")
    model.train()
    for _ in range(updates):
        losses_u, losses_b = [], []
        for sample in samples:
            graph, support, row = sample["graph"], sample["support"], sample["row"]
            raw = model(raster=sample["raster"], globals_=sample["globals"], **sample["graph_inputs"])
            logits = gather_pose_behavior_logits(track, raw, graph, sample["raster"], support,
                field_length=sample["field_length"], field_width=sample["field_width"])
            if track == "A1":
                utility = raw["utility"]
                h, w = sample["raster"].shape[-2:]
                col = (graph.xy[:, 0] / sample["field_length"] * w).floor().long().clamp(0, w-1)
                rr = (graph.xy[:, 1] / sample["field_width"] * h).floor().long().clamp(0, h-1)
                node_u = utility[0, graph.heading_bin, rr, col]
            else:
                node_u = raw["utility"][0]
            all_target_records = row["node_u_targets"]
            target_records = [item for item in all_target_records if
                item.get("status") == "available" and item.get("u_target") is not None]
            if target_records:
                node_ids = torch.tensor([x["node_id"] for x in target_records], device=device, dtype=torch.long)
                target_u = torch.tensor([x["u_target"] for x in target_records], device=device, dtype=node_u.dtype)
                node_weights = torch.tensor([x["node_inverse_probability_weight"] for x in target_records],
                    device=device, dtype=target_u.dtype)
                loss_u = (F.smooth_l1_loss(node_u[node_ids], target_u, beta=1., reduction="none") *
                    node_weights).sum() / node_weights.sum()
            else:
                loss_u = node_u.new_zeros(())
            terms_by_node = {}
            pair_index_by_behavior = {}
            for pair in row["pairs"]:
                if pair.get("status") == "available":
                    pair_index_by_behavior[(pair["node_id"], pair["behavior_id"])] = pair["pair_index"]
            centered_behavior_logits = {}
            for node in row["selected_node_ids"]:
                behavior_pairs = [(behavior, pair_index) for (node_id, behavior), pair_index
                    in pair_index_by_behavior.items() if node_id == node]
                if behavior_pairs:
                    values = torch.stack([logits[pair_index] for _, pair_index in behavior_pairs])
                    centered = values - values.mean()
                    centered_behavior_logits.update({(node, behavior): centered[i]
                        for i, (behavior, _) in enumerate(behavior_pairs)})
            for pref in row["behavior_preferences"]:
                if pref["status"] != "comparable":
                    continue
                node = pref["node_id"]
                # Pick one valid pair for each compared behavior; U cancels in
                # the paired logits, leaving the centered behavior contrast.
                left = centered_behavior_logits[(node, pref["left_behavior_id"])]
                right = centered_behavior_logits[(node, pref["right_behavior_id"])]
                sign = 1. if pref["preferred_behavior_id"] == pref["left_behavior_id"] else -1.
                terms_by_node.setdefault(node, []).append((
                    F.softplus(-sign * (left - right)), 1.0))
            if terms_by_node:
                node_terms, node_term_weights = [], []
                target_by_node = {item["node_id"]: item for item in all_target_records}
                for node, values in terms_by_node.items():
                    weights = torch.tensor([weight for _, weight in values], device=device,
                        dtype=values[0][0].dtype)
                    node_terms.append((torch.stack([term for term, _ in values]) * weights).sum() / weights.sum())
                    node_term_weights.append(float(target_by_node[node]["node_inverse_probability_weight"]))
                weights_b = torch.tensor(node_term_weights, device=device, dtype=node_terms[0].dtype)
                loss_b = (torch.stack(node_terms) * weights_b).sum() / weights_b.sum()
            else:
                loss_b = loss_u.new_zeros(())
            if not target_records and not terms_by_node:
                continue
            loss = loss_u + loss_b
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            if target_records: losses_u.append(float(loss_u.detach().cpu()))
            if terms_by_node: losses_b.append(float(loss_b.detach().cpu()))
        value_losses.append(sum(losses_u)/len(losses_u) if losses_u else 0.)
        preference_losses.append(sum(losses_b)/len(losses_b) if losses_b else 0.)
    _initialize_unlabelled_plan_a_penalty_head(model, track)
    payload = {"schema": PLAN_A_LEGAL_NODE_WEIGHTS_SCHEMA, "schema_version": 1,
        "artifact_kind": "weights_only_plan_a_legal_node_u_b_v1", "track": track,
        "model_schema_version": 1, "model": model.state_dict(), "dataset_sha256": data_sha,
        "dataset_manifest_sha256": manifest_sha,
        "source_dataset_sha256": manifest["source_dataset_sha256"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "dataset_path": str(Path(dataset).resolve()),
        "source_dataset_path": str(Path(source_dataset).resolve()),
        "sources": manifest["sources"],
        "source_identity_sha256": manifest["source_identity_sha256"],
        "protocol_id": manifest["protocol_id"], "protocol_sha256": manifest["qualification_protocol_sha256"],
        "target_schema": "meco-plan-a-legal-node-utility-and-feasible-preference-v1",
        "dataset_schema": "meco-counterfactual-plan-a-legal-node-v1",
        "u_oracle_version": manifest["u_oracle_version"],
        "u_oracle_semantics": manifest["u_oracle_semantics"],
        "u_label_provenance": "strictly replayed matched simulator branch returns; bounded sampled support; no global optimality claim",
        "source_dataset_schema": "meco-counterfactual-trajectory-v3",
        "candidate_raw_feature_count": MECO_CANDIDATE_RAW_FEATURE_COUNT,
        "candidate_model_input_width": MECO_CANDIDATE_MODEL_INPUT_WIDTH,
        "candidate_feature_names": list(tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)),
        "split": "train", "prediction_mode": manifest["prediction_mode"],
        "coverage_rule": manifest["coverage_rule"], "u_loss": "inverse-node-inclusion-weighted smooth_l1(U(node), equal-feasible-behavior mean of uniform descriptor-sample returns)",
        "b_loss": "pairwise_softplus on behavior logits centered over sampled feasible behaviors per node; uniformly sampled descriptor alternatives estimate behavior marginals; equal node weighting with node inclusion weights",
        "p_target": None, "p_prior": 0.0, "p_mode": "p0",
        "optimizer_state_included": False, "resume_compatible": False,
        "device": manifest["device"], "updates": updates, "learning_rate": learning_rate,
        "node_target_count": node_count, "comparable_behavior_pair_count": preference_count,
        "selector_digest": _canonical_hash([x["row"]["selection_digest"] for x in samples]),
        "behavior_descriptor_sampling": "one deterministic uniform hash-ranked descriptor per feasible node-behavior; descriptor inclusion probabilities/weights are provenance, while U and B estimands are expectations over uniform descriptor draws",
        "value_losses": value_losses, "preference_losses": preference_losses}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {"path": str(output), "schema": PLAN_A_LEGAL_NODE_WEIGHTS_SCHEMA,
        "node_targets": node_count, "comparable_pairs": preference_count}


def warm_start_plan_a_joint_ubp_v1(track: str, legal_dataset: Path,
        source_dataset: Path, p_dataset: Path, output: Path, *,
        qualification_protocol: Path, updates=10, learning_rate=1e-4,
        device=None, route_margin=0.02, map_regularization=0.001,
        minimum_coverage_fraction=0.01):
    """Fit one shared A model against replay-bound legal-node U/B and oracle P."""
    if (track not in ("A1", "A2", "A3") or type(updates) is not int or updates < 1 or
            not _finite_number(learning_rate) or learning_rate <= 0 or
            not _finite_number(route_margin) or route_margin <= 0 or
            not _finite_number(map_regularization) or map_regularization < 0 or
            not _finite_number(minimum_coverage_fraction) or
            not 0 < minimum_coverage_fraction <= 1):
        raise ValueError("invalid joint Plan A U+B+P warm-start configuration")
    _prepare_native_gpu_runtime(device)
    from .meco_device import require_indexed_cuda_device
    device = str(require_indexed_cuda_device(device, purpose="joint Plan A U+B+P fitting"))
    from .curriculum_training import source_identity
    from .meco_path_penalty import project_pose_penalty, integrate_frozen_routes
    from .meco_plan_a_policy import gather_pose_behavior_logits
    from .meco_counterfactual import (PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA,
        PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA, PLAN_A_P_ROUTE_ORACLE_VERSION,
        _device_identity)
    sources = source_identity()
    ub_samples, ub_manifest, ub_data_sha, ub_manifest_sha = (
        _read_and_replay_plan_a_legal_node_dataset(legal_dataset, source_dataset,
            qualification_protocol, device, track))
    p_rows, p_data_sha, p_manifest_sha, _, protocol_id, protocol_sha, p_manifest = (
        _read_plan_a_p_route_dataset(p_dataset,
            qualification_protocol=qualification_protocol, expected_sources=sources))
    if p_manifest.get("schema") != PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA:
        raise ValueError("joint U+B+P fitter requires independent oracle-v3 P-route labels")
    if (ub_manifest.get("source_dataset_sha256") !=
            p_manifest.get("source_counterfactual_dataset_sha256") or
            ub_manifest.get("source_manifest_sha256") !=
            p_manifest.get("source_counterfactual_manifest_sha256")):
        raise ValueError("joint U+B and P datasets must bind identical trajectory-v3 source bytes and manifest")
    if (ub_manifest.get("protocol_id") != protocol_id or
            ub_manifest.get("qualification_protocol_sha256") != protocol_sha or
            ub_manifest.get("sources") != p_manifest.get("sources") or
            ub_manifest.get("source_identity_sha256") != p_manifest.get("source_identity_sha256") or
            ub_manifest.get("prediction_mode") != p_manifest.get("prediction_mode") or
            ub_manifest.get("prediction_contract") != p_manifest.get("prediction_contract") or
            ub_manifest.get("split") != "train" or p_manifest.get("split") != "train" or
            ub_manifest.get("device") != p_manifest.get("device") or
            ub_manifest.get("device", {}).get("device") != device or
            p_manifest.get("device") != _device_identity(device)):
        raise ValueError("joint U+B/P protocol, source, mode, split, or device identities differ")
    p_samples = _materialize_plan_a_p_route_rows(p_rows, p_manifest,
        track=track, train_device=device)
    if not ub_samples or not p_samples:
        raise ValueError("joint Plan A datasets must both contain replayed training rows")
    model = make_meco_track(track,
        candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT).to(device)
    layer, channel_mask = _plan_a_p_output_parameters(model, track)
    weight_mask = channel_mask.reshape(-1, *([1] * (layer.weight.ndim - 1)))
    with torch.no_grad():
        layer.weight.mul_(weight_mask)
        initial_logit = torch.log(torch.expm1(torch.tensor(0.02,
            device=device, dtype=layer.bias.dtype)))
        if track == "A1":
            layer.bias[1::2] = initial_logit
        else:
            layer.bias[1] = initial_logit
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    p_cost_scale = float(p_manifest["p_intervention_scale"])
    if any(abs(sample["p_cost_scale"] - p_cost_scale) > 1e-12 for sample in p_samples):
        raise ValueError("replayed P rows disagree on the frozen P scale")
    legal_mask = p_samples[0]["legal_xy_mask"]
    if any(not torch.equal(s["legal_xy_mask"], legal_mask) for s in p_samples):
        raise ValueError("P dataset mixes legal XY grid geometries")
    legal_cell_count = int(legal_mask.sum().item())
    visited_cells = set()
    counts = {"u": 0, "b": 0, "p": 0}
    history = []
    model.train()
    steps_per_epoch = max(len(ub_samples), len(p_samples))
    for epoch in range(updates):
        epoch_history = []
        for step in range(steps_per_epoch):
            ub = ub_samples[step % len(ub_samples)]
            ps = p_samples[step % len(p_samples)]
            raw = model(raster=ub["raster"], globals_=ub["globals"], **ub["graph_inputs"])
            graph, support, row = ub["graph"], ub["support"], ub["row"]
            logits = gather_pose_behavior_logits(track, raw, graph, ub["raster"], support,
                field_length=ub["field_length"], field_width=ub["field_width"])
            if track == "A1":
                h, w = ub["raster"].shape[-2:]
                col = (graph.xy[:, 0] / ub["field_length"] * w).floor().long().clamp(0, w - 1)
                rr = (graph.xy[:, 1] / ub["field_width"] * h).floor().long().clamp(0, h - 1)
                node_u = raw["utility"][0, graph.heading_bin, rr, col]
            else:
                node_u = raw["utility"][0]
            targets = [x for x in row["node_u_targets"]
                if x.get("status") == "available" and x.get("u_target") is not None]
            if targets:
                ids = torch.tensor([x["node_id"] for x in targets], device=device, dtype=torch.long)
                target_u = torch.tensor([x["u_target"] for x in targets], device=device, dtype=node_u.dtype)
                iw = torch.tensor([x["node_inverse_probability_weight"] for x in targets],
                    device=device, dtype=node_u.dtype)
                loss_u = (F.smooth_l1_loss(node_u[ids], target_u, beta=1., reduction="none") * iw).sum() / iw.sum()
                counts["u"] += len(targets)
            else:
                loss_u = node_u.new_zeros(())
            pair_by_behavior = {}
            for pair in row["pairs"]:
                if pair.get("status") == "available":
                    pair_by_behavior[(pair["node_id"], pair["behavior_id"])] = pair["pair_index"]
            centered = {}
            for node in row["selected_node_ids"]:
                vals = [(behavior, idx) for (nid, behavior), idx in pair_by_behavior.items() if nid == node]
                if vals:
                    scores = torch.stack([logits[idx] for _, idx in vals])
                    scores = scores - scores.mean()
                    centered.update({(node, behavior): scores[i] for i, (behavior, _) in enumerate(vals)})
            by_node = {}
            target_by_node = {x["node_id"]: x for x in row["node_u_targets"]}
            for pref in row["behavior_preferences"]:
                if pref.get("status") != "comparable":
                    continue
                node = pref["node_id"]
                left, right = (node, pref["left_behavior_id"]), (node, pref["right_behavior_id"])
                if left not in centered or right not in centered or node not in target_by_node:
                    continue
                sign = 1. if pref["preferred_behavior_id"] == pref["left_behavior_id"] else -1.
                by_node.setdefault(node, []).append(F.softplus(-sign * (centered[left] - centered[right])))
            if by_node:
                node_terms, node_weights = [], []
                for node, terms in by_node.items():
                    node_terms.append(torch.stack(terms).mean())
                    node_weights.append(float(target_by_node[node]["node_inverse_probability_weight"]))
                bw = torch.tensor(node_weights, device=device, dtype=node_terms[0].dtype)
                loss_b = (torch.stack(node_terms) * bw).sum() / bw.sum()
                counts["b"] += sum(len(x) for x in by_node.values())
            else:
                loss_b = loss_u.new_zeros(())

            raw_p = model(raster=ps["raster"], globals_=ps["globals"], **ps["graph"])
            pmap = project_pose_penalty(track, raw_p["penalty"], ps["graph_object"].lookup,
                ps["planner"], ps["field_length"], ps["field_width"])
            route_terms = []
            for outcomes in ps["route_groups"]:
                unique = {}
                for outcome in outcomes:
                    native = outcome["native_route"]
                    unique.setdefault(native["route_trace_sha256"], outcome)
                    for snap in native["route_trace"]:
                        points = snap["route_points_m"]
                        for point in points:
                            ix = min(legal_mask.shape[0]-1,
                                max(0, int(float(point[0]) / ps["resolution_m"])))
                            iy = min(legal_mask.shape[1]-1,
                                max(0, int(float(point[1]) / ps["resolution_m"])))
                            if bool(legal_mask[ix, iy]):
                                visited_cells.add((ix, iy))
                        for first, second in zip(points, points[1:]):
                            dx, dy = float(second[0])-float(first[0]), float(second[1])-float(first[1])
                            n = max(1, math.ceil(math.hypot(dx, dy) / (ps["resolution_m"] * .5)))
                            for i in range(n + 1):
                                f = i / n
                                ix = min(legal_mask.shape[0]-1, max(0, int((float(first[0])+f*dx)/ps["resolution_m"])))
                                iy = min(legal_mask.shape[1]-1, max(0, int((float(first[1])+f*dy)/ps["resolution_m"])))
                                if bool(legal_mask[ix, iy]):
                                    visited_cells.add((ix, iy))
                variants = list(unique.values())
                for i, left in enumerate(variants):
                    for right in variants[i+1:]:
                        lc = left["native_route"]["independent_cost_oracle"]["route_cost_grid_edges"]
                        rc = right["native_route"]["independent_cost_oracle"]["route_cost_grid_edges"]
                        if lc == rc:
                            continue
                        better, worse = (left, right) if lc < rc else (right, left)
                        costs = []
                        for outcome in (better, worse):
                            records = [{"path_points_m": s["route_points_m"]}
                                for s in outcome["native_route"]["route_trace"]]
                            learned = integrate_frozen_routes(pmap, records,
                                ps["resolution_m"], map_scale=p_cost_scale,
                                points_per_grid_edge=1.).mean()
                            base = outcome.get("_replayed_base_route_cost_grid_edges")
                            if not _finite_number(base) or base < 0:
                                raise ValueError("oracle P route lacks replayed base cost")
                            costs.append(learned + float(base))
                        route_terms.append(F.softplus(costs[0] - costs[1] + route_margin))
            if route_terms:
                loss_p = torch.stack(route_terms).mean() + map_regularization * pmap.mean()
                counts["p"] += len(route_terms)
            else:
                loss_p = raw_p["penalty"].new_zeros(())
            if not targets and not by_node and not route_terms:
                continue
            loss = loss_u + loss_b + loss_p
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            epoch_history.append({"u": float(loss_u.detach().cpu()) if targets else None,
                "b": float(loss_b.detach().cpu()) if by_node else None,
                "p": float(loss_p.detach().cpu()) if route_terms else None,
                "joint": float(loss.detach().cpu())})
        if not epoch_history:
            raise ValueError("joint epoch has no estimable U, B, or P supervision")
        history.append({key: (sum(item[key] for item in epoch_history if item[key] is not None) /
                sum(item[key] is not None for item in epoch_history)
                if key != "joint" and any(item[key] is not None for item in epoch_history)
                else (sum(item[key] for item in epoch_history) / len(epoch_history)
                    if key == "joint" else None))
            for key in ("u", "b", "p", "joint")})
    coverage = len(visited_cells) / max(1, legal_cell_count)
    if any(counts[key] < 1 for key in ("u", "b", "p")):
        raise ValueError("joint datasets require estimable U, B, and non-tied oracle P targets")
    if coverage < minimum_coverage_fraction:
        raise ValueError("P executed paths do not meet the configured legal-grid coverage floor")
    if source_identity() != sources:
        raise RuntimeError("source/runtime changed during joint Plan A fitting")
    from .meco_counterfactual import PLAN_A_LEGAL_NODE_DATASET_SCHEMA, PLAN_A_LEGAL_NODE_TARGET_SCHEMA
    _, _, _, protocol_path = _load_protocol(qualification_protocol)
    payload = {"schema": PLAN_A_JOINT_UBP_WEIGHTS_SCHEMA, "schema_version": 1,
        "artifact_kind": "weights_only_plan_a_joint_u_b_p_v1", "track": track,
        "model_schema_version": 1, "model": model.state_dict(),
        "legal_dataset_path": str(Path(legal_dataset).resolve()),
        "source_dataset_path": str(Path(source_dataset).resolve()),
        "p_dataset_path": str(Path(p_dataset).resolve()),
        "legal_dataset_sha256": ub_data_sha, "legal_manifest_sha256": ub_manifest_sha,
        "p_dataset_sha256": p_data_sha, "p_manifest_sha256": p_manifest_sha,
        "source_dataset_sha256": ub_manifest["source_dataset_sha256"],
        "source_manifest_sha256": ub_manifest["source_manifest_sha256"],
        "legal_dataset_schema": PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
        "legal_target_schema": PLAN_A_LEGAL_NODE_TARGET_SCHEMA,
        "p_dataset_schema": PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA,
        "p_target_schema": PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA,
        "p_oracle_version": PLAN_A_P_ROUTE_ORACLE_VERSION,
        "sources": sources, "source_identity_sha256": _canonical_hash(sources),
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "qualification_protocol_file": protocol_path.name, "split": "train",
        "prediction_mode": ub_manifest["prediction_mode"],
        "prediction_contract": ub_manifest["prediction_contract"],
        "candidate_raw_feature_count": MECO_CANDIDATE_RAW_FEATURE_COUNT,
        "candidate_model_input_width": MECO_CANDIDATE_MODEL_INPUT_WIDTH,
        "candidate_feature_names": list(tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)),
        "collection_device": ub_manifest["device"], "training_device": _device_identity(device),
        "loss_semantics": {"u": "inverse_node_inclusion_weighted_smooth_l1 on available replay-bound legal-node U targets",
            "b": "node inclusion weighted pairwise softplus on behavior logits centered over available feasible behaviors; unavailable/tied/uncertain excluded",
            "p": "independent oracle-v3 pairwise softplus on replayed base bump plus learned scaled P route integral, plus mean-P regularization",
            "joint": "L_U + L_B + L_P, one shared encoder/model and one optimizer step per aligned batch"},
        "task_loss_weights": {"u": 1.0, "b": 1.0, "p": 1.0},
        "target_masks": {"u": "status=available and finite u_target",
            "b": "status=comparable; both behavior logits available and centered within feasible node support",
            "p": "distinct route trace pairs with non-tied independent oracle-v3 costs"},
        "p_mode": "learned", "p_cost_scale": p_cost_scale,
        "p_projection": "mean_over_legal_heading_bins_xy_grid_v1",
        "p_map_units": "dimensionless_additive_penalty_per_cardinal_grid_step",
        "p_path_units": "normalized_grid_edge_units",
        "route_margin": route_margin, "map_regularization": map_regularization,
        "p_route_pair_count": counts["p"], "p_route_legal_xy_coverage_fraction": coverage,
        "p_route_minimum_coverage_fraction": minimum_coverage_fraction,
        "steps_per_epoch": steps_per_epoch,
        "batch_schedule": "each epoch visits every legal-node and P dataset row at least once; shorter dataset cycles deterministically",
        "updates": updates, "learning_rate": learning_rate,
        "joint_loss_history": history,
        "optimizer_state_included": False, "resume_compatible": False}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    validate_plan_a_joint_ubp_artifact(track, output,
        qualification_protocol=qualification_protocol, expected_sources=sources,
        expected_prediction_mode=ub_manifest["prediction_mode"])
    return {"path": str(output), "schema": PLAN_A_JOINT_UBP_WEIGHTS_SCHEMA,
        "u_target_count": counts["u"], "b_pair_count": counts["b"],
        "p_route_pair_count": counts["p"], "p_route_coverage": coverage,
        "joint_loss_history": history}


def load_plan_a_legal_node_weights(model, checkpoint: Path, *, track: str, device,
        qualification_protocol: Path, expected_sources: dict,
        expected_prediction_mode: str | None = None):
    """Load only validated legal-node weights on an explicit CUDA device."""
    from .meco_device import require_indexed_cuda_device
    selected = str(require_indexed_cuda_device(device, purpose="Plan A legal-node weights consumer"))
    validate_plan_a_legal_node_artifact(track, checkpoint,
        qualification_protocol=qualification_protocol,
        expected_sources=expected_sources,
        expected_prediction_mode=expected_prediction_mode)
    payload = torch.load(Path(checkpoint), map_location=selected, weights_only=True)
    if (not isinstance(payload, dict) or payload.get("schema") != PLAN_A_LEGAL_NODE_WEIGHTS_SCHEMA or
            payload.get("track") != track or payload.get("device", {}).get("device") != selected or
            payload.get("p_target", "missing") is not None or payload.get("p_prior") != 0.0 or
            not isinstance(payload.get("model"), dict)):
        raise ValueError("legal-node weights artifact identity or P-null contract mismatch")
    model.load_state_dict(payload["model"], strict=True)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--scenario", required=True)
    collect.add_argument("--category", required=True)
    collect.add_argument("--seeds", type=int, nargs="+", required=True)
    collect.add_argument("--qualification-protocol", type=Path, required=True)
    collect.add_argument("--device", required=True,
        help="explicit indexed CUDA/ROCm device, for example cuda:0")
    collect.add_argument("--horizon", type=int, default=160)
    collect.add_argument("--action-ticks", type=int, default=12)
    collect.add_argument("--candidate-k", type=int, choices=(8, 16, 32), default=16)
    collect.add_argument("--prediction-mode", choices=PREDICTION_MODES, default="cv")
    counterfactual = commands.add_parser("collect-counterfactual-v2")
    counterfactual.add_argument("--output", type=Path, required=True)
    counterfactual.add_argument("--scenario", required=True)
    counterfactual.add_argument("--category", required=True)
    counterfactual.add_argument("--seeds", type=int, nargs="+", required=True)
    counterfactual.add_argument("--qualification-protocol", type=Path, required=True)
    counterfactual.add_argument("--device", required=True,
        help="explicit indexed CUDA/ROCm device, for example cuda:0")
    counterfactual.add_argument("--horizon-ticks", type=int, default=160)
    counterfactual.add_argument("--action-ticks", type=int, default=12)
    counterfactual.add_argument("--candidate-k", type=int, choices=(8, 16, 32), default=16)
    counterfactual.add_argument("--prediction-mode", choices=PREDICTION_MODES, default="cv")
    counterfactual.add_argument("--max-c-candidates", type=int, default=8)
    counterfactual.add_argument("--max-a-pairs", type=int, default=8)
    counterfactual.add_argument("--gamma-per-macro", type=float, default=.993,
        help="discount applied once per decision macro, including a final partial macro")
    warm = commands.add_parser("warm-start")
    warm.add_argument("--track", choices=MECO_TRACKS, required=True)
    warm.add_argument("--dataset", type=Path, required=True)
    warm.add_argument("--output", type=Path, required=True)
    warm.add_argument("--qualification-protocol", type=Path, required=True)
    warm.add_argument("--updates", type=int, default=10)
    warm.add_argument("--learning-rate", type=float, default=1e-4)
    warm.add_argument("--device", required=True,
        help="explicit indexed CUDA/ROCm device, for example cuda:0")
    cf_warm = commands.add_parser("warm-start-counterfactual-v2")
    cf_warm.add_argument("--track", choices=MECO_TRACKS, required=True)
    cf_warm.add_argument("--dataset", type=Path, required=True)
    cf_warm.add_argument("--output", type=Path, required=True)
    cf_warm.add_argument("--qualification-protocol", type=Path, required=True)
    cf_warm.add_argument("--updates", type=int, default=10)
    cf_warm.add_argument("--learning-rate", type=float, default=1e-4)
    cf_warm.add_argument("--device", required=True,
        help="explicit indexed CUDA/ROCm device, for example cuda:0")
    plan_a_v3 = commands.add_parser("warm-start-plan-a-v3")
    plan_a_v3.add_argument("--track", choices=("A1", "A2", "A3"), required=True)
    plan_a_v3.add_argument("--dataset", type=Path, required=True)
    plan_a_v3.add_argument("--output", type=Path, required=True)
    plan_a_v3.add_argument("--qualification-protocol", type=Path, required=True)
    plan_a_v3.add_argument("--updates", type=int, default=10)
    plan_a_v3.add_argument("--learning-rate", type=float, default=1e-4)
    plan_a_v3.add_argument("--device", required=True,
        help="explicit indexed CUDA/ROCm device, for example cuda:0")
    legal_collect = commands.add_parser("collect-plan-a-legal-nodes-v1")
    legal_collect.add_argument("--dataset", type=Path, required=True,
        help="strictly replayable counterfactual v2 training dataset")
    legal_collect.add_argument("--output", type=Path, required=True)
    legal_collect.add_argument("--qualification-protocol", type=Path, required=True)
    legal_collect.add_argument("--expert-goal-source", type=Path,
        help="include each replay-bound explicit selected goal as a certainty unit")
    legal_collect.add_argument("--device", required=True,
        help="explicit CUDA/ROCm device matching the frozen source dataset")
    expert_goal_collect = commands.add_parser("collect-plan-a-expert-goal-source-v1")
    expert_goal_collect.add_argument("--dataset", type=Path, required=True,
        help="strictly replayable counterfactual trajectory-v3 dataset")
    expert_goal_collect.add_argument("--output", type=Path, required=True)
    expert_goal_collect.add_argument("--qualification-protocol", type=Path, required=True)
    expert_goal_collect.add_argument("--device", required=True,
        help="explicit indexed CUDA/ROCm device matching the frozen source dataset")
    expert_goal_warm = commands.add_parser("warm-start-plan-a-expert-goals-v1")
    expert_goal_warm.add_argument("--track", choices=("A1", "A2", "A3"), required=True)
    expert_goal_warm.add_argument("--dataset", type=Path, required=True,
        help="strict annotation JSONL with replay-bound manifest")
    expert_goal_warm.add_argument("--legal-node-dataset", type=Path, required=True)
    expert_goal_warm.add_argument("--source-dataset", type=Path, required=True,
        help="replay-bound explicit expert-goal source v1")
    expert_goal_warm.add_argument("--annotation-source", type=Path, required=True,
        help="immutable authority file whose SHA-256 is recorded in the manifest")
    expert_goal_warm.add_argument("--output", type=Path, required=True)
    expert_goal_warm.add_argument("--qualification-protocol", type=Path, required=True)
    expert_goal_warm.add_argument("--updates", type=int, default=10)
    expert_goal_warm.add_argument("--learning-rate", type=float, default=1e-4)
    expert_goal_warm.add_argument("--device", required=True,
        help="explicit indexed CUDA/ROCm device matching frozen data; CPU is not supported")
    preference_v2_warm = commands.add_parser("warm-start-plan-a-preferences-v2")
    preference_v2_warm.add_argument("--track", choices=("A1", "A2", "A3"), required=True)
    preference_v2_warm.add_argument("--dataset", type=Path, required=True)
    preference_v2_warm.add_argument("--legal-node-dataset", type=Path, required=True)
    preference_v2_warm.add_argument("--source-dataset", type=Path, required=True)
    preference_v2_warm.add_argument("--authority-json", type=Path, required=True)
    preference_v2_warm.add_argument("--output", type=Path, required=True)
    preference_v2_warm.add_argument("--qualification-protocol", type=Path, required=True)
    preference_v2_warm.add_argument("--updates", type=int, default=10)
    preference_v2_warm.add_argument("--learning-rate", type=float, default=1e-4)
    preference_v2_warm.add_argument("--device", required=True,
        help="explicit indexed CUDA/ROCm device matching frozen data; CPU is not supported")
    legal_warm = commands.add_parser("warm-start-plan-a-legal-nodes-v1")
    legal_warm.add_argument("--track", choices=("A1", "A2", "A3"), required=True)
    legal_warm.add_argument("--dataset", type=Path, required=True)
    legal_warm.add_argument("--source-dataset", type=Path, required=True)
    legal_warm.add_argument("--output", type=Path, required=True)
    legal_warm.add_argument("--qualification-protocol", type=Path, required=True)
    legal_warm.add_argument("--updates", type=int, default=10)
    legal_warm.add_argument("--learning-rate", type=float, default=1e-4)
    legal_warm.add_argument("--device", required=True,
        help="explicit indexed CUDA device; CPU is not supported")
    joint_warm = commands.add_parser("warm-start-plan-a-joint-ubp-v1")
    joint_warm.add_argument("--track", choices=("A1", "A2", "A3"), required=True)
    joint_warm.add_argument("--legal-dataset", type=Path, required=True)
    joint_warm.add_argument("--source-dataset", type=Path, required=True)
    joint_warm.add_argument("--p-dataset", type=Path, required=True,
        help="independent oracle-v3 executed-route dataset")
    joint_warm.add_argument("--output", type=Path, required=True)
    joint_warm.add_argument("--qualification-protocol", type=Path, required=True)
    joint_warm.add_argument("--updates", type=int, default=10)
    joint_warm.add_argument("--learning-rate", type=float, default=1e-4)
    joint_warm.add_argument("--device", required=True,
        help="explicit indexed CUDA device; CPU is not supported")
    joint_warm.add_argument("--route-margin", type=float, default=0.02)
    joint_warm.add_argument("--map-regularization", type=float, default=0.001)
    joint_warm.add_argument("--minimum-coverage-fraction", type=float, default=0.01)
    p_routes = commands.add_parser("collect-plan-a-p-routes-v1")
    p_routes.add_argument("--dataset", type=Path, required=True,
        help="strictly replayable counterfactual v2 training dataset")
    p_routes.add_argument("--output", type=Path, required=True)
    p_routes.add_argument("--qualification-protocol", type=Path, required=True)
    p_routes.add_argument("--p-cost-scale", type=float, default=1.0)
    p_routes.add_argument("--intervention-maps", type=int, default=3)
    p_routes.add_argument("--device", required=True,
        help="explicit indexed CUDA/ROCm device matching the source dataset")
    plan_a_p = commands.add_parser("warm-start-plan-a-p-v1")
    plan_a_p.add_argument("--track", choices=("A1", "A2", "A3"), required=True)
    plan_a_p.add_argument("--dataset", type=Path, required=True)
    plan_a_p.add_argument("--value-checkpoint", type=Path, required=True,
        help="approved v3 Plan A U+B weights used as the frozen base")
    plan_a_p.add_argument("--output", type=Path, required=True)
    plan_a_p.add_argument("--qualification-protocol", type=Path, required=True)
    plan_a_p.add_argument("--updates", type=int, default=10)
    plan_a_p.add_argument("--learning-rate", type=float, default=1e-4)
    plan_a_p.add_argument("--device", required=True,
        help="explicit indexed CUDA/ROCm device, for example cuda:0")
    plan_a_p.add_argument("--route-margin", type=float, default=0.02)
    plan_a_p.add_argument("--map-regularization", type=float, default=0.001)
    plan_a_p.add_argument("--minimum-coverage-fraction", type=float, default=0.01)
    p_routes_v3 = commands.add_parser("collect-plan-a-p-routes-v3")
    p_routes_v3.add_argument("--dataset", type=Path, required=True)
    p_routes_v3.add_argument("--output", type=Path, required=True)
    p_routes_v3.add_argument("--qualification-protocol", type=Path, required=True)
    p_routes_v3.add_argument("--p-cost-scale", type=float, default=1.0)
    p_routes_v3.add_argument("--intervention-maps", type=int, default=3)
    p_routes_v3.add_argument("--device", required=True,
        help="explicit indexed CUDA/ROCm device matching the source dataset")
    plan_a_p_v3 = commands.add_parser("warm-start-plan-a-p-v3")
    plan_a_p_v3.add_argument("--track", choices=("A1", "A2", "A3"), required=True)
    plan_a_p_v3.add_argument("--dataset", type=Path, required=True)
    plan_a_p_v3.add_argument("--value-checkpoint", type=Path, required=True)
    plan_a_p_v3.add_argument("--output", type=Path, required=True)
    plan_a_p_v3.add_argument("--qualification-protocol", type=Path, required=True)
    plan_a_p_v3.add_argument("--updates", type=int, default=10)
    plan_a_p_v3.add_argument("--learning-rate", type=float, default=1e-4)
    plan_a_p_v3.add_argument("--device", required=True,
        help="explicit CUDA device, for example cuda:0; CPU is not supported")
    plan_a_p_v3.add_argument("--route-margin", type=float, default=0.02)
    plan_a_p_v3.add_argument("--map-regularization", type=float, default=0.001)
    plan_a_p_v3.add_argument("--minimum-coverage-fraction", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.command == "collect":
        result = collect_dataset(args.output, scenario_id=args.scenario,
            category=args.category, seeds=args.seeds,
            qualification_protocol=args.qualification_protocol,
            device=args.device, horizon=args.horizon, action_ticks=args.action_ticks,
            candidate_k=args.candidate_k, prediction_mode=args.prediction_mode)
    elif args.command == "collect-counterfactual-v2":
        from .meco_counterfactual import collect_counterfactual_v2
        result = collect_counterfactual_v2(args.output,
            scenario_id=args.scenario, category=args.category, seeds=args.seeds,
            qualification_protocol=args.qualification_protocol, device=args.device,
            horizon_ticks=args.horizon_ticks, action_ticks=args.action_ticks,
            candidate_k=args.candidate_k, prediction_mode=args.prediction_mode,
            max_c_candidates=args.max_c_candidates, max_a_pairs=args.max_a_pairs,
            gamma_per_macro=args.gamma_per_macro)
    elif args.command == "warm-start":
        result = warm_start(args.track, args.dataset, args.output,
            qualification_protocol=args.qualification_protocol, updates=args.updates,
            learning_rate=args.learning_rate, device=args.device)
    elif args.command == "warm-start-plan-a-v3":
        result = warm_start_counterfactual_plan_a_v3(args.track, args.dataset, args.output,
            qualification_protocol=args.qualification_protocol, updates=args.updates,
            learning_rate=args.learning_rate, device=args.device)
    elif args.command == "collect-plan-a-legal-nodes-v1":
        from .meco_counterfactual import collect_plan_a_legal_nodes_v1
        result = collect_plan_a_legal_nodes_v1(args.output, dataset=args.dataset,
            qualification_protocol=args.qualification_protocol, device=args.device,
            expert_goal_source=args.expert_goal_source)
    elif args.command == "collect-plan-a-expert-goal-source-v1":
        from .meco_counterfactual import collect_plan_a_expert_goal_source_v1
        result = collect_plan_a_expert_goal_source_v1(args.output, dataset=args.dataset,
            qualification_protocol=args.qualification_protocol, device=args.device)
    elif args.command == "warm-start-plan-a-expert-goals-v1":
        result = warm_start_plan_a_expert_goals_v1(args.track, args.dataset,
            args.legal_node_dataset, args.source_dataset, args.annotation_source,
            args.output, qualification_protocol=args.qualification_protocol,
            updates=args.updates, learning_rate=args.learning_rate, device=args.device)
    elif args.command == "warm-start-plan-a-preferences-v2":
        result = warm_start_plan_a_preferences_v2(args.track, args.dataset,
            args.legal_node_dataset, args.source_dataset, args.authority_json,
            args.output, qualification_protocol=args.qualification_protocol,
            updates=args.updates, learning_rate=args.learning_rate, device=args.device)
    elif args.command == "warm-start-plan-a-legal-nodes-v1":
        result = warm_start_plan_a_legal_nodes_v1(args.track, args.dataset,
            args.source_dataset, args.output,
            qualification_protocol=args.qualification_protocol,
            updates=args.updates, learning_rate=args.learning_rate, device=args.device)
    elif args.command == "warm-start-plan-a-joint-ubp-v1":
        result = warm_start_plan_a_joint_ubp_v1(args.track, args.legal_dataset,
            args.source_dataset, args.p_dataset, args.output,
            qualification_protocol=args.qualification_protocol,
            updates=args.updates, learning_rate=args.learning_rate, device=args.device,
            route_margin=args.route_margin,
            map_regularization=args.map_regularization,
            minimum_coverage_fraction=args.minimum_coverage_fraction)
    elif args.command == "collect-plan-a-p-routes-v1":
        from .meco_counterfactual import collect_plan_a_p_routes_v1
        result = collect_plan_a_p_routes_v1(args.output, dataset=args.dataset,
            qualification_protocol=args.qualification_protocol,
            p_cost_scale=args.p_cost_scale,
            intervention_maps=args.intervention_maps, device=args.device)
    elif args.command == "warm-start-plan-a-p-v1":
        result = warm_start_counterfactual_plan_a_p_v1(args.track, args.dataset,
            args.value_checkpoint, args.output,
            qualification_protocol=args.qualification_protocol,
            updates=args.updates, learning_rate=args.learning_rate,
            device=args.device, route_margin=args.route_margin,
            map_regularization=args.map_regularization,
            minimum_coverage_fraction=args.minimum_coverage_fraction)
    elif args.command == "collect-plan-a-p-routes-v3":
        from .meco_counterfactual import collect_plan_a_p_routes_v3
        result = collect_plan_a_p_routes_v3(args.output, dataset=args.dataset,
            qualification_protocol=args.qualification_protocol,
            p_cost_scale=args.p_cost_scale,
            intervention_maps=args.intervention_maps, device=args.device)
    elif args.command == "warm-start-plan-a-p-v3":
        result = warm_start_counterfactual_plan_a_p_v1(args.track, args.dataset,
            args.value_checkpoint, args.output,
            qualification_protocol=args.qualification_protocol,
            updates=args.updates, learning_rate=args.learning_rate,
            device=args.device, route_margin=args.route_margin,
            map_regularization=args.map_regularization,
            minimum_coverage_fraction=args.minimum_coverage_fraction,
            oracle_labels=True)
    else:
        result = warm_start_counterfactual_v2(args.track, args.dataset, args.output,
            qualification_protocol=args.qualification_protocol, updates=args.updates,
            learning_rate=args.learning_rate, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
