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
COUNTERFACTUAL_LABEL_PROVENANCE = "within-prefix matched simulator gross_horizon_return pairwise comparisons"
COUNTERFACTUAL_DISCOUNT_SEMANTICS = (
    "one discount transition per macro decision, independent of macro duration; "
    "first branch reward exponent 0 and each continuation decision increments "
    "exponent once, including a final partial macro")
LABEL_PROVENANCE = "curriculum_scenarios.teacher_actions; sensed observations only"
ACTION_TO_BEHAVIOR = {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 6: 3, 5: 4}


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
                    seeds: list[int], qualification_protocol: Path, device: str = "cpu",
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
               updates: int = 10, learning_rate: float = 1e-4, device: str = "cpu"):
    """Fit observed action preferences and save model weights only."""
    if track not in MECO_TRACKS or type(updates) is not int or updates < 1:
        raise ValueError("invalid track or update count")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be positive and finite")
    from .curriculum_training import source_identity
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


def _read_counterfactual_dataset(path: Path, *, qualification_protocol: Path,
                                 expected_sources: dict):
    """Validate v2 bytes and frozen identities before any replay is attempted."""
    from .meco_counterfactual import (DATASET_SCHEMA as cf_dataset_schema,
        TARGET_SCHEMA as cf_target_schema, CONTINUATION_POLICY,
        _device_identity)

    path = Path(path).resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    protocol, protocol_id, protocol_sha, protocol_path = _load_protocol(qualification_protocol)
    device_metadata = manifest.get("device")
    if not isinstance(device_metadata, dict) or not isinstance(device_metadata.get("device"), str):
        raise ValueError("counterfactual dataset has no frozen device identity")
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
            manifest.get("state_digest_schema") != "meco-full-environment-state-v1" or
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
    for row in rows:
        if (not isinstance(row, dict) or row.get("schema") != cf_dataset_schema or
                row.get("target_schema") != cf_target_schema or
                row.get("scenario_id") != scenario_id or row.get("category") != category or
                row.get("protocol_id") != protocol_id or
                row.get("qualification_protocol_sha256") != protocol_sha or
                row.get("source_identity_sha256") != manifest["source_identity_sha256"] or
                row.get("device") != expected_device):
            raise ValueError("mixed schema, source, protocol, task, or device in v2 data")
        seed, index = row.get("seed"), row.get("decision_index")
        if type(seed) is not int or seed not in seeds or seed not in allowed_seeds or type(index) is not int or index < 0:
            raise ValueError("counterfactual row is outside the frozen train assignment")
        key = (scenario_id, category, seed, index)
        if key in keys:
            raise ValueError("duplicate counterfactual branch identity")
        keys.add(key)
        by_seed.setdefault(seed, []).append(row)
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
           list(range(len(seed_rows))) for seed_rows in by_seed.values()):
        raise ValueError("counterfactual decision indices are not contiguous per train seed")
    if set(by_seed) != set(seeds):
        raise ValueError("counterfactual dataset omitted one or more declared train seeds")
    return rows, data_sha, manifest_sha, protocol, protocol_id, protocol_sha, manifest


def _materialize_counterfactual_rows(rows, manifest, *, track, train_device):
    from .meco_counterfactual import replay_counterfactual_branch, _hash
    from .meco_track_training import _graph_inputs

    device = torch.device(train_device)
    result = []
    for row in rows:
        replay = replay_counterfactual_branch(row, manifest)
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
                result.append({"kind": "C", "raster": raster, "globals": globals_,
                    "table": table, "route": route, "risk": risk,
                    "pairs": pairs})
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    return result


def warm_start_counterfactual_v2(track: str, dataset: Path, output: Path, *,
        qualification_protocol: Path, updates: int = 10,
        learning_rate: float = 1e-4, device: str = "cpu"):
    """Fit only within-prefix gross-return pairwise preferences from v2 data."""
    if track not in MECO_TRACKS or type(updates) is not int or updates < 1:
        raise ValueError("invalid track or update count")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be positive and finite")
    from .curriculum_training import source_identity
    from .meco_counterfactual import _device_identity
    from .meco_plan_a_policy import gather_pose_behavior_logits

    sources = source_identity()
    rows, data_hash, manifest_hash, _, protocol_id, protocol_sha, manifest = _read_counterfactual_dataset(
        dataset, qualification_protocol=qualification_protocol,
        expected_sources=sources)
    if _device_identity(device) != manifest["device"]:
        raise ValueError("counterfactual replay/training device must match the frozen dataset runtime")
    if (track.startswith("A") and track not in ("A1", "A2", "A3") or
            track.startswith("C") and track not in ("C1", "C2")):
        raise ValueError("v2 counterfactual fitting supports only Plan A1-A3 or Plan C1-C2")
    rows = [row for row in rows if row["scenario_id"] == manifest["scenario_id"]]
    replayed = _materialize_counterfactual_rows(rows, manifest, track=track, train_device=device)
    tie_count = 0
    pair_count = 0
    for sample in replayed:
        values = sample["pairs"]
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
    payload = {"schema": COUNTERFACTUAL_WEIGHTS_SCHEMA, "schema_version": 2,
        "artifact_kind": "weights_only_counterfactual_pairwise_v2", "track": track,
        "model_schema_version": 1, "model": model.state_dict(),
        "dataset_sha256": data_hash, "dataset_schema": manifest["schema"],
        "dataset_manifest_sha256": manifest_hash,
        "target_schema": manifest["target_schema"],
        "label_provenance": COUNTERFACTUAL_LABEL_PROVENANCE,
        "target_semantics": "within_prefix_pairwise_preference_from_gross_horizon_return",
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
            "mean_loss_by_epoch": losses}


def validate_warm_start_artifact(track: str, checkpoint: Path, *,
                                 qualification_protocol: Path,
                                 expected_sources: dict,
                                 expected_prediction_mode: str | None = None):
    """Validate immutable metadata without restoring any training state."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
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


def load_warm_start(track: str, model, checkpoint: Path, *, qualification_protocol: Path,
                    expected_sources: dict,
                    expected_prediction_mode: str | None = None):
    """Load only compatible model weights; never restore optimizer, RNG, or cursor."""
    payload = validate_warm_start_artifact(track, checkpoint,
        qualification_protocol=qualification_protocol, expected_sources=expected_sources,
        expected_prediction_mode=expected_prediction_mode)
    model.load_state_dict(payload["model"], strict=True)
    return payload


def validate_counterfactual_warm_start_artifact(track: str, checkpoint: Path, *,
        qualification_protocol: Path, expected_sources: dict,
        expected_prediction_mode: str | None = None):
    """Validate a v2 pairwise artifact without accepting v1 imitation weights."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mode = payload.get("prediction_mode")
    protocol, protocol_id, protocol_sha, protocol_path = _load_protocol(qualification_protocol)
    from .meco_counterfactual import CONTINUATION_POLICY
    valid_track = track in MECO_TRACKS and (
        track in ("A1", "A2", "A3") or track in ("C1", "C2"))
    expected_target = "within_prefix_pairwise_preference_from_gross_horizon_return"
    expected_scoring = ("raw_U_node_plus_B_behavior; no P gradient; p_mode=p0"
        if track.startswith("A") else
        "raw_candidate_utility; route_and_risk_costs_excluded_for_gross_return_target")
    losses = payload.get("mean_loss_by_epoch")
    pairs = payload.get("non_tied_pair_count")
    ties = payload.get("tied_pair_count")
    if (not valid_track or payload.get("schema") != COUNTERFACTUAL_WEIGHTS_SCHEMA or
            payload.get("schema_version") != 2 or
            payload.get("artifact_kind") != "weights_only_counterfactual_pairwise_v2" or
            payload.get("track") != track or payload.get("model_schema_version") != 1 or
            payload.get("dataset_schema") != "meco-counterfactual-trajectory-v2" or
            payload.get("target_schema") != "meco-matched-prefix-gross-return-v2" or
            payload.get("label_provenance") != COUNTERFACTUAL_LABEL_PROVENANCE or
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


def load_counterfactual_warm_start(track: str, model, checkpoint: Path, *,
        qualification_protocol: Path, expected_sources: dict,
        expected_prediction_mode: str | None = None):
    """Load v2 pairwise model weights only; optimizer/RNG/cursor stay fresh."""
    payload = validate_counterfactual_warm_start_artifact(track, checkpoint,
        qualification_protocol=qualification_protocol, expected_sources=expected_sources,
        expected_prediction_mode=expected_prediction_mode)
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
    collect.add_argument("--device", default="cpu")
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
    counterfactual.add_argument("--device", default="cpu")
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
    warm.add_argument("--device", default="cpu")
    cf_warm = commands.add_parser("warm-start-counterfactual-v2")
    cf_warm.add_argument("--track", choices=MECO_TRACKS, required=True)
    cf_warm.add_argument("--dataset", type=Path, required=True)
    cf_warm.add_argument("--output", type=Path, required=True)
    cf_warm.add_argument("--qualification-protocol", type=Path, required=True)
    cf_warm.add_argument("--updates", type=int, default=10)
    cf_warm.add_argument("--learning-rate", type=float, default=1e-4)
    cf_warm.add_argument("--device", default="cpu")
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
    else:
        result = warm_start_counterfactual_v2(args.track, args.dataset, args.output,
            qualification_protocol=args.qualification_protocol, updates=args.updates,
            learning_rate=args.learning_rate, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
