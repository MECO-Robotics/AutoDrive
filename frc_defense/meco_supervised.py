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
from .meco_candidates import build_route_feasible_candidates
from .meco_candidates import CANDIDATE_BOUNDARY, EXTRA_FEATURES
from .meco_pose_graph import build_legal_pose_graph
from .meco_tracks import (MECO_TRACKS, MECO_CANDIDATE_RAW_FEATURE_COUNT,
    MECO_CANDIDATE_MODEL_INPUT_WIDTH,
    SAFE_ACTION_FEATURES, make_meco_track, score_candidates,
    PREDICTION_MODES, prediction_contract)
from .training_curriculum import get_scenario

DATASET_SCHEMA = "meco-supervised-trajectory-v1"
TARGET_SCHEMA = "meco-observed-action-target-v1"
WEIGHTS_SCHEMA = "meco-warm-start-weights-v1"
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
    counterfactual.add_argument("--gamma", type=float, default=.993)
    warm = commands.add_parser("warm-start")
    warm.add_argument("--track", choices=MECO_TRACKS, required=True)
    warm.add_argument("--dataset", type=Path, required=True)
    warm.add_argument("--output", type=Path, required=True)
    warm.add_argument("--qualification-protocol", type=Path, required=True)
    warm.add_argument("--updates", type=int, default=10)
    warm.add_argument("--learning-rate", type=float, default=1e-4)
    warm.add_argument("--device", default="cpu")
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
            gamma=args.gamma)
    else:
        result = warm_start(args.track, args.dataset, args.output,
            qualification_protocol=args.qualification_protocol, updates=args.updates,
            learning_rate=args.learning_rate, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
