"""Versioned provenance contract for serialized A/C track model inputs."""
from __future__ import annotations

import hashlib
import json


SCHEMA = "meco-model-input-provenance-v1"
TRACKS = ("A1", "A2", "A3", "C1", "C2")


_INPUTS = {
    "raster": {"shape": "[B,6,H,W]", "tracks": ["A1", "C2"],
        "origin": "six-channel observation snapshot", "observed": True,
        "boundary": "fuel/opponent channels are sensor tracks; ally channel is the shared policy-localization feed; no physical sim.pose is rasterized"},
    "globals_": {"shape": "[B,7]", "tracks": list(TRACKS),
        "origin": "observation snapshot global scalars", "observed": False,
        "boundary": "mixed: own pose is the policy localization estimate (integrated from idealized sim.velocity odometry); load uses simulator piece ownership; hub timer and remaining match time use simulator match state; pose is not physical sim.pose"},
    "hidden": {"shape": "[B,64]", "tracks": list(TRACKS),
        "origin": "recurrent model state from prior decisions", "observed": False,
        "boundary": "internal model state; reset/continued by episode and tracking availability rules"},
    "node_features": {"shape": "[B,N,9]", "tracks": ["A2", "A3"],
        "origin": "six raster values sampled at each legal pose plus normalized x/y and heading bin", "observed": False,
        "boundary": "raster portion inherits its sensor/telemetry boundary; pose geometry comes from planner-derived legal graph"},
    "edges": {"shape": "[B,N,10]", "tracks": ["A2", "A3"],
        "origin": "directed legal SE(2) graph transitions", "observed": False,
        "boundary": "derived from active planner grid and footprint-clearance checks, not sensor observations"},
    "edge_mask": {"shape": "[B,N,10] bool", "tracks": ["A2", "A3"],
        "origin": "legal SE(2) graph transition validity", "observed": False,
        "boundary": "planner-derived transition feasibility"},
    "region_ids": {"shape": "[B,N]", "tracks": ["A2", "A3"],
        "origin": "connected partition of legal graph", "observed": False,
        "boundary": "deterministic derived topology; no game-state observation"},
    "coarse_edges": {"shape": "[B,R,Dc]", "tracks": ["A2"],
        "origin": "inter-region legal graph transitions", "observed": False,
        "boundary": "planner-derived transition topology"},
    "coarse_edge_mask": {"shape": "[B,R,Dc] bool", "tracks": ["A2"],
        "origin": "inter-region transition validity", "observed": False,
        "boundary": "planner-derived transition feasibility"},
    "candidate_features": {"shape": "[B,C,40]", "tracks": ["C1", "C2"],
        "origin": "20 ordered candidate values followed by 20 missingness indicators", "observed": False,
        "boundary": "candidate table is built from safe actions, sensor tracks, public field geometry, planner routes, opponent sensor tracks, and controller commitment; simulator-owned action mask remains authoritative"},
}


_FEATURE_ORIGINS = (
    ("action_index", "safe action/candidate builder", "fixed simulator action slot; not simulator truth observation"),
    ("sensor_track_buffer_index", "sensor track buffer", "identity of a currently sensed fuel target"),
    ("goal_x_m", "safe action or route-feasible candidate builder", "sensed target geometry, fixed scoring/ferry geometry, observed opponent track, or current shared robot pose"),
    ("goal_y_m", "safe action or route-feasible candidate builder", "sensed target geometry, fixed scoring/ferry geometry, observed opponent track, or current shared robot pose"),
    ("heading_bin", "candidate goal geometry", "discretized heading to the selected goal or target"),
    ("behavior_index", "safe action mapping", "fixed action-to-behavior mapping"),
    ("field_length_m", "field configuration", "public static field dimension"),
    ("field_width_m", "field configuration", "public static field dimension"),
    ("target_distance_m", "safe action builder", "distance from shared robot pose to sensed or fixed target"),
    ("nearby_sensed_track_count", "sensor track buffer", "count of sensed fuel tracks within the configured radius"),
    ("track_confidence", "sensor track buffer", "perception confidence for the selected sensed target"),
    ("track_age_s", "sensor track buffer", "perception age for the selected sensed target"),
    ("opponent_distance_m", "opponent sensor track", "distance to sensed opponent; missing if no valid sensed opponent"),
    ("route_time_s", "route-feasible candidate builder", "route-derived ETA using planner path, robot speed/turn limits, and macro duration"),
    ("legal_node_index", "legal pose graph and route builder", "index into planner-derived legal pose support"),
    ("route_distance_m", "route evaluator", "planner route distance; missing if infeasible or unavailable"),
    ("risk_penalty_points", "opponent forecast", "bounded forecast penalty from sensed opponent track and route ETA; missing when no usable forecast"),
    ("target_heading_rad", "candidate goal geometry", "terminal heading derived from candidate goal/behavior constraints"),
    ("is_continue", "controller commitment state", "whether the row continues the revalidated prior commitment"),
    ("commitment_switch_cost", "controller commitment state", "fixed penalty indicator for switching away from an active commitment"),
)
_RAW_FEATURE_NAMES = tuple(item[0] for item in _FEATURE_ORIGINS)


def registry():
    """Return the versioned, JSON-safe registry in canonical declaration order."""
    return {"schema": SCHEMA, "tracks": list(TRACKS),
        "model_inputs": {name: dict(spec) for name, spec in _INPUTS.items()},
        "serialized_inputs_by_track": {
            "A1": ["raster", "globals_", "hidden"],
            "A2": ["node_features", "edges", "edge_mask", "region_ids", "globals_",
                "coarse_edges", "coarse_edge_mask", "hidden"],
            "A3": ["node_features", "edges", "edge_mask", "region_ids", "globals_", "hidden"],
            "C1": ["globals_", "candidate_features", "hidden"],
            "C2": ["raster", "globals_", "candidate_features", "hidden"]},
        "candidate_features": [{"name": name, "origin": origin,
            "boundary": boundary, "tracks": ["C1", "C2"], "observed": False}
            for name, origin, boundary in _FEATURE_ORIGINS],
        "candidate_model_feature_names": list(_RAW_FEATURE_NAMES) +
            [f"{name}__missing" for name in _RAW_FEATURE_NAMES],
        "candidate_schema_by_track": {"C1": list(_RAW_FEATURE_NAMES) +
            [f"{name}__missing" for name in _RAW_FEATURE_NAMES],
            "C2": list(_RAW_FEATURE_NAMES) +
            [f"{name}__missing" for name in _RAW_FEATURE_NAMES]},
        "forbidden_provenance_rule": "simulator truth and simulator-derived state must never be labeled observed"}


def canonical_bytes():
    return json.dumps(validate_registry(), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def input_contract_digest():
    return hashlib.sha256(canonical_bytes()).hexdigest()


def candidate_feature_names():
    return tuple(validate_registry()["candidate_model_feature_names"])


def validate_registry():
    """Fail closed if registry completeness/order diverges from runtime contracts."""
    from .meco_tracks import (MECO_INPUT_NAMES, SAFE_ACTION_FEATURES, EXTRA_FEATURES,
        MECO_CANDIDATE_RAW_FEATURE_COUNT, MECO_CANDIDATE_MODEL_INPUT_WIDTH)
    data = registry()
    input_names = set().union(*(set(names) for names in MECO_INPUT_NAMES.values()))
    expected_raw = tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)
    actual_raw = tuple(item["name"] for item in data["candidate_features"])
    if (tuple(data["tracks"]) != tuple(TRACKS) or
            set(data["serialized_inputs_by_track"]) != set(TRACKS) or
            set(data["model_inputs"]) != input_names or
            any(tuple(data["serialized_inputs_by_track"][track]) != tuple(MECO_INPUT_NAMES[track])
                for track in TRACKS)):
        raise ValueError("model input registry differs from serialized track input order")
    if (actual_raw != expected_raw or len(actual_raw) != MECO_CANDIDATE_RAW_FEATURE_COUNT or
            len(data["candidate_model_feature_names"]) != MECO_CANDIDATE_MODEL_INPUT_WIDTH or
            data["candidate_model_feature_names"] !=
                list(expected_raw) + [f"{name}__missing" for name in expected_raw]):
        raise ValueError("candidate feature registry differs from runtime feature order or width")
    for name, spec in data["model_inputs"].items():
        if (not isinstance(spec.get("origin"), str) or not spec["origin"] or
                not isinstance(spec.get("boundary"), str) or not spec["boundary"] or
                type(spec.get("observed")) is not bool or
                any(track not in TRACKS for track in spec.get("tracks", []))):
            raise ValueError(f"incomplete model input registry entry: {name}")
        provenance_text = f"{spec['origin']} {spec['boundary']}".lower()
        if (spec.get("observed") is True and
                any(token in provenance_text for token in ("simulator truth", "simulator-derived", "simulator derived"))):
            raise ValueError("simulator truth/state cannot be labeled observed")
    visible = {name: tuple(spec["tracks"]) for name, spec in data["model_inputs"].items()}
    expected_visibility = {name: tuple(track for track in TRACKS
        if name in MECO_INPUT_NAMES[track]) for name in input_names}
    if any(visible[name] != expected_visibility[name] for name in input_names):
        raise ValueError("model input track visibility differs from serialized use")
    for track in TRACKS:
        serialized = data["serialized_inputs_by_track"][track]
        if any(track not in visible[name] for name in serialized):
            raise ValueError(f"serialized input visibility is incomplete for {track}")
    if any(type(feature.get("observed")) is not bool or
            (feature["observed"] and any(token in
             f"{feature.get('origin', '')} {feature.get('boundary', '')}".lower()
             for token in ("simulator truth", "simulator-derived", "simulator derived")))
            for feature in data["candidate_features"]):
        raise ValueError("simulator truth/state cannot be labeled observed in candidate features")
    if (tuple(data["candidate_schema_by_track"].get("C1", ())) !=
            tuple(data["candidate_schema_by_track"].get("C2", ())) or
            tuple(data["candidate_schema_by_track"].get("C1", ())) !=
            tuple(data["candidate_model_feature_names"])):
        raise ValueError("C1/C2 candidate schema mismatch")
    return data


def validate_serialized_inputs(track: str, inputs):
    data = validate_registry()
    expected = tuple(data["serialized_inputs_by_track"].get(track, ()))
    if not expected or tuple(inputs) != expected:
        raise ValueError(f"{track} serialized model inputs do not match registered names/order")
    return input_contract_digest()
