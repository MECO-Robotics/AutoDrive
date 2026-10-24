"""Frozen Phase 4 adversarial/generalization factor definitions."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from .training_curriculum import SCENARIOS


PHASE4_FACTORS = {
    "feinting_opponent": {
        "kind": "feinting_opponent",
        "reversal_tick": 360,
        "telegraph_distance_m": 1.2,
        "tick_semantics": "20ms physics ticks from scenario reset",
    },
    "observation_delay": {
        "kind": "observation_delay",
        "delay_ticks": 6,
        "tick_semantics": "20ms physics ticks; historical policy observation buffer",
    },
    "independent_low_confidence": {
        "kind": "independent_low_confidence",
        "fuel_confidence_cap": 0.2,
        "modified_fields": ["track_confidence only"],
    },
    "abrupt_opponent_policy_change": {
        "kind": "abrupt_opponent_policy_change",
        "switch_tick": 360,
        "behavior_a": "orbit",
        "behavior_b": "direct_pursuit",
        "tick_semantics": "20ms physics ticks from scenario reset",
    },
}

PHASE4_TRAIN_EXPOSURE_VERSION = 1
PHASE4_TRAIN_SCENARIOS = ("offense_vs_scripted_pressure", "full_match_offense")


def phase4_train_protocol_id() -> str:
    return hashlib.sha256(json.dumps({
        "version": PHASE4_TRAIN_EXPOSURE_VERSION,
        "split": "train",
        "factors": PHASE4_FACTORS,
        "scenarios": list(PHASE4_TRAIN_SCENARIOS),
        "selection": "least-exposed factor/category/seed tuple after curriculum prerequisites",
        "held_out_splits": "rejected by training assignment validator",
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def phase4_train_categories(scenario_id: str) -> tuple[str, ...]:
    if scenario_id not in PHASE4_TRAIN_SCENARIOS:
        return ()
    return tuple(category for category in SCENARIOS[scenario_id].required_categories
        if any(part.startswith(tuple(f"level_{level}" for level in range(1, 6)))
               for part in category.split("/")))


def phase4_training_assignment_id(*, cycle: int, scenario_id: str,
                                  category: str, seed: int,
                                  training_seed_pool_id: str,
                                  factor: dict) -> str:
    factor = validate_phase4_factor(factor)
    payload = {"version": PHASE4_TRAIN_EXPOSURE_VERSION,
        "split": "train", "cycle": cycle, "scenario_id": scenario_id,
        "category": category, "seed": seed,
        "training_seed_pool_id": training_seed_pool_id,
        "factor": factor, "factor_sha256": phase4_factor_digest(factor),
        "protocol_id": phase4_train_protocol_id()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_phase4_training_assignment(assignment: dict | None) -> dict | None:
    """Validate opt-in Phase 4 training provenance; held-out splits fail closed."""
    if not isinstance(assignment, dict):
        raise ValueError("Phase 4 training assignment must be an object")
    factor = assignment.get("phase4_factor")
    phase_fields = ("phase4_training_version", "phase4_split", "phase4_factor_sha256",
                    "phase4_training_assignment_id", "phase4_training_protocol_id")
    if factor is None:
        if any(field in assignment for field in phase_fields):
            raise ValueError("incomplete Phase 4 training assignment")
        return None
    if assignment.get("phase4_split") != "train":
        raise ValueError("Phase 4 training factors are restricted to the train split")
    if assignment.get("phase4_training_version") != PHASE4_TRAIN_EXPOSURE_VERSION:
        raise ValueError("unsupported Phase 4 training assignment version")
    if any(type(assignment.get(name)) is not int or assignment[name] < 1
           for name in ("cycle",)):
        raise ValueError("Phase 4 training assignment has invalid cycle")
    factor = validate_phase4_factor(factor)
    scenario_id = assignment.get("scenario_id")
    category = assignment.get("category")
    if (scenario_id not in PHASE4_TRAIN_SCENARIOS
            or category not in phase4_train_categories(scenario_id)):
        raise ValueError("Phase 4 training factor is outside the registered train domain")
    seed = assignment.get("seed")
    pool_id = assignment.get("training_seed_pool_id")
    if type(seed) is not int or seed < 0 or not isinstance(pool_id, str) or not pool_id:
        raise ValueError("Phase 4 training assignment has invalid seed provenance")
    if assignment.get("phase4_factor_sha256") != phase4_factor_digest(factor):
        raise ValueError("Phase 4 training factor digest mismatch")
    if assignment.get("phase4_training_protocol_id") != phase4_train_protocol_id():
        raise ValueError("Phase 4 training protocol identity mismatch")
    expected_id = phase4_training_assignment_id(cycle=assignment.get("cycle"),
        scenario_id=scenario_id, category=category, seed=seed,
        training_seed_pool_id=pool_id, factor=factor)
    if assignment.get("phase4_training_assignment_id") != expected_id:
        raise ValueError("Phase 4 training assignment identity mismatch")
    return factor


def phase4_training_metadata(assignment: dict) -> dict | None:
    factor = validate_phase4_training_assignment(assignment)
    if factor is None:
        return None
    return {"version": assignment["phase4_training_version"],
        "split": assignment["phase4_split"], "factor": factor,
        "factor_sha256": assignment["phase4_factor_sha256"],
        "assignment_id": assignment["phase4_training_assignment_id"],
        "protocol_id": assignment["phase4_training_protocol_id"]}


def phase4_factor_config(name: str | None) -> dict | None:
    """Return a copy of a frozen factor spec, or None for its paired baseline."""
    if name is None:
        return None
    if name not in PHASE4_FACTORS:
        raise ValueError(f"unknown Phase 4 factor: {name}")
    return deepcopy(PHASE4_FACTORS[name])


def validate_phase4_factor(value: dict | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise ValueError("Phase 4 factor must be a registered factor object")
    kind = value["kind"]
    if kind not in PHASE4_FACTORS or value != PHASE4_FACTORS[kind]:
        raise ValueError("Phase 4 factor differs from its frozen registered contract")
    return deepcopy(value)


def phase4_factor_digest(value: dict | None) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"),
                           allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
