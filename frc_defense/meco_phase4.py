"""Frozen Phase 4 adversarial/generalization factor definitions."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json


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
