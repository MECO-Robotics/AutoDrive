"""Resumable task exposure scheduler for matched MECO training lanes.

This scheduler unlocks DAG stages from training exposure only. It does not
write curriculum promotion state; that still requires the separate qualified
evaluation gates in ``training_curriculum``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import random

from .training_curriculum import (CURRICULUM_VERSION, CurriculumState, SCENARIOS,
    _ancestors, eligible_scenarios, seed_partition, task_mixture)


SCHEDULER_VERSION = 1


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


@dataclass
class MecoTaskScheduler:
    master_seed: int
    exposure_chunks_per_scenario: int = 10
    state: dict | None = None

    def __post_init__(self):
        if type(self.master_seed) is not int or self.master_seed < 0:
            raise ValueError("master_seed must be nonnegative")
        if type(self.exposure_chunks_per_scenario) is not int or self.exposure_chunks_per_scenario < 1:
            raise ValueError("exposure_chunks_per_scenario must be positive")
        if self.state is None:
            self.state = {"schema_version": SCHEDULER_VERSION,
                "curriculum_version": CURRICULUM_VERSION,
                "master_seed": self.master_seed,
                "exposure_chunks_per_scenario": self.exposure_chunks_per_scenario,
                "completed_cycle": 0, "pending_assignment": None,
                "exposure_chunks": {}, "category_exposures": {},
                "recovery_pressure": {}, "history": []}
        self._validate_state()

    def _validate_state(self):
        state = self.state
        if (not isinstance(state, dict) or state.get("schema_version") != SCHEDULER_VERSION
                or state.get("curriculum_version") != CURRICULUM_VERSION
                or state.get("master_seed") != self.master_seed
                or state.get("exposure_chunks_per_scenario") != self.exposure_chunks_per_scenario):
            raise ValueError("MECO task scheduler state/configuration mismatch")
        for key in ("exposure_chunks", "category_exposures", "recovery_pressure"):
            if not isinstance(state.get(key), dict):
                raise ValueError(f"invalid scheduler {key}")
        if type(state.get("completed_cycle")) is not int or state["completed_cycle"] < 0:
            raise ValueError("invalid scheduler cycle")
        pending = state.get("pending_assignment")
        if pending is not None:
            self._validate_assignment(pending)

    def _validate_assignment(self, assignment):
        if not isinstance(assignment, dict):
            raise ValueError("invalid task assignment")
        scenario_id, category = assignment.get("scenario_id"), assignment.get("category")
        if scenario_id not in SCENARIOS or category not in SCENARIOS[scenario_id].required_categories:
            raise ValueError("task assignment is not registered in the curriculum DAG")
        if type(assignment.get("seed")) is not int or assignment["seed"] < 0:
            raise ValueError("invalid task seed")
        if type(assignment.get("cycle")) is not int or assignment["cycle"] < 1:
            raise ValueError("invalid task cycle")
        target = assignment.get("target_update")
        if target is not None and (type(target) is not int or target < 1):
            raise ValueError("invalid task update frontier")
        assignment_id = assignment.get("assignment_id")
        expected_id = _digest({"cycle": assignment["cycle"],
            "scenario_id": scenario_id, "category": category,
            "seed": assignment["seed"], "master_seed": self.master_seed})
        if assignment_id != expected_id:
            raise ValueError("task assignment identity is invalid")

    def _exposure_state(self):
        """Build an ephemeral prerequisite view from exposure, never promotion."""
        completed = {scenario for scenario in SCENARIOS
            if self.state["exposure_chunks"].get(scenario, 0)
               >= self.exposure_chunks_per_scenario
            and all(self.state["category_exposures"].get(
                f"{scenario}/{category}", 0) >= 1
                for category in SCENARIOS[scenario].required_categories)}
        # A later stage cannot unlock until every DAG ancestor has received
        # the configured exposure. This set is intentionally not persisted as
        # CurriculumState.qualified.
        unlocked = {scenario for scenario in completed
                    if _ancestors(scenario) <= completed}
        return CurriculumState(qualified=unlocked)

    def schedule(self, cycle: int, target_update: int | None = None):
        pending = self.state["pending_assignment"]
        if pending is not None:
            if pending["cycle"] != cycle:
                raise ValueError("pending task assignment belongs to a different cycle")
            if target_update is not None and pending.get("target_update") != target_update:
                raise ValueError("pending task assignment has a different update frontier")
            return dict(pending)
        if type(cycle) is not int or cycle != self.state["completed_cycle"] + 1:
            raise ValueError("task cycles must advance sequentially")
        curriculum_state = self._exposure_state()
        unlocked = set(eligible_scenarios(curriculum_state))
        if not unlocked:
            raise RuntimeError("curriculum DAG has no exposure-eligible task")
        recovery = {name for name, pressure in self.state["recovery_pressure"].items()
                    if pressure > 0 and name in unlocked}
        if recovery:
            current = set(recovery)
            for scenario_id in tuple(recovery):
                current.update(_ancestors(scenario_id))
            current &= unlocked
        else:
            incomplete = [name for name in SCENARIOS
                if name in unlocked and name not in curriculum_state.qualified]
            current = {incomplete[0] if incomplete else
                       next(name for name in SCENARIOS if name in unlocked)}
        upcoming = tuple(name for name in SCENARIOS
            if name in unlocked and name not in current
            and name not in curriculum_state.qualified)
        mixture = task_mixture(curriculum_state, tuple(sorted(current)), upcoming)
        rng = random.Random((self.master_seed << 32) ^ cycle)
        scenario_id = rng.choices(tuple(mixture), weights=tuple(mixture.values()), k=1)[0]
        categories = SCENARIOS[scenario_id].required_categories
        recovery_categories = [category for key, pressure in self.state["recovery_pressure"].items()
            if pressure > 0 for name, category in (key.split("/", 1),)
            if name == scenario_id and category in categories]
        category = (rng.choice(sorted(recovery_categories)) if recovery_categories
                    else rng.choice(categories))
        selected_recovery = scenario_id in recovery and category in recovery_categories
        cycle_seed = int.from_bytes(hashlib.sha256(
            f"{self.master_seed}/meco-task-cycle/{cycle}".encode()).digest()[:8], "big")
        seed = seed_partition(scenario_id, "train", 1,
                              master_seed=cycle_seed)[0]
        assignment = {"cycle": cycle, "scenario_id": scenario_id,
            "category": category, "seed": seed,
            "assignment_id": _digest({"cycle": cycle, "scenario_id": scenario_id,
                "category": category, "seed": seed, "master_seed": self.master_seed}),
            "sampling_role": ("recovery" if selected_recovery
                else "ancestor_retention" if recovery
                else "stage_exposure" if scenario_id not in curriculum_state.qualified
                else "ancestor_retention"),
            "mixture": mixture,
            "exposure_progress": {name: self.state["exposure_chunks"].get(name, 0)
                for name in SCENARIOS},
            "category_exposure_progress": dict(self.state["category_exposures"]),
            "recovery_pressure": dict(self.state["recovery_pressure"]),
            "promotion_status": "not_evaluated_by_exposure_scheduler"}
        if target_update is not None:
            if type(target_update) is not int or target_update < 1:
                raise ValueError("target_update must be a positive integer")
            assignment["target_update"] = target_update
        self.state["pending_assignment"] = assignment
        return dict(assignment)

    def complete(self, assignment: dict, lane_results: dict[str, dict]):
        self._validate_assignment(assignment)
        pending = self.state["pending_assignment"]
        if pending is None or pending["assignment_id"] != assignment.get("assignment_id"):
            raise ValueError("completed task does not match pending scheduler assignment")
        scenario_id, category = assignment["scenario_id"], assignment["category"]
        self.state["exposure_chunks"][scenario_id] = (
            self.state["exposure_chunks"].get(scenario_id, 0) + 1)
        category_key = f"{scenario_id}/{category}"
        self.state["category_exposures"][category_key] = (
            self.state["category_exposures"].get(category_key, 0) + 1)
        successes = failures = 0
        for track, result in sorted(lane_results.items()):
            if not isinstance(result, dict):
                continue
            successes += int(result.get("successes", 0))
            failures += int(result.get("failures", 0))
        pressure = self.state["recovery_pressure"].get(category_key, 0)
        if failures > successes:
            pressure += 1
        elif successes:
            pressure = max(0, pressure - 1)
        if pressure:
            self.state["recovery_pressure"][category_key] = pressure
        else:
            self.state["recovery_pressure"].pop(category_key, None)
        self.state["completed_cycle"] = assignment["cycle"]
        self.state["history"].append({"assignment": assignment,
            "lane_results": lane_results,
            "completed_utc": datetime.now(timezone.utc).isoformat()})
        self.state["pending_assignment"] = None
        return self.state
