"""Strict, diagnostic-only outcome records for Phase 4 pool factors.

The functions in this module bind evaluator rows to the simulator's pair
contact ledger. They do not promote aggregate score/count metrics to
per-robot contributions and they never enable qualification.
"""
from __future__ import annotations

import hashlib
import json
import math


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def simulator_pair_contact_ledger(sim, pairs: list[tuple[int, int]]) -> dict:
    """Read exact SAT contact counters for the requested robot-slot pairs."""
    required = ("_collision_pair_i", "_collision_pair_j", "pair_contact_event_count",
        "pair_contact_solver_report_count", "pair_contact_first_substep",
        "pair_contact_first_time_s", "pair_contact_last_substep",
        "pair_contact_last_time_s", "pair_contact_impulse_sum",
        "physics_substep_index", "physics_time_s")
    if getattr(sim, "num_robots", None) != 6 or not all(hasattr(sim, name) for name in required):
        raise ValueError("simulator lacks the six-robot pair-contact ledger")
    pair_table = list(zip(sim._collision_pair_i.tolist(), sim._collision_pair_j.tolist()))
    rows = []
    for requested in pairs:
        if (not isinstance(requested, tuple) or len(requested) != 2 or
                any(type(slot) is not int or slot not in range(6) for slot in requested) or
                requested[0] == requested[1]):
            raise ValueError("contact ledger request contains invalid robot slots")
        target = tuple(sorted(requested))
        matches = [i for i, pair in enumerate(pair_table)
                   if tuple(sorted((int(pair[0]), int(pair[1])))) == target]
        if len(matches) != 1:
            raise ValueError("simulator collision table lacks a unique requested robot pair")
        index = matches[0]
        count = int(sim.pair_contact_event_count[0, index].item())
        reports = int(sim.pair_contact_solver_report_count[0, index].item())
        first_tick = int(sim.pair_contact_first_substep[0, index].item())
        last_tick = int(sim.pair_contact_last_substep[0, index].item())
        first_time = float(sim.pair_contact_first_time_s[0, index].item())
        last_time = float(sim.pair_contact_last_time_s[0, index].item())
        impulse = [float(x) for x in sim.pair_contact_impulse_sum[0, index].tolist()]
        if (count < 0 or reports < count or
                (count == 0 and (first_tick != 0 or last_tick != 0 or
                    first_time != 0. or last_time != 0.)) or
                (count > 0 and (first_tick <= 0 or last_tick < first_tick or
                    not math.isfinite(first_time) or not math.isfinite(last_time) or
                    first_time < 0. or last_time < first_time)) or
                not all(math.isfinite(x) for x in impulse)):
            raise ValueError("simulator pair-contact ledger is internally inconsistent")
        rows.append({"robot_slot_ids": list(target),
            "distinct_contact_substep_count": count,
            "contact_solver_report_count": reports,
            "first_contact_physics_tick": first_tick if count else None,
            "first_contact_physics_time_s": first_time if count else None,
            "last_contact_physics_tick": last_tick if count else None,
            "last_contact_physics_time_s": last_time if count else None,
            "accumulated_impulse_n_s": impulse,
            "source": "simulator_robot_pair_sat_contact_ledger"})
    ticks = int(sim.physics_substep_index[0].item())
    seconds = float(sim.physics_time_s[0].item())
    if ticks <= 0 or not math.isfinite(seconds) or seconds <= 0.:
        raise ValueError("simulator contact ledger has no episode exposure")
    return {"schema": "meco-simulator-pair-contact-ledger-v1",
        "pairs": rows, "physics_ticks": ticks, "physics_seconds": seconds,
        "qualification_eligible": False}


def reactive_opponent_episode_record(*, env, schedule: dict, policy_id: str,
        scenario_id: str, category: str, qualification_case: dict, seed: int,
        condition: str, episode_id: str, condition_id: str | None,
        source_identity_sha256: str, checkpoint_sha256: str | None,
        execution_device: dict, metrics: dict, episode_censored: bool) -> dict:
    """Bind the realized opponent slots and simulator contacts to one episode."""
    if (not isinstance(schedule, dict) or schedule.get("schema") !=
            "meco-phase4-opponent-evaluator-schedule-v1" or
            schedule.get("factor_id") != "defender.reactive_pool" or
            schedule.get("policy_id") != policy_id or schedule.get("scenario_id") != scenario_id or
            category not in schedule.get("categories", ()) or seed not in schedule.get("seed_list", ()) or
            schedule.get("qualification_case") != qualification_case or
            schedule.get("split") != "validation" or
            condition not in ("no_factor", "factor_treatment")):
        raise ValueError("reactive-opponent capture differs from the exact validation assignment")
    for digest in (source_identity_sha256,):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("reactive-opponent capture requires source identity")
    if checkpoint_sha256 is not None and (not isinstance(checkpoint_sha256, str) or
            len(checkpoint_sha256) != 64):
        raise ValueError("reactive-opponent capture has invalid checkpoint identity")
    slots = schedule.get("configured_pressure_slots", {}).get(category)
    if (not isinstance(slots, list) or not slots or slots != sorted(set(slots)) or
            any(type(slot) is not int or slot not in (0, 1, 2, 4, 5) for slot in slots)):
        raise ValueError("reactive-opponent schedule has invalid assigned slots")
    runtime = getattr(env, "phase4_opponent_runtime", None)
    treatment = condition == "factor_treatment"
    if treatment:
        if (not isinstance(runtime, dict) or runtime.get("policy_id") != policy_id or
                runtime.get("factor_id") != schedule["factor_id"] or
                runtime.get("category") != category or runtime.get("seed") != seed or
                runtime.get("configured_slots") != slots or
                runtime.get("applied_slots") != slots):
            raise ValueError("reactive-opponent runtime does not prove exact slot application")
        calls = runtime.get("applied_command_calls_by_slot")
        total_ticks = int(env.sim.physics_substep_index[0].item())
        if (not isinstance(calls, dict) or
                any(calls.get(str(slot)) != total_ticks for slot in slots)):
            raise ValueError("reactive-opponent commands do not cover full episode exposure")
    elif runtime is not None:
        raise ValueError("reactive-opponent control unexpectedly applied the treatment schedule")
    contact = simulator_pair_contact_ledger(env.sim,
        [(3, slot) for slot in slots]) if treatment else simulator_pair_contact_ledger(
            env.sim, [(3, slot) for slot in slots])
    identity_rows = [row for row in schedule.get("per_episode_assignment", ())
                     if row.get("seed") == seed]
    if len(identity_rows) != 1:
        raise ValueError("reactive-opponent schedule lacks one exact per-seed assignment")
    assignment = identity_rows[0]
    if (assignment.get("policy_id") != policy_id or
            assignment.get("policy_family") != schedule.get("policy_family") or
            assignment.get("configured_pressure_slots") != slots):
        raise ValueError("reactive-opponent per-seed assignment changed policy or slots")
    contact_count = sum(row["distinct_contact_substep_count"] for row in contact["pairs"])
    intercepted = bool(treatment and contact_count > 0)
    return {"schema": "meco-phase4-reactive-opponent-outcome-v1",
        "factor_id": schedule["factor_id"], "policy_id": policy_id,
        "policy_family": schedule["policy_family"],
        "policy_config_sha256": schedule["policy_config_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "assignment_sha256": assignment["assignment_sha256"],
        "scenario_id": scenario_id, "category": category,
        "qualification_case": qualification_case, "split": "validation",
        "seed": seed, "condition": condition, "condition_id": condition_id,
        "episode_id": episode_id, "source_identity_sha256": source_identity_sha256,
        "checkpoint_sha256": checkpoint_sha256, "execution_device": execution_device,
        "configured_slots": slots,
        "applied_slots": runtime.get("applied_slots") if treatment else [],
        "applied_command_calls_by_slot": runtime.get(
            "applied_command_calls_by_slot") if treatment else {str(slot): 0 for slot in slots},
        "event_coverage_complete": bool(treatment and not episode_censored),
        "simulator_contact_ledger": contact,
        "interception_event_criterion": "at least one solver-confirmed SAT contact substep between focal slot 3 and an assigned policy slot during an uncensored treatment episode",
        "intercepted": intercepted if not episode_censored else None,
        "interception_episode_numerator": int(intercepted) if not episode_censored else None,
        "interception_eligible_episode_denominator": int(not episode_censored and treatment),
        "episode_outcome_metrics": {name: metrics.get(name) for name in (
            "success", "scores", "useful_throughput", "completed_cycles")},
        "episode_outcome_semantics": "whole-team simulator episode metrics while this policy was assigned; not a per-policy causal contribution",
        "qualification_eligible": False}


def reduce_reactive_opponent_pair(records_by_arm: dict, *, schedule: dict,
        expected_keys: set[tuple[str, str, int]], source_by_arm: dict,
        checkpoint_by_arm: dict, device_by_arm: dict) -> dict:
    """Fail-closed paired per-policy/family summary without group averaging."""
    if set(records_by_arm) != {"no_factor", "factor_treatment"}:
        raise ValueError("reactive-opponent reducer requires both paired arms")
    if any(set(records_by_arm[arm]) != expected_keys for arm in records_by_arm):
        raise ValueError("reactive-opponent pair has missing or unexpected policy/category/seed rows")
    paired, family_groups = [], {}
    for key in sorted(expected_keys):
        control = records_by_arm["no_factor"][key]
        treatment = records_by_arm["factor_treatment"][key]
        for arm, record in (("no_factor", control), ("factor_treatment", treatment)):
            if (not isinstance(record, dict) or
                    record.get("schema") != "meco-phase4-reactive-opponent-outcome-v1" or
                    record.get("policy_id") != key[0] or record.get("category") != key[1] or
                    record.get("seed") != key[2] or record.get("condition") != arm or
                    record.get("schedule_sha256") != schedule.get("schedule_sha256") or
                    record.get("source_identity_sha256") != source_by_arm[arm] or
                    record.get("checkpoint_sha256") != checkpoint_by_arm[arm] or
                    record.get("execution_device") != device_by_arm[arm] or
                    record.get("qualification_eligible") is not False):
                raise ValueError(f"reactive-opponent row provenance mismatch: {arm} {key}")
        if (control.get("policy_family") != treatment.get("policy_family") or
                treatment.get("event_coverage_complete") is not True or
                control.get("applied_slots") != []):
            raise ValueError(f"reactive-opponent pair is incomplete or mismatched: {key}")
        family = treatment["policy_family"]
        family_groups.setdefault(family, []).append((key, control, treatment))
        paired.append({"policy_id": key[0], "policy_family": family,
            "category": key[1], "seed": key[2],
            "control_episode_outcome": control["episode_outcome_metrics"],
            "treatment_episode_outcome": treatment["episode_outcome_metrics"],
            "intercepted": treatment["intercepted"],
            "contact_counts_by_slot": {tuple(row["robot_slot_ids"]):
                row["distinct_contact_substep_count"] for row in
                treatment["simulator_contact_ledger"]["pairs"]}})
    families = {family: {"policy_ids": sorted({entry[0][0] for entry in entries}),
        "row_count": len(entries), "intercepted_episodes": sum(
            entry[2]["interception_episode_numerator"] for entry in entries),
        "eligible_episodes": sum(entry[2]["interception_eligible_episode_denominator"]
            for entry in entries), "paired_rows": [row for row in paired
                if row["policy_family"] == family]}
        for family, entries in sorted(family_groups.items())}
    return {"schema": "meco-phase4-reactive-opponent-pair-outcomes-v1",
        "policy_family_coverage": {family: {"policy_ids": value["policy_ids"],
            "row_count": value["row_count"]} for family, value in families.items()},
        "interception_rate_by_family": {family: {
            "numerator_intercepted_episodes": value["intercepted_episodes"],
            "denominator_eligible_episodes": value["eligible_episodes"],
            "rate": (value["intercepted_episodes"] / value["eligible_episodes"]
                if value["eligible_episodes"] else None)}
            for family, value in families.items()},
        "paired_outcome_by_family": {family: value["paired_rows"]
            for family, value in families.items()},
        "policy_family_groups": families, "qualification_eligible": False}
