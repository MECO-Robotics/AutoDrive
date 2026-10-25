"""Pure contract for replay-reconstructable MECO simulated score outcomes.

This module deliberately does not implement official FRC scoring. Outcome
collection is blocked unless a human approval artifact binds this exact
simulator source and the limited FUEL-only scoring contract.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path


LEDGER_SCHEMA = "meco-simulated-score-ledger-v1"
RULES_ID = "meco_simulated_fuel_score_v1"
EVENT_STREAM_SCHEMA = "meco-score-events-v1"
ESTIMAND_SCHEMA = "meco-simulated-score-estimand-review-v1"
MAX_SCORE_EVENTS_PER_EPISODE = 32768
OUTCOME_NAME = "simulated_match_score"
SOURCE_PLAN_SHA256 = "718ef9d76f5525bf8e4759d3a0a46790a3c978e2b146efec6436c89dbbb49b10"
PHASE0_PLAN_SHA256 = "95e1b9185509259af952fab7fc956c12974d50aacf43525302275f859f917d49"
RULES = {
    "schema": "meco-simulated-score-rules-v1",
    "rules_id": RULES_ID,
    "event_type": "fuel_score_awarded",
    "score_units_per_event": 1,
    "unit": "simulated_score_unit",
    "award_semantics": "one unit for each accepted MECO FUEL score mutation",
    "eligible_when": ["valid_score_intent", "active_hub", "alliance_zone",
                       "robot_readiness", "score_cadence", "active_world"],
    "included_outcomes": ["active_hub_fuel_score_award"],
    "excluded_outcomes": ["tower_award", "foul_penalty", "disqualification",
                           "ranking_points", "end_game_bonus", "advancement_points"],
    "official_match_points": None,
}


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


RULES_SHA256 = sha256(RULES)


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _float32(value):
    return struct.unpack("=f", struct.pack("=f", float(value)))[0]


def load_estimand_approval(path: str | Path, *, source_identity_sha256: str):
    """Read and validate the human-authored, source-bound approval artifact."""
    raw = Path(path).read_bytes()
    try:
        artifact = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("score estimand approval is not valid JSON") from exc
    if raw != canonical_bytes(artifact):
        raise ValueError("score estimand approval must use canonical immutable bytes")
    validate_estimand_approval(artifact, source_identity_sha256=source_identity_sha256)
    return artifact, hashlib.sha256(raw).hexdigest()


def estimand_approval_digest(artifact):
    """Digest the exact canonical approval artifact accepted by this contract."""
    return hashlib.sha256(canonical_bytes(artifact)).hexdigest()


def validate_estimand_approval(artifact, *, source_identity_sha256: str):
    if not isinstance(artifact, dict) or artifact.get("schema") != ESTIMAND_SCHEMA:
        raise ValueError("human score estimand approval is required")
    if artifact.get("approved") is not True:
        raise ValueError("score estimand approval is not approved")
    required_text = ("approval_id", "approver", "approved_at", "intended_use",
                     "interpretation_limits", "primary_estimand", "unit")
    if any(not isinstance(artifact.get(key), str) or not artifact[key].strip()
           for key in required_text):
        raise ValueError("score estimand approval is incomplete")
    if (artifact.get("source_plan_sha256") != SOURCE_PLAN_SHA256
            or artifact.get("phase0_plan_sha256") != PHASE0_PLAN_SHA256):
        raise ValueError("score estimand approval is bound to a different plan")
    if artifact.get("simulator_source_sha256") != source_identity_sha256:
        raise ValueError("score estimand approval is bound to different simulator source")
    if artifact.get("rules_id") != RULES_ID or artifact.get("rules_sha256") != RULES_SHA256:
        raise ValueError("score estimand approval is bound to different score rules")
    if artifact.get("primary_estimand") != OUTCOME_NAME or artifact.get("unit") != RULES["unit"]:
        raise ValueError("score estimand approval names a different outcome")
    if artifact.get("included_outcomes") != RULES["included_outcomes"]:
        raise ValueError("score estimand approval includes unsupported outcomes")
    if artifact.get("excluded_outcomes") != RULES["excluded_outcomes"]:
        raise ValueError("score estimand approval does not preserve excluded outcomes")
    formula = artifact.get("formula")
    if formula != "sum score_units by alliance over accepted FUEL score-award events":
        raise ValueError("score estimand approval formula is unsupported")
    return True


def _incomplete_ledger(context, approval_sha256, missing):
    context_copy = dict(context) if isinstance(context, dict) else {}
    ledger = {
        "schema": LEDGER_SCHEMA,
        "status": "incomplete",
        "incomplete_reasons": sorted(set(missing)),
        "context": context_copy,
        "estimand_approval_sha256": approval_sha256,
        "rules": RULES,
        "rules_sha256": RULES_SHA256,
        "event_rows": [],
        "event_digests": [],
        "events_sha256": sha256([]),
        "simulated_match_score": {"schema": LEDGER_SCHEMA, "rules_id": RULES_ID,
            "rules_sha256": RULES_SHA256, "red": None, "blue": None,
            "complete": False},
        "official_match_points": None,
        "qualification_eligible": False,
    }
    ledger["ledger_sha256"] = sha256(ledger)
    return ledger


def _invalid_ledger(context, approval_sha256, reason):
    ledger = {
        "schema": LEDGER_SCHEMA,
        "status": "invalid",
        "invalid_reason": reason,
        "context": dict(context),
        "estimand_approval_sha256": approval_sha256,
        "rules": RULES,
        "rules_sha256": RULES_SHA256,
        "event_rows": [],
        "event_digests": [],
        "events_sha256": sha256([]),
        "simulated_match_score": {"schema": LEDGER_SCHEMA, "rules_id": RULES_ID,
            "rules_sha256": RULES_SHA256, "red": None, "blue": None,
            "complete": False},
        "official_match_points": None,
        "qualification_eligible": False,
    }
    ledger["ledger_sha256"] = sha256(ledger)
    return ledger


def _validate_context(context, approval_sha256):
    required = ("episode_id", "seed", "scenario_id", "robot_team_slots", "world_count",
                "source_identity_sha256", "effective_scoring_configuration",
                "effective_scoring_configuration_sha256", "terminal_state",
                "terminal_tick", "fuel_count")
    if not isinstance(context, dict):
        return ["episode context missing"]
    missing = [f"missing context field: {key}" for key in required if key not in context]
    if missing:
        return missing
    if not isinstance(context["episode_id"], str) or not context["episode_id"]:
        missing.append("invalid episode identity")
    if type(context["seed"]) is not int or context["seed"] < 0:
        missing.append("invalid episode seed")
    if not isinstance(context["scenario_id"], str) or not context["scenario_id"]:
        missing.append("invalid scenario identity")
    teams = context["robot_team_slots"]
    if (not isinstance(teams, list) or len(teams) != 6
            or any(type(team) is not int or team not in (0, 1) for team in teams)
            or teams.count(0) != 3 or teams.count(1) != 3):
        missing.append("invalid six-robot alliance map")
    if type(context["world_count"]) is not int or context["world_count"] < 1:
        missing.append("invalid simulator world count")
    if not _is_sha256(context["source_identity_sha256"]):
        missing.append("invalid source identity")
    config = context["effective_scoring_configuration"]
    if not isinstance(config, dict) or not config:
        missing.append("effective scoring configuration missing")
    else:
        required_config = ("dt", "robot_control_modes", "robot_types", "robot_roles",
            "robot_fuel_capacities", "robot_score_interval_s", "robot_readiness",
            "score_intent_source", "alliance_zone_depth_m", "hub_schedule",
            "robot_dimensions_m", "field_geometry", "robot_team_slots", "fuel_count")
        missing.extend(f"effective scoring configuration missing: {key}"
                       for key in required_config if key not in config)
        if context["effective_scoring_configuration_sha256"] != sha256(config):
            missing.append("effective scoring configuration digest mismatch")
        if config.get("robot_team_slots") != teams:
            missing.append("top-level alliance map differs from effective configuration")
        if config.get("fuel_count") != context.get("fuel_count"):
            missing.append("top-level FUEL count differs from effective configuration")
        for key, allowed in (("robot_control_modes", {"none", "nn", "deterministic"}),
                              ("robot_types", {"dumper", "turret"}),
                              ("robot_roles", {"offense", "defense"})):
            values = config.get(key)
            if (not isinstance(values, list) or len(values) != 6
                    or any(value not in allowed for value in values)):
                missing.append(f"invalid effective {key}")
        for key in ("robot_fuel_capacities", "robot_score_interval_s"):
            values = config.get(key)
            if (not isinstance(values, list) or len(values) != 6
                    or any(type(value) not in (int, float)
                           or not math.isfinite(value) or value <= 0 for value in values)):
                missing.append(f"invalid effective {key}")
        zone_depth = config.get("alliance_zone_depth_m")
        if (type(zone_depth) not in (int, float) or not math.isfinite(zone_depth)
                or zone_depth <= 0):
            missing.append("invalid effective alliance zone depth")
        if (not isinstance(config.get("score_intent_source"), str)
                or not config["score_intent_source"]):
            missing.append("effective score intent source missing")
        dimensions = config.get("robot_dimensions_m")
        if (not isinstance(dimensions, dict)
                or any(not isinstance(dimensions.get(key), list)
                       or len(dimensions[key]) != 6
                       or any(type(value) not in (int, float)
                              or not math.isfinite(value) or value <= 0
                              for value in dimensions[key])
                       for key in ("length", "width"))):
            missing.append("invalid effective robot dimensions")
        geometry = config.get("field_geometry")
        if (not isinstance(geometry, dict)
                or any(type(geometry.get(key)) not in (int, float)
                       or not math.isfinite(geometry[key]) or geometry[key] <= 0
                       for key in ("field_length_m", "field_width_m"))
                or not isinstance(geometry.get("hub_centers_m"), list)
                or len(geometry["hub_centers_m"]) != 2
                or not isinstance(geometry.get("bump_regions"), list)):
            missing.append("invalid effective field scoring geometry")
        readiness = config.get("robot_readiness")
        if (not isinstance(readiness, dict)
                or any(type(readiness.get(key)) not in (int, float)
                       or not math.isfinite(readiness[key]) or readiness[key] < 0
                       for key in ("dumper_stationary_linear_mps",
                                   "dumper_stationary_angular_radps",
                                   "dumper_aim_tolerance_rad"))
                or not isinstance(readiness.get("bump_clearance_source"), str)
                or not readiness["bump_clearance_source"]):
            missing.append("invalid effective robot readiness configuration")
    if context["terminal_state"] not in ("complete", "censored"):
        missing.append("invalid terminal or censoring state")
    if type(context["terminal_tick"]) is not int or context["terminal_tick"] < 0:
        missing.append("invalid terminal tick")
    if not _is_sha256(approval_sha256):
        missing.append("estimand approval digest missing")
    fuel_count = context.get("fuel_count")
    if type(fuel_count) is not int or fuel_count < 1:
        missing.append("invalid configured FUEL count")
    config = context.get("effective_scoring_configuration")
    if isinstance(config, dict):
        dt = config.get("dt")
        if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
            missing.append("effective simulator timestep missing")
    return missing


def _validate_event_rows(context, events):
    if not isinstance(events, list):
        raise ValueError("score event rows must be a list")
    order_rows = [(row.get("tick"), row.get("event_id")) for row in events
                  if isinstance(row, dict)]
    if len(order_rows) == len(events) and all(
            type(tick) is int and isinstance(event_id, str)
            and 0 <= tick <= context.get("terminal_tick", -1)
            for tick, event_id in order_rows) and len({key for _, key in order_rows}) == len(events):
        if any(current <= previous for previous, current in
               zip(order_rows, order_rows[1:])):
            raise ValueError("score event rows are reordered")
    teams = context["robot_team_slots"]
    normalized = []
    seen_ids = set()
    seen_incarnations = set()
    next_generation = {}
    expected = {"event_id", "episode_id", "tick", "match_elapsed_s", "world_slot",
                "alliance", "robot_slot", "piece_slot", "piece_generation",
                "event_type", "score_units", "source_site"}
    dt = context["effective_scoring_configuration"]["dt"]
    dt32 = _float32(dt)
    elapsed_by_world = {}
    fuel_count = context["fuel_count"]
    for source in events:
        if not isinstance(source, dict) or set(source) != expected:
            raise ValueError("score event row has an invalid schema")
        row = dict(source)
        if not isinstance(row["event_id"], str) or not row["event_id"]:
            raise ValueError("score event ID is invalid")
        if row["event_id"] in seen_ids:
            raise ValueError("duplicate score event ID")
        seen_ids.add(row["event_id"])
        if row["episode_id"] != context["episode_id"]:
            raise ValueError("score event episode differs from replay context")
        if type(row["tick"]) is not int or not 0 <= row["tick"] <= context["terminal_tick"]:
            raise ValueError("score event tick is outside the episode")
        if (type(row["world_slot"]) is not int or row["world_slot"] < 0
                or row["world_slot"] >= context["world_count"]):
            raise ValueError("score event world slot is invalid")
        elapsed = row["match_elapsed_s"]
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= 160:
            raise ValueError("score event clock is invalid")
        # `tick` is zero-based and match_elapsed is a float32 accumulator that
        # advances before scoring. Reproduce that recurrence exactly.
        elapsed32, elapsed_tick = elapsed_by_world.get(row["world_slot"], (0.0, -1))
        while elapsed_tick < row["tick"]:
            elapsed32 = min(_float32(elapsed32 + dt32), 160.0)
            elapsed_tick += 1
        elapsed_by_world[row["world_slot"]] = (elapsed32, elapsed_tick)
        if elapsed != elapsed32:
            raise ValueError("score event clock differs from simulator tick and timestep")
        if type(row["robot_slot"]) is not int or not 0 <= row["robot_slot"] < 6:
            raise ValueError("score event robot slot is invalid")
        if type(row["alliance"]) is not int or row["alliance"] != teams[row["robot_slot"]]:
            raise ValueError("score event alliance differs from robot map")
        if (type(row["piece_slot"]) is not int
                or not 0 <= row["piece_slot"] < fuel_count):
            raise ValueError("score event piece slot is invalid")
        if type(row["piece_generation"]) is not int or row["piece_generation"] < 0:
            raise ValueError("score event piece generation is invalid")
        if row["event_type"] != RULES["event_type"]:
            raise ValueError("unrecognized score event type")
        if type(row["score_units"]) is not int or row["score_units"] != RULES["score_units_per_event"]:
            raise ValueError("score event units differ from pinned rules")
        if row["source_site"] not in (
                "gamepiece_actions.update_fuel_actions.vectorized_score_mutation",
                "gamepiece_actions.update_fuel_actions.robot_loop_score_mutation"):
            raise ValueError("score event source site is unsupported")
        incarnation = (row["episode_id"], row["world_slot"], row["piece_slot"],
                       row["piece_generation"])
        if incarnation in seen_incarnations:
            raise ValueError("duplicate award for piece incarnation")
        seen_incarnations.add(incarnation)
        slot_key = (row["episode_id"], row["world_slot"], row["piece_slot"])
        expected_generation = next_generation.get(slot_key, 0)
        if row["piece_generation"] != expected_generation:
            raise ValueError("piece incarnation generation is not the next scorer lifecycle")
        next_generation[slot_key] = expected_generation + 1
        normalized.append(row)
    return normalized


def build_score_ledger(context, events, approval, *, approval_sha256):
    """Validate events and derive the simulated team score from event rows."""
    source_hash = (context.get("source_identity_sha256")
                   if isinstance(context, dict) else None)
    validate_estimand_approval(approval, source_identity_sha256=source_hash)
    if approval_sha256 != estimand_approval_digest(approval):
        return _invalid_ledger(context, approval_sha256,
                               "estimand approval digest differs from exact artifact")
    missing = _validate_context(context, approval_sha256)
    if context.get("terminal_state") == "censored":
        missing.append("episode is censored")
    if missing:
        return _incomplete_ledger(context, approval_sha256, missing)
    try:
        rows = _validate_event_rows(context, events)
    except ValueError as exc:
        return _invalid_ledger(context, approval_sha256, str(exc))
    totals = [0, 0]
    for row in rows:
        totals[row["alliance"]] += row["score_units"]
    event_digests = [sha256(row) for row in rows]
    ledger = {
        "schema": LEDGER_SCHEMA,
        "status": "complete",
        "incomplete_reasons": [],
        "context": dict(context),
        "estimand_approval_sha256": approval_sha256,
        "rules": RULES,
        "rules_sha256": RULES_SHA256,
        "event_rows": rows,
        "event_digests": event_digests,
        "events_sha256": sha256(rows),
        "simulated_match_score": {"schema": LEDGER_SCHEMA, "rules_id": RULES_ID,
            "rules_sha256": RULES_SHA256, "red": totals[0], "blue": totals[1],
            "complete": True},
        "official_match_points": None,
        "qualification_eligible": False,
    }
    ledger["ledger_sha256"] = sha256(ledger)
    return ledger


def make_event_stream(context, events):
    normalized = _validate_event_rows(context, events)
    return {"schema": EVENT_STREAM_SCHEMA, "context": dict(context),
            "rules": RULES, "rules_sha256": RULES_SHA256,
            "events": normalized, "events_sha256": sha256(normalized)}


class MecoScoreEventRecorder:
    """Opt-in bounded-by-episode capture for accepted simulator score events."""

    def __init__(self, context, approval, approval_sha256):
        validate_estimand_approval(approval,
            source_identity_sha256=context.get("source_identity_sha256"))
        if approval_sha256 != estimand_approval_digest(approval):
            raise ValueError("score estimand digest differs from exact approval artifact")
        missing = _validate_context(context, approval_sha256)
        if missing:
            raise ValueError("score event recorder context is incomplete: " + "; ".join(missing))
        self.context = dict(context)
        self.approval = dict(approval)
        self.approval_sha256 = approval_sha256
        self.events = []
        self._sequence = 0
        self.invalid_reason = None

    def invalidate(self, reason):
        self.invalid_reason = str(reason)

    def record_award(self, *, tick, match_elapsed_s, world_slot, alliance,
                     robot_slot, piece_slot, piece_generation, source_site):
        if len(self.events) >= MAX_SCORE_EVENTS_PER_EPISODE:
            raise ValueError("score event capture exceeded its fixed episode bound")
        if not 0 <= world_slot < self.context["world_count"]:
            raise ValueError("score capture world slot is outside context")
        event_id = (f"{self.context['episode_id']}:w{world_slot:06d}:"
                    f"e{self._sequence:012d}")
        self._sequence += 1
        self.events.append({
            "event_id": event_id,
            "episode_id": self.context["episode_id"],
            "tick": int(tick),
            "match_elapsed_s": float(match_elapsed_s),
            "world_slot": int(world_slot),
            "alliance": int(alliance),
            "robot_slot": int(robot_slot),
            "piece_slot": int(piece_slot),
            "piece_generation": int(piece_generation),
            "event_type": RULES["event_type"],
            "score_units": RULES["score_units_per_event"],
            "source_site": source_site,
        })

    def finish(self, *, terminal_tick, terminal_state, terminal_score_counts):
        if self.invalid_reason is not None:
            raise ValueError("score event collection crossed an episode reset")
        context = dict(self.context)
        context["terminal_tick"] = int(terminal_tick)
        context["terminal_state"] = terminal_state
        ledger = build_score_ledger(context, self.events, self.approval,
                                    approval_sha256=self.approval_sha256)
        stream = make_event_stream(context, self.events)
        if ledger["status"] != "complete":
            raise ValueError("score event collection is blocked for an incomplete episode")
        if ledger["status"] == "complete":
            expected = [ledger["simulated_match_score"]["red"],
                        ledger["simulated_match_score"]["blue"]]
            if terminal_score_counts != expected:
                raise ValueError("terminal simulator counters differ from score-event ledger")
        return {"context": context, "event_stream": stream,
                "ledger": ledger, "terminal_score_counts": list(terminal_score_counts)}

    def finish_incomplete(self, *, terminal_tick, terminal_state,
                          terminal_score_counts, reason):
        """Report a fail-closed capture gap without presenting partial events as complete."""
        context = dict(self.context)
        context["terminal_tick"] = int(terminal_tick)
        context["terminal_state"] = terminal_state
        reasons = [str(reason)]
        if self.invalid_reason is not None:
            reasons.append(str(self.invalid_reason))
        ledger = _incomplete_ledger(context, self.approval_sha256, reasons)
        return {"context": context, "event_stream": None, "ledger": ledger,
                "terminal_score_counts": list(terminal_score_counts)}


def reconstruct_replay_score(replay, approval):
    """Rebuild score using replay context and event rows, never summary totals."""
    if not isinstance(replay, dict):
        raise ValueError("MECO score replay is malformed")
    stream = replay.get("score_event_stream")
    if (not isinstance(stream, dict) or stream.get("schema") != EVENT_STREAM_SCHEMA
            or stream.get("rules") != RULES or stream.get("rules_sha256") != RULES_SHA256):
        raise ValueError("MECO score replay has no pinned event stream")
    if replay.get("source_identity_sha256") != stream.get("context", {}).get(
            "source_identity_sha256"):
        raise ValueError("MECO score replay source identity differs from event context")
    approval_digest = estimand_approval_digest(approval)
    if replay.get("estimand_approval_sha256") != approval_digest:
        raise ValueError("MECO score replay approval digest differs from exact artifact")
    events = stream.get("events")
    if stream.get("events_sha256") != sha256(events):
        raise ValueError("MECO score replay event digest mismatch")
    ledger = build_score_ledger(stream.get("context"), events, approval,
                                approval_sha256=replay.get("estimand_approval_sha256"))
    if ledger["status"] != "complete":
        raise ValueError("MECO score replay outcome is incomplete")
    return ledger


def validate_replay_score_binding(replay, evaluation_record, terminal_score_counts,
                                  approval):
    rebuilt = reconstruct_replay_score(replay, approval)
    if replay.get("simulated_score_ledger") != rebuilt:
        raise ValueError("replay score ledger differs from event reconstruction")
    replay_record = replay.get("evaluation_record")
    if not isinstance(replay_record, dict):
        raise ValueError("replay evaluation score binding is missing")
    if replay_record.get("simulated_score_ledger_sha256") != rebuilt["ledger_sha256"]:
        raise ValueError("replay evaluation score digest differs from event reconstruction")
    if replay_record.get("simulated_match_score") != rebuilt["simulated_match_score"]:
        raise ValueError("replay evaluation score summary differs from event reconstruction")
    if replay_record.get("source_identity_sha256") != rebuilt["context"][
            "source_identity_sha256"]:
        raise ValueError("replay evaluation score source differs from event reconstruction")
    if evaluation_record.get("simulated_score_ledger") != rebuilt:
        raise ValueError("evaluation score ledger differs from replay reconstruction")
    if evaluation_record.get("simulated_match_score") != rebuilt["simulated_match_score"]:
        raise ValueError("evaluation score summary differs from replay reconstruction")
    if evaluation_record.get("source_identity_sha256") != rebuilt["context"][
            "source_identity_sha256"]:
        raise ValueError("evaluation score source identity differs from replay reconstruction")
    if (not isinstance(terminal_score_counts, list) or len(terminal_score_counts) != 2
            or terminal_score_counts != [rebuilt["simulated_match_score"]["red"],
                                         rebuilt["simulated_match_score"]["blue"]]):
        raise ValueError("terminal simulator counters differ from reconstructed score")
    if evaluation_record.get("official_match_points") is not None:
        raise ValueError("simulated score cannot populate official match points")
    if evaluation_record.get("qualification_eligible") is not False:
        raise ValueError("simulated score cannot establish qualification eligibility")
    return rebuilt
