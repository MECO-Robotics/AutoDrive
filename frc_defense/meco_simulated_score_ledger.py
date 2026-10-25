"""Pure contract for replay-reconstructable MECO simulated score outcomes.

This module deliberately does not implement official FRC scoring. Outcome
collection is blocked unless a human approval artifact binds this exact
simulator source and the limited FUEL-only scoring contract.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


LEDGER_SCHEMA = "meco-simulated-score-ledger-v1"
RULES_ID = "meco_simulated_fuel_score_v1"
EVENT_STREAM_SCHEMA = "meco-score-events-v1"
ESTIMAND_SCHEMA = "meco-simulated-score-estimand-review-v1"
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


def load_estimand_approval(path: str | Path, *, source_identity_sha256: str):
    """Read and validate the human-authored, source-bound approval artifact."""
    raw = Path(path).read_bytes()
    try:
        artifact = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("score estimand approval is not valid JSON") from exc
    validate_estimand_approval(artifact, source_identity_sha256=source_identity_sha256)
    return artifact, hashlib.sha256(raw).hexdigest()


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


def _validate_context(context, approval_sha256):
    required = ("episode_id", "seed", "scenario_id", "robot_team_slots", "world_count",
                "source_identity_sha256", "effective_scoring_configuration",
                "effective_scoring_configuration_sha256", "terminal_state",
                "terminal_tick")
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
    elif context["effective_scoring_configuration_sha256"] != sha256(config):
        missing.append("effective scoring configuration digest mismatch")
    if context["terminal_state"] not in ("complete", "censored"):
        missing.append("invalid terminal or censoring state")
    if type(context["terminal_tick"]) is not int or context["terminal_tick"] < 0:
        missing.append("invalid terminal tick")
    if not _is_sha256(approval_sha256):
        missing.append("estimand approval digest missing")
    return missing


def _validate_event_rows(context, events):
    if not isinstance(events, list):
        raise ValueError("score event rows must be a list")
    teams = context["robot_team_slots"]
    normalized = []
    seen_ids = set()
    seen_incarnations = set()
    previous = None
    expected = {"event_id", "episode_id", "tick", "match_elapsed_s", "world_slot",
                "alliance", "robot_slot", "piece_slot", "piece_generation",
                "event_type", "score_units", "source_site"}
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
        elapsed = row["match_elapsed_s"]
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= 160:
            raise ValueError("score event clock is invalid")
        if (type(row["world_slot"]) is not int or row["world_slot"] < 0
                or row["world_slot"] >= context["world_count"]):
            raise ValueError("score event world slot is invalid")
        if type(row["robot_slot"]) is not int or not 0 <= row["robot_slot"] < 6:
            raise ValueError("score event robot slot is invalid")
        if type(row["alliance"]) is not int or row["alliance"] != teams[row["robot_slot"]]:
            raise ValueError("score event alliance differs from robot map")
        if type(row["piece_slot"]) is not int or row["piece_slot"] < 0:
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
        order = (row["tick"], row["event_id"])
        if previous is not None and order <= previous:
            raise ValueError("score event rows are reordered")
        previous = order
        normalized.append(row)
    return normalized


def build_score_ledger(context, events, approval, *, approval_sha256):
    """Validate events and derive the simulated team score from event rows."""
    source_hash = (context.get("source_identity_sha256")
                   if isinstance(context, dict) else None)
    validate_estimand_approval(approval, source_identity_sha256=source_hash)
    missing = _validate_context(context, approval_sha256)
    if context.get("terminal_state") == "censored":
        missing.append("episode is censored")
    if missing:
        return _incomplete_ledger(context, approval_sha256, missing)
    rows = _validate_event_rows(context, events)
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


def reconstruct_replay_score(replay, approval):
    """Rebuild score using replay context and event rows, never summary totals."""
    if not isinstance(replay, dict):
        raise ValueError("MECO score replay is malformed")
    stream = replay.get("score_event_stream")
    if (not isinstance(stream, dict) or stream.get("schema") != EVENT_STREAM_SCHEMA
            or stream.get("rules") != RULES or stream.get("rules_sha256") != RULES_SHA256):
        raise ValueError("MECO score replay has no pinned event stream")
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
    if evaluation_record.get("simulated_score_ledger") != rebuilt:
        raise ValueError("evaluation score ledger differs from replay reconstruction")
    if evaluation_record.get("simulated_match_score") != rebuilt["simulated_match_score"]:
        raise ValueError("evaluation score summary differs from replay reconstruction")
    if (not isinstance(terminal_score_counts, list) or len(terminal_score_counts) != 2
            or terminal_score_counts != [rebuilt["simulated_match_score"]["red"],
                                         rebuilt["simulated_match_score"]["blue"]]):
        raise ValueError("terminal simulator counters differ from reconstructed score")
    if evaluation_record.get("official_match_points") is not None:
        raise ValueError("simulated score cannot populate official match points")
    if evaluation_record.get("qualification_eligible") is not False:
        raise ValueError("simulated score cannot establish qualification eligibility")
    return rebuilt
