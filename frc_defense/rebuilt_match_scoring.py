"""Pure, source-pinned scoring contracts for 2026 REBUILT.

This module does not construct a simulator.  ``rebuilt_active_hub_fuel_proxy_v1``
accepts the current simulator's abstract score-action events; only separately
validated sensor, volunteer, referee, roster, and match-context evidence can
produce ``official_match_points``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


RULES_SNAPSHOT_PATH = (Path(__file__).resolve().parents[1]
                       / "qualifications" / "meco" / "rebuilt-rules-TU22.json")
RULES_SNAPSHOT_SHA256 = "18cf92041290e8fa1df15d7e967fe5e107118a6d443fe2b6526c1c05b73f47a8"
PROXY_CONTRACT_ID = "rebuilt_active_hub_fuel_proxy_v1"
OFFICIAL_SCORE_ID = "official_match_points"
ALLIANCES = ("red", "blue")
_COVERAGE_FIELDS = frozenset({
    "match_context_complete",
    "alliance_rosters_complete",
    "active_lineups_complete",
    "six_robot_slots_complete",
    "fuel_sensor_events_complete",
    "score_window_coverage_complete",
    "tower_adjudications_complete",
    "referee_penalty_coverage_complete",
    "disqualification_coverage_complete",
})


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible values using canonical UTF-8 encoding."""
    try:
        body = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical finite JSON data") from exc
    return hashlib.sha256(body).hexdigest()


def _rules_snapshot() -> dict[str, Any]:
    try:
        raw = RULES_SNAPSHOT_PATH.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("pinned REBUILT rules snapshot is unavailable or invalid") from exc
    if hashlib.sha256(raw).hexdigest() != RULES_SNAPSHOT_SHA256:
        raise RuntimeError("pinned REBUILT rules snapshot digest mismatch")
    authorities = value.get("authorities") if isinstance(value, dict) else None
    if (not isinstance(value, dict)
            or value.get("schema") != "rebuilt-rules-snapshot-v1"
            or not isinstance(authorities, list)
            or {item.get("sha256") for item in authorities if isinstance(item, dict)} != {
                "5c67300fc412a1eed8ca9dc37fb6644aee05795a2befed1f69736643cbf95139",
                "8ec2754d157b7f35cf93b4be301ebdf576f2abf2650975731dcc3cdf5ab0ecb1",
            }):
        raise RuntimeError("pinned REBUILT rule authorities are incomplete")
    return value


RULES_SNAPSHOT = _rules_snapshot()


def rules_snapshot_sha256() -> str:
    """Return the pinned snapshot's raw-byte digest."""
    return RULES_SNAPSHOT_SHA256


def derive_simulation_tie_selection(
    simulation_match_id: str,
    seed: int,
    *,
    world_index: int = 0,
    reset_generation: int = 0,
) -> dict[str, Any]:
    """Derive a reproducible simulation tie choice without consuming an RNG."""
    match_id = _require_text(simulation_match_id, "simulation_match_id")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(world_index) is not int or world_index < 0:
        raise ValueError("world_index must be a nonnegative integer")
    if type(reset_generation) is not int or reset_generation < 0:
        raise ValueError("reset_generation must be a nonnegative integer")
    base_payload = {
        "algorithm": "sha256_match_seed_world_low_bit_v1",
        "simulation_match_id": match_id,
        "seed": seed,
        "world_index": world_index,
    }
    base_digest = canonical_sha256(base_payload)
    selected_bit = (int(base_digest[-1], 16) & 1) ^ (reset_generation & 1)
    return {
        "alliance": ALLIANCES[selected_bit],
        "simulation_only": True,
        "derivation": "sha256_match_seed_world_low_bit_xor_reset_generation_parity_v1",
        "base_derivation_sha256": base_digest,
        "simulation_match_id": match_id,
        "seed": seed,
        "world_index": world_index,
        "reset_generation": reset_generation,
    }


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _require_time(value: Any, field: str, maximum: float = 160.0) -> float:
    if not _number(value) or not 0 <= value <= maximum:
        raise ValueError(f"{field} must be a finite time in [0, {maximum:g}]")
    return float(value)


def match_period(time_s: float) -> str:
    """Return the half-open official match-clock period containing ``time_s``."""
    value = _require_time(time_s, "time_s")
    for period in RULES_SNAPSHOT["match_clock_seconds"]["periods"]:
        if period["start_inclusive"] <= value < period["end_exclusive"]:
            return period["id"]
    raise ValueError("time_s is at match end and is not inside a scoring period")


def shift_1_inactive_alliance(
    auto_fuel_counts: Mapping[str, int],
    *,
    fms_tie_selection: str | None = None,
) -> str:
    """Resolve which HUB is inactive in SHIFT 1 from AUTO totals/FMS tie result."""
    if (not isinstance(auto_fuel_counts, Mapping)
            or set(auto_fuel_counts) != set(ALLIANCES)
            or any(type(auto_fuel_counts[a]) is not int or auto_fuel_counts[a] < 0
                   for a in ALLIANCES)):
        raise ValueError("auto_fuel_counts must contain nonnegative red and blue integers")
    red, blue = auto_fuel_counts["red"], auto_fuel_counts["blue"]
    if red == blue:
        if fms_tie_selection not in ALLIANCES:
            raise ValueError("an AUTO FUEL tie requires the FMS-selected inactive alliance")
        return fms_tie_selection
    if fms_tie_selection is not None:
        raise ValueError("FMS tie selection is only valid when AUTO FUEL totals tie")
    return "red" if red > blue else "blue"


def hub_is_active(
    alliance: str,
    time_s: float,
    auto_fuel_counts: Mapping[str, int],
    *,
    fms_tie_selection: str | None = None,
) -> bool:
    """Derive a HUB's active state from TU22 periods and AUTO/FMS result."""
    if alliance not in ALLIANCES:
        raise ValueError("alliance must be red or blue")
    period = match_period(time_s)
    if period in ("auto", "transition", "end_game"):
        return True
    first_inactive = shift_1_inactive_alliance(
        auto_fuel_counts, fms_tie_selection=fms_tie_selection)
    shift_number = int(period.removeprefix("shift_"))
    inactive = first_inactive if shift_number % 2 == 1 else (
        "blue" if first_inactive == "red" else "red")
    return alliance != inactive


def _validate_context(context: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "match_id", "seed", "competition_context", "alliance_rosters",
        "active_lineups", "robot_slots", "fms_tie_selection",
    }
    if not isinstance(context, Mapping) or set(context) != keys:
        raise ValueError("match context must provide the exact roster, lineup, slot, and FMS fields")
    match_id = _require_text(context["match_id"], "match_id")
    if type(context["seed"]) is not int or context["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    competition = context["competition_context"]
    if competition not in ("qualification", "playoff", "championship_playoff"):
        raise ValueError("competition_context must be qualification, playoff, or championship_playoff")
    rosters, lineups = context["alliance_rosters"], context["active_lineups"]
    if (not isinstance(rosters, Mapping) or set(rosters) != set(ALLIANCES)
            or not isinstance(lineups, Mapping) or set(lineups) != set(ALLIANCES)):
        raise ValueError("red and blue alliance rosters and active lineups are required")
    clean_rosters: dict[str, list[str]] = {}
    clean_lineups: dict[str, list[str]] = {}
    for alliance in ALLIANCES:
        roster, lineup = rosters[alliance], lineups[alliance]
        if (not isinstance(roster, (list, tuple)) or len(roster) not in (3, 4)
                or any(not isinstance(team, str) or not team.strip() for team in roster)
                or len(set(roster)) != len(roster)):
            raise ValueError(f"{alliance} alliance roster must contain 3 or 4 unique team IDs")
        if (competition == "championship_playoff" and len(roster) != 4):
            raise ValueError("Championship playoff roster must contain four teams")
        if (not isinstance(lineup, (list, tuple)) or len(lineup) != 3
                or any(not isinstance(team, str) or not team.strip() for team in lineup)
                or len(set(lineup)) != 3 or not set(lineup) <= set(roster)):
            raise ValueError(f"{alliance} active lineup must select exactly three roster teams")
        clean_rosters[alliance] = [team.strip() for team in roster]
        clean_lineups[alliance] = [team.strip() for team in lineup]
    if set(clean_rosters["red"]) & set(clean_rosters["blue"]):
        raise ValueError("a team cannot be rostered on both alliances")
    slots = context["robot_slots"]
    if not isinstance(slots, (list, tuple)) or len(slots) != 6:
        raise ValueError("exactly six active robot slots are required for official scoring")
    clean_slots = []
    seen_robot_ids: set[str] = set()
    slot_teams: dict[str, set[str]] = {alliance: set() for alliance in ALLIANCES}
    for slot in slots:
        if not isinstance(slot, Mapping) or set(slot) != {"alliance", "team_id", "robot_id"}:
            raise ValueError("robot slots need exactly alliance, team_id, and robot_id")
        alliance = slot["alliance"]
        team = _require_text(slot["team_id"], "robot slot team_id")
        robot = _require_text(slot["robot_id"], "robot slot robot_id")
        if alliance not in ALLIANCES or team not in clean_lineups[alliance]:
            raise ValueError("each active robot slot must bind a team in its alliance lineup")
        if robot in seen_robot_ids or team in slot_teams[alliance]:
            raise ValueError("active robot slots must uniquely identify robots and teams")
        seen_robot_ids.add(robot)
        slot_teams[alliance].add(team)
        clean_slots.append({"alliance": alliance, "team_id": team, "robot_id": robot})
    if any(slot_teams[a] != set(clean_lineups[a]) for a in ALLIANCES):
        raise ValueError("six robot slots must cover each alliance's exact three-team lineup")
    fms = context["fms_tie_selection"]
    if fms is not None and fms not in ALLIANCES:
        raise ValueError("fms_tie_selection must be red, blue, or null")
    return {
        "match_id": match_id,
        "seed": context["seed"],
        "competition_context": competition,
        "alliance_rosters": clean_rosters,
        "active_lineups": clean_lineups,
        "robot_slots": sorted(clean_slots, key=lambda row: (row["alliance"], row["team_id"])),
        "fms_tie_selection": fms,
    }


def _event_time(event: Mapping[str, Any]) -> float:
    if event["event_type"] == "fuel_sensor_crossing":
        return float(event["crossing_time_s"])
    return float(event["occurred_time_s"])


def _normalize_events(events: Sequence[Mapping[str, Any]], context: dict[str, Any],
                      auto_counts: Mapping[str, int]) -> list[dict[str, Any]]:
    if not isinstance(events, (list, tuple)):
        raise ValueError("events must be an ordered list")
    slot_by_robot = {row["robot_id"]: row for row in context["robot_slots"]}
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    last_key: tuple[float, str] | None = None
    tower_events: dict[tuple[str, str], dict[str, set[int]]] = {}
    disqualified: set[str] = set()

    for raw in events:
        if not isinstance(raw, Mapping):
            raise ValueError("each event must be an object")
        kind = raw.get("event_type")
        common = {"event_id", "event_type"}
        if kind == "fuel_sensor_crossing":
            expected = common | {"crossing_time_s", "assessed_time_s", "alliance", "team_id",
                                 "robot_id", "sensor_event_id", "sensor_confirmed", "rule_ref"}
        elif kind == "tower_award":
            expected = common | {"occurred_time_s", "assessed_time_s", "alliance", "team_id",
                                 "robot_id", "phase", "level", "adjudicator_id",
                                 "adjudication_id", "evidence_ref", "rule_ref"}
        elif kind == "foul_award":
            expected = common | {"occurred_time_s", "violating_alliance", "beneficiary_alliance",
                                 "team_id", "severity", "adjudicator_id", "adjudication_id",
                                 "evidence_ref", "rule_ref"}
        elif kind == "disqualification":
            expected = common | {"occurred_time_s", "alliance", "team_id", "adjudicator_id",
                                 "adjudication_id", "evidence_ref", "rule_ref"}
        elif kind == "ranking_points":
            expected = common | {"occurred_time_s", "alliance", "team_id", "rp_type", "points",
                                 "source_ref", "rule_ref"}
        else:
            raise ValueError(f"unsupported scoring event type: {kind!r}")
        if set(raw) != expected:
            raise ValueError(f"{kind} event has missing or unexpected fields")
        event = dict(raw)
        event_id = _require_text(event["event_id"], "event_id")
        if event_id in seen_ids:
            raise ValueError(f"duplicate event_id: {event_id}")
        seen_ids.add(event_id)
        event["event_id"] = event_id
        event_time = _event_time(event)
        sort_key = (event_time, event_id)
        if last_key is not None and sort_key < last_key:
            raise ValueError("events must be ordered by event time then event_id")
        last_key = sort_key

        if kind == "fuel_sensor_crossing":
            crossing = _require_time(event["crossing_time_s"], "crossing_time_s")
            assessed = _require_time(event["assessed_time_s"], "assessed_time_s", 163.)
            if assessed < crossing:
                raise ValueError("FUEL assessment cannot precede sensor crossing")
            alliance = event["alliance"]
            if alliance not in ALLIANCES or event["sensor_confirmed"] is not True:
                raise ValueError("official FUEL scoring requires a confirmed red/blue sensor crossing")
            sensor_id = _require_text(event["sensor_event_id"], "sensor_event_id")
            _require_text(event["rule_ref"], "rule_ref")
            team, robot = event["team_id"], event["robot_id"]
            if team is not None or robot is not None:
                if not isinstance(team, str) or not isinstance(robot, str):
                    raise ValueError("FUEL attribution needs both team_id and robot_id or neither")
                slot = slot_by_robot.get(robot)
                if slot is None or slot["alliance"] != alliance or slot["team_id"] != team:
                    raise ValueError("FUEL team/robot attribution is outside the active slot map")
            active = hub_is_active(alliance, crossing, auto_counts,
                                   fms_tie_selection=context["fms_tie_selection"])
            deadline = _fuel_assessment_deadline(
                alliance, crossing, auto_counts, context["fms_tie_selection"])
            if assessed > deadline:
                raise ValueError("FUEL assessment falls outside the pinned scoring window")
            event.update(crossing_time_s=crossing, assessed_time_s=assessed,
                         sensor_event_id=sensor_id)
        elif kind == "tower_award":
            occurred = _require_time(event["occurred_time_s"], "occurred_time_s")
            assessed = _require_time(event["assessed_time_s"], "assessed_time_s", 163.)
            alliance = event["alliance"]
            team = _require_text(event["team_id"], "team_id")
            robot = _require_text(event["robot_id"], "robot_id")
            phase, level = event["phase"], event["level"]
            if alliance not in ALLIANCES or team not in context["active_lineups"][alliance]:
                raise ValueError("TOWER award must identify a team in the active match lineup")
            slot = slot_by_robot.get(robot)
            if (slot is None or slot["alliance"] != alliance or slot["team_id"] != team):
                raise ValueError("TOWER alliance, team, and robot must match one active slot")
            if type(level) is not int:
                raise ValueError("TOWER level must be an integer")
            if phase == "auto":
                if not 0 <= occurred < 20 or not 20 <= assessed <= 23 or level != 1:
                    raise ValueError("AUTO TOWER award must be Level 1 and assessed after AUTO")
                value = 15
            elif phase == "teleop":
                if not 20 <= occurred < 160 or assessed < 160 or assessed > 163 or level not in (1, 2, 3):
                    raise ValueError("TELEOP TOWER award has invalid time, assessment, or level")
                value = {1: 10, 2: 20, 3: 30}[level]
            else:
                raise ValueError("TOWER phase must be auto or teleop")
            _require_text(event["adjudicator_id"], "adjudicator_id")
            _require_text(event["adjudication_id"], "adjudication_id")
            _require_text(event["evidence_ref"], "evidence_ref")
            _require_text(event["rule_ref"], "rule_ref")
            record = tower_events.setdefault((alliance, phase), {"robots": set(), "auto_l1": set()})
            if robot in record["robots"]:
                raise ValueError("a robot may receive at most one TOWER award per phase")
            record["robots"].add(robot)
            if phase == "auto":
                record["auto_l1"].add(robot)
                if len(record["auto_l1"]) > 2:
                    raise ValueError("AUTO Level 1 is capped at two robots per alliance")
            event.update(occurred_time_s=occurred, assessed_time_s=assessed)
        elif kind == "foul_award":
            occurred = _require_time(event["occurred_time_s"], "occurred_time_s")
            violating, beneficiary = event["violating_alliance"], event["beneficiary_alliance"]
            team, severity = event["team_id"], event["severity"]
            if (violating not in ALLIANCES or beneficiary not in ALLIANCES
                    or beneficiary == violating
                    or team not in context["active_lineups"][violating]):
                raise ValueError("FOUL award must credit the opponent of the violating alliance")
            points = {"minor": 5, "major": 15}.get(severity)
            if points is None:
                raise ValueError("FOUL severity must be minor or major")
            for field in ("adjudicator_id", "adjudication_id", "evidence_ref", "rule_ref"):
                _require_text(event[field], field)
            event["occurred_time_s"] = occurred
        elif kind == "disqualification":
            occurred = _require_time(event["occurred_time_s"], "occurred_time_s")
            alliance, team = event["alliance"], event["team_id"]
            if alliance not in ALLIANCES or team not in context["active_lineups"][alliance]:
                raise ValueError("DISQUALIFICATION must identify a team in the active match lineup")
            if team in disqualified:
                raise ValueError("team has duplicate DISQUALIFICATION events")
            disqualified.add(team)
            for field in ("adjudicator_id", "adjudication_id", "evidence_ref", "rule_ref"):
                _require_text(event[field], field)
            event["occurred_time_s"] = occurred
        elif kind == "ranking_points":
            occurred = _require_time(event["occurred_time_s"], "occurred_time_s", 163.)
            alliance, team = event["alliance"], event["team_id"]
            if alliance not in ALLIANCES or team not in context["active_lineups"][alliance]:
                raise ValueError("RANKING POINTS must identify a team in the active match lineup")
            if type(event["points"]) is not int or event["points"] < 0:
                raise ValueError("ranking-point awards must be nonnegative integers")
            _require_text(event["rp_type"], "rp_type")
            _require_text(event["source_ref"], "source_ref")
            _require_text(event["rule_ref"], "rule_ref")
            event["occurred_time_s"] = occurred
        normalized.append(event)
    return normalized


def _auto_counts(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {alliance: 0 for alliance in ALLIANCES}
    for event in events:
        if (event.get("event_type") == "fuel_sensor_crossing"
                and event.get("crossing_time_s", 160) < 20
                and event.get("sensor_confirmed") is True):
            counts[event["alliance"]] += 1
    return counts


def _shift_boundaries() -> tuple[float, ...]:
    return (30., 55., 80., 105., 130., 160.)


def _fuel_assessment_deadline(
    alliance: str,
    crossing: float,
    auto_counts: Mapping[str, int],
    fms: str | None,
) -> float:
    if crossing < 20.:
        return 23.
    # Find the first boundary at which this active HUB deactivates. The rule
    # grants three seconds for processing after that boundary; absent a later
    # deactivation, TELEOP's end assessment is the final bound.
    for boundary in _shift_boundaries():
        if boundary <= crossing or boundary >= 160.:
            continue
        before = hub_is_active(alliance, boundary - 1e-6, auto_counts,
                               fms_tie_selection=fms)
        after = hub_is_active(alliance, boundary, auto_counts,
                              fms_tie_selection=fms)
        if before and not after:
            return boundary + 3.
    return 163.


def _normalize_proxy_events(proxy_events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(proxy_events, (list, tuple)):
        raise ValueError("proxy_events must be an ordered list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous: tuple[float, str] | None = None
    for raw in proxy_events:
        if not isinstance(raw, Mapping) or set(raw) != {
                "event_id", "event_type", "event_time_s", "alliance", "source_event_id",
                "event_semantics"}:
            raise ValueError("proxy event has missing or unexpected fields")
        if raw["event_type"] != "fuel_action_score":
            raise ValueError("proxy accepts only explicit fuel_action_score events")
        event_id = _require_text(raw["event_id"], "proxy event_id")
        if event_id in seen:
            raise ValueError(f"duplicate proxy event_id: {event_id}")
        seen.add(event_id)
        time_s = _require_time(raw["event_time_s"], "proxy event_time_s")
        if raw["alliance"] not in ALLIANCES:
            raise ValueError("proxy event alliance must be red or blue")
        source_id = _require_text(raw["source_event_id"], "source_event_id")
        if raw["event_semantics"] != "abstract_score_action_not_sensor_crossing":
            raise ValueError("proxy event must declare its abstract score-action semantics")
        key = (time_s, event_id)
        if previous is not None and key < previous:
            raise ValueError("proxy events must be ordered by event time then event_id")
        previous = key
        result.append({"event_id": event_id, "event_type": "fuel_action_score",
            "event_time_s": time_s, "alliance": raw["alliance"],
            "source_event_id": source_id,
            "event_semantics": "abstract_score_action_not_sensor_crossing"})
    return result


def _normalize_fuel_proxy_events(
    events: Sequence[Mapping[str, Any]],
    *,
    slot_alliances: Sequence[str],
    auto_counts: Mapping[str, int],
    tie_selection: str,
    schedule_override: bool,
    time_step_s: float,
) -> list[dict[str, Any]]:
    if not isinstance(events, (list, tuple)):
        raise ValueError("fuel proxy events must be an ordered list")
    expected_keys = {
        "event_id", "event_type", "event_time_s", "source_tick",
        "source_event_id", "alliance", "robot_slot", "piece_id",
        "hub_active", "event_semantics",
    }
    normalized: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    source_ids: set[str] = set()
    last_key: tuple[int, int, int] | None = None
    for raw in events:
        if (not isinstance(raw, Mapping)
                or set(raw) not in (expected_keys, expected_keys | {"proxy_points"})):
            raise ValueError("fuel proxy event has missing or unexpected fields")
        if "proxy_points" in raw and raw["proxy_points"] != 1:
            raise ValueError("each abstract score-action event has exactly one proxy point")
        if raw["event_type"] != "fuel_action_score":
            raise ValueError("fuel proxy accepts only explicit abstract fuel score actions")
        if raw["event_semantics"] != "abstract_score_action_not_sensor_crossing":
            raise ValueError("fuel proxy event must identify abstract score-action semantics")
        event_id = _require_text(raw["event_id"], "event_id")
        source_id = _require_text(raw["source_event_id"], "source_event_id")
        if event_id in event_ids or source_id in source_ids:
            raise ValueError("fuel proxy event and source IDs must be unique")
        event_ids.add(event_id)
        source_ids.add(source_id)
        if type(raw["source_tick"]) is not int or raw["source_tick"] < 0:
            raise ValueError("source_tick must be a nonnegative integer")
        time_s = _require_time(raw["event_time_s"], "event_time_s")
        expected_time = (raw["source_tick"] + 1) * time_step_s
        if time_s != expected_time:
            raise ValueError("event_time_s must match its exact source physics tick")
        if time_s >= 160.:
            raise ValueError("proxy score-action event must be inside the half-open match clock")
        slot = raw["robot_slot"]
        if type(slot) is not int or not 0 <= slot < len(slot_alliances):
            raise ValueError("robot_slot must identify one of the six simulation slots")
        alliance = raw["alliance"]
        if alliance not in ALLIANCES or slot_alliances[slot] != alliance:
            raise ValueError("proxy event alliance must match its simulation slot")
        if type(raw["piece_id"]) is not int or raw["piece_id"] < 0:
            raise ValueError("piece_id must be a nonnegative simulator-local integer")
        if raw["hub_active"] is not True:
            raise ValueError("a scored abstract FUEL action must have an active HUB")
        key = (raw["source_tick"], slot, raw["piece_id"])
        if last_key is not None and key < last_key:
            raise ValueError("fuel proxy events must be in source tick/slot/piece order")
        last_key = key
        if not schedule_override:
            expected_active = hub_is_active(
                alliance, time_s, auto_counts,
                fms_tie_selection=tie_selection)
            if not expected_active:
                raise ValueError("fuel proxy event is scored in an inactive HUB")
        normalized.append({
            "event_id": event_id,
            "event_type": "fuel_action_score",
            "event_time_s": time_s,
            "source_tick": raw["source_tick"],
            "source_event_id": source_id,
            "alliance": alliance,
            "robot_slot": slot,
            "piece_id": raw["piece_id"],
            "hub_active": True,
            "event_semantics": "abstract_score_action_not_sensor_crossing",
            "proxy_points": 1,
        })
    return normalized


def build_fuel_proxy_ledger(
    *,
    simulation_episode_id: str,
    simulation_match_id: str,
    seed: int,
    world_index: int,
    reset_generation: int,
    time_step_s: float,
    slot_alliances: Sequence[str],
    events: Sequence[Mapping[str, Any]],
    schedule_override: str | None = None,
) -> dict[str, Any]:
    """Build an abstract six-slot proxy without official roster or match context.

    This API deliberately has no team IDs, official FMS result, or official
    score field. Its tie selection is a deterministic simulation stand-in.
    """
    episode_id = _require_text(simulation_episode_id, "simulation_episode_id")
    match_id = _require_text(simulation_match_id, "simulation_match_id")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(world_index) is not int or world_index < 0:
        raise ValueError("world_index must be a nonnegative integer")
    if type(reset_generation) is not int or reset_generation < 0:
        raise ValueError("reset_generation must be a nonnegative integer")
    if not _number(time_step_s) or time_step_s <= 0:
        raise ValueError("time_step_s must be finite and positive")
    if (not isinstance(slot_alliances, (list, tuple)) or len(slot_alliances) != 6
            or set(slot_alliances) != {"red", "blue"}
            or slot_alliances.count("red") != 3
            or slot_alliances.count("blue") != 3):
        raise ValueError("slot_alliances must map exactly three of six simulation slots to each alliance")
    if schedule_override is not None:
        schedule_override = _require_text(schedule_override, "schedule_override")
        if schedule_override not in ("curriculum_schedule", "behavior_probe"):
            raise ValueError("schedule_override must be a known diagnostic override or null")
    tie = derive_simulation_tie_selection(
        match_id, seed, world_index=world_index,
        reset_generation=reset_generation)
    preliminary = {alliance: 0 for alliance in ALLIANCES}
    if not isinstance(events, (list, tuple)):
        raise ValueError("fuel proxy events must be an ordered list")
    for raw in events:
        if not isinstance(raw, Mapping):
            raise ValueError("fuel proxy event must be an object")
        # The row is validated below; only count candidates with a valid time
        # and alliance here so malformed events still fail closed in order.
        time_s = _require_time(raw.get("event_time_s"), "event_time_s")
        if time_s < 20. and raw.get("alliance") in ALLIANCES:
            preliminary[raw["alliance"]] += 1
    clean_events = _normalize_fuel_proxy_events(
        events, slot_alliances=slot_alliances, auto_counts=preliminary,
        tie_selection=tie["alliance"], schedule_override=schedule_override is not None,
        time_step_s=float(time_step_s))
    points = {alliance: sum(
        1 for row in clean_events if row["alliance"] == alliance)
        for alliance in ALLIANCES}
    missing_evidence = [
        "official_match_context", "official_roster_identity",
        "official_fms_tie_record", "fuel_sensor_crossings",
        "fuel_score_window_coverage", "tower_adjudications",
        "referee_penalty_coverage", "disqualification_coverage",
    ]
    body = {
        "schema": "rebuilt-active-hub-fuel-proxy-ledger-v1",
        "contract_id": PROXY_CONTRACT_ID,
        "scope": "six_slot_2d_simulation",
        "rules_snapshot_sha256": RULES_SNAPSHOT_SHA256,
        "simulation_identity": {
            "episode_id": episode_id,
            "match_id": match_id,
            "seed": seed,
            "world_index": world_index,
            "reset_generation": reset_generation,
            "simulation_only": True,
        },
        "time_step_s": float(time_step_s),
        "slot_alliances": list(slot_alliances),
        "simulation_tie_selection": tie,
        "auto_fuel_counts": preliminary,
        "schedule_override": schedule_override,
        "rules_compliant_schedule": schedule_override is None,
        "event_semantics": "abstract_score_action_not_sensor_crossing",
        "events": clean_events,
        "events_sha256": canonical_sha256(clean_events),
        "event_count": len(clean_events),
        "alliance_points": points,
        "tower_points_complete": False,
        "fouls_complete": False,
        "alliance_roster_complete": False,
        "fuel_sensor_crossings_complete": False,
        "score_window_coverage_complete": False,
        "official_match_context_complete": False,
        "official_fms_record_present": False,
        "missing_official_evidence": missing_evidence,
        "qualification_eligible": False,
    }
    body["ledger_sha256"] = canonical_sha256(body)
    return body


def validate_fuel_proxy_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild and compare a simulation-only proxy ledger."""
    if not isinstance(ledger, Mapping):
        raise ValueError("fuel proxy ledger must be an object")
    forbidden = {"match_points", "official_match_points", "team_id",
                 "alliance_rosters", "active_lineups", "competition_context",
                 "fms_tie_selection"}
    if forbidden & set(ledger):
        raise ValueError("simulation-only proxy ledger contains official context or score fields")
    identity = ledger.get("simulation_identity")
    if not isinstance(identity, Mapping) or set(identity) != {
            "episode_id", "match_id", "seed", "world_index",
            "reset_generation", "simulation_only"}:
        raise ValueError("simulation-only identity is incomplete")
    if identity.get("simulation_only") is not True:
        raise ValueError("proxy identity must be simulation-only")
    expected = build_fuel_proxy_ledger(
        simulation_episode_id=identity["episode_id"],
        simulation_match_id=identity["match_id"], seed=identity["seed"],
        world_index=identity["world_index"],
        reset_generation=identity["reset_generation"],
        time_step_s=ledger.get("time_step_s"),
        slot_alliances=ledger.get("slot_alliances"),
        events=ledger.get("events"),
        schedule_override=ledger.get("schedule_override"),
    )
    if dict(ledger) != expected:
        raise ValueError("fuel proxy ledger digest, rows, or scope changed")
    return dict(ledger)


def _validate_coverage(coverage: Mapping[str, Any]) -> dict[str, bool]:
    if not isinstance(coverage, Mapping) or set(coverage) != _COVERAGE_FIELDS:
        raise ValueError("official score coverage must declare every completeness field exactly")
    if any(type(value) is not bool for value in coverage.values()):
        raise ValueError("official score completeness fields must be booleans")
    return {key: coverage[key] for key in sorted(_COVERAGE_FIELDS)}


def build_score_ledger(
    *,
    match_context: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    proxy_events: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate rows and produce separate proxy and complete-score outputs.

    Missing coverage never becomes a zero or a qualified result. This function
    is pure and accepts caller-provided records; it does not read simulator
    state or infer referee/TOWER decisions.
    """
    context = _validate_context(match_context)
    clean_proxy = _normalize_proxy_events(proxy_events)
    proxy_auto = {alliance: sum(
        1 for row in clean_proxy if row["alliance"] == alliance and row["event_time_s"] < 20.)
        for alliance in ALLIANCES}
    proxy_fms = context["fms_tie_selection"] if proxy_auto["red"] == proxy_auto["blue"] else None
    if proxy_auto["red"] == proxy_auto["blue"]:
        shift_1_inactive_alliance(proxy_auto, fms_tie_selection=proxy_fms)
    proxy_points = {alliance: 0 for alliance in ALLIANCES}
    for row in clean_proxy:
        if hub_is_active(row["alliance"], row["event_time_s"], proxy_auto,
                         fms_tie_selection=proxy_fms):
            proxy_points[row["alliance"]] += 1

    # Official AUTO FUEL counts determine the official Hub schedule. Resolve
    # AUTO events first, then validate all remaining rows against that schedule.
    if not isinstance(events, (list, tuple)):
        raise ValueError("events must be an ordered list")
    preliminary_auto = {alliance: 0 for alliance in ALLIANCES}
    for raw in events:
        if isinstance(raw, Mapping) and raw.get("event_type") == "fuel_sensor_crossing":
            crossing = _require_time(raw.get("crossing_time_s"), "crossing_time_s")
            if crossing < 20 and raw.get("sensor_confirmed") is True and raw.get("alliance") in ALLIANCES:
                preliminary_auto[raw["alliance"]] += 1
    if preliminary_auto["red"] == preliminary_auto["blue"]:
        shift_1_inactive_alliance(preliminary_auto,
            fms_tie_selection=context["fms_tie_selection"])
    elif context["fms_tie_selection"] is not None:
        raise ValueError("FMS tie selection is only valid when official AUTO FUEL totals tie")
    clean_events = _normalize_events(events, context, preliminary_auto)
    # Recount scored AUTO fuel, excluding malformed/inactive records; both HUBs
    # are active in AUTO, so this must equal the preliminary tally.
    auto_counts = _auto_counts(clean_events)
    if auto_counts != preliminary_auto:
        raise ValueError("AUTO FUEL schedule counts differ from validated sensor events")
    clean_coverage = _validate_coverage(coverage)
    completeness_reasons = [f"missing_{field}" for field in sorted(_COVERAGE_FIELDS)
                            if not clean_coverage[field]]

    alliance_points = {alliance: 0 for alliance in ALLIANCES}
    team_points = {team: 0 for alliance in ALLIANCES
                   for team in context["alliance_rosters"][alliance]}
    ranking_points = {team: 0 for alliance in ALLIANCES
                      for team in context["alliance_rosters"][alliance]}
    disqualified: dict[str, str] = {}
    for event in clean_events:
        kind = event["event_type"]
        if kind == "fuel_sensor_crossing":
            alliance = event["alliance"]
            points = 1 if hub_is_active(alliance, event["crossing_time_s"], auto_counts,
                                        fms_tie_selection=context["fms_tie_selection"]) else 0
            alliance_points[alliance] += points
            if event.get("team_id") is not None:
                team_points[event["team_id"]] += points
        elif kind == "tower_award":
            points = 15 if event["phase"] == "auto" else {1: 10, 2: 20, 3: 30}[event["level"]]
            alliance_points[event["alliance"]] += points
            team_points[event["team_id"]] += points
        elif kind == "foul_award":
            alliance_points[event["beneficiary_alliance"]] += {"minor": 5, "major": 15}[event["severity"]]
        elif kind == "disqualification":
            disqualified[event["team_id"]] = event["alliance"]
        elif kind == "ranking_points":
            ranking_points[event["team_id"]] += event["points"]

    team_match_points = dict(team_points)
    if context["competition_context"] == "qualification":
        for team_id in disqualified:
            team_match_points[team_id] = 0
            ranking_points[team_id] = 0
    else:
        for alliance in set(disqualified.values()):
            alliance_points[alliance] = 0
            for team in context["alliance_rosters"][alliance]:
                team_match_points[team] = 0

    official: dict[str, Any] | None = None
    if completeness_reasons:
        official_reason = completeness_reasons
    else:
        official_reason = []
        official = {
            "red": alliance_points["red"],
            "blue": alliance_points["blue"],
        }
    event_digests = [canonical_sha256(event) for event in clean_events]
    proxy_digests = [canonical_sha256(event) for event in clean_proxy]
    body = {
        "schema": "rebuilt-match-points-ledger-v1",
        "rules_snapshot_sha256": RULES_SNAPSHOT_SHA256,
        "match_context": context,
        "events": clean_events,
        "event_sha256": event_digests,
        "events_sha256": canonical_sha256(clean_events),
        "proxy_events": clean_proxy,
        "proxy_event_sha256": proxy_digests,
        "proxy_events_sha256": canonical_sha256(clean_proxy),
        "coverage": clean_coverage,
        PROXY_CONTRACT_ID: {
            "contract_id": PROXY_CONTRACT_ID,
            "rules_snapshot_sha256": RULES_SNAPSHOT_SHA256,
            "scope": "six_slot_2d_simulation",
            "event_semantics": "abstract_score_action_not_sensor_crossing",
            "alliance_points": proxy_points,
            "event_count": len(clean_proxy),
            "event_sha256": canonical_sha256(clean_proxy),
            "tower_points_complete": False,
            "fouls_complete": False,
            "alliance_roster_complete": False,
            "qualification_eligible": False,
        },
        OFFICIAL_SCORE_ID: official,
        "official_match_points_complete": official is not None,
        "official_match_points_incompleteness": official_reason,
        "team_match_points": team_match_points,
        "ranking_points": ranking_points,
        "ranking_points_separate": True,
        "disqualified_teams": sorted(disqualified),
        "qualification_eligible": False,
    }
    body["ledger_sha256"] = canonical_sha256(body)
    return body


def validate_score_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild and compare a finalized ledger, rejecting changed event rows."""
    if not isinstance(ledger, Mapping):
        raise ValueError("score ledger must be an object")
    if "match_points" in ledger:
        raise ValueError("ambiguous match_points field is forbidden")
    expected = build_score_ledger(
        match_context=ledger.get("match_context"),
        events=ledger.get("events"),
        proxy_events=ledger.get("proxy_events"),
        coverage=ledger.get("coverage"),
    )
    if dict(ledger) != expected:
        raise ValueError("score ledger digest, event rows, or canonical form changed")
    return dict(ledger)
