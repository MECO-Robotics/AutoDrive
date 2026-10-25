"""Deterministic contract tests; these do not construct or run the simulator."""
from copy import deepcopy

import pytest

from frc_defense.rebuilt_match_scoring import (
    ALLIANCES,
    PROXY_CONTRACT_ID,
    RULES_SNAPSHOT_SHA256,
    build_score_ledger,
    canonical_sha256,
    hub_is_active,
    match_period,
    shift_1_inactive_alliance,
    validate_score_ledger,
)


def context(*, competition="qualification", fms="red"):
    rosters = {
        "red": ["R1", "R2", "R3", "R4"] if competition == "championship_playoff" else ["R1", "R2", "R3"],
        "blue": ["B1", "B2", "B3", "B4"] if competition == "championship_playoff" else ["B1", "B2", "B3"],
    }
    lineups = {a: rosters[a][:3] for a in ALLIANCES}
    slots = [{"alliance": a, "team_id": team, "robot_id": f"{a}-{i}"}
             for a in ALLIANCES for i, team in enumerate(lineups[a])]
    return {"match_id": "test-match", "seed": 42, "competition_context": competition,
            "alliance_rosters": rosters, "active_lineups": lineups, "robot_slots": slots,
            "fms_tie_selection": fms}


def complete_coverage(value=True):
    return {
        "match_context_complete": value,
        "alliance_rosters_complete": value,
        "active_lineups_complete": value,
        "six_robot_slots_complete": value,
        "fuel_sensor_events_complete": value,
        "score_window_coverage_complete": value,
        "tower_adjudications_complete": value,
        "referee_penalty_coverage_complete": value,
        "disqualification_coverage_complete": value,
    }


def fuel(event_id, alliance, at, *, assessed=None, team=None, robot=None):
    return {"event_id": event_id, "event_type": "fuel_sensor_crossing",
            "crossing_time_s": at, "assessed_time_s": at if assessed is None else assessed,
            "alliance": alliance, "team_id": team, "robot_id": robot,
            "sensor_event_id": f"sensor-{event_id}", "sensor_confirmed": True,
            "rule_ref": "GM 6.5"}


def tower(event_id, alliance, team, robot, phase, level, at, assessed):
    return {"event_id": event_id, "event_type": "tower_award", "occurred_time_s": at,
            "assessed_time_s": assessed, "alliance": alliance, "team_id": team,
            "robot_id": robot, "phase": phase, "level": level,
            "adjudicator_id": "volunteer-1", "adjudication_id": f"adj-{event_id}",
            "evidence_ref": f"frame-{event_id}", "rule_ref": "GM 6.6"}


def foul(event_id, violating="red", severity="minor", team="R1"):
    return {"event_id": event_id, "event_type": "foul_award", "occurred_time_s": 40,
            "violating_alliance": violating,
            "beneficiary_alliance": "blue" if violating == "red" else "red",
            "team_id": team, "severity": severity, "adjudicator_id": "ref-1",
            "adjudication_id": f"adj-{event_id}", "evidence_ref": f"video-{event_id}",
            "rule_ref": "GM 6.5"}


def proxy(event_id, alliance, at):
    return {"event_id": event_id, "event_type": "fuel_action_score", "event_time_s": at,
            "alliance": alliance, "source_event_id": f"action-{event_id}",
            "event_semantics": "abstract_score_action_not_sensor_crossing"}


def build(events=(), *, ctx=None, proxy_events=(), coverage=None):
    return build_score_ledger(match_context=ctx or context(), events=list(events),
                              proxy_events=list(proxy_events),
                              coverage=coverage or complete_coverage())


def test_pinned_rules_and_half_open_period_boundaries():
    assert RULES_SNAPSHOT_SHA256 == "18cf92041290e8fa1df15d7e967fe5e107118a6d443fe2b6526c1c05b73f47a8"
    assert [match_period(t) for t in (0, 19.999, 20, 29.999, 30, 54.999,
                                      55, 79.999, 80, 104.999, 105, 129.999, 130, 159.999)] == [
        "auto", "auto", "transition", "transition", "shift_1", "shift_1",
        "shift_2", "shift_2", "shift_3", "shift_3", "shift_4", "shift_4", "end_game", "end_game"]
    with pytest.raises(ValueError):
        match_period(160)


def test_fms_tie_and_alternating_shift_activity():
    assert shift_1_inactive_alliance({"red": 2, "blue": 1}) == "red"
    with pytest.raises(ValueError, match="tie"):
        shift_1_inactive_alliance({"red": 1, "blue": 1})
    assert shift_1_inactive_alliance({"red": 1, "blue": 1}, fms_tie_selection="blue") == "blue"
    counts = {"red": 4, "blue": 2}
    assert hub_is_active("red", 29.9, counts)
    assert not hub_is_active("red", 30, counts)
    assert hub_is_active("red", 55, counts)
    assert not hub_is_active("blue", 55, counts)
    assert hub_is_active("blue", 80, counts)
    assert hub_is_active("red", 130, counts)


def test_proxy_is_explicitly_not_official_points_or_qualification():
    result = build(proxy_events=[proxy("p1", "red", 10), proxy("p2", "red", 31),
                                 proxy("p3", "blue", 31)])
    proxy_result = result[PROXY_CONTRACT_ID]
    assert proxy_result["alliance_points"] == {"red": 1, "blue": 1}
    assert proxy_result["scope"] == "two_robot_2d"
    assert proxy_result["tower_points_complete"] is False
    assert proxy_result["qualification_eligible"] is False
    assert result["official_match_points"] == {"red": 0, "blue": 0}
    assert result["qualification_eligible"] is False


def test_fuel_points_follow_auto_lead_shift_and_assessment_grace():
    rows = [fuel("a1", "red", 5), fuel("a2", "blue", 6),
            fuel("inactive", "red", 40), fuel("late", "blue", 54, assessed=57)]
    result = build(rows)
    assert result["official_match_points"] == {"red": 1, "blue": 2}
    assert result["team_match_points"]["B1"] == 0
    bad = [fuel("a1", "red", 5), fuel("a2", "blue", 6), fuel("late", "blue", 54, assessed=59)]
    with pytest.raises(ValueError, match="scoring window"):
        build(bad)


def test_fms_tie_uses_reported_selection_and_tower_values_caps():
    ctx = context(fms="blue")
    rows = [fuel("ra", "red", 1), fuel("ba", "blue", 2),
            tower("ta1", "red", "R1", "red-0", "auto", 1, 19, 20),
            tower("tt1", "red", "R1", "red-0", "teleop", 3, 150, 160),
            tower("tt2", "blue", "B1", "blue-0", "teleop", 2, 151, 160)]
    result = build(rows, ctx=ctx)
    assert result["official_match_points"] == {"red": 46, "blue": 21}
    too_many_auto = [tower(f"a{i}", "red", f"R{i+1}", f"red-{i}", "auto", 1, 19, 20)
                     for i in range(3)]
    with pytest.raises(ValueError, match="capped"):
        build(too_many_auto, ctx=ctx)
    with pytest.raises(ValueError, match="one TOWER award"):
        build([tower("t1", "red", "R1", "red-0", "teleop", 1, 100, 160),
               tower("t2", "red", "R1", "red-0", "teleop", 2, 101, 160)], ctx=ctx)


def test_foul_disqualification_and_ranking_points_stay_separate():
    ranking = {"event_id": "rp", "event_type": "ranking_points", "occurred_time_s": 160,
               "alliance": "red", "team_id": "R1", "rp_type": "bonus", "points": 2,
               "source_ref": "match-report", "rule_ref": "GM 6.6"}
    result = build([foul("f1"), foul("f2", severity="major"), ranking])
    assert result["official_match_points"] == {"red": 0, "blue": 20}
    assert result["ranking_points"]["R1"] == 2
    dq = {"event_id": "dq", "event_type": "disqualification", "occurred_time_s": 159,
          "alliance": "red", "team_id": "R1", "adjudicator_id": "ref-1",
          "adjudication_id": "dq-1", "evidence_ref": "record-1", "rule_ref": "GM 10.6"}
    qualified = build([foul("f1"), dq, ranking])
    assert qualified["team_match_points"]["R1"] == 0
    assert qualified["ranking_points"]["R1"] == 0
    playoff = build([foul("f1"), dq], ctx=context(competition="playoff"))
    assert playoff["official_match_points"]["red"] == 0


def test_championship_four_team_roster_has_three_slots_and_mutation_rejected():
    ledger = build([foul("mutation-check")],
                   ctx=context(competition="championship_playoff"))
    assert len(ledger["match_context"]["alliance_rosters"]["red"]) == 4
    assert len(ledger["match_context"]["active_lineups"]["red"]) == 3
    assert len(ledger["match_context"]["robot_slots"]) == 6
    assert validate_score_ledger(ledger) == ledger
    altered = deepcopy(ledger)
    altered["events"][0]["severity"] = "major"
    with pytest.raises(ValueError, match="changed"):
        validate_score_ledger(altered)


def test_incomplete_coverage_returns_null_not_zero_complete_score():
    coverage = complete_coverage(False)
    ledger = build(coverage=coverage)
    assert ledger["official_match_points"] is None
    assert ledger["official_match_points_complete"] is False
    assert len(ledger["official_match_points_incompleteness"]) == len(coverage)
    assert ledger["qualification_eligible"] is False
