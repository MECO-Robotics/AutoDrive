"""Pure schema and replay reconstruction tests; no simulator is imported."""
from copy import deepcopy
import struct

import pytest

from frc_defense.meco_simulated_score_ledger import (
    ESTIMAND_SCHEMA, OUTCOME_NAME, PHASE0_PLAN_SHA256, RULES, RULES_ID,
    RULES_SHA256, SOURCE_PLAN_SHA256, build_score_ledger, make_event_stream,
    MecoScoreEventRecorder, canonical_bytes, estimand_approval_digest,
    load_estimand_approval, reconstruct_replay_score, sha256, validate_estimand_approval,
    validate_replay_score_binding,
)


SOURCE = "a" * 64


def _approval():
    return {
        "schema": ESTIMAND_SCHEMA,
        "approval_id": "human-review-001",
        "approved": True,
        "approver": "reviewer@example.org",
        "approved_at": "2026-10-25T12:00:00Z",
        "source_plan_sha256": SOURCE_PLAN_SHA256,
        "phase0_plan_sha256": PHASE0_PLAN_SHA256,
        "simulator_source_sha256": SOURCE,
        "rules_id": RULES_ID,
        "rules_sha256": RULES_SHA256,
        "primary_estimand": OUTCOME_NAME,
        "formula": "sum score_units by alliance over accepted FUEL score-award events",
        "unit": RULES["unit"],
        "included_outcomes": RULES["included_outcomes"],
        "excluded_outcomes": RULES["excluded_outcomes"],
        "interpretation_limits": "Whole-alliance modeled outcome; not official FRC points or causal robot contribution.",
        "intended_use": "Descriptive simulator outcome research only.",
    }


APPROVAL_SHA = estimand_approval_digest(_approval())


def _context(**overrides):
    value = {
        "episode_id": "episode-91",
        "seed": 91,
        "scenario_id": "pickup_simple",
        "robot_team_slots": [0, 0, 0, 1, 1, 1],
        "world_count": 1,
        "source_identity_sha256": SOURCE,
        "effective_scoring_configuration": {
            "dt": 0.02,
            "alliance_zone_depth_m": 3.0,
            "hub_schedule": "source-bound-current",
            "robot_control_modes": ["deterministic"] * 6,
            "robot_types": ["dumper"] * 6,
            "robot_roles": ["offense", "offense", "offense", "defense", "defense", "defense"],
            "robot_fuel_capacities": [8] * 6,
            "robot_score_interval_s": [0.2] * 6,
            "robot_dimensions_m": {"length": [1.] * 6, "width": [.7] * 6},
            "field_geometry": {"field_length_m": 16., "field_width_m": 8.,
                "hub_centers_m": [[4., 4.], [12., 4.]], "bump_regions": []},
            "robot_readiness": {"dumper_stationary_linear_mps": 0.2,
                "dumper_stationary_angular_radps": 0.2,
                "dumper_aim_tolerance_rad": 3.14159 / 9,
                "bump_clearance_source": "TensorThreeVsThreeEnv._robots_clear_of_bumps"},
            "score_intent_source": "effective policy action trace passed to update_fuel_actions",
            "score_rules_id": RULES_ID,
        },
        "effective_scoring_configuration_sha256": None,
        "fuel_count": 32,
        "terminal_state": "complete",
        "terminal_tick": 8,
    }
    value.update(overrides)
    if value["effective_scoring_configuration_sha256"] is None:
        value["effective_scoring_configuration_sha256"] = sha256(
            value["effective_scoring_configuration"])
    return value


def _event(event_id, tick, robot, piece, generation, **overrides):
    elapsed = _simulator_elapsed(tick)
    value = {
        "event_id": event_id,
        "episode_id": "episode-91",
        "tick": tick,
        "match_elapsed_s": elapsed,
        "world_slot": 0,
        "alliance": 0 if robot < 3 else 1,
        "robot_slot": robot,
        "piece_slot": piece,
        "piece_generation": generation,
        "event_type": "fuel_score_awarded",
        "score_units": 1,
        "source_site": "gamepiece_actions.update_fuel_actions.vectorized_score_mutation",
    }
    value.update(overrides)
    return value


def _simulator_elapsed(tick):
    elapsed = 0.
    dt = struct.unpack("=f", struct.pack("=f", .02))[0]
    for _ in range(tick + 1):
        elapsed = min(struct.unpack("=f", struct.pack("=f", elapsed + dt))[0], 160.)
    return elapsed


def _events():
    return [
        _event("event-0001", 1, 0, 4, 0),
        _event("event-0002", 1, 3, 7, 0),
        _event("event-0003", 4, 0, 4, 1),
    ]


def test_ledger_aggregates_only_versioned_score_events_deterministically():
    context, events, approval = _context(), _events(), _approval()
    first = build_score_ledger(context, events, approval, approval_sha256=APPROVAL_SHA)
    second = build_score_ledger(context, events, approval, approval_sha256=APPROVAL_SHA)

    assert first == second
    assert first["status"] == "complete"
    assert first["simulated_match_score"] == {
        "schema": "meco-simulated-score-ledger-v1", "rules_id": RULES_ID,
        "rules_sha256": RULES_SHA256, "red": 2, "blue": 1, "complete": True}
    assert first["official_match_points"] is None
    assert first["qualification_eligible"] is False
    assert first["ledger_sha256"] == sha256({
        key: value for key, value in first.items() if key != "ledger_sha256"})


def test_recorder_assigns_stable_ids_and_keeps_pre_respawn_generation():
    recorder = MecoScoreEventRecorder(_context(), _approval(), APPROVAL_SHA)
    recorder.record_award(tick=0, match_elapsed_s=_simulator_elapsed(0), world_slot=0,
        alliance=0, robot_slot=0, piece_slot=4, piece_generation=0,
        source_site="gamepiece_actions.update_fuel_actions.vectorized_score_mutation")
    recorder.record_award(tick=3, match_elapsed_s=_simulator_elapsed(3), world_slot=0,
        alliance=0, robot_slot=0, piece_slot=4, piece_generation=1,
        source_site="gamepiece_actions.update_fuel_actions.robot_loop_score_mutation")
    captured = recorder.finish(terminal_tick=8, terminal_state="complete",
                               terminal_score_counts=[2, 0])
    assert [row["piece_generation"] for row in captured["event_stream"]["events"]] == [0, 1]
    assert [row["event_id"] for row in captured["event_stream"]["events"]] == [
        "episode-91:w000000:e000000000000",
        "episode-91:w000000:e000000000001"]


def test_recorder_rejects_an_episode_reset_during_capture():
    recorder = MecoScoreEventRecorder(_context(), _approval(), APPROVAL_SHA)
    recorder.invalidate("episode reset during score event capture")
    with pytest.raises(ValueError, match="crossed an episode reset"):
        recorder.finish(terminal_tick=8, terminal_state="complete",
                        terminal_score_counts=[0, 0])


def test_approval_digest_is_bound_to_exact_canonical_artifact():
    approval = _approval()
    bad_digest = "b" * 64
    ledger = build_score_ledger(_context(), _events(), approval,
                                approval_sha256=bad_digest)
    assert ledger["status"] == "invalid"
    assert "exact artifact" in ledger["invalid_reason"]
    with pytest.raises(ValueError, match="exact approval artifact"):
        MecoScoreEventRecorder(_context(), approval, bad_digest)


def test_loaded_approval_digest_binds_canonical_file_bytes(tmp_path):
    approval = _approval()
    path = tmp_path / "estimand.json"
    path.write_bytes(canonical_bytes(approval))
    loaded, digest = load_estimand_approval(path, source_identity_sha256=SOURCE)
    assert loaded == approval
    assert digest == estimand_approval_digest(approval)
    path.write_text('{ "schema": "' + ESTIMAND_SCHEMA + '" }', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical immutable bytes"):
        load_estimand_approval(path, source_identity_sha256=SOURCE)


@pytest.mark.parametrize("mutation, message", [
    (lambda events: events.append(deepcopy(events[0])), "duplicate score event ID"),
    (lambda events: events.__setitem__(2, {**events[2], "piece_generation": 0}),
     "duplicate award for piece incarnation"),
    (lambda events: events.reverse(), "score event rows are reordered"),
    (lambda events: events[0].update(tick=99), "score event tick is outside"),
    (lambda events: events[0].update(match_elapsed_s=.031), "differs from simulator tick"),
    (lambda events: events[0].update(piece_slot=32), "piece slot is invalid"),
    (lambda events: events[0].update(piece_generation=1), "not the next scorer lifecycle"),
    (lambda events: events[0].update(match_elapsed_s=float("nan")), "clock is invalid"),
    (lambda events: events[0].update(alliance=1), "alliance differs"),
    (lambda events: events[0].update(score_units=2), "units differ"),
    (lambda events: events[0].update(source_site="untrusted"), "source site is unsupported"),
])
def test_invalid_event_mutations_fail_closed(mutation, message):
    events = deepcopy(_events())
    mutation(events)
    ledger = build_score_ledger(_context(), events, _approval(),
                                approval_sha256=APPROVAL_SHA)
    assert ledger["status"] == "invalid"
    if message:
        assert message in ledger["invalid_reason"]


def test_missing_context_and_censoring_are_incomplete_not_zero_scores():
    approval = _approval()
    missing = _context()
    missing.pop("scenario_id")
    incomplete = build_score_ledger(missing, _events(), approval,
                                    approval_sha256=APPROVAL_SHA)
    censored = build_score_ledger(_context(terminal_state="censored"), _events(),
                                  approval, approval_sha256=APPROVAL_SHA)
    for ledger in (incomplete, censored):
        assert ledger["status"] == "incomplete"
        assert ledger["simulated_match_score"]["red"] is None
        assert ledger["simulated_match_score"]["blue"] is None
        assert ledger["simulated_match_score"]["complete"] is False


@pytest.mark.parametrize("field,value,message", [
    ("approved", False, "not approved"),
    ("rules_sha256", "0" * 64, "different score rules"),
    ("simulator_source_sha256", "0" * 64, "different simulator source"),
    ("included_outcomes", ["tower_award"], "unsupported outcomes"),
    ("primary_estimand", "official_match_points", "different outcome"),
])
def test_approval_mutations_block_collection(field, value, message):
    approval = _approval()
    approval[field] = value
    with pytest.raises(ValueError, match=message):
        validate_estimand_approval(approval, source_identity_sha256=SOURCE)


def test_replay_reconstruction_uses_events_and_detects_summary_or_stream_mutation():
    context, events, approval = _context(), _events(), _approval()
    ledger = build_score_ledger(context, events, approval,
                                approval_sha256=APPROVAL_SHA)
    replay = {
        "score_event_stream": make_event_stream(context, events),
        "simulated_score_ledger": ledger,
        "estimand_approval_sha256": APPROVAL_SHA,
        "source_identity_sha256": SOURCE,
        "evaluation_record": {
            "simulated_score_ledger_sha256": ledger["ledger_sha256"],
            "simulated_match_score": ledger["simulated_match_score"],
            "source_identity_sha256": SOURCE,
        },
    }
    record = {
        "simulated_score_ledger": ledger,
        "simulated_match_score": ledger["simulated_match_score"],
        "official_match_points": None,
        "qualification_eligible": False,
        "source_identity_sha256": SOURCE,
    }
    assert validate_replay_score_binding(replay, record, [2, 1], approval) == ledger

    changed = deepcopy(replay)
    changed["score_event_stream"]["events"].pop()
    # Recompute the event digest and an unrelated outer digest: the ledger still
    # fails because replay-only reconstruction no longer matches the row.
    changed["score_event_stream"]["events_sha256"] = sha256(
        changed["score_event_stream"]["events"])
    changed["outer_file_sha256"] = sha256(changed)
    with pytest.raises(ValueError, match="score ledger differs"):
        validate_replay_score_binding(changed, record, [2, 1], approval)

    bad_record = deepcopy(record)
    bad_record["simulated_match_score"] = {**record["simulated_match_score"], "red": 99}
    with pytest.raises(ValueError, match="summary differs"):
        validate_replay_score_binding(replay, bad_record, [2, 1], approval)
    changed_approval_binding = deepcopy(replay)
    changed_approval_binding["estimand_approval_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="approval digest differs"):
        reconstruct_replay_score(changed_approval_binding, approval)


def test_terminal_counters_and_official_or_qualification_claims_are_not_aliases():
    context, events, approval = _context(), _events(), _approval()
    ledger = build_score_ledger(context, events, approval,
                                approval_sha256=APPROVAL_SHA)
    replay = {"score_event_stream": make_event_stream(context, events),
              "simulated_score_ledger": ledger,
              "estimand_approval_sha256": APPROVAL_SHA,
              "source_identity_sha256": SOURCE,
              "evaluation_record": {
                  "simulated_score_ledger_sha256": ledger["ledger_sha256"],
                  "simulated_match_score": ledger["simulated_match_score"],
                  "source_identity_sha256": SOURCE,
              }}
    record = {"simulated_score_ledger": ledger,
              "simulated_match_score": ledger["simulated_match_score"],
              "official_match_points": None, "qualification_eligible": False,
              "source_identity_sha256": SOURCE,
              "fuel_score_count": [2, 1], "useful_throughput": 0.0,
              "reward_total": 5.}
    with pytest.raises(ValueError, match="terminal simulator counters"):
        validate_replay_score_binding(replay, record, [0, 0], approval)
    record["official_match_points"] = 3
    with pytest.raises(ValueError, match="official match points"):
        validate_replay_score_binding(replay, record, [2, 1], approval)
