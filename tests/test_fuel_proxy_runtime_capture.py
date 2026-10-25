"""Offline fixtures for the scorer-to-proxy capture seam; no simulator runs."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from frc_defense.gamepiece_actions import (
    _capture_fuel_proxy_score_slots,
    _capture_fuel_proxy_step_events,
)
from frc_defense.rebuilt_match_scoring import validate_fuel_proxy_ledger
from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


def test_scorer_slot_capture_builds_ordered_proxy_ledger_without_raw_mutations():
    env = SimpleNamespace(
        fuel_proxy_enabled=True,
        simulation_episode_id="offline-episode",
        simulation_match_id="offline-match",
        simulation_seed=74,
        n=1,
        horizon=4,
        dt=.02,
        _fuel_proxy_reset_generation=torch.tensor([0]),
        _fuel_proxy_piece_events=torch.full((1, 4, 6), -1, dtype=torch.long),
        _fuel_proxy_step_piece_ids=torch.full((1, 6), -1, dtype=torch.long),
        _world_indices=torch.arange(1),
        steps=torch.tensor([1]),
        curriculum_schedule=None,
        behavior_probe=None,
        fuel_score_count=torch.tensor([[2, 3]]),
        auto_fuel_scores=torch.tensor([[1, 2]]),
        reward=torch.tensor([[.25, .5]]),
    )
    raw_before = (env.fuel_score_count.clone(), env.auto_fuel_scores.clone(),
                  env.reward.clone())

    # These are the selected local piece IDs returned from the real scorer
    # seam; slots 0 and 3 map to red and blue in the six-slot runtime.
    _capture_fuel_proxy_score_slots(
        env, torch.tensor([True]), torch.tensor([[11, -1, -1, 17, -1, -1]]))
    _capture_fuel_proxy_step_events(env, torch.tensor([True]))
    ledger = TensorThreeVsThreeEnv.build_fuel_proxy_ledgers(env)[0]

    assert [row["robot_slot"] for row in ledger["events"]] == [0, 3]
    assert [row["piece_id"] for row in ledger["events"]] == [11, 17]
    assert [row["event_time_s"] for row in ledger["events"]] == [.02, .02]
    assert all(row["event_semantics"] ==
               "abstract_score_action_not_sensor_crossing"
               for row in ledger["events"])
    assert len({row["event_id"] for row in ledger["events"]}) == 2
    assert ledger["qualification_eligible"] is False
    assert "official_match_points" not in ledger
    assert validate_fuel_proxy_ledger(ledger) == ledger
    torch.testing.assert_close(env.fuel_score_count, raw_before[0])
    torch.testing.assert_close(env.auto_fuel_scores, raw_before[1])
    torch.testing.assert_close(env.reward, raw_before[2])
