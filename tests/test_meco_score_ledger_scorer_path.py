"""Production scorer capture fixtures; intentionally gated on simulator E2E review."""
import pytest
import torch

from frc_defense.meco_simulated_score_ledger import (
    ESTIMAND_SCHEMA, OUTCOME_NAME, PHASE0_PLAN_SHA256, RULES, RULES_ID,
    RULES_SHA256, SOURCE_PLAN_SHA256, MecoScoreEventRecorder,
    estimand_approval_digest,
)
from frc_defense.meco_track_training import _meco_score_ledger_context
from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


def _approved_fixture(source_sha256):
    return {
        "schema": ESTIMAND_SCHEMA,
        "approval_id": "scorer-path-test-fixture",
        "approved": True,
        "approver": "test-only",
        "approved_at": "2026-10-25T00:00:00Z",
        "source_plan_sha256": SOURCE_PLAN_SHA256,
        "phase0_plan_sha256": PHASE0_PLAN_SHA256,
        "simulator_source_sha256": source_sha256,
        "rules_id": RULES_ID,
        "rules_sha256": RULES_SHA256,
        "primary_estimand": OUTCOME_NAME,
        "formula": "sum score_units by alliance over accepted FUEL score-award events",
        "unit": RULES["unit"],
        "included_outcomes": RULES["included_outcomes"],
        "excluded_outcomes": RULES["excluded_outcomes"],
        "interpretation_limits": "Test fixture only; not official points or causal contribution.",
        "intended_use": "Unit fixture only.",
    }


def _reward_from_score_events(env, active):
    red = env.fuel_scored_event[:, :3].sum(-1).float()
    blue = env.fuel_scored_event[:, 3:].sum(-1).float()
    reward = torch.where(env.team_ids[None] == 0,
                         (red - blue)[:, None], (blue - red)[:, None]).expand(-1, 6)
    return torch.where(active[:, None], reward, torch.zeros_like(reward))


@pytest.mark.skip(reason=(
    "Run only with the required seeded simulator E2E evaluation and fresh dashboard-currentness evidence."))
@pytest.mark.parametrize("vectorized", [True, False], ids=["vectorized", "robot-loop"])
def test_update_fuel_actions_capture_reuse_reset_and_reward_parity(vectorized):
    def make_env():
        # `cuda:0` is the first explicitly assigned supported WX9100 target;
        # this fixture has no CPU/P600 fallback.
        env = TensorThreeVsThreeEnv(num_envs=2, device="cuda:0", seed=91,
            fuel_count=96, randomize=False, perception_dropout=0,
            position_noise=0, velocity_noise=0, fuel_physics=False,
            robot_types=("turret",) * 6)
        env._vectorized_scoring_enabled = vectorized
        env.piece_active.zero_()
        env.piece_owner.fill_(-1)
        env.piece_active[0, 0] = True
        env.piece_owner[0, 0] = 0
        hub = env.hub_centers[0]
        env.sim.pose[0, 0] = torch.stack(
            (hub.new_tensor(env.alliance_zone_depth - 0.5),
             hub[1], hub.new_zeros(())))
        env.sim.velocity[0, 0].zero_()
        assert env.sim.pose[0, 0, 0] <= env.alliance_zone_depth
        assert env._robots_clear_of_bumps(env.sim.pose[:, :, :2])[0, 0]
        env.match_elapsed.zero_()
        env.steps.zero_()
        env.next_score.zero_()
        env.hub_active.fill_(True)
        return env

    captured_env = make_env()
    baseline_env = make_env()
    source_sha256 = "a" * 64
    approval = _approved_fixture(source_sha256)
    context = _meco_score_ledger_context(captured_env, episode_id="fixture", seed=91,
        scenario_id="pickup_simple", source_identity_sha256=source_sha256)
    recorder = MecoScoreEventRecorder(context, approval,
                                      estimand_approval_digest(approval))
    captured_env._meco_score_event_recorder = recorder
    active = torch.tensor([True, False], device="cuda:0")
    actions = torch.full((2, 6), 4, dtype=torch.long, device="cuda:0")

    def score_step(env):
        return env.step(actions, active_mask=active, capture_observation=False,
                        capture_info=True)

    first_captured = score_step(captured_env)
    first_baseline = score_step(baseline_env)
    first_captured_reward = _reward_from_score_events(captured_env, active)
    first_baseline_reward = _reward_from_score_events(baseline_env, active)
    assert captured_env._meco_piece_generation[0, 0].item() == 1
    assert captured_env.piece_owner[0, 0].item() == -1
    for env in (captured_env, baseline_env):
        env.piece_active[0, 0] = True
        env.piece_owner[0, 0] = 0
        env.next_score.zero_()

    second_captured = score_step(captured_env)
    second_baseline = score_step(baseline_env)
    second_captured_reward = _reward_from_score_events(captured_env, active)
    second_baseline_reward = _reward_from_score_events(baseline_env, active)
    assert [row["piece_generation"] for row in recorder.events] == [0, 1]
    assert all(row["piece_slot"] == 0 for row in recorder.events)
    assert captured_env._meco_piece_generation[0, 0].item() == 2
    for captured, baseline, expected_capture_reward, expected_baseline_reward in (
            (first_captured, first_baseline, first_captured_reward, first_baseline_reward),
            (second_captured, second_baseline, second_captured_reward, second_baseline_reward)):
        torch.testing.assert_close(captured[4]["team_score"], baseline[4]["team_score"],
                                   rtol=0, atol=0)
        torch.testing.assert_close(captured[4]["fuel_scored_event"],
                                   baseline[4]["fuel_scored_event"], rtol=0, atol=0)
        torch.testing.assert_close(captured[1], baseline[1], rtol=0, atol=0)
        torch.testing.assert_close(captured[1], expected_capture_reward, rtol=0, atol=0)
        torch.testing.assert_close(baseline[1], expected_baseline_reward, rtol=0, atol=0)

    captured_env._meco_piece_generation[1, 0] = 5
    captured_env.reset_done(torch.tensor([True, False], device="cuda:0"))
    assert captured_env._meco_piece_generation[0, 0].item() == 0
    assert captured_env._meco_piece_generation[1, 0].item() == 5
    assert captured_env._meco_world_reset_generation.tolist() == [1, 0]
    with pytest.raises(ValueError, match="crossed an episode reset"):
        recorder.finish(terminal_tick=2, terminal_state="complete",
                        terminal_score_counts=[2, 0])
