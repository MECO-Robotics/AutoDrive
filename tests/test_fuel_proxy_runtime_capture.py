"""GPU-targeted scorer-to-proxy regressions; do not substitute CPU fixtures."""
import pytest

torch = pytest.importorskip("torch")

from frc_defense.rebuilt_match_scoring import validate_fuel_proxy_ledger
from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


def _wx9100_device():
    if not torch.cuda.is_available():
        pytest.skip("requires the supported WX9100 GPU")
    name = torch.cuda.get_device_name(0).lower().replace(" ", "")
    if "wx9100" not in name:
        pytest.skip(f"requires the supported WX9100 GPU, found {torch.cuda.get_device_name(0)}")
    return "cuda:0"


def _make_meco_env(worlds=1, horizon=8):
    return TensorThreeVsThreeEnv(
        num_envs=worlds,
        device=_wx9100_device(),
        seed=74,
        horizon=horizon,
        randomize=False,
        fuel_count=96,
        control_modes=("deterministic", "none", "none", "none", "none", "none"),
        robot_roles=("offense",) * 6,
        robot_types=("turret",) * 6,
        fuel_proxy_enabled=True,
        simulation_episode_id="gpu-score-episode",
        simulation_match_id="gpu-score-match",
        simulation_seed=74,
    )


def _prepare_score_piece(env, world, piece_id, event_time_s):
    scorer_pose = torch.tensor(
        [2.0, 1.0, 0.0], device=env.device, dtype=env.sim.pose.dtype)
    env.sim.pose[world, 0] = scorer_pose
    env.policy_pose[world, 0] = scorer_pose
    env.sim.velocity[world, 0].zero_()
    env.piece_active[world].zero_()
    env.piece_owner[world].fill_(-1)
    env.piece_active[world, piece_id] = True
    env.piece_owner[world, piece_id] = 0
    env.piece_pos[world, piece_id] = env.sim.pose[world, 0, :2]
    env.match_elapsed[world] = event_time_s
    env.hub_active[world].fill_(True)
    env.next_score[world, 0] = 0.


def _run_scorer_tick(env, active):
    score_intent = torch.zeros((env.n, 6), device=env.device, dtype=torch.bool)
    score_intent[:, 0] = True
    env._update_fuel(active, score_intent)
    # This is the production post-scorer persistence seam used by step().
    from frc_defense.gamepiece_actions import _capture_fuel_proxy_step_events
    _capture_fuel_proxy_step_events(env, active)


def _run_scoring_action_step(env):
    actions = torch.full((env.n, 6), 7, device=env.device, dtype=torch.long)
    actions[:, 0] = 4
    return env.step(actions)


@pytest.mark.parametrize(("event_time_s", "expected_auto"), [
    (19.98, 1), (20.0, 0), (20.02, 0),
])
def test_cuda_meco_scorer_uses_pinned_half_open_auto_boundary(
        event_time_s, expected_auto):
    event_tick = round(event_time_s / .02)
    env = _make_meco_env(horizon=max(8, event_tick + 1))
    assert env.auto_period_end_time_s == 20.0
    assert expected_auto == int(event_time_s < env.auto_period_end_time_s)
    _prepare_score_piece(env, 0, 11, event_time_s)
    env.steps[0] = event_tick
    active = torch.ones(1, device=env.device, dtype=torch.bool)

    score_before = env.fuel_score_count.clone()
    auto_before = env.auto_fuel_scores.clone()
    _run_scorer_tick(env, active)

    assert env.fuel_scored_event[0, 0].item() == 1
    assert env.fuel_score_count[0, 0].item() == score_before[0, 0].item() + 1
    assert env.auto_fuel_scores[0, 0].item() == auto_before[0, 0].item() + expected_auto
    score_after_scorer = env.fuel_score_count.clone()
    auto_after_scorer = env.auto_fuel_scores.clone()
    ledger = validate_fuel_proxy_ledger(env.build_fuel_proxy_ledgers()[0])
    assert len(ledger["events"]) == 1
    event = ledger["events"][0]
    assert event["event_id"] == (
        "fuel-score:gpu-score-episode:w0:g0:t0:r0:p11")
    assert event["alliance"] == "red"
    assert event["robot_slot"] == 0
    assert event["piece_id"] == 11
    assert event["event_time_s"] == event_time_s
    assert ledger["qualification_eligible"] is False
    assert "official_match_points" not in ledger
    torch.testing.assert_close(env.fuel_score_count, score_after_scorer)
    torch.testing.assert_close(env.auto_fuel_scores, auto_after_scorer)


def test_cuda_meco_step_reward_and_scores_are_unchanged_by_ledger_export():
    env = _make_meco_env()
    _prepare_score_piece(env, 0, 11, .02)
    _, reward, _, _, _ = _run_scoring_action_step(env)

    assert env.fuel_scored_event[0, 0].item() == 1
    assert reward[0].tolist() == [1., 1., 1., -1., -1., -1.]
    score_after_step = env.fuel_score_count.clone()
    auto_after_step = env.auto_fuel_scores.clone()
    events_after_step = env.fuel_scored_event.clone()
    reward_after_step = reward.clone()

    ledger = validate_fuel_proxy_ledger(env.build_fuel_proxy_ledgers()[0])
    assert [row["event_id"] for row in ledger["events"]] == [
        "fuel-score:gpu-score-episode:w0:g0:t0:r0:p11"]
    torch.testing.assert_close(env.fuel_score_count, score_after_step)
    torch.testing.assert_close(env.auto_fuel_scores, auto_after_step)
    torch.testing.assert_close(env.fuel_scored_event, events_after_step)
    torch.testing.assert_close(reward, reward_after_step)


def test_cuda_mask_and_partial_reset_preserve_ledger_rows_and_refresh_ids():
    env = _make_meco_env(worlds=2)
    active_both = torch.ones(2, device=env.device, dtype=torch.bool)
    _prepare_score_piece(env, 0, 11, .02)
    _prepare_score_piece(env, 1, 17, .02)
    env.steps[:] = 1
    _run_scorer_tick(env, active_both)

    initial = env.build_fuel_proxy_ledgers()
    assert [row["event_id"] for row in initial[0]["events"]] == [
        "fuel-score:gpu-score-episode:w0:g0:t0:r0:p11"]
    assert [row["event_id"] for row in initial[1]["events"]] == [
        "fuel-score:gpu-score-episode:w1:g0:t0:r0:p17"]

    _prepare_score_piece(env, 0, 12, .04)
    env.steps[:] = 2
    active_world_zero = torch.tensor([True, False], device=env.device)
    _run_scorer_tick(env, active_world_zero)
    masked = env.build_fuel_proxy_ledgers()
    assert [row["event_id"] for row in masked[0]["events"]] == [
        "fuel-score:gpu-score-episode:w0:g0:t0:r0:p11",
        "fuel-score:gpu-score-episode:w0:g0:t1:r0:p12",
    ]
    assert [row["event_id"] for row in masked[1]["events"]] == [
        "fuel-score:gpu-score-episode:w1:g0:t0:r0:p17"]

    env.reset_done(torch.tensor([True, False], device=env.device))
    _prepare_score_piece(env, 0, 13, .02)
    env.steps[0] = 1
    _run_scorer_tick(env, active_world_zero)
    after_reset = env.build_fuel_proxy_ledgers()
    assert [row["event_id"] for row in after_reset[0]["events"]] == [
        "fuel-score:gpu-score-episode:w0:g1:t0:r0:p13"]
    assert [row["event_id"] for row in after_reset[1]["events"]] == [
        "fuel-score:gpu-score-episode:w1:g0:t0:r0:p17"]
    assert after_reset[0]["simulation_identity"]["reset_generation"] == 1
    assert after_reset[1]["simulation_identity"]["reset_generation"] == 0
