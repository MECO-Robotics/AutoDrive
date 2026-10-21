"""Native scenario integration checks; accompany each run with simulator smoke."""
import torch
import pytest
from frc_defense.curriculum_scenarios import make_curriculum_env, reset_curriculum_scenario, teacher_actions
from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


def test_nn_offense_is_opt_in():
    with pytest.raises(ValueError, match='only for defense'):
        TensorThreeVsThreeEnv(num_envs=1, device='cpu', control_modes=('none',)*3+('nn','none','none'),
                              robot_roles=('offense',)*6, fuel_count=96)


def test_phases_masks_reset_and_sensor_teacher():
    torch.set_num_threads(1)
    env = make_curriculum_env('inactive_ferry_preloaded','inactive/preloaded',1,'cpu',17,horizon=20)
    assert not env.hub_active[0,1]
    assert env._last_obs.shape == (1,6,137)
    assert not env._last_obs[0,3,-8:-2].bool().any()
    actions = teacher_actions(env,env._last_obs)
    assert actions.shape == (1,6) and actions.dtype == torch.long
    assert actions[0,3] == 6
    env.step(actions)
    assert not env.hub_active[0,1]
    reset_curriculum_scenario(env,17)
    assert env.curriculum_metrics.elapsed.item() == 0
    assert (env.piece_owner == 3).sum().item() == 6
    assert not env.curriculum_finished.any()


def test_seeded_layouts_capacity_and_physical_initial_transition_stock():
    torch.set_num_threads(1)
    env = make_curriculum_env('pickup_scattered','dense',1,'cpu',17,horizon=10)
    positions = env.piece_pos.clone()
    assert env.curriculum['targets']['amount'] <= env.robot_fuel_capacity_values[3]
    reset_curriculum_scenario(env,17)
    assert torch.equal(positions, env.piece_pos)
    transition = make_curriculum_env('hub_transition_cycles',
        'inactive_to_active/empty/enough_time/staged_at_home',1,'cpu',17,horizon=10)
    assert transition.curriculum_metrics.initial_staged.sum() == 4
    assert (transition.piece_owner[:,80:84] == -1).all()
    assert not transition.hub_active[0,1]


def test_teacher_ignores_hidden_unobserved_fuel_truth():
    torch.set_num_threads(1)
    env = make_curriculum_env('pickup_simple','single',1,'cpu',17,horizon=10)
    first = teacher_actions(env,env._last_obs)
    env.piece_pos.add_(1000.)  # sensor tracks remain unchanged
    second = teacher_actions(env,env._last_obs)
    assert torch.equal(first,second)


def test_pickup_completion_bonus_once_and_success_freezes_physics():
    torch.set_num_threads(1)
    env = make_curriculum_env('pickup_simple','single',1,'cpu',17,horizon=50)
    reward = 0.
    for _ in range(50):
        _,_,done,_,_ = env.step(teacher_actions(env,env._last_obs))
        reward += env.curriculum_metrics.reward_delta().item()
        if done.all(): break
    assert done.all()
    assert reward == pytest.approx(5.02)
    position = env.sim.pose.clone()
    env.step(teacher_actions(env,env._last_obs))
    assert torch.equal(position,env.sim.pose)
    assert env.curriculum_metrics.reward_delta().item() == 0


def test_loaded_pickup_search_cannot_release_required_load():
    torch.set_num_threads(1)
    env = make_curriculum_env('pickup_scattered','initially_hidden',1,'cpu',17,horizon=10)
    env.piece_owner[:,0] = 3
    indices,valid,_ = env._candidates()
    mask,_ = env._action_mask(indices,valid)
    assert not mask[0,3,6]
    assert teacher_actions(env,env._last_obs)[0,3] != 6


def test_airborne_release_requires_fuel_physics():
    with pytest.raises(ValueError, match='requires fuel_physics'):
        make_curriculum_env('hub_transition_cycles','inactive_to_active/empty/enough_time/after_release',1,'cpu',17,fuel_physics=False)


def test_team_inactive_schedule_includes_receiver_activation():
    torch.set_num_threads(1)
    env = make_curriculum_env('team_delivery_cycles','inactive/open/receiver',1,'cpu',17,horizon=10)
    assert env.curriculum['schedule'] == ((0.,False),(45.,True))
    assert not env.hub_active[0,1]
    assert env.control_modes[4] == 'deterministic'


@pytest.mark.parametrize('device', ['cuda:1', 'cuda', 'cpu'])
def test_requested_cuda_device_selected_before_environment_allocation(monkeypatch, device):
    from frc_defense import curriculum_scenarios
    calls = []
    class StopBeforeAllocation(Exception):
        pass
    def construct(**kwargs):
        calls.append(('construct', kwargs['device']))
        raise StopBeforeAllocation
    monkeypatch.setattr(torch.cuda, 'set_device', lambda requested: calls.append(('select', str(requested))))
    monkeypatch.setattr(curriculum_scenarios, 'CurriculumEnv', construct)
    with pytest.raises(StopBeforeAllocation):
        make_curriculum_env('pickup_simple', 'single', 1, device, 17)
    expected = [('select', 'cuda:1'), ('construct', 'cuda:1')] if device == 'cuda:1' else [('construct', device)]
    assert calls == expected
