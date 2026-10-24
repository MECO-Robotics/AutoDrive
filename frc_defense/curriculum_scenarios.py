"""Seeded draft offense tasks using normal physics, sensors and NN execution.

All thresholds/budgets remain unqualified. The observation-only scripted teacher
is a diagnostic baseline, not a qualified reference or imitation dataset.
"""
from __future__ import annotations
import math
import torch

from .architecture_scenarios import reset_scenario
from .curriculum_metrics import CurriculumMetrics
from .fuel_physics_runtime import reset_fuel_physics
from .meco_phase4 import validate_phase4_factor
from .tensor_3v3 import TensorThreeVsThreeEnv
from .training_curriculum import get_scenario

_PHASE4_SENSOR_STATE_FIELDS = (
    'policy_pose',
    'track_pos', 'track_vel', 'track_age', 'track_mask', 'track_confidence',
    'track_spatial_extent', 'track_angular_extent', 'track_merged',
    'perception_quality_state', 'opponent_pose', 'opponent_velocity',
    'opponent_size', 'opponent_valid', 'opponent_age',
)


class CurriculumEnv(TensorThreeVsThreeEnv):
    def _phase4_apply_confidence_cap(self):
        """Cap only sensed FUEL confidence for the learner robot."""
        cap = getattr(self, 'phase4_confidence_cap', None)
        if cap is None:
            return
        sensed = self.track_mask[:, 3]
        confidence = self.track_confidence[:, 3]
        reduced = sensed & (confidence > float(cap))
        confidence.copy_(torch.where(sensed, confidence.clamp(max=float(cap)), confidence))
        metrics = getattr(self, 'phase4_runtime_metrics', None)
        if metrics is not None:
            metrics['confidence_cap_refresh_count'] += 1
            metrics['confidence_cap_sensed_track_count'] = int(sensed.sum().item())
            metrics['confidence_cap_reduced_track_values'] += int(reduced.sum().item())
            metrics['confidence_cap_final_mean'] = (
                float(confidence[sensed].mean().item()) if bool(sensed.any()) else None)

    def _update_perception(self, active, active_count=None):
        result = super()._update_perception(active, active_count=active_count)
        self._phase4_apply_confidence_cap()
        return result

    def _phase4_capture_sensor_frame(self, observation=None):
        frame = {name: getattr(self, name).clone()
                 for name in _PHASE4_SENSOR_STATE_FIELDS}
        frame['observation'] = (self._last_obs if observation is None else observation).clone()
        return frame

    def _phase4_restore_sensor_frame(self, frame):
        for name in _PHASE4_SENSOR_STATE_FIELDS:
            getattr(self, name).copy_(frame[name])
        self._last_obs = frame['observation'].clone()

    def step(self, actions, active_mask=None, **kwargs):
        requested = torch.ones(self.n, device=self.device, dtype=torch.bool) if active_mask is None else torch.as_tensor(active_mask, device=self.device, dtype=torch.bool)
        active_requested = requested & ~self.curriculum_finished
        delay = int(getattr(self, 'phase4_observation_delay_ticks', 0))
        observation, reward, done, truncated, info = super().step(actions, active_requested, **kwargs)
        if delay:
            current = self._phase4_capture_sensor_frame(observation)
            self._phase4_current_sensor_frame = current
            history = self._phase4_observation_history
            history.append(current)
            if len(history) > delay + 1:
                history.pop(0)
            delayed = history[0]
            self._phase4_restore_sensor_frame(delayed)
            observation = delayed['observation']
            self.phase4_runtime_metrics['delayed_observation_ticks'] += int(active_requested.sum().item())
        success = self.curriculum_metrics.outcomes()['success']
        newly_finished = success & ~self.curriculum_finished & requested
        self.curriculum_metrics.delta += 5. * newly_finished.float()
        self.curriculum_finished |= success
        return observation, reward, done | self.curriculum_finished, truncated, info

    def _update_fuel(self, active, score_intent):
        result = super()._update_fuel(active, score_intent)
        if hasattr(self, 'curriculum_metrics'):
            self.curriculum_metrics.update(active)
        return result

    def _curriculum_commands(self, commands, active):
        if not hasattr(self, 'curriculum'):
            return commands
        pressure = self.curriculum['pressure']
        if pressure['level'] == 0:
            return commands
        # Scripted blockers still use ordinary chassis/contact physics. They
        # have NN controller slots so deterministic offense cannot overwrite
        # these scripts; their strategic action is hold.
        for slot in pressure['slots']:
            commands[:, slot] = 0.
            level = pressure['level']
            factor = getattr(self, 'phase4_factor_config', None)
            if factor and slot == 0 and factor['kind'] in (
                    'feinting_opponent', 'abrupt_opponent_policy_change'):
                tick = factor['reversal_tick'] if factor['kind'] == 'feinting_opponent' else factor['switch_tick']
                after_switch = self.steps >= tick
                if factor['kind'] == 'feinting_opponent':
                    sign = torch.where(after_switch, -1., 1.)
                    anchor = self.phase4_defender_anchor[:, slot, :2]
                    target = anchor.clone()
                    target[:, 1] += sign * float(factor['telegraph_distance_m'])
                    commands[:, slot, :2] = (target - self.policy_pose[:, slot, :2]).clamp(-1., 1.)
                    if bool(after_switch.any()) and self.phase4_runtime_metrics['feint_reversal_tick'] is None:
                        self.phase4_runtime_metrics['feint_reversal_tick'] = int(tick)
                else:
                    if factor['behavior_a'] == 'orbit':
                        phase = self.match_elapsed * 1.2 + pressure['phase']
                        target = self.phase4_defender_anchor[:, slot, :2].clone()
                        target[:, 1] += torch.sin(phase) * .6
                        target[:, 0] += torch.cos(phase * .7 + slot) * .8
                        behavior_a_commands = (target - self.policy_pose[:, slot, :2]).clamp(-1., 1.)
                    else:
                        behavior_a_commands = torch.zeros_like(commands[:, slot, :2])
                    delta = self.opponent_pose[:, slot, :2] - self.policy_pose[:, slot, :2]
                    valid = self.opponent_valid[:, slot]
                    behavior_b_commands = torch.where(valid[:, None], delta.clamp(-1.5, 1.5), 0.)
                    commands[:, slot, :2] = torch.where(after_switch[:, None],
                        behavior_b_commands, behavior_a_commands)
                    if bool(after_switch.any()) and self.phase4_runtime_metrics['policy_change_tick'] is None:
                        self.phase4_runtime_metrics['policy_change_tick'] = int(tick)
                continue
            if level == 2 or (level == 5 and slot == 0) or pressure['family'] == 'orbiting_interceptor':
                phase = self.match_elapsed * 1.2 + pressure['phase']
                target = self.policy_pose[:, slot, :2].clone()
                target[:, 1] = pressure['anchor_y'] + torch.sin(phase) * .6
                if pressure['family'] == 'orbiting_interceptor':
                    target[:, 0] = 10.8 + torch.cos(phase * .7 + slot) * .8
                    target[:, 1] += slot * .45
                commands[:, slot, :2] = (target - self.policy_pose[:, slot, :2]).clamp(-1., 1.)
            elif pressure['family'] == 'split_chaser':
                delta = self.opponent_pose[:, slot, :2] - self.policy_pose[:, slot, :2]
                valid = self.opponent_valid[:, slot]
                phase = self.match_elapsed * .65 + pressure['phase']
                offset = torch.stack((torch.cos(phase), torch.sin(phase)), -1) * .8
                commands[:, slot, :2] = torch.where(valid[:, None],
                    (delta + offset).clamp(-1.5, 1.5), 0.)
            elif level == 4 or (level == 5 and slot == 1):
                # Only the defender's simulated opponent observation is used.
                delta = self.opponent_pose[:, slot, :2] - self.policy_pose[:, slot, :2]
                valid = self.opponent_valid[:, slot]
                commands[:, slot, :2] = torch.where(valid[:, None], delta.clamp(-1.5, 1.5), 0.)
        return torch.where(active[:, None, None], commands, 0.)

    def _reset_phase4_observation_buffer(self, observation=None):
        delay = int(getattr(self, 'phase4_observation_delay_ticks', 0))
        current = self._phase4_capture_sensor_frame(observation)
        self._phase4_current_sensor_frame = current
        self._phase4_observation_history = [self._phase4_capture_sensor_frame(
            current['observation']) for _ in range(delay + 1)]


def make_curriculum_env(scenario_id, category, num_envs, device, seed, *, horizon=None, fuel_physics=True,
                        production_d0=False, layout_id='2026_rebuilt',
                        defender_family='category_scripted', start_condition='scenario_defined',
                        sensor_profile='standard', phase4_factor=None):
    """Build a curriculum environment, optionally running production D0 in slot 3.

    ``production_d0`` changes only the learner slot's controller from the
    default NN controller to the native deterministic offense controller.
    Scenario-specific teammate and scripted pressure/blocker assignments are
    still applied below, and the same reset path is used in either mode.
    """
    definition = get_scenario(scenario_id)
    if defender_family not in ('none', 'category_scripted', 'orbiting_interceptor', 'split_chaser'):
        raise ValueError('unknown scripted defender family')
    if start_condition not in ('scenario_defined', 'offset_upper_lane', 'offset_lower_lane'):
        raise ValueError('unknown qualification start condition')
    if sensor_profile not in ('standard', 'noisy'):
        raise ValueError('unknown qualification sensor profile')
    if category.endswith('/after_release') and not fuel_physics:
        raise ValueError('after_release requires fuel_physics=True for an actual airborne initial release')
    if category not in definition.required_categories:
        raise ValueError(f'unknown category for {scenario_id}: {category}')
    phase4_factor = validate_phase4_factor(phase4_factor)
    pressure_level = _pressure_level(category)
    if phase4_factor is not None and (scenario_id not in (
            'offense_vs_scripted_pressure', 'full_match_offense')
            or pressure_level not in (1, 2, 3, 4, 5)):
        raise ValueError('Phase 4 adversarial factors require a registered level_1 through level_5 pressure category')
    requested_device = torch.device(device)
    if requested_device.type == 'cuda' and requested_device.index is not None:
        # Native simulator kernels use the current device/stream at setup.
        torch.cuda.set_device(requested_device)
    modes = ['none'] * 6
    modes[3] = 'deterministic' if production_d0 else 'nn'
    pressure = _pressure(category, seed, defender_family) if scenario_id in ('offense_vs_scripted_pressure', 'full_match_offense') else _pressure('', seed, defender_family)
    if scenario_id == 'team_delivery_cycles':
        modes[4] = 'deterministic'
        if category.endswith(('multiple_teammates', 'opponents')):
            modes[5] = 'deterministic'
        if category.endswith('opponents'):
            modes[0] = 'deterministic'
    if scenario_id == 'full_match_offense':
        modes[:3] = ['deterministic'] * 3
        modes[4:6] = ['deterministic', 'deterministic']
    for slot in pressure['slots']:
        modes[slot] = 'nn'
    env = CurriculumEnv(num_envs=num_envs, device=device, seed=seed, control_modes=modes,
        robot_roles=('defense', 'offense', 'offense', 'offense', 'offense', 'offense'),
        allow_nn_offense=True, dt=.02, horizon=horizon or math.ceil(definition.time_budget_s/.02),
        randomize=False, fuel_count=96, max_fuel_capacity=(30 if scenario_id.startswith('pickup_') else 6), fuel_physics=fuel_physics,
        perception_dropout=.04 if sensor_profile == 'noisy' else .02,
        position_noise=.03 if sensor_profile == 'noisy' else .015,
        velocity_noise=.10 if sensor_profile == 'noisy' else .05,
        field_layout=layout_id,
        replan_interval=10)
    env.phase4_factor_config = phase4_factor
    env.phase4_observation_delay_ticks = (phase4_factor['delay_ticks']
        if phase4_factor and phase4_factor['kind'] == 'observation_delay' else 0)
    env.phase4_confidence_cap = (phase4_factor['fuel_confidence_cap']
        if phase4_factor and phase4_factor['kind'] == 'independent_low_confidence' else None)
    env.phase4_runtime_metrics = {'delayed_observation_ticks': 0,
        'feint_reversal_tick': None, 'policy_change_tick': None,
        'confidence_cap_requested': env.phase4_confidence_cap,
        'confidence_cap_applied': None,
        'confidence_cap_refresh_count': 0,
        'confidence_cap_sensed_track_count': 0,
        'confidence_cap_reduced_track_values': 0,
        'confidence_cap_final_mean': None}
    if env.phase4_confidence_cap is not None:
        env.phase4_runtime_metrics['confidence_cap_applied'] = float(env.phase4_confidence_cap)
    env.curriculum = {'scenario_id': scenario_id, 'scenario_version': definition.version,
        'category': category, 'learner_slot': 3, 'seed': int(seed),
        'time_budget_s': definition.time_budget_s, 'pressure': pressure,
        'targets': {}, 'allowed_actions': (), 'schedule': None,
        'route': next((r for r in ('trench', 'bump', 'open') if r in category.split('/')), 'open'),
        'reference_qualified': False,
        'execution': 'production_deterministic_d0' if production_d0 else 'generic_nn',
        'layout_id': layout_id, 'defender_family': defender_family,
        'start_condition': start_condition, 'sensor_profile': sensor_profile,
        'limitations': ['Draft budgets and targets require scripted-reference qualification.',
                        'Uncontrolled robot bodies remain off-field and are restored after each step.']}
    reset_curriculum_scenario(env, seed)
    return env


def _pressure(category, seed, family='category_scripted'):
    marker = next((part for part in category.split('/') if part.startswith('level_')), 'level_0')
    level = _pressure_level(category)
    if family in ('orbiting_interceptor', 'split_chaser'):
        level = max(level, 4)
    return {'level': level, 'style': marker, 'slots': (0, 1) if level == 5 else ((0,) if level else ()),
            'phase': (int(seed) % 1000) / 1000 * 2 * math.pi, 'anchor_y': 4.035,
            'family': family}


def _pressure_level(category):
    marker = next((part for part in category.split('/') if part.startswith('level_')), 'level_0')
    try:
        return int(marker.split('_')[1])
    except (IndexError, ValueError):
        return 0


def _refresh(env):
    env.synchronize_policy_pose()
    env.sim.velocity.zero_()
    env._target_collect_heading.copy_(env.policy_pose[:, :, 2])
    env._target_turn_anchor.copy_(env.policy_pose[:, :, :2])
    env._raster_cursor.copy_(env._closest_raster_indices(env.policy_pose[:, :, :2]))
    env._raster_reanchor.fill_(True)
    for name in ('last_path', 'last_lengths', 'last_intercept', 'last_intercept_time',
                 'last_start', 'last_goal', 'last_heading', 'last_progress',
                 'last_trench_alignment', 'last_speed_profile', 'potential'):
        getattr(env.planner, name).zero_()
    env._planner_controller_modes_dirty = True
    cells_x = (env.piece_pos[..., 0] / env._pickup_grid_cell_size).floor().long()
    cells_y = (env.piece_pos[..., 1] / env._pickup_grid_cell_size).floor().long()
    cells = (cells_y.clamp(0, env._pickup_grid_ny-1) * env._pickup_grid_nx + cells_x.clamp(0, env._pickup_grid_nx-1))
    env._pickup_possible_cells[:, :, 0] = cells.to(torch.int16)
    reset_fuel_physics(env)
    for name in ('track_pos', 'track_vel', 'track_mask', 'track_confidence',
                 'track_spatial_extent', 'track_angular_extent', 'track_merged',
                 '_current_fuel_visibility', 'perception_quality_state',
                 'opponent_valid', 'opponent_pose', 'opponent_velocity', 'opponent_size', '_perception_rng_ticks'):
        getattr(env, name).zero_()
    env.track_age.fill_(float('inf'))
    env.opponent_age.fill_(float('inf'))
    env._update_perception(torch.ones(env.n, device=env.device, dtype=torch.bool))
    env._last_obs = env.observe()
    return env._last_obs


def reset_curriculum_scenario(env, seed):
    """Clear all physics/planner/sensor state, then expose ordinary perception."""
    config = env.curriculum
    config['seed'] = int(seed)
    # Architecture reset already clears the ordinary core state and caches.
    # Disable old metrics while it resets; no old lifecycle survives an episode.
    if hasattr(env, 'curriculum_metrics'):
        del env.curriculum_metrics
    env.curriculum_schedule = None
    reset_scenario(env, 'pickup_scattered', seed)
    env.architecture_pickup_only = False
    name, category, route = config['scenario_id'], config['category'], config['route']
    generator = torch.Generator(device=env.device).manual_seed(int(seed))
    def rand(shape):
        return torch.rand(shape, device=env.device, generator=generator)
    env.piece_active.zero_()
    env.piece_owner.fill_(-1)
    env.piece_zone.fill_(-1)
    env.piece_pos.zero_()
    env.piece_vel.zero_()
    y = {'open': 4.035, 'trench': .8, 'bump': 2.5}[route]
    env.sim.pose[:, 3] = env.sim.pose.new_tensor((8., y, 0.))
    env.curriculum_finished = torch.zeros(env.n, device=env.device, dtype=torch.bool)
    env.curriculum_teacher_delivering = torch.zeros_like(env.curriculum_finished)
    config['targets'] = {'amount': 4, 'batch_size': 2, 'cycles': 2}
    config['allowed_actions'] = (0, 1, 2, 3, 4, 6, 7)
    active = not (name.startswith('inactive_') or category.startswith('inactive/'))
    schedule = ((0., active),)
    preload = 0
    count, origin = 18, (8.8, y)
    if name == 'pickup_simple':
        env.sim.pose[:, 3, 0] += (rand((env.n,))-.5)*.3
        env.sim.pose[:, 3, 2] = (rand((env.n,))-.5)*.6
        count = 1 if category == 'single' else 6
        config['targets']['amount'] = count
        config['allowed_actions'] = (0, 1, 2, 3, 7)
    elif name == 'pickup_scattered':
        count = 12 if category == 'sparse' else 24
        if category == 'sparse':
            origin = (7.5, 4.035)
        config['targets']['amount'] = math.ceil(count * .5)
        config['allowed_actions'] = (0, 1, 2, 3, 7)
        if category == 'initially_hidden':
            config['allowed_actions'] = (0, 1, 2, 3, 6, 7)
            env.sim.pose[:, 3, 2] = math.pi
    elif name == 'pickup_multi_zone':
        count = 24
        origin = (7.8, .8)
        env.sim.pose[:, 3, :2] = env.sim.pose.new_tensor((7., .8))
        config['allowed_actions'] = (0, 1, 2, 3, 6, 7)
        config['targets']['amount'] = 14
    elif name == 'active_carry_to_zone':
        count, preload = 0, 6
        config['allowed_actions'] = (4, 7)
        env.sim.pose[:, 3, 0] = 10.5
    elif name == 'active_shoot_preloaded':
        count, preload = 0, 6
        config['allowed_actions'] = (4, 7)
        env.sim.pose[:, 3, :2] = env.sim.pose.new_tensor((14., 4.035))
        env.sim.pose[:, 3, 2] = math.pi
        if category.endswith('approach'):
            env.sim.pose[:, 3, :2] = env.sim.pose.new_tensor((15., 6.7))
        if category.endswith('heading_error'):
            env.sim.pose[:, 3, 2] = math.pi/2
    elif name == 'inactive_ferry_preloaded':
        count, preload = 0, (3 if category.endswith('partial_load') else 6)
        config['targets']['amount'] = 2 if preload == 3 else 4
        config['allowed_actions'] = (6, 7)
        env.sim.pose[:, 3, :2] = env.sim.pose.new_tensor((8.8, 4.035))
        if category.endswith('approach'):
            env.sim.pose[:, 3, 0] = 6.8
        if category.endswith('alignment'):
            env.sim.pose[:, 3, 2] = math.pi
        if category.endswith('landing'):
            env.sim.pose[:, 3, 1] = 6.8
    elif name == 'active_collect_carry_score' and route == 'open':
        # Open collection/scoring is within the alliance zone. Cross-boundary
        # carry gates separately require actual trench or bump traversal.
        env.sim.pose[:, 3, :2] = env.sim.pose.new_tensor((14., 6.8))
        env.sim.pose[:, 3, 2] = math.pi
        origin = (13., 6.8)
    elif name == 'hub_transition_cycles':
        if '/' not in category:
            # Explicit recovery tasks, with useful stock available to retain
            # until activation instead of demanding an impossible transfer.
            preload = 4
            count = 0
            env.sim.pose[:, 3, :2] = env.sim.pose.new_tensor((14., 6.8))
            schedule = ((0., False), (4., True))
            config['targets']['cycles'] = 1
        else:
            direction, load, timing, situation = category.split('/')
            preload = {'empty': 0, 'partial': 3, 'full': 6}[load]
            first_active = direction == 'active_to_inactive'
            transition = 8. if timing == 'enough_time' else .4
            schedule = ((0., first_active), (transition, not first_active),
                        (transition + 20., first_active), (transition + 40., True))
            if situation in ('approaching_shot', 'staged_at_home'):
                env.sim.pose[:, 3, :2] = env.sim.pose.new_tensor((14., 6.8))
                env.sim.pose[:, 3, 2] = math.pi
                origin = (13., 6.8)
            if situation == 'carrying_home':
                env.sim.pose[:, 3, :2] = env.sim.pose.new_tensor((10.5, .8))
            if situation == 'staged_at_home':
                origin = (13.2, 6.8)
            # after_release is initialized as physical airborne outside-origin
            # delivery below, not as already-credited staging.
            if situation == 'after_release' and load == 'empty':
                preload = 0
            config['targets']['cycles'] = 1
    if name in ('offense_vs_scripted_pressure', 'team_delivery_cycles', 'full_match_offense'):
        if route == 'open' and active:
            env.sim.pose[:, 3, :2] = env.sim.pose.new_tensor((14., 6.8))
            env.sim.pose[:, 3, 2] = math.pi
            origin = (13., 6.8)
        if name == 'full_match_offense':
            schedule = None
        elif name == 'team_delivery_cycles' and not active:
            schedule = ((0., False), (45., True))
        for slot in (4, 5):
            if env.control_modes[slot] != 'none':
                env.sim.pose[:, slot] = env.sim.pose.new_tensor((14.5, 2. + (slot-4)*4., math.pi))
        if name == 'team_delivery_cycles' and env.control_modes[0] != 'none':
            env.sim.pose[:, 0] = env.sim.pose.new_tensor((7., 4., 0.))
    # Separated FUEL grids with seeded jitter; category density and zone gaps
    # change actual layouts rather than only metadata labels.
    if count:
        columns = 6
        spacing = .65 if category == 'sparse' else .32
        if name == 'pickup_simple' and count == 1:
            spacing = 0.
        idx = torch.arange(count, device=env.device)
        offsets = torch.stack(((idx % columns).float()*spacing,
                               ((idx//columns).float() - (count//columns-1)/2)*.32), -1)
        positions = offsets[None] + env.piece_pos.new_tensor(origin)
        positions = positions.expand(env.n, -1, -1).clone()
        positions += (rand((env.n, count, 2))-.5)*.04
        if name == 'pickup_multi_zone':
            positions[:, count//2:, 0] = 13.5 + (idx[count//2:] % 4).float()*.3
            positions[:, count//2:, 1] = .6 + (idx[count//2:]//4-3).float()*.3
            if category in ('empty', 'depleted'):
                positions[:, :count//2, 0] = 9.2 + (idx[:count//2] % 6).float()*.32
                positions[:, :count//2, 1] = 3.7 + (idx[:count//2]//6).float()*.32
                env.sim.pose[:, 3, :2] = env.sim.pose.new_tensor((6.5, 6.8))
            if category == 'depleted':
                # A small remnant in the starting zone; the required amount
                # forces leaving it and finding a productive zone.
                positions[:, :2] = env.sim.pose[:, 3, None, :2] + env.piece_pos.new_tensor((.8, 0.))
        env.piece_pos[:, :count] = positions
        env.piece_active[:, :count] = True
        env.piece_zone[:, :count] = 0
    if preload:
        env.piece_active[:, count:count+preload] = True
        env.piece_owner[:, count:count+preload] = 3
        env.piece_pos[:, count:count+preload] = env.sim.pose[:, 3, None, :2]
        env.piece_zone[:, count:count+preload] = 6
    if route in ('trench', 'bump') and not active and name in ('inactive_collect_ferry', 'offense_vs_scripted_pressure', 'team_delivery_cycles', 'full_match_offense'):
        env.sim.pose[:, 3, :2] = env.sim.pose.new_tensor((13.3, y))
        env.sim.pose[:, 3, 2] = math.pi
        if count:
            env.piece_pos[:, :count, 0] = 9.8 + (torch.arange(count, device=env.device)%6).float()*.3
    if name == 'full_match_offense':
        for slot in range(3):
            if env.control_modes[slot] == 'deterministic':
                env.sim.pose[:, slot] = env.sim.pose.new_tensor((5.5+slot*.9, .8+slot*2.5, math.pi))
    env.planner.avoid_bumps = route != 'bump'
    pressure = config['pressure'] = _pressure(category, seed, config['defender_family']) if name in ('offense_vs_scripted_pressure', 'full_match_offense') else _pressure('', seed, config['defender_family'])
    if pressure['level'] == 3 and not active and route == 'open':
        blocked_y = .8 if pressure['style'] != 'level_3_bump' else 2.5
        if pressure['style'] == 'level_3_seeded_choice':
            blocked_y = .8 if int(seed)%2 else 2.5
        env.sim.pose[:, 3] = env.sim.pose.new_tensor((13.3, blocked_y, math.pi))
        env.piece_pos[:, :count, 1] = blocked_y
        env.piece_pos[:, :count, 0] = 9.8 + (torch.arange(count, device=env.device)%6).float()*.3
    pressure['anchor_y'] = float(env.sim.pose[0, 3, 1])
    for slot in pressure['slots']:
        blocker_y = pressure['anchor_y']
        if pressure['style'] == 'level_3_trench': blocker_y = .8
        if pressure['style'] == 'level_3_bump': blocker_y = 2.5
        if pressure['style'] == 'level_3_seeded_choice': blocker_y = .8 if int(seed)%2 else 2.5
        blocker_x = 11.9 if pressure['level'] == 3 else (13.6 if active and route == 'open' else 9.8)
        env.sim.pose[:, slot] = env.sim.pose.new_tensor((blocker_x, blocker_y + (slot*.9), math.pi))
    env.curriculum_action_mask = torch.ones((env.n, 6, 8), device=env.device, dtype=torch.bool)
    env.curriculum_action_mask[:, 3] = False
    env.curriculum_action_mask[:, 3, list(config['allowed_actions'])] = True
    if config['start_condition'] in ('offset_upper_lane', 'offset_lower_lane'):
        offset = .25 if config['start_condition'] == 'offset_upper_lane' else -.25
        on_field = ((env.sim.pose[..., 0] >= 0.) & (env.sim.pose[..., 0] <= env.field_length)
                    & (env.sim.pose[..., 1] >= 0.) & (env.sim.pose[..., 1] <= env.sim.field_width))
        env.sim.pose[..., 1] += on_field.to(env.sim.pose.dtype) * offset
        env.piece_pos[..., 1] += env.piece_active.to(env.piece_pos.dtype) * offset
    config['schedule'] = schedule
    env.curriculum_schedule = schedule
    env.match_elapsed.fill_(-env.dt)
    env._update_match(torch.ones(env.n, device=env.device, dtype=torch.bool))
    _refresh(env)
    env.curriculum_metrics = CurriculumMetrics(env)
    if name == 'hub_transition_cycles' and '/' in category:
        situation = category.split('/')[-1]
        if situation == 'staged_at_home':
            # Initial useful stock represents a documented earlier external
            # delivery; no acquisition or delivery reward is paid at reset.
            staged = slice(80, 84)
            env.piece_active[:, staged] = True
            env.piece_owner[:, staged] = -1
            env.piece_pos[:, staged, 0] = 13.4
            env.piece_pos[:, staged, 1] = 6.2 + torch.arange(4, device=env.device)*.3
            env.curriculum_metrics.ferry_credit[:, staged] = True
            env.curriculum_metrics.initial_staged[:, staged] = True
            env.curriculum_metrics.inactive_origin[:, staged] = True
        elif situation == 'after_release':
            from .fuel_physics_runtime import launch_fuel
            selected = torch.zeros_like(env.piece_active)
            selected[:, 80:82] = True
            env.piece_active |= selected
            env.piece_pos[:, 80:82] = env.piece_pos.new_tensor((9., 4.))
            env.curriculum_metrics.external |= selected
            owners = torch.full_like(env.piece_owner, 3)
            env.curriculum_metrics.released(selected, owners)
            launch_fuel(env, selected, env.piece_pos.new_tensor((13.6, 6.8)),
                        origins=env.piece_pos, height=.4, flight_time=1.)
        # Refresh normal sensing after additional physically staged pieces.
        if situation == 'staged_at_home':
            _refresh(env)
        else:
            env._update_perception(torch.ones(env.n, device=env.device, dtype=torch.bool))
            env._last_obs = env.observe()
    env.phase4_defender_anchor = env.policy_pose.clone()
    env._reset_phase4_observation_buffer(env._last_obs)
    return env._last_obs


def teacher_actions(env, obs):
    """Diagnostic teacher using sensed candidates and own pose/load telemetry.

    ``obs`` is the ordinary observation returned by this env. Candidate positions
    below are the same noisy sensor tracks used by that observation, never
    piece truth. Generic strategic execution remains unchanged.
    """
    del obs
    indices, valid, _ = env._candidates()
    mask, load = env._action_mask(indices, valid)
    points = env.track_pos.gather(2, indices[..., None].expand(-1, -1, -1, 2))
    candidate = valid[:, 3] & mask[:, 3, :4]
    active = env.hub_active[:, 1]
    candidate &= active[:, None] | (points[:, 3, :, 0] < env.field_length-env.alliance_zone_depth)
    distances = (points[:, 3] - env.policy_pose[:, 3, None, :2]).norm(dim=-1)
    choice = distances.masked_fill(~candidate, float('inf')).argmin(-1)
    choice = torch.where(candidate.any(-1), choice, 6)
    loaded = load[:, 3] >= min(4, env.robot_fuel_capacity_values[3])
    # Finish an acquired partial load if no useful sensed pickup is available.
    loaded |= (load[:, 3] > 0) & ~candidate.any(-1)
    env.curriculum_teacher_delivering &= load[:, 3] > 0
    env.curriculum_teacher_delivering |= loaded
    delivery = torch.where(active, 4, 6)
    delivery_legal = mask[:, 3].gather(1, delivery[:, None]).squeeze(1)
    choice = torch.where(env.curriculum_teacher_delivering & delivery_legal, delivery, choice)
    at_home = env.policy_pose[:, 3, 0] >= env.field_length-env.alliance_zone_depth
    choice = torch.where(~active & at_home & (load[:, 3] > 0), 7, choice)
    chosen_legal = mask[:, 3].gather(1, choice[:, None]).squeeze(1)
    choice = torch.where(chosen_legal, choice, 7)
    actions = torch.full((env.n, 6), 7, device=env.device, dtype=torch.long)
    actions[:, 3] = choice
    return actions
