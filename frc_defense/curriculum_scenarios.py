"""Seeded draft offense tasks using normal physics, sensors and NN execution.

All thresholds/budgets remain unqualified. The observation-only scripted teacher
is a diagnostic baseline, not a qualified reference or imitation dataset.
"""
from __future__ import annotations
import math
import base64
import hashlib
import json
import torch

from .architecture_scenarios import reset_scenario
from .curriculum_metrics import CurriculumMetrics
from .fuel_physics_runtime import reset_fuel_physics
from .meco_phase4 import validate_phase4_factor
from .meco_phase4_opponents import POLICY_POOL, opponent_velocity_command
from .meco_phase4_context import (CONTACT_GEOMETRIES,
    contact_exposure_actions, contact_exposure_profile,
    apply_full_team_start_profile, phase4_context_identity,
    full_team_start_profile_legality,
    validate_phase4_start_profile_assignment,
    validate_phase4_focal_pose_assignment, apply_phase4_focal_pose_assignment,
    apply_phase4_focal_pose_assignment_reset)
from .tensor_3v3 import TensorThreeVsThreeEnv
from .training_curriculum import get_scenario

_PHASE4_SENSOR_STATE_FIELDS = (
    'policy_pose',
    'track_pos', 'track_vel', 'track_age', 'track_mask', 'track_confidence',
    'track_spatial_extent', 'track_angular_extent', 'track_merged',
    'perception_quality_state', 'opponent_pose', 'opponent_velocity',
    'opponent_size', 'opponent_valid', 'opponent_age',
)
PRESSURE_ANCHOR_SCHEMA = 'meco-phase4-pressure-anchor-v1'


class CurriculumEnv(TensorThreeVsThreeEnv):
    def _phase4_apply_confidence_cap(self, *, reset_mask=None):
        """Cap only sensed FUEL confidence for the learner robot."""
        cap = getattr(self, 'phase4_confidence_cap', None)
        if cap is None:
            return
        if reset_mask is not None:
            if (not isinstance(reset_mask, torch.Tensor) or
                    reset_mask.device != self.device or reset_mask.dtype is not torch.bool or
                    tuple(reset_mask.shape) != (self.n,)):
                raise ValueError('confidence reset mask must be device bool [world]')
            selected = reset_mask
            sensed_all = self.track_mask[:, 3]
            sensed = sensed_all & selected[:, None]
            confidence = self.track_confidence[:, 3]
            reduced = sensed & (confidence > float(cap))
            capped = torch.where(sensed_all, confidence.clamp(max=float(cap)), confidence)
            confidence.copy_(torch.where(selected[:, None], capped, confidence))
            count = sensed.sum(-1)
            reduced_count = reduced.sum(-1)
            final_sum = (confidence * sensed.to(confidence.dtype)).sum(-1)
            final_mean = torch.where(count > 0,
                final_sum / count.clamp_min(1).to(final_sum.dtype),
                torch.full_like(final_sum, float('nan')))
            defaults = (
                ('phase4_reset_confidence_cap_refresh_count', torch.zeros(
                    self.n, device=self.device, dtype=torch.long)),
                ('phase4_reset_confidence_cap_sensed_track_count', torch.zeros(
                    self.n, device=self.device, dtype=torch.long)),
                ('phase4_reset_confidence_cap_reduced_track_count', torch.zeros(
                    self.n, device=self.device, dtype=torch.long)),
                ('phase4_reset_confidence_cap_final_mean', torch.zeros(
                    self.n, device=self.device, dtype=confidence.dtype)),
            )
            updates = (torch.ones_like(count), count, reduced_count, final_mean)
            for (name, default), update in zip(defaults, updates):
                prior = getattr(self, name, default)
                setattr(self, name, torch.where(selected, prior + update, prior))
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

    def _update_perception_for_reset(self, active):
        """Refresh selected reset worlds without host confidence reductions."""
        result = super()._update_perception(active)
        self._phase4_apply_confidence_cap(reset_mask=active)
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
        geometry = getattr(self, 'contact_geometry_id', None)
        if geometry is not None:
            # Replace only the scripted opponent's slot-2 command. Slot 3 and
            # all other actions remain those selected by the active policy.
            actions = replace_contact_opponent_action(actions,
                contact_exposure_actions(self))
        teammate_assignment = getattr(self, 'phase4_teammate_behavior_episode', None)
        teammate_details = None
        policy_action_override = None
        if teammate_assignment is not None:
            from .meco_phase4_context import teammate_behavior_action_details
            teammate_details = teammate_behavior_action_details(self, teammate_assignment)
            override_mask = torch.zeros((self.n, 6), device=self.device,
                                        dtype=torch.bool)
            for slot in teammate_assignment:
                override_mask[:, slot] = active_requested
            policy_action_override = (teammate_details['actions'], override_mask)
        route_seen_before = (self.curriculum_metrics.route_seen.clone()
            if geometry is not None and hasattr(self, 'curriculum_metrics') else None)
        observation, reward, done, truncated, info = super().step(
            actions, active_requested, policy_action_override=policy_action_override,
            **kwargs)
        if teammate_details is not None:
            runtime = self.phase4_teammate_behavior_runtime
            actual = self.last_actions.detach().cpu()
            requested = teammate_details['actions'].detach().cpu()
            desired = teammate_details['requested_actions'].detach().cpu()
            legal = teammate_details['requested_legal'].detach().cpu()
            helper_fallback = teammate_details['helper_fallback'].detach().cpu()
            rows = []
            for slot in teammate_assignment:
                rows.append({"slot": slot,
                    "requested_behavior_action": int(desired[0, slot]),
                    "helper_selected_action": int(requested[0, slot]),
                    "actually_applied_action": int(actual[0, slot]),
                    "requested_legal": bool(legal[0, slot]),
                    "helper_fallback": bool(helper_fallback[0, slot]),
                    "environment_remask_fallback": bool(
                        actual[0, slot] != requested[0, slot]),
                    "active": bool(active_requested[0])})
            runtime['events'].append({"tick": len(runtime['events']),
                                     "slots": rows})
        if route_seen_before is not None:
            self.contact_geometry_runtime['approach_action_ticks'] += active_requested.long()
            crossed = self.curriculum_metrics.route_seen & ~route_seen_before
            if bool(crossed.any()):
                self.contact_geometry_runtime['route_seen'] |= crossed
                unset = crossed & (self.contact_geometry_runtime['route_seen_time_s'] < 0.)
                self.contact_geometry_runtime['route_seen_time_s'][unset] = (
                    self.curriculum_metrics.elapsed[unset])
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
            return torch.where(active[:, None, None], commands, 0.)
        factor = getattr(self, 'phase4_factor_config', None)
        stationary_slot = (factor['blocker_slot']
            if factor and factor['kind'] == 'stationary_blockage' else None)
        if stationary_slot is not None:
            commands[:, stationary_slot] = 0.
            target = getattr(self, 'stationary_blockage_assigned_pose', None)
            if target is not None:
                pose = self.sim.pose[:, stationary_slot]
                distance = torch.linalg.vector_norm(pose[:, :2] - target[:, :2], dim=-1)
                angle_delta = torch.atan2(torch.sin(pose[:, 2] - target[:, 2]),
                                          torch.cos(pose[:, 2] - target[:, 2])).abs()
                valid = ((distance <= factor['pose_tolerance_m']) &
                         (angle_delta <= factor['heading_tolerance_rad']))
                prior = self.phase4_runtime_metrics['stationary_blockage_pose_valid']
                self.phase4_runtime_metrics['stationary_blockage_pose_valid'] = prior & valid
        pressure = self.curriculum['pressure']
        if pressure['level'] == 0:
            return torch.where(active[:, None, None], commands, 0.)
        # Scripted blockers still use ordinary chassis/contact physics. They
        # have NN controller slots so deterministic offense cannot overwrite
        # these scripts; their strategic action is hold.
        opponent_runtime = getattr(self, 'phase4_opponent_runtime', None)
        opponent_slots = (set(opponent_runtime['configured_slots'])
                          if opponent_runtime is not None else set())
        direct_pursuit = (opponent_runtime is not None and
                          opponent_runtime.get('policy_id') == 'direct_pursuit')
        if opponent_runtime is not None:
            pressure_slots = set(pressure['slots'])
            if direct_pursuit:
                registered = set(opponent_runtime.get('registered_pressure_slots', ()))
                preserved = set(opponent_runtime.get('preserved_scripted_slots', ()))
                if (len(opponent_slots) != 1 or not opponent_slots.issubset(pressure_slots)
                        or registered != pressure_slots or preserved != pressure_slots - opponent_slots):
                    raise RuntimeError('direct-pursuit assignment must replace one registered pressure slot')
            elif pressure_slots != opponent_slots:
                raise RuntimeError('runtime pressure slots differ from validation-bound opponent assignment')
        for slot in pressure['slots']:
            if slot == stationary_slot:
                commands[:, slot] = 0.
                continue
            commands[:, slot] = 0.
            policy_id = getattr(self, 'phase4_opponent_policy_id', None)
            if policy_id is not None and (not direct_pursuit or slot in opponent_slots):
                # The policy sees its own controller pose and only its noisy,
                # validity-masked opponent track. Static anchor/objective data
                # are scenario configuration, not simulator truth state.
                own_xy = self.policy_pose[:, slot, :2]
                sensed = {
                    'own_xy': own_xy,
                    'target_xy': self.opponent_pose[:, slot, :2],
                    'target_velocity_xy': self.opponent_velocity[:, slot, :2],
                    'target_valid': self.opponent_valid[:, slot],
                    'anchor_xy': self.phase4_defender_anchor[:, slot, :2],
                    'objective_xy': torch.stack((
                        torch.full_like(own_xy[:, 0], self.field_length),
                        torch.full_like(own_xy[:, 1], self.field_width * .5)), dim=-1),
                    'phase': self.match_elapsed + self.phase4_pressure_phase / 1.2,
                }
                commands[:, slot, :2] = opponent_velocity_command(
                    policy_id, sensed, tick=self.steps,
                    max_speed_mps=self.sim.speed[:, slot])
                if opponent_runtime is not None:
                    opponent_runtime['applied_slots'] = sorted(set(
                        opponent_runtime['applied_slots']) | {int(slot)})
                    opponent_runtime['applied_command_calls'] += 1
                    calls_by_slot = opponent_runtime['applied_command_calls_by_slot']
                    calls_by_slot[str(slot)] = calls_by_slot.get(str(slot), 0) + int(
                        active_requested.sum().item())
                continue
            level = pressure['level']
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
                        phase = self.match_elapsed * 1.2 + self.phase4_pressure_phase
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
                phase = self.match_elapsed * 1.2 + self.phase4_pressure_phase
                target = self.policy_pose[:, slot, :2].clone()
                target[:, 1] = self.phase4_pressure_anchor_y_m + torch.sin(phase) * .6
                if pressure['family'] == 'orbiting_interceptor':
                    target[:, 0] = 10.8 + torch.cos(phase * .7 + slot) * .8
                    target[:, 1] += slot * .45
                commands[:, slot, :2] = (target - self.policy_pose[:, slot, :2]).clamp(-1., 1.)
            elif pressure['family'] == 'split_chaser':
                delta = self.opponent_pose[:, slot, :2] - self.policy_pose[:, slot, :2]
                valid = self.opponent_valid[:, slot]
                phase = self.match_elapsed * .65 + self.phase4_pressure_phase
                offset = torch.stack((torch.cos(phase), torch.sin(phase)), -1) * .8
                commands[:, slot, :2] = torch.where(valid[:, None],
                    (delta + offset).clamp(-1.5, 1.5), 0.)
            elif level == 4 or (level == 5 and slot == 1):
                # Only the defender's simulated opponent observation is used.
                delta = self.opponent_pose[:, slot, :2] - self.policy_pose[:, slot, :2]
                valid = self.opponent_valid[:, slot]
                commands[:, slot, :2] = torch.where(valid[:, None], delta.clamp(-1.5, 1.5), 0.)
        return torch.where(active[:, None, None], commands, 0.)

    def _reset_phase4_observation_buffer(self, observation=None, *, mask=None):
        delay = int(getattr(self, 'phase4_observation_delay_ticks', 0))
        current = self._phase4_capture_sensor_frame(observation)
        if mask is None:
            self._phase4_current_sensor_frame = current
            self._phase4_observation_history = [self._phase4_capture_sensor_frame(
                current['observation']) for _ in range(delay + 1)]
            return
        if (not isinstance(mask, torch.Tensor) or mask.device != self.device or
                mask.dtype is not torch.bool or tuple(mask.shape) != (self.n,)):
            raise ValueError('sensor history reset mask must be device bool [world]')
        if not hasattr(self, '_phase4_observation_history'):
            self._phase4_current_sensor_frame = current
            self._phase4_observation_history = [self._phase4_capture_sensor_frame(
                current['observation']) for _ in range(delay + 1)]
            return
        history = self._phase4_observation_history
        if len(history) != delay + 1:
            raise RuntimeError('sensor history depth differs from the static delay contract')
        row_mask = mask.reshape((self.n,) + (1,) * (self._last_obs.ndim - 1))

        def merge_frame(destination, source):
            for name in _PHASE4_SENSOR_STATE_FIELDS:
                field = getattr(self, name)
                field_mask = mask.reshape((self.n,) + (1,) * (field.ndim - 1))
                destination[name].copy_(torch.where(field_mask, source[name],
                                                     destination[name]))
            destination['observation'].copy_(torch.where(row_mask,
                source['observation'], destination['observation']))

        if getattr(self, '_phase4_current_sensor_frame', None) is None:
            self._phase4_current_sensor_frame = current
        else:
            merge_frame(self._phase4_current_sensor_frame, current)
        for frame in history:
            merge_frame(frame, current)


def make_curriculum_env(scenario_id, category, num_envs, device, seed, *, horizon=None, fuel_physics=None,
                        production_d0=None, layout_id=None,
                        defender_family=None, start_condition=None,
                        sensor_profile=None, phase4_factor=None,
                        contact_geometry_id=None, contact_geometry_config=None,
                        phase4_opponent_policy_id=None,
                        phase4_opponent_schedule=None, phase4_opponent_seed=None,
                        phase4_start_profile_assignment=None,
                        phase4_focal_pose_assignment=None,
                        phase4_teammate_behavior_assignment=None,
                        _effective_snapshot_only=False,
                        _effective_snapshot_runtime_identity=None,
                        _effective_snapshot_repo_root=None,
                        _effective_snapshot_source_plan_path="docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md",
                        evaluation_protocol_content_base64=None,
                        evaluation_protocol_object=None,
                        evaluation_protocol_content_sha256=None):
    """Build a curriculum environment, optionally running production D0 in slot 3.

    ``production_d0`` changes only the learner slot's controller from the
    default NN controller to the native deterministic offense controller.
    Scenario-specific teammate and scripted pressure/blocker assignments are
    still applied below, and the same reset path is used in either mode.
    """
    requested_values = {"fuel_physics": fuel_physics, "production_d0": production_d0,
        "layout_id": layout_id, "defender_family": defender_family,
        "start_condition": start_condition, "sensor_profile": sensor_profile}
    if fuel_physics is None:
        fuel_physics = True
    if production_d0 is None:
        production_d0 = False
    if layout_id is None:
        layout_id = '2026_rebuilt'
    if defender_family is None:
        defender_family = 'category_scripted'
    if start_condition is None:
        start_condition = 'scenario_defined'
    if sensor_profile is None:
        sensor_profile = 'standard'
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
    if (phase4_factor and phase4_factor['kind'] == 'stationary_blockage' and
            contact_geometry_id is not None):
        raise ValueError('stationary blockage cannot be combined with contact geometry')
    if phase4_opponent_policy_id is not None:
        if phase4_opponent_policy_id not in POLICY_POOL:
            raise ValueError('unknown Phase 4 opponent policy ID')
        supported_levels = ((0, 1, 2, 3, 4, 5)
            if phase4_opponent_policy_id == 'trench_interceptor' else (1, 2, 3, 4, 5))
        if (scenario_id not in ('offense_vs_scripted_pressure', 'full_match_offense')
                or _pressure_level(category) not in supported_levels):
            raise ValueError('Phase 4 opponent policy requires a registered pressure category')
        if (phase4_opponent_policy_id in ('trench_interceptor', 'direct_pursuit')
                and phase4_opponent_schedule is None):
            raise ValueError('scheduled Phase 4 opponent requires its protocol-bound validation schedule')
    elif phase4_opponent_schedule is not None:
        raise ValueError('opponent validation schedule requires its assigned policy ID')
    trench_schedule = None
    if phase4_opponent_schedule is not None:
        from .meco_phase4_opponents import validate_opponent_policy_schedule
        if phase4_opponent_seed is None:
            raise ValueError('scheduled opponent policy requires its validation seed')
        trench_schedule = validate_opponent_policy_schedule(
            phase4_opponent_schedule, scenario_id=scenario_id,
            categories=phase4_opponent_schedule.get('categories', ()),
            qualification_case=phase4_opponent_schedule.get('qualification_case', {}),
            seeds=phase4_opponent_schedule.get('seed_list', ()), split='validation',
            factor_id=phase4_opponent_schedule.get('factor_id'),
            protocol_id=phase4_opponent_schedule.get('protocol_id'))
        if (scenario_id != trench_schedule['scenario_id'] or
                category not in trench_schedule['categories'] or
                phase4_opponent_seed not in trench_schedule['seed_list'] or
                phase4_opponent_policy_id != trench_schedule['policy_id'] or
                {'layout_id': layout_id,
                    'defender_family': defender_family,
                    'start_condition': start_condition,
                    'sensor_profile': sensor_profile} !=
                    trench_schedule['qualification_case']):
            raise ValueError('opponent schedule does not bind this exact validation episode')
    if contact_geometry_id is None:
        if contact_geometry_config is not None:
            raise ValueError('contact geometry config requires a registered geometry ID')
    else:
        if contact_geometry_id not in CONTACT_GEOMETRIES:
            raise ValueError('unknown contact geometry ID')
        expected_contact_config = phase4_context_identity(
            'robot_contact_geometry', contact_geometry_id)
        if contact_geometry_config != expected_contact_config:
            raise ValueError('contact geometry config differs from its frozen registry identity')
        if (scenario_id not in ('offense_vs_scripted_pressure', 'full_match_offense')
                or _pressure_level(category) not in (1, 2, 3, 4, 5)):
            raise ValueError('contact geometry is only registered for pressure/match scenarios')
    if phase4_start_profile_assignment is None:
        start_profile_assignment = None
    else:
        start_profile_assignment = validate_phase4_start_profile_assignment(
            phase4_start_profile_assignment)
        if (start_profile_assignment['scenario_id'] != scenario_id or
                scenario_id != 'full_match_offense'):
            raise ValueError('legal full-team start profiles require full_match_offense')
        if contact_geometry_id is not None:
            raise ValueError('start profile and contact geometry cannot both set full-team poses')
    pressure_level = _pressure_level(category)
    if phase4_factor is not None and (scenario_id not in (
            'offense_vs_scripted_pressure', 'full_match_offense')
            or pressure_level not in (1, 2, 3, 4, 5)):
        raise ValueError('Phase 4 adversarial factors require a registered level_1 through level_5 pressure category')
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
    if contact_geometry_id is not None:
        # The evaluator supplies slot 2's sensed-target action 5 directly.
        # NN mode makes TensorThreeVsThreeEnv consume that action rather than
        # replacing it with an unrelated scripted deterministic offense action.
        modes[2] = 'nn'
    env_kwargs = dict(num_envs=num_envs, device=device, seed=seed, control_modes=modes,
        robot_roles=('defense', 'offense', 'offense', 'offense', 'offense', 'offense'),
        allow_nn_offense=True, dt=.02, horizon=horizon or math.ceil(definition.time_budget_s/.02),
        randomize=False, fuel_count=96, max_fuel_capacity=(30 if scenario_id.startswith('pickup_') else 6), fuel_physics=fuel_physics,
        defer_initial_perception=True,
        perception_dropout=.04 if sensor_profile == 'noisy' else .02,
        position_noise=.03 if sensor_profile == 'noisy' else .015,
        velocity_noise=.10 if sensor_profile == 'noisy' else .05,
        field_layout=layout_id,
        replan_interval=10)
    if _effective_snapshot_only:
        if any(value is None for value in requested_values.values()):
            raise ValueError('effective capture requires explicit overrides for every material setting')
        if phase4_focal_pose_assignment is not None or phase4_teammate_behavior_assignment is not None:
            raise ValueError('effective capture for focal/teammate validation assignments is not registered')
        snapshot_locals = locals()
        factory_arguments = {name: snapshot_locals[name]
            for name, parameter in __import__('inspect').signature(make_curriculum_env).parameters.items()
            if not name.startswith('_')}
        selected_runtime_identity = _discover_effective_runtime_identity(device,
            _effective_snapshot_repo_root)
        return _effective_curriculum_snapshot(scenario_id, category, seed, definition,
            defender_family, start_condition, sensor_profile, production_d0, pressure,
            phase4_factor, phase4_opponent_policy_id, trench_schedule,
            contact_geometry_id, contact_geometry_config, env_kwargs,
            requested_values=requested_values,
            evaluation_protocol_content_base64=evaluation_protocol_content_base64,
            evaluation_protocol_object=evaluation_protocol_object,
            evaluation_protocol_content_sha256=evaluation_protocol_content_sha256,
            factory_arguments=factory_arguments,
            selected_runtime_identity=selected_runtime_identity,
            runtime_identity=_effective_snapshot_runtime_identity,
            repo_root=_effective_snapshot_repo_root,
            source_plan_path=_effective_snapshot_source_plan_path)
    requested_device = torch.device(device)
    if requested_device.type == 'cuda' and requested_device.index is not None:
        # Native simulator kernels use the current device/stream at setup.
        torch.cuda.set_device(requested_device)
    env = CurriculumEnv(**env_kwargs)
    env.phase4_pressure_anchor_y_m = torch.full((env.n,), 4.035,
        device=env.device, dtype=env.sim.pose.dtype)
    env.phase4_pressure_phase = torch.full((env.n,), float(pressure['phase']),
        device=env.device, dtype=env.sim.pose.dtype)
    env.phase4_factor_config = phase4_factor
    env.phase4_opponent_policy_id = phase4_opponent_policy_id
    env.phase4_opponent_schedule = trench_schedule
    env.phase4_opponent_runtime = (None if trench_schedule is None else {
        'schema': 'meco-phase4-opponent-runtime-v1',
        'policy_id': trench_schedule['policy_id'],
        'factor_id': trench_schedule['factor_id'],
        'category': category, 'seed': int(phase4_opponent_seed),
        'configured_slots': trench_schedule['configured_pressure_slots'][category],
        'registered_pressure_slots': trench_schedule.get(
            'registered_pressure_slots', {}).get(category,
            trench_schedule['configured_pressure_slots'][category]),
        'preserved_scripted_slots': trench_schedule.get(
            'preserved_scripted_slots', {}).get(category, []),
        'applied_slots': [], 'applied_command_calls': 0,
        'applied_command_calls_by_slot': {
            str(slot): 0 for slot in trench_schedule['configured_pressure_slots'][category]},
        'interception_rate': None, 'alternate_crossing_success_rate': None,
        'trench_crossing_time_s': None, 'qualification_eligible': False})
    env.contact_geometry_id = contact_geometry_id
    env.contact_geometry_config = contact_geometry_config
    env.phase4_start_profile_assignment = start_profile_assignment
    if phase4_focal_pose_assignment is None:
        focal_pose_assignment = None
    else:
        focal_pose_assignment = validate_phase4_focal_pose_assignment(
            phase4_focal_pose_assignment)
        if (focal_pose_assignment['scenario_id'] != scenario_id or
                focal_pose_assignment['split'] != 'validation' or
                category not in focal_pose_assignment['categories'] or
                int(seed) not in focal_pose_assignment['seeds']):
            raise ValueError('focal pose assignment does not bind this validation episode')
    env.phase4_focal_pose_assignment = focal_pose_assignment
    env.phase4_teammate_behavior_episode = None
    env.phase4_teammate_behavior_runtime = None
    if phase4_teammate_behavior_assignment is not None:
        from .meco_phase4_context import validate_phase4_teammate_behavior_assignment
        assignment = validate_phase4_teammate_behavior_assignment(
            phase4_teammate_behavior_assignment)
        if (scenario_id != 'full_match_offense' or assignment['split'] != 'validation'
                or assignment['scenario_id'] != scenario_id or int(seed) not in assignment['seeds']
                or category not in assignment['categories']):
            raise ValueError('teammate behavior override requires its exact full-match validation assignment')
        env.phase4_teammate_behavior_assignment = assignment
        if assignment['arm'] == 'behavior_pool':
            episode_index = assignment['seeds'].index(int(seed))
            env.phase4_teammate_behavior_episode = assignment['behavior_schedule'][episode_index]['assignments']
        env.phase4_teammate_behavior_runtime = {"schema": "phase4-teammate-behavior-runtime-v1",
            "arm": assignment['arm'],
            "assignments": ({str(slot): behavior for slot, behavior in
                            env.phase4_teammate_behavior_episode.items()}
                            if env.phase4_teammate_behavior_episode is not None else None),
            "assignment_applied": env.phase4_teammate_behavior_episode is not None,
            "events": [], "qualification_eligible": False}
    env.phase4_start_profile_identity = (None if start_profile_assignment is None else
        phase4_context_identity('legal_team_pose_sets',
            start_profile_assignment['registry_profile_id']))
    env.contact_geometry_runtime = {
        'geometry_id': contact_geometry_id,
        'factor_id': 'contact.robot_bump' if contact_geometry_id is not None else None,
        'exposure_assigned': contact_geometry_id is not None,
        'pair_slots': [2, 3] if contact_geometry_id is not None else None,
        'opponent_action': 5 if contact_geometry_id is not None else None,
        'opponent_action_source': ('fresh_sensed_track_action_5'
            if contact_geometry_id is not None else None),
        'slot2_env_control_mode': 'nn' if contact_geometry_id is not None else None,
        'approach_action_ticks': torch.zeros(num_envs, device=device, dtype=torch.long),
        'focal_action_overridden': False,
        'route_seen': torch.zeros(num_envs, device=device, dtype=torch.bool),
        'route_seen_time_s': torch.full((num_envs,), -1., device=device),
        'qualification_eligible': False,
        'recovery_time_after_bump_s': None,
        'route_completion_rate': None,
    }
    env.phase4_pressure_anchor_capture_stage = (
        'post_start_profile_pose' if (start_profile_assignment is not None or
                                      contact_geometry_id is not None)
        else 'pre_blocker_pose')
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
        'confidence_cap_final_mean': None,
        'stationary_blockage_assigned': False,
        'stationary_blockage_pose_valid': torch.ones(
            env.n, device=env.device, dtype=torch.bool)}
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


def _discover_effective_runtime_identity(device, repo_root=None):
    """Collect runtime identity from loaded Python/Torch and the selected accelerator."""
    import importlib.metadata
    import platform
    import re
    from pathlib import Path
    selected = torch.device(device)
    if selected.type != 'cuda' or selected.index is None:
        raise ValueError('effective capture requires an explicitly indexed CUDA/ROCm device')
    try:
        properties = torch.cuda.get_device_properties(selected.index)
    except Exception as exc:
        raise ValueError('effective capture could not discover the selected accelerator') from exc
    uuid = getattr(properties, 'uuid', None)
    name = getattr(properties, 'name', None)
    if not isinstance(uuid, str) or not uuid.strip() or not isinstance(name, str) or not name.strip():
        raise ValueError('selected accelerator must expose a stable UUID and model name')
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    simulator_paths = ('frc_defense/tensor_3v3.py', 'frc_defense/tensor_sim.py',
        'frc_defense/tensor_physics.py')
    simulator_hashes = {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in simulator_paths}
    pyproject_bytes = (root / 'pyproject.toml').read_bytes()
    project_match = re.search(r"^version\s*=\s*['\"]([^'\"]+)['\"]\s*$",
        pyproject_bytes.decode('utf-8'), re.MULTILINE)
    if project_match is None:
        raise ValueError('effective capture cannot resolve the simulator project version')
    simulator_version = project_match.group(1)
    dependency_specs = {'pyproject.toml': hashlib.sha256(pyproject_bytes).hexdigest()}
    requirements_path = root / 'requirements-rocm-gfx900.txt'
    if requirements_path.is_file():
        dependency_specs['requirements-rocm-gfx900.txt'] = hashlib.sha256(
            requirements_path.read_bytes()).hexdigest()
    installed = sorted((distribution.metadata.get('Name', '').lower(), distribution.version)
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get('Name'))
    dependency_digest = hashlib.sha256(json.dumps({"spec_artifacts": dependency_specs,
        "installed_distributions": installed}, sort_keys=True, separators=(',', ':'))
        .encode('utf-8')).hexdigest()
    if torch.version.hip:
        backend, backend_version = 'ROCm', str(torch.version.hip)
    elif torch.version.cuda:
        backend, backend_version = 'CUDA', str(torch.version.cuda)
    else:
        raise ValueError('effective capture requires an initialized CUDA/ROCm runtime')
    return {'simulator_id': 'autodrive.TensorThreeVsThreeEnv',
        'simulator_version': simulator_version,
        'simulator_build_sha256': hashlib.sha256(json.dumps(simulator_hashes,
            sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest(),
        'python_version': platform.python_version(), 'torch_version': str(torch.__version__),
        'backend': backend, 'backend_version': backend_version,
        'device_id': uuid.strip(), 'device_index': selected.index,
        'device_model': name.strip(), 'dependency_lock_sha256': dependency_digest}


def _effective_curriculum_snapshot(scenario_id, category, seed, definition,
        defender_family, start_condition, sensor_profile, production_d0, pressure,
        phase4_factor, phase4_opponent_policy_id, opponent_schedule,
        contact_geometry_id, contact_geometry_config, env_kwargs, *,
        requested_values, evaluation_protocol_content_base64,
        evaluation_protocol_object,
        evaluation_protocol_content_sha256,
        factory_arguments,
        selected_runtime_identity,
        runtime_identity, repo_root, source_plan_path):
    """Export the normalized object passed to CurriculumEnv without constructing it."""
    from pathlib import Path
    from .meco_baseline_freeze import (EFFECTIVE_SNAPSHOT_SCHEMA,
        PRESSURE_ANCHOR_RUNTIME_STATE_SCHEMA,
        canonical_sha256, capture_source_identity, _validate_effective_runtime)

    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    source = capture_source_identity(root, source_plan_path)
    observed_runtime = _validate_effective_runtime(selected_runtime_identity)
    runtime = observed_runtime
    if runtime_identity is not None and _validate_effective_runtime(runtime_identity) != observed_runtime:
        raise ValueError('caller runtime identity differs from runtime discovered by the factory')
    resolved = {"scenario_id": scenario_id, "scenario_version": definition.version,
        "scenario_time_budget_s": definition.time_budget_s, "category": category,
        "seed": int(seed), "defender_family": defender_family,
        "start_condition": start_condition, "sensor_profile": sensor_profile,
        "production_d0": bool(production_d0), "pressure": pressure,
        "phase4_factor": phase4_factor, "phase4_opponent_policy_id": phase4_opponent_policy_id,
        "phase4_opponent_schedule": opponent_schedule,
        "contact_geometry_id": contact_geometry_id,
        "contact_geometry_config": contact_geometry_config,
        "environment_constructor": env_kwargs,
        "requested_values": requested_values}
    dynamics = {key: env_kwargs[key] for key in (
        "dt", "horizon", "fuel_physics", "fuel_count", "max_fuel_capacity")}
    field = {key: env_kwargs[key] for key in ("field_layout",)}
    geometry = {**field, "field_length_m": 16.54, "field_width_m": 8.07}
    values = {
        "robot_dynamics": {**dynamics, "constructor": "TensorThreeVsThreeEnv"},
        "field_geometry": geometry,
        "legal_connectivity": {"field_layout": env_kwargs["field_layout"],
            "replan_interval_ticks": env_kwargs["replan_interval"],
            "navigation_implementation": "TensorThreeVsThreeNavigationMixin"},
        "opponent_controllers": {"control_modes": list(env_kwargs["control_modes"]),
            "pressure": pressure, "family": defender_family,
            "phase4_policy_id": phase4_opponent_policy_id,
            "schedule": opponent_schedule},
        "action_update_cadence": {"physics_dt_s": env_kwargs["dt"],
            "action_replan_interval_ticks": env_kwargs["replan_interval"],
            "horizon_ticks": env_kwargs["horizon"]},
        "sensor_noise_profiles": {"profile": sensor_profile,
            "perception_dropout": env_kwargs["perception_dropout"],
            "position_noise_m": env_kwargs["position_noise"],
            "velocity_noise_mps": env_kwargs["velocity_noise"]},
        "scoring_observation_semantics": {"simulator_metric": "fuel_score_count",
            "metric_semantics": "scored_fuel_piece_count",
            "official_match_points_available": False,
            "fuel_count": env_kwargs["fuel_count"],
            "max_fuel_capacity": env_kwargs["max_fuel_capacity"]},
        "safety_event_semantics": {"phase4_factor": phase4_factor,
            "contact_geometry_id": contact_geometry_id,
            "contact_geometry_config": contact_geometry_config,
            "native_collision_and_route_guards": True}}
    resolved_fields = {}
    for field_name, value in values.items():
        resolved_fields[field_name] = {"value": value,
            "selector": f"make_curriculum_env.effective.{field_name}",
            "runtime_owner": "CurriculumEnv/TensorThreeVsThreeEnv",
            "resolution_state": "overridden" if field_name == "sensor_noise_profiles" else "resolved",
            "derivation": {"kind": "runtime_object",
                "input_sha256": canonical_sha256({"effective_configuration": resolved,
                    "field": field_name, "value": value})}}
    config_values = {
        "campaign_assignment": {"scenario_id": scenario_id, "category": category,
            "seed": int(seed), "control_modes": list(env_kwargs["control_modes"]),
            "robot_roles": list(env_kwargs["robot_roles"]), "production_d0": bool(production_d0)},
        "evaluation_protocol": {"scenario_id": scenario_id, "category": category,
            "seed": int(seed), "time_budget_s": definition.time_budget_s,
            "horizon_ticks": env_kwargs["horizon"]},
        "drivetrain_physics": dynamics,
        "gamepiece_rules": {"fuel_count": env_kwargs["fuel_count"],
            "max_fuel_capacity": env_kwargs["max_fuel_capacity"],
            "fuel_physics": env_kwargs["fuel_physics"]},
        "field_layout": geometry,
        "scenario_registry": {"scenario_id": scenario_id, "version": definition.version,
            "category": category, "required_categories": list(definition.required_categories),
            "time_budget_s": definition.time_budget_s},
        "opponent_schedule": {"defender_family": defender_family, "pressure": pressure,
            "policy_id": phase4_opponent_policy_id, "schedule": opponent_schedule},
        "sensor_profile": values["sensor_noise_profiles"]}
    if evaluation_protocol_content_base64 is None:
        raise ValueError('effective capture requires the exact evaluator-loaded protocol bytes')
    try:
        evaluation_protocol_content = base64.b64decode(
            evaluation_protocol_content_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError('evaluator-loaded protocol bytes are invalid') from exc
    if not isinstance(evaluation_protocol_object, dict):
        raise ValueError('evaluator-loaded protocol must be an object')
    def reject_duplicate_json_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate key in evaluator-loaded protocol: {key}')
            result[key] = value
        return result
    try:
        decoded_protocol_object = json.loads(evaluation_protocol_content,
            object_pairs_hook=reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f'invalid JSON constant: {value}')))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('evaluator-loaded protocol bytes are not valid JSON') from exc
    if decoded_protocol_object != evaluation_protocol_object:
        raise ValueError('evaluator-loaded protocol object differs from its exact bytes')
    if hashlib.sha256(evaluation_protocol_content).hexdigest() != evaluation_protocol_content_sha256:
        raise ValueError('evaluator-loaded protocol bytes differ from their loaded digest')
    config_values["evaluation_protocol"] = evaluation_protocol_object
    artifacts = {}
    for name, value in config_values.items():
        content = (evaluation_protocol_content if name == "evaluation_protocol" else
            json.dumps(value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False).encode("utf-8"))
        artifacts[name] = {"content_base64": base64.b64encode(content).decode("ascii"),
            "content_sha256": hashlib.sha256(content).hexdigest()}
    producer = {"module": "frc_defense.curriculum_scenarios",
        "function": "make_curriculum_env", "source_identity_sha256": canonical_sha256(source)}
    dynamic_runtime_state = {"phase4_pressure_anchor": {
        "schema": PRESSURE_ANCHOR_RUNTIME_STATE_SCHEMA,
        "shape": ["world"], "dtype": "same_as_sim_pose",
        "device_owner": "CurriculumEnv.device",
        "tensor_owner": "CurriculumEnv.phase4_pressure_anchor_y_m",
        "capture_stages": ["pre_blocker_pose", "post_start_profile_pose"],
        "reset_mask_semantics": "selected_world_rows_only",
        "static_snapshot_contains_numeric_values": False}}
    body = {"schema": EFFECTIVE_SNAPSHOT_SCHEMA, "producer": producer,
        "factory_arguments": factory_arguments,
        "resolved_fields": resolved_fields, "configuration_artifacts": artifacts,
        "dynamic_runtime_state": dynamic_runtime_state,
        "runtime_identity": runtime}
    body["snapshot_sha256"] = canonical_sha256(body)
    return body


def _pressure(category, seed, family='category_scripted'):
    marker = next((part for part in category.split('/') if part.startswith('level_')), 'level_0')
    level = _pressure_level(category)
    if family in ('orbiting_interceptor', 'split_chaser'):
        level = max(level, 4)
    return {'level': level, 'style': marker, 'slots': (0, 1) if level == 5 else ((0,) if level else ()),
            'phase': (int(seed) % 1000) / 1000 * 2 * math.pi,
            'family': family}


def replace_contact_opponent_action(actions, contact_actions):
    """Return actions with only registered opponent slot 2 replaced."""
    if (not isinstance(actions, torch.Tensor) or actions.ndim != 2 or
            actions.shape[1] != 6 or not isinstance(contact_actions, torch.Tensor) or
            contact_actions.shape != actions.shape):
        raise ValueError('contact action override requires matching [world, 6] tensors')
    result = actions.clone()
    result[:, 2] = contact_actions[:, 2].to(device=result.device, dtype=result.dtype)
    return result


def _pressure_level(category):
    marker = next((part for part in category.split('/') if part.startswith('level_')), 'level_0')
    try:
        return int(marker.split('_')[1])
    except (IndexError, ValueError):
        return 0


def _refresh(env, *, mask=None):
    if mask is not None:
        if (not isinstance(mask, torch.Tensor) or mask.device != env.device or
                mask.dtype is not torch.bool or tuple(mask.shape) != (env.n,)):
            raise ValueError('reset refresh mask must be device bool [world]')
        env.synchronize_policy_pose(mask)
        focal_pose_assignment = getattr(env, 'phase4_focal_pose_assignment', None)
        if focal_pose_assignment is not None:
            applied, truth_preserved = apply_phase4_focal_pose_assignment_reset(
                env, focal_pose_assignment, mask)
        else:
            applied = mask
            truth_preserved = torch.ones_like(mask)
            prior = getattr(env, 'phase4_focal_pose_reset_applied', None)
            if not isinstance(prior, torch.Tensor) or tuple(prior.shape) != (env.n,):
                prior = torch.zeros_like(mask)
            env.phase4_focal_pose_reset_applied = torch.where(mask, applied, prior)
            prior_truth = getattr(env, 'phase4_focal_pose_truth_preserved', None)
            if not isinstance(prior_truth, torch.Tensor) or tuple(prior_truth.shape) != (env.n,):
                prior_truth = torch.zeros_like(mask)
            env.phase4_focal_pose_truth_preserved = torch.where(mask, truth_preserved,
                                                                prior_truth)
        env.sim.velocity.copy_(torch.where(mask[:, None, None],
            torch.zeros_like(env.sim.velocity), env.sim.velocity))
        env._target_collect_heading.copy_(torch.where(mask[:, None],
            env.policy_pose[:, :, 2], env._target_collect_heading))
        env._target_turn_anchor.copy_(torch.where(mask[:, None, None],
            env.policy_pose[:, :, :2], env._target_turn_anchor))
        cursor = env._closest_raster_indices(env.policy_pose[:, :, :2])
        env._raster_cursor.copy_(torch.where(mask[:, None], cursor, env._raster_cursor))
        env._raster_reanchor.copy_(torch.where(mask[:, None],
            torch.ones_like(env._raster_reanchor), env._raster_reanchor))
        for name in ('last_path', 'last_lengths', 'last_intercept', 'last_intercept_time',
                     'last_start', 'last_goal', 'last_heading', 'last_progress',
                     'last_trench_alignment', 'last_speed_profile', 'potential'):
            value = getattr(env.planner, name)
            row_mask = mask.reshape((env.n,) + (1,) * (value.ndim - 1))
            value.copy_(torch.where(row_mask, torch.zeros_like(value), value))
        env._planner_controller_modes_dirty_mask |= mask
        cells_x = (env.piece_pos[..., 0] / env._pickup_grid_cell_size).floor().long()
        cells_y = (env.piece_pos[..., 1] / env._pickup_grid_cell_size).floor().long()
        cells = (cells_y.clamp(0, env._pickup_grid_ny-1) * env._pickup_grid_nx + cells_x.clamp(0, env._pickup_grid_nx-1))
        env._pickup_possible_cells[:, :, 0].copy_(torch.where(mask[:, None],
            cells.to(torch.int16), env._pickup_possible_cells[:, :, 0]))
        reset_fuel_physics(env, mask)
        for name in ('track_pos', 'track_vel', 'track_mask', 'track_confidence',
                     'track_spatial_extent', 'track_angular_extent', 'track_merged',
                     '_current_fuel_visibility', 'perception_quality_state',
                     'opponent_valid', 'opponent_pose', 'opponent_velocity', 'opponent_size'):
            value = getattr(env, name)
            row_mask = mask.reshape((env.n,) + (1,) * (value.ndim - 1))
            value.copy_(torch.where(row_mask, torch.zeros_like(value), value))
        env.track_age.copy_(torch.where(mask[:, None],
            torch.full_like(env.track_age, float('inf')), env.track_age))
        env.opponent_age.copy_(torch.where(mask[:, None],
            torch.full_like(env.opponent_age, float('inf')), env.opponent_age))
        env._update_perception_for_reset(mask)
        fresh = env.observe()
        if env._last_obs is None:
            env._last_obs = fresh
        else:
            env._last_obs = torch.where(mask[:, None, None], fresh, env._last_obs)
        env._reset_phase4_observation_buffer(env._last_obs, mask=mask)
        return env._last_obs

    # Preserve the existing unmasked refresh API for non-reset callers.
    env.synchronize_policy_pose()
    focal_pose_assignment = getattr(env, 'phase4_focal_pose_assignment', None)
    if focal_pose_assignment is not None:
        apply_phase4_focal_pose_assignment(env, focal_pose_assignment)
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


def reset_curriculum_scenario(env, seed, *, mask=None):
    """Clear all physics/planner/sensor state, then expose ordinary perception."""
    config = env.curriculum
    if mask is None:
        reset_mask = torch.ones((env.n,), device=env.device, dtype=torch.bool)
    elif (not isinstance(mask, torch.Tensor) or mask.device != env.device or
            mask.dtype is not torch.bool or tuple(mask.shape) != (env.n,)):
        raise ValueError('curriculum reset mask must be device bool [world]')
    else:
        reset_mask = mask
    def write_rows(destination, values):
        values = torch.as_tensor(values, device=destination.device,
                                 dtype=destination.dtype).expand_as(destination)
        row_mask = reset_mask.reshape((env.n,) + (1,) * (destination.ndim - 1))
        destination.copy_(torch.where(row_mask, values, destination))

    config['seed'] = int(seed)
    if getattr(env, 'contact_geometry_id', None) is not None:
        write_rows(env.contact_geometry_runtime['route_seen'], False)
        write_rows(env.contact_geometry_runtime['route_seen_time_s'], -1.)
        write_rows(env.contact_geometry_runtime['approach_action_ticks'], 0)
    # Architecture reset already clears the ordinary core state and caches.
    env.curriculum_schedule = None
    # The curriculum reset performs its own final _refresh below. Defer the
    # base reset's perception/observation so a registered full-team start
    # profile can be applied before any policy-facing state is generated.
    reset_scenario(env, 'pickup_scattered', seed, refresh_perception=False,
                   mask=(reset_mask if mask is not None else None))
    env.architecture_pickup_only = False
    name, category, route = config['scenario_id'], config['category'], config['route']
    def rand(shape):
        shape = tuple(shape)
        return env._random_active(reset_mask, shape[1:],
                                  stream='scenario_reset')
    write_rows(env.piece_active, False)
    write_rows(env.piece_owner, -1)
    write_rows(env.piece_zone, -1)
    write_rows(env.piece_pos, 0.)
    write_rows(env.piece_vel, 0.)
    y = {'open': 4.035, 'trench': .8, 'bump': 2.5}[route]
    write_rows(env.sim.pose[:, 3], env.sim.pose.new_tensor((8., y, 0.)))
    write_rows(env.curriculum_finished, False)
    write_rows(env.curriculum_teacher_delivering, False)
    config['targets'] = {'amount': 4, 'batch_size': 2, 'cycles': 2}
    config['allowed_actions'] = (0, 1, 2, 3, 4, 6, 7)
    active = not (name.startswith('inactive_') or category.startswith('inactive/'))
    schedule = ((0., active),)
    preload = 0
    count, origin = 18, (8.8, y)
    if name == 'pickup_simple':
        write_rows(env.sim.pose[:, 3, 0], env.sim.pose[:, 3, 0] + ((rand((env.n,))-.5)*.3))
        write_rows(env.sim.pose[:, 3, 2], (rand((env.n,))-.5)*.6)
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
            write_rows(env.sim.pose[:, 3, 2], math.pi)
    elif name == 'pickup_multi_zone':
        count = 24
        origin = (7.8, .8)
        write_rows(env.sim.pose[:, 3, :2], env.sim.pose.new_tensor((7., .8)))
        config['allowed_actions'] = (0, 1, 2, 3, 6, 7)
        config['targets']['amount'] = 14
    elif name == 'active_carry_to_zone':
        count, preload = 0, 6
        config['allowed_actions'] = (4, 7)
        write_rows(env.sim.pose[:, 3, 0], 10.5)
    elif name == 'active_shoot_preloaded':
        count, preload = 0, 6
        config['allowed_actions'] = (4, 7)
        write_rows(env.sim.pose[:, 3, :2], env.sim.pose.new_tensor((14., 4.035)))
        write_rows(env.sim.pose[:, 3, 2], math.pi)
        if category.endswith('approach'):
            write_rows(env.sim.pose[:, 3, :2], env.sim.pose.new_tensor((15., 6.7)))
        if category.endswith('heading_error'):
            write_rows(env.sim.pose[:, 3, 2], math.pi/2)
    elif name == 'inactive_ferry_preloaded':
        count, preload = 0, (3 if category.endswith('partial_load') else 6)
        config['targets']['amount'] = 2 if preload == 3 else 4
        config['allowed_actions'] = (6, 7)
        write_rows(env.sim.pose[:, 3, :2], env.sim.pose.new_tensor((8.8, 4.035)))
        if category.endswith('approach'):
            write_rows(env.sim.pose[:, 3, 0], 6.8)
        if category.endswith('alignment'):
            write_rows(env.sim.pose[:, 3, 2], math.pi)
        if category.endswith('landing'):
            write_rows(env.sim.pose[:, 3, 1], 6.8)
    elif name == 'active_collect_carry_score' and route == 'open':
        # Open collection/scoring is within the alliance zone. Cross-boundary
        # carry gates separately require actual trench or bump traversal.
        write_rows(env.sim.pose[:, 3, :2], env.sim.pose.new_tensor((14., 6.8)))
        write_rows(env.sim.pose[:, 3, 2], math.pi)
        origin = (13., 6.8)
    elif name == 'hub_transition_cycles':
        if '/' not in category:
            # Explicit recovery tasks, with useful stock available to retain
            # until activation instead of demanding an impossible transfer.
            preload = 4
            count = 0
            write_rows(env.sim.pose[:, 3, :2], env.sim.pose.new_tensor((14., 6.8)))
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
                write_rows(env.sim.pose[:, 3, :2], env.sim.pose.new_tensor((14., 6.8)))
                write_rows(env.sim.pose[:, 3, 2], math.pi)
                origin = (13., 6.8)
            if situation == 'carrying_home':
                write_rows(env.sim.pose[:, 3, :2], env.sim.pose.new_tensor((10.5, .8)))
            if situation == 'staged_at_home':
                origin = (13.2, 6.8)
            # after_release is initialized as physical airborne outside-origin
            # delivery below, not as already-credited staging.
            if situation == 'after_release' and load == 'empty':
                preload = 0
            config['targets']['cycles'] = 1
    if name in ('offense_vs_scripted_pressure', 'team_delivery_cycles', 'full_match_offense'):
        if route == 'open' and active:
            write_rows(env.sim.pose[:, 3, :2], env.sim.pose.new_tensor((14., 6.8)))
            write_rows(env.sim.pose[:, 3, 2], math.pi)
            origin = (13., 6.8)
        if name == 'full_match_offense':
            schedule = None
        elif name == 'team_delivery_cycles' and not active:
            schedule = ((0., False), (45., True))
        for slot in (4, 5):
            if env.control_modes[slot] != 'none':
                write_rows(env.sim.pose[:, slot], env.sim.pose.new_tensor((14.5, 2. + (slot-4)*4., math.pi)))
        if name == 'team_delivery_cycles' and env.control_modes[0] != 'none':
            write_rows(env.sim.pose[:, 0], env.sim.pose.new_tensor((7., 4., 0.)))
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
                write_rows(env.sim.pose[:, 3, :2], env.sim.pose.new_tensor((6.5, 6.8)))
            if category == 'depleted':
                # A small remnant in the starting zone; the required amount
                # forces leaving it and finding a productive zone.
                positions[:, :2] = env.sim.pose[:, 3, None, :2] + env.piece_pos.new_tensor((.8, 0.))
        write_rows(env.piece_pos[:, :count], positions)
        write_rows(env.piece_active[:, :count], True)
        write_rows(env.piece_zone[:, :count], 0)
    if preload:
        write_rows(env.piece_active[:, count:count+preload], True)
        write_rows(env.piece_owner[:, count:count+preload], 3)
        write_rows(env.piece_pos[:, count:count+preload], env.sim.pose[:, 3, None, :2])
        write_rows(env.piece_zone[:, count:count+preload], 6)
    if route in ('trench', 'bump') and not active and name in ('inactive_collect_ferry', 'offense_vs_scripted_pressure', 'team_delivery_cycles', 'full_match_offense'):
        write_rows(env.sim.pose[:, 3, :2], env.sim.pose.new_tensor((13.3, y)))
        write_rows(env.sim.pose[:, 3, 2], math.pi)
        if count:
            write_rows(env.piece_pos[:, :count, 0], 9.8 + (torch.arange(count, device=env.device)%6).float()*.3)
    if name == 'full_match_offense':
        for slot in range(3):
            if env.control_modes[slot] == 'deterministic':
                write_rows(env.sim.pose[:, slot], env.sim.pose.new_tensor((5.5+slot*.9, .8+slot*2.5, math.pi)))
    env.planner.avoid_bumps = route != 'bump'
    pressure = config['pressure'] = _pressure(category, seed, config['defender_family']) if name in ('offense_vs_scripted_pressure', 'full_match_offense') else _pressure('', seed, config['defender_family'])
    if pressure['level'] == 3 and not active and route == 'open':
        blocked_y = .8 if pressure['style'] != 'level_3_bump' else 2.5
        if pressure['style'] == 'level_3_seeded_choice':
            blocked_y = .8 if int(seed)%2 else 2.5
        write_rows(env.sim.pose[:, 3], env.sim.pose.new_tensor((13.3, blocked_y, math.pi)))
        write_rows(env.piece_pos[:, :count, 1], blocked_y)
        write_rows(env.piece_pos[:, :count, 0], 9.8 + (torch.arange(count, device=env.device)%6).float()*.3)
    phase = torch.full((env.n,), float(pressure['phase']), device=env.device,
                       dtype=env.phase4_pressure_phase.dtype)
    write_rows(env.phase4_pressure_phase, phase)
    action_mask = torch.ones((env.n, 6, 8), device=env.device, dtype=torch.bool)
    action_mask[:, 3] = False
    action_mask[:, 3, list(config['allowed_actions'])] = True
    if not isinstance(getattr(env, 'curriculum_action_mask', None), torch.Tensor):
        env.curriculum_action_mask = action_mask
    else:
        write_rows(env.curriculum_action_mask, action_mask)
    if config['start_condition'] in ('offset_upper_lane', 'offset_lower_lane'):
        offset = .25 if config['start_condition'] == 'offset_upper_lane' else -.25
        on_field = ((env.sim.pose[..., 0] >= 0.) & (env.sim.pose[..., 0] <= env.field_length)
                    & (env.sim.pose[..., 1] >= 0.) & (env.sim.pose[..., 1] <= env.sim.field_width))
        write_rows(env.sim.pose[..., 1], env.sim.pose[..., 1] + (on_field.to(env.sim.pose.dtype) * offset))
        write_rows(env.piece_pos[..., 1], env.piece_pos[..., 1] + (env.piece_active.to(env.piece_pos.dtype) * offset))
    start_assignment = getattr(env, 'phase4_start_profile_assignment', None)
    def reset_status(name, initial):
        prior = getattr(env, name, None)
        if not isinstance(prior, torch.Tensor) or tuple(prior.shape) != tuple(initial.shape):
            prior = initial
        row_mask = reset_mask.reshape((env.n,) + (1,) * (prior.ndim - 1))
        setattr(env, name, torch.where(row_mask, initial, prior))
    reset_status('phase4_start_profile_applied', torch.zeros_like(reset_mask))
    reset_status('phase4_start_profile_legality', torch.zeros_like(reset_mask))
    reset_status('phase4_start_profile_failure_mask', torch.zeros_like(reset_mask))
    reset_status('phase4_reset_failure_mask', torch.zeros_like(reset_mask))
    reset_status('phase4_contact_profile_legality', torch.zeros_like(reset_mask))
    reset_status('phase4_start_profile_validation_rate', torch.zeros(
        (env.n,), device=env.device, dtype=torch.float32))
    reset_status('phase4_start_profile_validated_world_count', torch.zeros(
        (env.n,), device=env.device, dtype=torch.long))
    if env.contact_geometry_id is not None:
        contact_poses = contact_exposure_profile(env, env.contact_geometry_id,
            mask=reset_mask)
        env.sim.pose.copy_(torch.where(reset_mask[:, None, None], contact_poses,
                                       env.sim.pose))
        env.sim.velocity.copy_(torch.where(reset_mask[:, None, None],
            torch.zeros_like(env.sim.velocity), env.sim.velocity))
        contact_legal = full_team_start_profile_legality(env, env.sim.pose)
        write_rows(env.phase4_contact_profile_legality, contact_legal)
        write_rows(env.phase4_reset_failure_mask,
            env.phase4_reset_failure_mask | ~contact_legal)
        write_rows(env.phase4_pressure_anchor_y_m, env.sim.pose[:, 3, 1])
    if start_assignment is not None:
        start_assignment = validate_phase4_start_profile_assignment(start_assignment)
        apply_full_team_start_profile(env,
            start_assignment['runtime_profile_name'], mask=reset_mask)
        write_rows(env.phase4_pressure_anchor_y_m, env.sim.pose[:, 3, 1])
        config['phase4_start_profile'] = {
            'factor_id': start_assignment['factor_id'],
            'registry_profile_id': start_assignment['registry_profile_id'],
            'runtime_profile_name': start_assignment['runtime_profile_name'],
            'profile_config_sha256': start_assignment['profile_config_sha256'],
            'assignment_sha256': start_assignment['assignment_sha256']}
    # Capture the selected world's actual reset pose only after every registered
    # contact/start profile has established it. Blockers consume this same
    # rowwise anchor, so their placement and later scripted targets agree.
    write_rows(env.phase4_pressure_anchor_y_m, env.sim.pose[:, 3, 1])
    blocker_y = env.phase4_pressure_anchor_y_m
    if pressure['style'] == 'level_3_trench':
        blocker_y = torch.full_like(blocker_y, .8)
    if pressure['style'] == 'level_3_bump':
        blocker_y = torch.full_like(blocker_y, 2.5)
    if pressure['style'] == 'level_3_seeded_choice':
        blocker_y = torch.full_like(blocker_y, .8 if int(seed) % 2 else 2.5)
    for slot in pressure['slots']:
        blocker_x = 11.9 if pressure['level'] == 3 else (13.6 if active and route == 'open' else 9.8)
        blocker_pose = torch.stack((torch.full_like(blocker_y, blocker_x),
            blocker_y + slot * .9, torch.full_like(blocker_y, math.pi)), dim=-1)
        write_rows(env.sim.pose[:, slot], blocker_pose)
    if (env.phase4_factor_config is not None and
            env.phase4_factor_config['kind'] == 'stationary_blockage'):
        factor = env.phase4_factor_config
        slot = factor['blocker_slot']
        anchor_y = env.phase4_pressure_anchor_y_m
        blocker_pose = torch.stack((torch.full_like(anchor_y, factor['blocker_x_m']),
            anchor_y, torch.full_like(anchor_y, math.pi)), dim=-1)
        write_rows(env.sim.pose[:, slot], blocker_pose)
        assigned_pose = env.sim.pose[:, slot].clone()
        prior_assigned_pose = getattr(env, 'stationary_blockage_assigned_pose',
                                      torch.zeros_like(assigned_pose))
        write_rows(prior_assigned_pose, assigned_pose)
        env.stationary_blockage_assigned_pose = prior_assigned_pose
        env.phase4_runtime_metrics['stationary_blockage_assigned'] = True
        prior_valid = env.phase4_runtime_metrics['stationary_blockage_pose_valid']
        env.phase4_runtime_metrics['stationary_blockage_pose_valid'] = torch.where(
            reset_mask, torch.ones_like(reset_mask), prior_valid)
    config['schedule'] = schedule
    env.curriculum_schedule = schedule
    write_rows(env.match_elapsed, -env.dt)
    env._update_match(reset_mask)
    prior_generation = getattr(env, 'curriculum_episode_generation', None)
    if not isinstance(prior_generation, torch.Tensor) or tuple(prior_generation.shape) != (env.n,):
        prior_generation = torch.zeros((env.n,), device=env.device, dtype=torch.long)
    env.curriculum_episode_generation = prior_generation + reset_mask.long()
    if hasattr(env, 'curriculum_metrics'):
        env.curriculum_metrics.reset(reset_mask)
    else:
        env.curriculum_metrics = CurriculumMetrics(env)
    _refresh(env, mask=reset_mask)
    if name == 'hub_transition_cycles' and '/' in category:
        situation = category.split('/')[-1]
        if situation == 'staged_at_home':
            # Initial useful stock represents a documented earlier external
            # delivery; no acquisition or delivery reward is paid at reset.
            staged = slice(80, 84)
            write_rows(env.piece_active[:, staged], True)
            write_rows(env.piece_owner[:, staged], -1)
            write_rows(env.piece_pos[:, staged, 0], 13.4)
            write_rows(env.piece_pos[:, staged, 1], 6.2 + torch.arange(4, device=env.device)*.3)
            write_rows(env.curriculum_metrics.ferry_credit[:, staged], True)
            write_rows(env.curriculum_metrics.initial_staged[:, staged], True)
            write_rows(env.curriculum_metrics.inactive_origin[:, staged], True)
        elif situation == 'after_release':
            from .fuel_physics_runtime import launch_fuel
            selected = torch.zeros_like(env.piece_active)
            selected[:, 80:82] = reset_mask[:, None]
            env.piece_active |= selected
            write_rows(env.piece_pos[:, 80:82], env.piece_pos.new_tensor((9., 4.)))
            env.curriculum_metrics.external |= selected
            owners = torch.full_like(env.piece_owner, 3)
            env.curriculum_metrics.released(selected, owners)
            launch_fuel(env, selected, env.piece_pos.new_tensor((13.6, 6.8)),
                        origins=env.piece_pos, height=.4, flight_time=1.)
        # Refresh normal sensing after additional physically staged pieces.
        if situation == 'staged_at_home':
            _refresh(env, mask=reset_mask)
        else:
            env._update_perception_for_reset(reset_mask)
            fresh = env.observe()
            env._last_obs = torch.where(reset_mask[:, None, None], fresh, env._last_obs)
    prior_defender_anchor = getattr(env, 'phase4_defender_anchor', None)
    if not isinstance(prior_defender_anchor, torch.Tensor) or tuple(prior_defender_anchor.shape) != tuple(env.policy_pose.shape):
        prior_defender_anchor = env.policy_pose.clone()
    env.phase4_defender_anchor = torch.where(reset_mask[:, None, None],
        env.policy_pose, prior_defender_anchor)
    env._reset_phase4_observation_buffer(env._last_obs, mask=reset_mask)
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
