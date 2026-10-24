"""Typed, provenance-aware per-episode metrics for MECO evaluation.

The ledger intentionally distinguishes sampled outer environment-step signals
from physics-substep totals and labels XY displacement as a chord proxy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

SAFETY_METRICS_SCHEMA = "meco-safety-rates-v1"
SAFETY_CONTACT_METRIC_NAMES = (
    "safety_robot_contact_rate",
    "safety_field_contact_rate",
    "safety_wall_contact_rate",
)


@dataclass
class MecoEpisodeMetricLedger:
    """Accumulate only focal-robot signals observed by the evaluator."""

    env: object = field(repr=False)
    route_tracking_available: bool = True
    goal_tracking_available: bool = True
    outer_env_steps: int = 0
    elapsed_simulated_seconds: float = 0.0
    xy_chord_proxy_m: float = 0.0
    fuel_acquired: int = 0
    fuel_scored: int = 0
    fuel_passed: int = 0
    fuel_denied: int = 0
    fuel_abandoned: int | None = None
    failed_collection: int | None = None
    fuel_event_steps_available: bool = True
    collection_bouts_available: bool = True
    collection_bouts: list[dict] = field(default_factory=list)
    _active_collection_bout: dict | None = field(default=None, repr=False)
    route_ids: list[str | None] = field(default_factory=list)
    goal_attempts: list[dict] = field(default_factory=list)
    _previous_xy: torch.Tensor | None = field(default=None, repr=False)

    def __post_init__(self):
        pose = getattr(getattr(self.env, "sim", None), "pose", None)
        if isinstance(pose, torch.Tensor) and pose.ndim >= 3 and pose.shape[1] > 3:
            self._previous_xy = pose[0, 3, :2].detach().clone()

    @staticmethod
    def _focal_event(env, name: str) -> int | None:
        value = getattr(env, name, None)
        if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[1] <= 3:
            return None
        return int(value[0, 3].detach().sum().item())

    @staticmethod
    def _focal_physics_counter(env, name: str) -> int | None:
        value = getattr(getattr(env, "sim", None), name, None)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] < 1 or value.shape[1] <= 3:
            return None
        return int(value[0, 3].detach().item())

    def observe_outer_step(self, executed_action: int | None = None) -> None:
        """Sample per-outer-step game events and evaluator-boundary distance."""
        pose = getattr(getattr(self.env, "sim", None), "pose", None)
        if isinstance(pose, torch.Tensor) and pose.ndim >= 3 and pose.shape[1] > 3:
            xy = pose[0, 3, :2].detach()
            if self._previous_xy is not None:
                self.xy_chord_proxy_m += float((xy - self._previous_xy).norm().item())
            self._previous_xy = xy.clone()

        self.outer_env_steps += 1
        self.elapsed_simulated_seconds += float(getattr(self.env, "dt", .02))
        acquired_this_step = self._focal_event(self.env, "fuel_acquired_event")
        if acquired_this_step is None:
            self.collection_bouts_available = False
        if executed_action is None or not isinstance(executed_action, int):
            self.collection_bouts_available = False
            self.close_collection_bout("action_signal_unavailable")
        elif 0 <= executed_action <= 3:
            if self._active_collection_bout is None:
                self._active_collection_bout = {
                    "start_tick": self.outer_env_steps - 1,
                    "end_tick": self.outer_env_steps,
                    "action_sequence": [],
                    "action_counts": {str(action): 0 for action in range(4)},
                    "acquired_quantity": 0,
                    "acquired_signal_available": acquired_this_step is not None,
                }
            bout = self._active_collection_bout
            bout["end_tick"] = self.outer_env_steps
            bout["action_sequence"].append(executed_action)
            bout["action_counts"][str(executed_action)] += 1
            if acquired_this_step is None:
                bout["acquired_signal_available"] = False
                self.collection_bouts_available = False
            else:
                bout["acquired_quantity"] += acquired_this_step
        else:
            self.close_collection_bout("non_collection_action")

        event_names = {
            "fuel_acquired": "fuel_acquired_event",
            "fuel_scored": "fuel_scored_event",
            "fuel_passed": "fuel_passed_event",
            "fuel_denied": "fuel_denied_event",
        }
        for field_name, tensor_name in event_names.items():
            value = self._focal_event(self.env, tensor_name)
            if value is None:
                self.fuel_event_steps_available = False
            else:
                setattr(self, field_name, getattr(self, field_name) + value)

    def close_collection_bout(self, reason: str = "episode_finalization") -> None:
        """Close the active contiguous action-0..3 bout, if one exists."""
        if self._active_collection_bout is None:
            return
        bout = self._active_collection_bout
        bout["closure_reason"] = reason
        bout["acquired"] = (
            bout["acquired_quantity"] > 0
            if bout["acquired_signal_available"] else None)
        self.collection_bouts.append(bout)
        self._active_collection_bout = None

    def observe_route(self, route_id: str | None) -> None:
        if self.route_tracking_available:
            self.route_ids.append(route_id)

    def observe_goal_attempt(self, *, action: int, route_id: str | None,
                             goal_xy: tuple[float, float] | None,
                             goal_heading: float | None,
                             terminal_goal_distance_m: float | None,
                             terminal_heading_error_rad: float | None,
                             heading_alignment_opportunity: bool,
                             selected_goal_id: str | None = None,
                             episode_id: str | None = None,
                             source_identity_sha256: str | None = None,
                             geometry_id: str | None = None,
                             execution_valid: bool | None = None,
                             execution_exclusion_reasons: tuple[str, ...] = ()) -> None:
        if not self.goal_tracking_available or goal_xy is None:
            return
        self.goal_attempts.append({
            "action": int(action),
            "route_id": route_id,
            "selected_goal_id": selected_goal_id,
            "episode_id": episode_id,
            "source_identity_sha256": source_identity_sha256,
            "geometry_id": geometry_id,
            "execution_valid": execution_valid,
            "execution_exclusion_reasons": list(execution_exclusion_reasons),
            "goal_xy_m": [float(goal_xy[0]), float(goal_xy[1])],
            "goal_heading_rad": (None if goal_heading is None else float(goal_heading)),
            "goal_heading_required": goal_heading is not None,
            "terminal_goal_distance_m": (None if terminal_goal_distance_m is None
                                          else float(terminal_goal_distance_m)),
            "goal_endpoint_within_0_04m": (None if terminal_goal_distance_m is None
                                           else terminal_goal_distance_m <= .04),
            "heading_alignment_opportunity": bool(heading_alignment_opportunity),
            "terminal_heading_error_rad": (None if terminal_heading_error_rad is None
                                            else float(terminal_heading_error_rad)),
            "heading_joint_endpoint_within_0_04m_and_0_1rad": (
                None if (terminal_goal_distance_m is None or
                         terminal_heading_error_rad is None)
                else terminal_goal_distance_m <= .04 and
                     terminal_heading_error_rad <= .1),
        })

    def to_record(self) -> dict:
        # Finalize even if the caller is serializing an episode after an
        # exception or timeout with a partial action macro.
        self.close_collection_bout("episode_finalization")
        route_switches = None
        route_aba_proxy = None
        if self.route_tracking_available:
            route_switches = sum(a is not None and b is not None and a != b
                                 for a, b in zip(self.route_ids, self.route_ids[1:]))
            route_aba_proxy = sum(a is not None and b is not None and c is not None
                                  and a == c and a != b
                                  for a, b, c in zip(self.route_ids,
                                      self.route_ids[1:], self.route_ids[2:]))
        goal_heading_attempts = [row for row in self.goal_attempts
                                 if row["goal_heading_rad"] is not None and
                                 row["heading_alignment_opportunity"]]
        active_physics_ticks = self._focal_physics_counter(
            self.env, "safety_active_physics_ticks")
        contact_counters = {
            "robot": self._focal_physics_counter(self.env, "safety_robot_contact_ticks"),
            "field": self._focal_physics_counter(self.env, "safety_field_contact_ticks"),
            "wall": self._focal_physics_counter(self.env, "safety_wall_contact_ticks"),
        }
        contact_rates = {
            f"safety_{name}_contact_rate": (
                count / active_physics_ticks
                if count is not None and active_physics_ticks is not None
                and active_physics_ticks > 0 else None)
            for name, count in contact_counters.items()
        }
        return {
            "schema_version": 4,
            "scope": "focal_slot_3_per_episode",
            "availability": {
                "route_selection": self.route_tracking_available,
                "selected_goal_endpoint": self.goal_tracking_available,
                "fuel_event_outer_step_sampling": self.fuel_event_steps_available,
                "collection_action_bouts": self.collection_bouts_available,
                "abandoned_fuel_event": False,
                "failed_collection_event": False,
                "safety_contact_tick_rates": (
                    active_physics_ticks is not None and active_physics_ticks > 0
                    and all(value is not None for value in contact_counters.values())),
                "illegal_transition_event": False,
                "footprint_violation_event": False,
            },
            "outer_env_steps": self.outer_env_steps,
            "elapsed_simulated_seconds": self.elapsed_simulated_seconds,
            "xy_chord_proxy_m": self.xy_chord_proxy_m,
            "route_switches": route_switches,
            "route_aba_proxy_count": route_aba_proxy,
            "selected_route_ids": self.route_ids if self.route_tracking_available else None,
            "goal_attempt_macro_count": (len(self.goal_attempts)
                                         if self.goal_tracking_available else None),
            "goal_endpoint_within_0_04m_count": (
                sum(row["goal_endpoint_within_0_04m"] is True
                    for row in self.goal_attempts)
                if self.goal_tracking_available else None),
            "heading_alignment_attempt_macro_count": (
                len(goal_heading_attempts) if self.goal_tracking_available else None),
            "heading_joint_endpoint_within_0_04m_and_0_1rad_count": (
                sum(row["heading_joint_endpoint_within_0_04m_and_0_1rad"] is True
                    for row in goal_heading_attempts)
                if self.goal_tracking_available else None),
            "goal_attempts": self.goal_attempts if self.goal_tracking_available else None,
            "fuel": {
                "acquired": self.fuel_acquired if self.fuel_event_steps_available else None,
                "scored": self.fuel_scored if self.fuel_event_steps_available else None,
                "passed": self.fuel_passed if self.fuel_event_steps_available else None,
                "denied": self.fuel_denied if self.fuel_event_steps_available else None,
                "abandoned": self.fuel_abandoned,
                "failed_collection": self.failed_collection,
                "provenance": {
                    "events": "sampled and accumulated once after each outer CurriculumEnv.step from simulator slot-3 event tensors",
                    "fuel_denied": "simulator denied-event counter; not interpreted as failed collection or failed intake",
                    "abandoned": "unavailable: no authoritative episode event writer is connected to this evaluation ledger",
                    "failed_collection": "unavailable: no authoritative failed-intake event is connected to this evaluation ledger",
                },
            },
            "collection_action_bouts": {
                "action_range_inclusive": [0, 3],
                "bouts": self.collection_bouts if self.collection_bouts_available else None,
                "count": len(self.collection_bouts) if self.collection_bouts_available else None,
                "acquired_count": (sum(row["acquired"] is True for row in self.collection_bouts)
                                   if self.collection_bouts_available else None),
                "unsuccessful_collection_action_bouts": (
                    sum(row["acquired"] is False for row in self.collection_bouts)
                    if self.collection_bouts_available else None),
                "acquired_proportion": (
                    sum(row["acquired"] is True for row in self.collection_bouts) /
                    len(self.collection_bouts)
                    if self.collection_bouts_available and self.collection_bouts else None),
                "provenance": "contiguous focal slot-3 executed actions 0..3 sampled after each outer CurriculumEnv.step; acquired iff focal fuel_acquired_event quantity increased during the bout; HUB denied remains a separate fuel.denied simulator event",
                "unavailable_reason": (None if self.collection_bouts_available else
                    "one or more outer steps lacked an executed-action or focal fuel_acquired_event signal"),
            },
            "contacts": {
                "sample_scope": "low_level_physics_tick",
                "active_physics_ticks": active_physics_ticks,
                "robot_contact_ticks": contact_counters["robot"],
                "field_contact_ticks": contact_counters["field"],
                "wall_contact_ticks": contact_counters["wall"],
                "rates_per_active_physics_tick": contact_rates,
                "provenance": "focal slot-3 contact flags accumulated once per active TensorPhysics.step after its collision passes; field and wall contacts are separate classes; these are observed contacts, not policy-rule violations",
                "missing_signal_semantics": "null; unavailable counters are never reported as zero",
            },
            "safety_metrics_schema": SAFETY_METRICS_SCHEMA,
            "safety_metrics": contact_rates,
            "unavailable_safety_metrics": {
                "illegal_transition_rate": None,
                "footprint_violation_rate": None,
                "reason": "no authoritative per-episode illegal-transition or footprint-violation event signal is connected to this evaluator",
            },
            "provenance": {
                "time": "outer_env_steps multiplied by env.dt; simulator elapsed time at evaluator outer-step cadence",
                "distance": "sum of consecutive focal XY pose displacements at outer environment-step boundaries; chord proxy, not curved path length",
                "route_switches": "selected route IDs across macro decisions; not AD* replans",
                "route_aba_proxy_count": "three consecutive selected IDs A-B-A; route-selection oscillation proxy, not replanning or feint detection",
                "goal_attempts": "one selected endpoint record per macro execution; Plan A continuation macro steps are included",
                "heading_alignment_attempts": "macro rows with a planned heading and any within-0.04m XY opportunity during that macro",
                "heading_joint_endpoint": "success requires both terminal XY distance <= 0.04m and terminal heading error <= 0.1 rad; an earlier in-macro XY opportunity alone is not endpoint success",
                "safety_contacts": "per-class focal contact ticks divided by focal active low-level physics ticks; not policy-rule violations",
                "claims_excluded": ["interception", "feint detection", "AD* replans", "illegal-transition or footprint-violation rates without authoritative event signals", "exact curved path length"],
            },
        }
