"""Typed, provenance-aware per-episode metrics for MECO evaluation.

The ledger intentionally distinguishes sampled outer environment-step signals
from physics-substep totals and labels XY displacement as a chord proxy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


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
    contact_outer_step_samples: int = 0
    robot_contact_outer_step_samples: int | None = 0
    static_contact_outer_step_samples: int | None = 0
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
    def _focal_contact(env, name: str) -> bool | None:
        value = getattr(getattr(env, "sim", None), name, None)
        if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[1] <= 3:
            return None
        return bool(value[0, 3].detach().any().item())

    def observe_outer_step(self) -> None:
        """Sample event/contact tensors after one completed ``env.step``."""
        pose = getattr(getattr(self.env, "sim", None), "pose", None)
        if isinstance(pose, torch.Tensor) and pose.ndim >= 3 and pose.shape[1] > 3:
            xy = pose[0, 3, :2].detach()
            if self._previous_xy is not None:
                self.xy_chord_proxy_m += float((xy - self._previous_xy).norm().item())
            self._previous_xy = xy.clone()

        self.outer_env_steps += 1
        self.elapsed_simulated_seconds += float(getattr(self.env, "dt", .02))
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

        robot_contact = self._focal_contact(self.env, "robot_contact")
        field_contact = self._focal_contact(self.env, "field_contact")
        wall_contact = self._focal_contact(self.env, "wall_contact")
        if robot_contact is None:
            self.robot_contact_outer_step_samples = None
        elif self.robot_contact_outer_step_samples is not None:
            self.robot_contact_outer_step_samples += int(robot_contact)
        if field_contact is None or wall_contact is None:
            self.static_contact_outer_step_samples = None
        elif self.static_contact_outer_step_samples is not None:
            self.static_contact_outer_step_samples += int(field_contact or wall_contact)
        if (robot_contact is not None or field_contact is not None or
                wall_contact is not None):
            self.contact_outer_step_samples += 1

    def observe_route(self, route_id: str | None) -> None:
        if self.route_tracking_available:
            self.route_ids.append(route_id)

    def observe_goal_attempt(self, *, action: int, route_id: str | None,
                             goal_xy: tuple[float, float] | None,
                             goal_heading: float | None,
                             terminal_goal_distance_m: float | None,
                             terminal_heading_error_rad: float | None,
                             heading_alignment_opportunity: bool) -> None:
        if not self.goal_tracking_available or goal_xy is None:
            return
        self.goal_attempts.append({
            "action": int(action),
            "route_id": route_id,
            "goal_xy_m": [float(goal_xy[0]), float(goal_xy[1])],
            "goal_heading_rad": (None if goal_heading is None else float(goal_heading)),
            "terminal_goal_distance_m": (None if terminal_goal_distance_m is None
                                          else float(terminal_goal_distance_m)),
            "goal_endpoint_within_0_04m": (None if terminal_goal_distance_m is None
                                           else terminal_goal_distance_m <= .04),
            "heading_alignment_opportunity": bool(heading_alignment_opportunity),
            "terminal_heading_error_rad": (None if terminal_heading_error_rad is None
                                            else float(terminal_heading_error_rad)),
            "heading_endpoint_within_0_1rad": (
                None if (not heading_alignment_opportunity or
                         terminal_heading_error_rad is None)
                else terminal_heading_error_rad <= .1),
        })

    def to_record(self) -> dict:
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
        return {
            "schema_version": 1,
            "scope": "focal_slot_3_per_episode",
            "availability": {
                "route_selection": self.route_tracking_available,
                "selected_goal_endpoint": self.goal_tracking_available,
                "fuel_event_outer_step_sampling": self.fuel_event_steps_available,
                "abandoned_fuel_event": False,
                "failed_collection_event": False,
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
            "heading_endpoint_within_0_1rad_count": (
                sum(row["heading_endpoint_within_0_1rad"] is True
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
            "contacts": {
                "sample_scope": "outer_env_step_sampled",
                "sampled_outer_steps": self.contact_outer_step_samples,
                "robot_contact_outer_step_samples": self.robot_contact_outer_step_samples,
                "static_contact_outer_step_samples": self.static_contact_outer_step_samples,
                "provenance": "focal slot-3 contact tensors sampled after each outer environment step; not physics-substep totals",
            },
            "provenance": {
                "time": "outer_env_steps multiplied by env.dt; simulator elapsed time at evaluator outer-step cadence",
                "distance": "sum of consecutive focal XY pose displacements at outer environment-step boundaries; chord proxy, not curved path length",
                "route_switches": "selected route IDs across macro decisions; not AD* replans",
                "route_aba_proxy_count": "three consecutive selected IDs A-B-A; route-selection oscillation proxy, not replanning or feint detection",
                "goal_attempts": "one selected endpoint record per macro execution; Plan A continuation macro steps are included",
                "claims_excluded": ["interception", "feint detection", "AD* replans", "physics-substep contact totals", "exact curved path length"],
            },
        }
