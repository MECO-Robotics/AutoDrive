"""Simulator-grounded Phase 4 trench interception and crossing evidence."""
from __future__ import annotations

import hashlib
import json
import math


TRENCH_OUTCOME_SCHEMA = "meco-phase4-trench-interception-outcome-v1"
TRENCH_POLICY_IDS = tuple(sorted(("A1", "A2", "A3", "C1", "C2",
    "A1_P0", "A2_P0", "A3_P0", "D0", "D1", "D2")))


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _finite_pair(value) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == 2 and
        all(type(item) in (int, float) and math.isfinite(item) for item in value))


def _clip_segment_to_band(a, b, xmin, xmax):
    """Return segment parameters inside an x band, or None for no overlap."""
    x0, x1 = a[0], b[0]
    if x0 == x1:
        return (0., 1.) if xmin <= x0 <= xmax else None
    ta, tb = (xmin - x0) / (x1 - x0), (xmax - x0) / (x1 - x0)
    lo, hi = max(0., min(ta, tb)), min(1., max(ta, tb))
    return (lo, hi) if lo <= hi else None


def _route_opening_id(points, openings):
    """Classify the first complete clear trench crossing on a frozen route."""
    for opening in openings:
        xmin, xmax = opening["centerline_band_x_m"]
        ymin, ymax = opening["clear_center_y_m"]
        in_band = False
        crossed = False
        direction = 0
        for a, b in zip(points, points[1:]):
            clipped = _clip_segment_to_band(a, b, xmin, xmax)
            if clipped is None:
                if in_band:
                    crossed = False
                    in_band = False
                continue
            t0, t1 = clipped
            y0 = a[1] + (b[1] - a[1]) * t0
            y1 = a[1] + (b[1] - a[1]) * t1
            if min(y0, y1) < ymin or max(y0, y1) > ymax:
                in_band = crossed = False
                continue
            if not in_band:
                dx = b[0] - a[0]
                if dx == 0:
                    continue
                direction = 1 if dx > 0 else -1
                in_band = True
                crossed = ((direction > 0 and a[0] <= xmin and b[0] >= xmax) or
                           (direction < 0 and a[0] >= xmax and b[0] <= xmin))
            else:
                crossed = crossed or (direction > 0 and b[0] >= xmax) or (
                    direction < 0 and b[0] <= xmin)
            if crossed:
                return opening["opening_id"]
    return None


def _crossings(samples, openings):
    """Find continuous, full-band crossings from contiguous truth-pose segments."""
    found = []
    state = {opening["opening_id"]: None for opening in openings}
    for left, right in zip(samples, samples[1:]):
        a, b = left["xy_m"], right["xy_m"]
        t0, t1 = left["physics_time_s"], right["physics_time_s"]
        for opening in openings:
            key = opening["opening_id"]
            xmin, xmax = opening["centerline_band_x_m"]
            ymin, ymax = opening["clear_center_y_m"]
            clipped = _clip_segment_to_band(a, b, xmin, xmax)
            if clipped is None:
                continue
            u0, u1 = clipped
            y0, y1 = a[1] + (b[1] - a[1]) * u0, a[1] + (b[1] - a[1]) * u1
            if min(y0, y1) < ymin or max(y0, y1) > ymax:
                state[key] = None
                continue
            dx = b[0] - a[0]
            if dx == 0:
                continue
            direction = 1 if dx > 0 else -1
            current = state[key]
            if current is None:
                enters = ((direction > 0 and a[0] < xmin <= b[0]) or
                          (direction < 0 and a[0] > xmax >= b[0]))
                if not enters:
                    continue
                entry_u = (xmin - a[0]) / dx if direction > 0 else (xmax - a[0]) / dx
                current = {"direction": direction,
                    "entry_time_s": t0 + (t1 - t0) * entry_u}
            elif current["direction"] != direction:
                state[key] = None
                continue
            exits = ((direction > 0 and a[0] <= xmax <= b[0]) or
                     (direction < 0 and a[0] >= xmin >= b[0]))
            if exits:
                exit_u = (xmax - a[0]) / dx if direction > 0 else (xmin - a[0]) / dx
                exit_time = t0 + (t1 - t0) * exit_u
                if exit_time > current["entry_time_s"]:
                    found.append({"opening_id": key,
                        "entry_time_s": current["entry_time_s"],
                        "exit_time_s": exit_time,
                        "direction": "increasing_x" if direction > 0 else "decreasing_x"})
                state[key] = None
            else:
                state[key] = current
    return sorted(found, key=lambda row: row["exit_time_s"])


def validate_trench_interception_record(record: dict, *, binding: dict,
        expected_condition: str, expected_policy_id: str, expected_seed: int,
        expected_source_sha256: str, expected_checkpoint_sha256: str,
        expected_device: dict) -> dict:
    """Recompute the row metrics and reject incomplete or mismatched evidence."""
    if not isinstance(record, dict) or record.get("schema") != TRENCH_OUTCOME_SCHEMA:
        raise ValueError("invalid trench-interception outcome record")
    reasons = []
    samples = record.get("truth_pose_samples")
    events = record.get("contact_events")
    horizon = record.get("horizon")
    if not isinstance(samples, list) or not isinstance(horizon, dict):
        raise ValueError("trench outcome lacks samples or horizon contract")
    expected_ticks = horizon.get("expected_physics_ticks")
    dt = horizon.get("physics_dt_s")
    full = (type(expected_ticks) is int and expected_ticks > 0 and
        type(dt) in (int, float) and math.isfinite(dt) and dt > 0 and
        len(samples) == expected_ticks + 1)
    if full:
        for tick, sample in enumerate(samples):
            if (not isinstance(sample, dict) or sample.get("physics_tick") != tick or
                    not _finite_pair(sample.get("xy_m")) or
                    type(sample.get("physics_time_s")) not in (int, float) or
                    not math.isfinite(sample["physics_time_s"]) or
                    not math.isclose(sample["physics_time_s"], tick * dt,
                        rel_tol=0., abs_tol=max(1e-9, dt * 1e-6))):
                full = False
                break
    if record.get("full_horizon_pose_coverage") is not full:
        raise ValueError("trench full-horizon pose coverage does not reconcile")
    if not full:
        reasons.append("incomplete_or_invalid_physics_pose_coverage")
    openings = record.get("opening_geometry")
    if not isinstance(openings, list) or not openings:
        raise ValueError("trench record lacks pinned opening geometry")
    route = record.get("initial_executable_route")
    if not isinstance(route, dict) or not isinstance(route.get("points_m"), list):
        reasons.append("initial_executable_route_unavailable")
        designated = None
    else:
        points = route["points_m"]
        if (len(points) < 2 or any(not _finite_pair(point) for point in points) or
                route.get("route_sha256") != _sha(points)):
            reasons.append("initial_executable_route_invalid")
            designated = None
        else:
            designated = _route_opening_id(points, openings)
            if designated is None:
                reasons.append("initial_route_does_not_designate_a_trench_opening")
    if record.get("designated_initial_opening_id") != designated:
        raise ValueError("trench initial opening differs from frozen route geometry")
    if record.get("invalid_reasons") != reasons:
        raise ValueError("trench invalid-row reasons do not reconcile")
    if not isinstance(events, list):
        raise ValueError("trench record lacks a pair-contact event list")
    if record.get("contact_capture_complete") is not full:
        reasons.append("incomplete_solver_contact_coverage")
    if record.get("simulator_contact_distinct_substep_count") != len(events):
        raise ValueError("trench contact stream does not reconcile to simulator pair counter")
    previous_tick = -1
    for index, event in enumerate(events):
        if (not isinstance(event, dict) or event.get("event_index") != index or
                event.get("robot_slot_ids") != [0, 3] or
                type(event.get("physics_tick")) is not int or
                event["physics_tick"] <= previous_tick or
                event["physics_tick"] > (expected_ticks or 0) or
                type(event.get("physics_time_s")) not in (int, float) or
                not math.isfinite(event["physics_time_s"]) or
                event.get("source") != "simulator_robot_pair_sat_contact_solver"):
            raise ValueError("trench slot-0/slot-3 contact event identity or clock is invalid")
        previous_tick = event["physics_tick"]
    if expected_condition == "factor_treatment":
        if record.get("configured_slots") != [0] or record.get("applied_slots") != [0]:
            raise ValueError("trench treatment did not apply exactly assigned slot 0")
        if full and record.get("interception_episode") is not bool(events):
            raise ValueError("trench treatment interception incidence differs from contact events")
        interception_numerator = int(bool(events)) if full and record.get(
            "contact_capture_complete") else None
    else:
        if record.get("applied_slots") != [] or record.get("interception_episode") is not None:
            raise ValueError("trench control must not apply or claim interceptor incidence")
        interception_numerator = None
    crossings = _crossings(samples, openings) if full else []
    alternate = (any(event["opening_id"] != designated for event in crossings)
                 if full and designated is not None else None)
    first_crossing = crossings[0] if crossings else None
    crossing_time = (first_crossing["exit_time_s"] if first_crossing else
                     horizon.get("horizon_elapsed_s") if full else None)
    crossing_censored = bool(first_crossing is None) if full else None
    if record.get("crossing_events") != crossings:
        raise ValueError("trench crossing events do not reconcile to continuous pose samples")
    if record.get("alternate_crossing_success") is not alternate:
        raise ValueError("trench alternate-crossing success does not reconcile")
    if (record.get("trench_crossing_time_s") != crossing_time or
            record.get("trench_crossing_right_censored") is not crossing_censored):
        raise ValueError("trench crossing time/censor does not reconcile")
    expected_binding = {
        "scenario_id": binding.get("scenario_id"),
        "category": binding.get("category"),
        "qualification_case": binding.get("qualification_case"),
        "protocol_id": binding.get("protocol_id"),
        "protocol_sha256": binding.get("protocol_sha256"),
        "seed_manifest_sha256": binding.get("seed_manifest_sha256"),
        "schedule_sha256": binding.get("schedule_sha256"),
        "pair_binding_id": binding.get("pair_binding_id"),
        "policy_id": expected_policy_id, "seed": expected_seed,
        "condition": expected_condition,
        "source_identity_sha256": expected_source_sha256,
        "policy_checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_bundle_sha256": binding.get("checkpoint_bundle_sha256"),
        "execution_device": expected_device}
    if any(record.get(key) != value for key, value in expected_binding.items()):
        raise ValueError("trench record differs from exact case/pair/source/checkpoint/device binding")
    if (record.get("factor_id") != "opponent.trench_interception" or
            record.get("opponent_policy_id") != "trench_interceptor" or
            not record.get("episode_id")):
        raise ValueError("trench record lacks exact factor or episode identity")
    route_valid = designated is not None and not any(reason in reasons for reason in (
        "initial_executable_route_unavailable", "initial_executable_route_invalid",
        "initial_route_does_not_designate_a_trench_opening"))
    return {"valid_full_horizon": full and not reasons,
        "full_horizon_pose_valid": full,
        "crossing_row_valid": full and route_valid,
        "interception_row_valid": full and record.get("contact_capture_complete") is True,
        "invalid_reasons": reasons, "interception_episode_numerator": interception_numerator,
        "alternate_crossing_success": alternate, "trench_crossing_time_s": crossing_time,
        "crossing_right_censored": crossing_censored,
        "crossing_events": crossings,
        "contact_event_count": len(events)}


class TrenchInterceptionEpisodeCapture:
    """Capture exact slot-3/slot-0 SAT events and continuous truth-pose coverage."""

    def __init__(self, env, binding: dict, *, condition: str, episode_id: str,
            source_identity_sha256: str, checkpoint_sha256: str, policy_id: str,
            seed: int, execution_device: dict, condition_id: str | None):
        if (binding.get("schema") != "meco-phase4-trench-interception-binding-v1" or
                binding.get("factor_id") != "opponent.trench_interception" or
                binding.get("policy_id") != "trench_interceptor" or
                binding.get("split") != "validation" or condition not in
                    ("no_factor", "factor_treatment") or
                policy_id not in TRENCH_POLICY_IDS or seed not in binding.get("seed_list", ())):
            raise ValueError("trench capture requires an exact held-out assignment")
        schedule = binding.get("schedule")
        if (not isinstance(schedule, dict) or schedule.get("schedule_sha256") !=
                binding.get("schedule_sha256") or schedule.get("configured_pressure_slots", {}).get(
                    binding.get("category")) != [0]):
            raise ValueError("trench binding must designate exactly pressure slot 0")
        pair_binding = binding.get("pair_binding")
        if (not isinstance(pair_binding, dict) or
                _sha(pair_binding) != binding.get("pair_binding_id") or
                pair_binding.get("schedule_sha256") != binding.get("schedule_sha256") or
                pair_binding.get("source_identity_sha256") != source_identity_sha256 or
                pair_binding.get("checkpoint_bundle_sha256") !=
                    binding.get("checkpoint_bundle_sha256") or
                pair_binding.get("execution_device") != execution_device):
            raise ValueError("trench stable pair identity differs from source/checkpoint/device")
        runtime = getattr(env, "phase4_opponent_runtime", None)
        if condition == "factor_treatment":
            if (not isinstance(runtime, dict) or runtime.get("policy_id") !=
                    "trench_interceptor" or runtime.get("configured_slots") != [0]):
                raise ValueError("trench treatment lacks applied policy runtime")
        elif runtime is not None:
            raise ValueError("trench no-factor control unexpectedly applies an opponent schedule")
        sim = env.sim
        if not all(hasattr(sim, name) for name in (
                "_collision_pair_i", "_collision_pair_j", "pair_contact_event_count",
                "pair_contact_last_substep", "pair_contact_last_time_s",
                "physics_substep_index", "physics_time_s")):
            raise ValueError("simulator lacks pair-identified contact counters and physics clock")
        pairs = list(zip(sim._collision_pair_i.tolist(), sim._collision_pair_j.tolist()))
        matches = [index for index, pair in enumerate(pairs)
                   if tuple(sorted((int(pair[0]), int(pair[1])))) == (0, 3)]
        if len(matches) != 1:
            raise ValueError("simulator collision table lacks unique slot-0/slot-3 pair")
        self.env, self.sim, self.binding = env, sim, binding
        self.condition, self.episode_id = condition, episode_id
        self.source_hash, self.checkpoint_hash = source_identity_sha256, checkpoint_sha256
        self.policy_id, self.seed, self.device = policy_id, seed, execution_device
        self.condition_id = condition_id
        self.pair_index = matches[0]
        self.previous_count = int(sim.pair_contact_event_count[0, self.pair_index].item())
        if self.previous_count != 0:
            raise ValueError("trench pair-contact counter must be clear at episode reset")
        self.last_tick = int(sim.physics_substep_index[0].item())
        self.last_time = float(sim.physics_time_s[0].item())
        if self.last_tick != 0 or self.last_time != 0.:
            raise ValueError("trench capture requires the reset physics clock")
        self.dt = float(sim.dt)
        self.openings = self._opening_geometry()
        self.route_points = None
        self.samples = [{"physics_tick": 0, "physics_time_s": 0.,
            "xy_m": [float(v) for v in sim.pose[0, 3, :2].tolist()]}]
        self.events = []

    def freeze_initial_route(self):
        """Freeze the first selected executable route before its first physics tick."""
        if self.route_points is None:
            self.route_points = self._initial_route()

    def _opening_geometry(self):
        boxes = getattr(self.env, "field_boxes", None)
        if not isinstance(boxes, (tuple, list)):
            raise ValueError("field geometry is not available for trench outcome capture")
        supports = {}
        for name in ("red_trench_support_lower", "red_trench_support_upper",
                     "blue_trench_support_lower", "blue_trench_support_upper"):
            matches = [box for box in boxes if getattr(box, "name", None) == name]
            if len(matches) != 1:
                raise ValueError("pinned field geometry lacks unique trench support boxes")
            supports[name] = matches[0]
        radius = math.hypot(float(self.sim.length[0, 3].item()),
                            float(self.sim.width[0, 3].item())) / 2
        result = []
        for color in ("red", "blue"):
            lower = supports[f"{color}_trench_support_lower"]
            upper = supports[f"{color}_trench_support_upper"]
            if (not math.isclose(float(lower.x), float(upper.x), abs_tol=1e-9) or
                    lower.y + lower.width / 2 >= upper.y - upper.width / 2):
                raise ValueError("trench support geometry has no contiguous legal opening")
            result.append({"opening_id": f"{color}_trench_opening",
                "trench_x_m": float(lower.x),
                "centerline_band_x_m": [float(lower.x - lower.length / 2 - radius),
                                         float(lower.x + lower.length / 2 + radius)],
                "clear_center_y_m": [float(lower.y + lower.width / 2 + radius),
                                     float(upper.y - upper.width / 2 - radius)],
                "source_support_box_ids": [lower.name, upper.name],
                "robot_conservative_radius_m": radius})
        if any(opening["clear_center_y_m"][0] >= opening["clear_center_y_m"][1]
               for opening in result):
            raise ValueError("trench opening is narrower than conservative robot clearance")
        return result

    def _initial_route(self):
        planner = getattr(self.env, "planner", None)
        if planner is None:
            return None
        try:
            length = int(planner.last_lengths.reshape(-1)[3].item())
            points = planner.last_path.reshape(
                -1, planner.last_path.shape[-2], 2)[3, :length].tolist()
            points = [[float(v) for v in point] for point in points]
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            return None
        if len(points) < 2 or any(not _finite_pair(point) for point in points):
            return None
        return points

    def observe_physics_step(self):
        tick = int(self.sim.physics_substep_index[0].item())
        time_s = float(self.sim.physics_time_s[0].item())
        pose = [float(v) for v in self.sim.pose[0, 3, :2].tolist()]
        if (tick != self.last_tick + 1 or not math.isfinite(time_s) or
                not math.isclose(time_s, self.last_time + self.dt,
                    rel_tol=0., abs_tol=max(1e-9, self.dt * 1e-6)) or
                not _finite_pair(pose)):
            raise RuntimeError("trench outcome sampler missed a physics tick or invalid pose")
        count = int(self.sim.pair_contact_event_count[0, self.pair_index].item())
        delta = count - self.previous_count
        if delta not in (0, 1):
            raise RuntimeError("trench pair-contact counter changed outside one substep")
        if delta:
            event_tick = int(self.sim.pair_contact_last_substep[0, self.pair_index].item())
            event_time = float(self.sim.pair_contact_last_time_s[0, self.pair_index].item())
            if (event_tick != tick or not math.isclose(event_time, time_s,
                    rel_tol=0., abs_tol=max(1e-9, self.dt * 1e-6))):
                raise RuntimeError("trench solver contact pair has mismatched substep/time")
            self.events.append({"event_index": len(self.events),
                "source": "simulator_robot_pair_sat_contact_solver",
                "robot_slot_ids": [0, 3], "physics_tick": tick,
                "physics_time_s": event_time})
        self.previous_count = count
        self.last_tick, self.last_time = tick, time_s
        self.samples.append({"physics_tick": tick, "physics_time_s": time_s,
            "xy_m": pose})

    def finish(self, *, episode_censored: bool = False) -> dict:
        expected_ticks = int(self.env.horizon)
        horizon_s = expected_ticks * self.dt
        full_coverage = (not episode_censored and self.last_tick == expected_ticks and
                         len(self.samples) == expected_ticks + 1)
        route = None if self.route_points is None else {
            "route_sha256": _sha(self.route_points), "points_m": self.route_points}
        designated = (None if route is None else
                      _route_opening_id(self.route_points, self.openings))
        invalid_reasons = []
        if not full_coverage:
            invalid_reasons.append("incomplete_or_invalid_physics_pose_coverage")
        if route is None:
            invalid_reasons.append("initial_executable_route_unavailable")
        elif designated is None:
            invalid_reasons.append("initial_route_does_not_designate_a_trench_opening")
        runtime = getattr(self.env, "phase4_opponent_runtime", None)
        applied = [] if self.condition == "no_factor" else (
            runtime.get("applied_slots") if isinstance(runtime, dict) else None)
        if self.condition == "factor_treatment" and isinstance(runtime, dict):
            calls = runtime.get("applied_command_calls_by_slot", {}).get("0")
            if calls != self.last_tick or applied != [0]:
                full_coverage = False
                if "incomplete_or_invalid_physics_pose_coverage" not in invalid_reasons:
                    invalid_reasons.append("incomplete_or_invalid_physics_pose_coverage")
                invalid_reasons.append("incomplete_treatment_slot_command_coverage")
        crossings = _crossings(self.samples, self.openings) if full_coverage else []
        first = crossings[0] if crossings else None
        time_s = first["exit_time_s"] if first else (horizon_s if full_coverage else None)
        return {"schema": TRENCH_OUTCOME_SCHEMA,
            "factor_id": "opponent.trench_interception",
            "policy_id": self.policy_id,
            "opponent_policy_id": self.binding["policy_id"],
            "policy_config_sha256": self.binding["policy_config_sha256"],
            "schedule_sha256": self.binding["schedule_sha256"],
            "assignment_identity_sha256": self.binding["per_episode_assignment_sha256"],
            "protocol_id": self.binding["protocol_id"],
            "protocol_sha256": self.binding["protocol_sha256"],
            "seed_manifest_sha256": self.binding["seed_manifest_sha256"],
            "scenario_id": self.binding["scenario_id"],
            "category": self.binding["category"],
            "qualification_case": self.binding["qualification_case"],
            "split": self.binding["split"], "seed": self.seed,
            "condition": self.condition, "condition_id": self.condition_id,
            "pair_binding_id": self.binding["pair_binding_id"],
            "episode_id": self.episode_id,
            "source_identity_sha256": self.source_hash,
            "policy_checkpoint_sha256": self.checkpoint_hash,
            "checkpoint_bundle_sha256": self.binding["checkpoint_bundle_sha256"],
            "execution_device": self.device,
            "policy_id_for_row": self.policy_id,
            "configured_slots": ([0] if self.condition == "factor_treatment" else []),
            "applied_slots": applied,
            "contact_capture_complete": (full_coverage and
                self.last_tick == len(self.samples) - 1),
            "simulator_contact_distinct_substep_count": self.previous_count,
            "contact_events": self.events,
            "opening_geometry": self.openings,
            "initial_executable_route": route,
            "designated_initial_opening_id": designated,
            "truth_pose_samples": self.samples,
            "horizon": {"expected_physics_ticks": expected_ticks,
                "physics_dt_s": self.dt, "horizon_elapsed_s": horizon_s,
                "observed_physics_ticks": self.last_tick,
                "full_horizon": full_coverage,
                "invalid_reason": ("episode_ended_before_horizon" if not full_coverage else None)},
            "full_horizon_pose_coverage": full_coverage,
            "invalid_reasons": invalid_reasons,
            "crossing_events": crossings,
            "interception_episode": (bool(self.events) if
                self.condition == "factor_treatment" and full_coverage else None),
            "alternate_crossing_success": (any(event["opening_id"] != designated
                for event in crossings) if full_coverage and designated is not None else None),
            "trench_crossing_time_s": time_s if full_coverage else None,
            "trench_crossing_right_censored": (first is None if full_coverage else None),
            "qualification_eligible": False}


def reduce_trench_interception_pair(control: dict, treatment: dict, *, binding: dict,
        policy_id: str, seed: int, source_identity_sha256: str,
        checkpoint_sha256: str, execution_device: dict) -> dict:
    """Validate one exact policy/seed pair and derive crossing contrasts."""
    rows = {}
    for condition, record in (("no_factor", control), ("factor_treatment", treatment)):
        rows[condition] = validate_trench_interception_record(record, binding=binding,
            expected_condition=condition, expected_policy_id=policy_id,
            expected_seed=seed, expected_source_sha256=source_identity_sha256,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_device=execution_device)
    if control.get("pair_binding_id") != treatment.get("pair_binding_id"):
        raise ValueError("trench arms differ in stable pair binding")
    if (not rows["no_factor"]["crossing_row_valid"] or
            not rows["factor_treatment"]["crossing_row_valid"] or
            not rows["factor_treatment"]["interception_row_valid"]):
        raise ValueError("invalid trench rows cannot satisfy paired outcome gate")
    return {"policy_id": policy_id, "seed": seed,
        "control_episode_id": control["episode_id"],
        "treatment_episode_id": treatment["episode_id"],
        "pair_binding_id": control["pair_binding_id"],
        "control_condition_id": control.get("condition_id"),
        "treatment_condition_id": treatment.get("condition_id"),
        "control_alternate_crossing_success": rows["no_factor"][
            "alternate_crossing_success"],
        "treatment_alternate_crossing_success": rows["factor_treatment"][
            "alternate_crossing_success"],
        "alternate_crossing_success_delta": int(rows["factor_treatment"][
            "alternate_crossing_success"]) - int(rows["no_factor"][
            "alternate_crossing_success"]),
        "control_trench_crossing_time_s": rows["no_factor"]["trench_crossing_time_s"],
        "treatment_trench_crossing_time_s": rows["factor_treatment"][
            "trench_crossing_time_s"],
        "trench_crossing_time_delta_s": rows["factor_treatment"][
            "trench_crossing_time_s"] - rows["no_factor"]["trench_crossing_time_s"],
        "treatment_interception_episode": rows["factor_treatment"][
            "interception_episode_numerator"] == 1,
        "interception_control_denominator": None,
        "qualification_eligible": False}
