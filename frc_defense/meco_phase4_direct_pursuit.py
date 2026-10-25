"""Simulator-grounded, episode-paired direct-pursuit evidence."""
from __future__ import annotations

import hashlib
import json
import math


DIRECT_PURSUIT_OUTCOME_SCHEMA = "meco-phase4-direct-pursuit-outcome-v1"
DIRECT_PURSUIT_TIME_LOSS_SCHEMA = "meco-phase4-direct-pursuit-time-loss-v1"
DIRECT_PURSUIT_PROGRESS_SAMPLE_TICKS = 12
DIRECT_PURSUIT_PROGRESS_THRESHOLD_FRACTION = 0.5


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


class DirectPursuitEpisodeCapture:
    """Capture exact pursuit-pair SAT events and focal progress at physics time."""

    def __init__(self, env, binding: dict, *, condition: str, episode_id: str,
            source_identity_sha256: str, checkpoint_sha256: str,
            scenario_id: str, category: str, qualification_case: dict,
            seed: int, split: str, policy_id: str, execution_device: dict,
            condition_id: str | None):
        if (not isinstance(binding, dict) or binding.get("policy_id") != "direct_pursuit"
                or binding.get("factor_id") != "opponent.direct_pursuit"
                or binding.get("split") != "validation"
                or split != "validation" or condition not in ("no_factor", "factor_treatment")):
            raise ValueError("direct-pursuit capture requires the exact validation schedule binding")
        slots = binding.get("configured_pressure_slots", {}).get(category)
        registered_slots = binding.get("registered_pressure_slots", {}).get(category)
        preserved_slots = binding.get("preserved_scripted_slots", {}).get(category)
        if (not isinstance(slots, list) or len(slots) != 1 or slots != sorted(set(slots))
                or any(type(slot) is not int or slot not in (0, 1, 2, 4, 5)
                       for slot in slots)
                or not isinstance(registered_slots, list)
                or registered_slots != sorted(set(registered_slots))
                or not isinstance(preserved_slots, list)
                or preserved_slots != sorted(set(preserved_slots))
                or set(registered_slots) != set(slots) | set(preserved_slots)
                or set(slots) & set(preserved_slots)):
            raise ValueError("direct-pursuit binding has invalid pressure slot identities")
        if (binding.get("scenario_id") != scenario_id
                or category not in binding.get("categories", ())
                or qualification_case != binding.get("qualification_case")
                or seed not in binding.get("seed_list", ())
                or policy_id not in (*("A1", "A2", "A3", "C1", "C2"),
                                     "A1_P0", "A2_P0", "A3_P0", "D0", "D1", "D2")
                or not all(isinstance(value, str) and value for value in (
                    episode_id, source_identity_sha256))
                or (checkpoint_sha256 is not None and
                    (not isinstance(checkpoint_sha256, str) or not checkpoint_sha256))):
            raise ValueError("direct-pursuit episode does not match its bound case/seed/source")
        self.env = env
        self.binding = binding
        self.condition = condition
        self.episode_id = episode_id
        self.source_hash = source_identity_sha256
        self.checkpoint_hash = checkpoint_sha256
        self.checkpoint_bundle = binding.get("checkpoint_bundle_sha256")
        if (not isinstance(self.checkpoint_bundle, dict) or
                not self.checkpoint_bundle or any(not isinstance(value, str) or not value
                    for value in self.checkpoint_bundle.values())):
            raise ValueError("direct-pursuit binding lacks exact checkpoint bundle identity")
        self.pair_binding_id = binding.get("pair_binding_id")
        if not isinstance(self.pair_binding_id, str) or not self.pair_binding_id:
            raise ValueError("direct-pursuit binding lacks stable paired-experiment identity")
        pair_binding = binding.get("pair_binding")
        if (not isinstance(pair_binding, dict)
                or _sha(pair_binding) != self.pair_binding_id
                or pair_binding.get("protocol_id") != binding.get("protocol_id")
                or pair_binding.get("protocol_sha256") != binding.get("protocol_sha256")
                or pair_binding.get("scenario_id") != scenario_id
                or pair_binding.get("categories") != [category]
                or pair_binding.get("qualification_case") != qualification_case
                or pair_binding.get("seed_list") != binding.get("seed_list")
                or pair_binding.get("schedule_sha256") != binding.get("schedule_sha256")
                or pair_binding.get("checkpoint_bundle_sha256") != self.checkpoint_bundle
                or pair_binding.get("source_identity_sha256") != source_identity_sha256
                or pair_binding.get("execution_device") != execution_device
                or pair_binding.get("conditions") != ["no_factor", "factor_treatment"]):
            raise ValueError("direct-pursuit paired identity differs from case/seed/source/checkpoint/device")
        self.scenario_id = scenario_id
        self.category = category
        self.qualification_case = dict(qualification_case)
        self.seed = seed
        self.split = split
        self.policy_id = policy_id
        self.device = execution_device
        self.condition_id = condition_id
        self.slots = tuple(slots)
        self.sim = env.sim
        if (getattr(self.sim, "num_robots", None) != 6 or
                not all(hasattr(self.sim, name) for name in (
                    "_collision_pair_i", "_collision_pair_j", "pair_contact_event_count",
                    "pair_contact_first_substep", "pair_contact_first_time_s",
                    "pair_contact_last_substep", "pair_contact_last_time_s",
                    "pair_contact_solver_report_count", "pair_contact_impulse_sum",
                    "physics_substep_index", "physics_time_s"))):
            raise ValueError("simulator lacks six-robot pair-contact identity/timing fields")
        pair_i = [int(value) for value in self.sim._collision_pair_i.tolist()]
        pair_j = [int(value) for value in self.sim._collision_pair_j.tolist()]
        self.pair_indices = {}
        for slot in self.slots:
            pair = tuple(sorted((3, slot)))
            matches = [index for index, value in enumerate(zip(pair_i, pair_j))
                       if value == pair]
            if len(matches) != 1:
                raise ValueError("simulator pair table lacks the exact focal/pursuer slot pair")
            self.pair_indices[slot] = matches[0]
        self.previous_counts = {slot: int(self.sim.pair_contact_event_count[0, index].item())
                                for slot, index in self.pair_indices.items()}
        if any(self.previous_counts.values()):
            raise ValueError("direct-pursuit capture must start from a clean simulator episode")
        field_length = float(self.sim.field_length)
        field_width = float(self.sim.field_width)
        self.objective_xy = [field_length, field_width * 0.5]
        self.start_xy = [float(value) for value in self.sim.pose[0, 3, :2].tolist()]
        dx = self.objective_xy[0] - self.start_xy[0]
        dy = self.objective_xy[1] - self.start_xy[1]
        self.route_length_m = math.hypot(dx, dy)
        if not math.isfinite(self.route_length_m) or self.route_length_m <= 1e-6:
            raise ValueError("direct-pursuit objective segment has no measurable length")
        self.direction = (dx / self.route_length_m, dy / self.route_length_m)
        self.objective_id = "offensive_field_end_center_v1"
        self.route_id = _sha({"schema": "meco-fixed-objective-segment-v1",
            "scenario_id": scenario_id, "category": category, "seed": seed,
            "start_xy_m": self.start_xy, "objective_id": self.objective_id,
            "objective_xy_m": self.objective_xy})
        self.last_tick = int(self.sim.physics_substep_index[0].item())
        self.last_time_s = float(self.sim.physics_time_s[0].item())
        if self.last_tick != 0 or self.last_time_s != 0.:
            raise ValueError("direct-pursuit capture requires reset simulator time")
        self.progress_samples = []
        self.contact_events = []
        self.threshold_crossing_sampled = False
        self._append_progress_sample()

    def _append_progress_sample(self, *, force=False):
        tick = int(self.sim.physics_substep_index[0].item())
        time_s = float(self.sim.physics_time_s[0].item())
        pose = [float(value) for value in self.sim.pose[0, 3, :2].tolist()]
        progress = ((pose[0] - self.start_xy[0]) * self.direction[0] +
                    (pose[1] - self.start_xy[1]) * self.direction[1])
        if not all(math.isfinite(value) for value in (time_s, *pose, progress)):
            raise ValueError("direct-pursuit progress sample is non-finite")
        sample = {"physics_tick": tick, "physics_time_s": time_s,
            "focal_xy_m": pose, "progress_along_objective_m": progress}
        if self.progress_samples and self.progress_samples[-1]["physics_tick"] == tick:
            self.progress_samples[-1] = sample
        elif force or tick % DIRECT_PURSUIT_PROGRESS_SAMPLE_TICKS == 0:
            self.progress_samples.append(sample)

    def observe_physics_step(self):
        tick = int(self.sim.physics_substep_index[0].item())
        time_s = float(self.sim.physics_time_s[0].item())
        if tick != self.last_tick + 1 or not math.isfinite(time_s) or time_s <= self.last_time_s:
            raise RuntimeError("direct-pursuit event sampler missed or reordered a physics step")
        for slot, index in self.pair_indices.items():
            count = int(self.sim.pair_contact_event_count[0, index].item())
            delta = count - self.previous_counts[slot]
            if delta < 0 or delta > 1:
                raise RuntimeError("pair-contact event count changed outside one sampled physics tick")
            if delta:
                event_tick = int(self.sim.pair_contact_last_substep[0, index].item())
                event_time = float(self.sim.pair_contact_last_time_s[0, index].item())
                if event_tick != tick or not math.isclose(event_time, time_s,
                        rel_tol=0., abs_tol=max(1e-9, float(self.sim.dt) * 1e-6)):
                    raise RuntimeError("simulator pair event identity/timing differs from observed tick")
                self.contact_events.append({"event_index": len(self.contact_events),
                    "source": "simulator_robot_pair_sat_contact_solver",
                    "robot_slot_ids": [3, slot], "physics_tick": event_tick,
                    "physics_time_s": event_time,
                    "distinct_contact_substep_index": count})
            self.previous_counts[slot] = count
        self.last_tick, self.last_time_s = tick, time_s
        if not self.threshold_crossing_sampled:
            pose = [float(value) for value in self.sim.pose[0, 3, :2].tolist()]
            progress = ((pose[0] - self.start_xy[0]) * self.direction[0] +
                        (pose[1] - self.start_xy[1]) * self.direction[1])
            if progress >= self.route_length_m * DIRECT_PURSUIT_PROGRESS_THRESHOLD_FRACTION:
                self._append_progress_sample(force=True)
                self.threshold_crossing_sampled = True
        if tick % DIRECT_PURSUIT_PROGRESS_SAMPLE_TICKS == 0:
            self._append_progress_sample()

    def finish(self, *, episode_censored: bool = False) -> dict:
        self._append_progress_sample(force=True)
        total_ticks = int(self.sim.physics_substep_index[0].item())
        elapsed_s = float(self.sim.physics_time_s[0].item())
        if total_ticks <= 0 or not math.isfinite(elapsed_s) or elapsed_s <= 0.:
            raise ValueError("direct-pursuit episode has no simulator exposure")
        runtime = getattr(self.env, "phase4_opponent_runtime", None)
        if self.condition == "factor_treatment":
            if not isinstance(runtime, dict):
                raise ValueError("direct-pursuit treatment lacks runtime command capture")
            configured = runtime.get("configured_slots")
            applied = runtime.get("applied_slots")
            calls_by_slot = runtime.get("applied_command_calls_by_slot")
            if (configured != list(self.slots) or applied != list(self.slots)
                    or not isinstance(calls_by_slot, dict)
                    or any(calls_by_slot.get(str(slot)) != total_ticks for slot in self.slots)):
                raise ValueError("direct-pursuit treatment command slots/calls do not cover exposure")
        else:
            applied, calls_by_slot = [], {str(slot): 0 for slot in self.slots}
            if runtime is not None:
                raise ValueError("no-factor direct-pursuit baseline unexpectedly applied a policy")
        pairs = []
        total_events = 0
        for slot, index in self.pair_indices.items():
            count = int(self.sim.pair_contact_event_count[0, index].item())
            reports = int(self.sim.pair_contact_solver_report_count[0, index].item())
            first_tick = int(self.sim.pair_contact_first_substep[0, index].item())
            last_tick = int(self.sim.pair_contact_last_substep[0, index].item())
            first_time = float(self.sim.pair_contact_first_time_s[0, index].item())
            last_time = float(self.sim.pair_contact_last_time_s[0, index].item())
            impulses = [float(value) for value in self.sim.pair_contact_impulse_sum[
                0, index].tolist()]
            if (count != self.previous_counts[slot] or
                    count != sum(event["robot_slot_ids"] == [3, slot]
                                 for event in self.contact_events)):
                raise ValueError("direct-pursuit event stream does not reconcile to simulator counters")
            total_events += count
            pairs.append({"robot_slot_ids": [3, slot],
                "distinct_contact_substep_count": count,
                "contact_solver_report_count": reports,
                "first_contact_physics_tick": first_tick if count else None,
                "first_contact_physics_time_s": first_time if count else None,
                "last_contact_physics_tick": last_tick if count else None,
                "last_contact_physics_time_s": last_time if count else None,
                "accumulated_impulse_n_s": impulses})
        exposure_pair_seconds = elapsed_s * len(self.slots)
        contact_rate = total_events / exposure_pair_seconds
        any_contact = total_events > 0
        progress_intervals = [{
            "start_physics_tick": left["physics_tick"],
            "start_physics_time_s": left["physics_time_s"],
            "start_progress_m": left["progress_along_objective_m"],
            "end_physics_tick": right["physics_tick"],
            "end_physics_time_s": right["physics_time_s"],
            "end_progress_m": right["progress_along_objective_m"]}
            for left, right in zip(self.progress_samples, self.progress_samples[1:])]
        route_censored = (episode_censored or not any(
            sample["progress_along_objective_m"] >=
            self.route_length_m * DIRECT_PURSUIT_PROGRESS_THRESHOLD_FRACTION
            for sample in self.progress_samples))
        return {"schema": DIRECT_PURSUIT_OUTCOME_SCHEMA,
            "policy_id": self.binding["policy_id"],
            "factor_id": self.binding["factor_id"],
            "policy_config_sha256": self.binding["policy_config_sha256"],
            "schedule_sha256": self.binding["schedule_sha256"],
            "assignment_identity": self.binding["per_episode_assignment_sha256"],
            "condition": self.condition, "scenario_id": self.scenario_id,
            "pair_binding_id": self.pair_binding_id,
            "category": self.category, "qualification_case": self.qualification_case,
            "split": self.split, "seed": self.seed, "episode_id": self.episode_id,
            "condition_id": self.condition_id,
            "policy_checkpoint_sha256": self.checkpoint_hash,
            "checkpoint_bundle_sha256": self.checkpoint_bundle,
            "source_identity_sha256": self.source_hash,
            "execution_device": self.device,
            "configured_slots": list(self.slots), "applied_slots": applied,
            "applied_command_calls_by_slot": calls_by_slot,
            "focal_robot_slot_id": 3, "pursuit_robot_slot_ids": list(self.slots),
            "contact_event_semantics": "one event per simulator-confirmed SAT-contact substep for the exact focal/pursuer pair",
            "contact_pairs": pairs, "contact_events": self.contact_events,
            "robot_contact_rate": contact_rate,
            "robot_contact_rate_numerator_distinct_substeps": total_events,
            "robot_contact_rate_denominator_pair_seconds": exposure_pair_seconds,
            "intercepted": any_contact,
            "interception_event_criterion": ("episode-level physical encounter: at least one simulator-confirmed SAT contact between focal slot 3 and an assigned pursuit slot; this binary incidence is distinct from contact frequency, which counts every distinct pair-contact substep"),
            "interception_episode_numerator": int(any_contact),
            "interception_eligible_episode_denominator": 1,
            "interception_events": [event for event in self.contact_events],
            "physics_exposure": {"active_physics_ticks": total_ticks,
                "active_physics_seconds": elapsed_s,
                "first_physics_tick": 0, "last_physics_tick": total_ticks,
                "first_physics_time_s": 0.0, "last_physics_time_s": elapsed_s,
                "censored": bool(episode_censored)},
            "objective_route": {"schema": "meco-fixed-objective-segment-v1",
                "route_id": self.route_id, "objective_id": self.objective_id,
                "start_xy_m": self.start_xy, "objective_xy_m": self.objective_xy,
                "route_length_m": self.route_length_m,
                "progress_measure": "signed focal displacement projected onto fixed start-to-objective segment; raw, unclamped",
                "progress_sample_interval_ticks": DIRECT_PURSUIT_PROGRESS_SAMPLE_TICKS,
                "progress_samples": self.progress_samples,
                "progress_intervals": progress_intervals,
                "threshold_fraction_for_paired_time_loss": DIRECT_PURSUIT_PROGRESS_THRESHOLD_FRACTION,
                "censored": bool(route_censored)},
            "time_lost_to_opponents_s": None,
            "time_loss_status": "requires_exact_same_seed_registered_script_pair",
            "time_loss_reference_semantics": ("factor-treatment crossing time minus same-seed crossing time with the registered category-script opponent in the assigned pressure slot"),
            "qualification_eligible": False}


def _time_at_route_fraction(record: dict) -> tuple[float | None, bool]:
    route = record.get("objective_route")
    if not isinstance(route, dict):
        return None, True
    length = route.get("route_length_m")
    fraction = route.get("threshold_fraction_for_paired_time_loss")
    intervals = route.get("progress_intervals")
    if (not isinstance(length, (float, int)) or length <= 0 or
            fraction != DIRECT_PURSUIT_PROGRESS_THRESHOLD_FRACTION or
            not isinstance(intervals, list) or not intervals):
        return None, True
    threshold = float(length) * float(fraction)
    for interval in intervals:
        start, end = interval.get("start_progress_m"), interval.get("end_progress_m")
        start_t, end_t = interval.get("start_physics_time_s"), interval.get("end_physics_time_s")
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in (start, end, start_t, end_t)) or end_t <= start_t:
            return None, True
        if start >= threshold:
            return float(start_t), False
        if start < threshold <= end:
            ratio = (threshold - start) / (end - start)
            return float(start_t + ratio * (end_t - start_t)), False
    return None, True


def paired_direct_pursuit_outcomes(control: dict, treatment: dict) -> dict:
    """Reconcile one exact control/treatment episode pair and derive outcomes."""
    identity_keys = ("scenario_id", "category", "qualification_case", "split",
        "seed", "policy_id", "factor_id", "policy_config_sha256",
            "assignment_identity", "source_identity_sha256", "checkpoint_bundle_sha256",
        "pair_binding_id", "execution_device",
        "policy_checkpoint_sha256", "configured_slots", "focal_robot_slot_id",
        "pursuit_robot_slot_ids")
    if (not isinstance(control, dict) or not isinstance(treatment, dict) or
            control.get("schema") != DIRECT_PURSUIT_OUTCOME_SCHEMA or
            treatment.get("schema") != DIRECT_PURSUIT_OUTCOME_SCHEMA or
            any(control.get(key) != treatment.get(key) for key in identity_keys) or
            control.get("condition") != "no_factor" or
            treatment.get("condition") != "factor_treatment" or
            not isinstance(control.get("episode_id"), str) or
            not isinstance(treatment.get("episode_id"), str) or
            control.get("episode_id") == treatment.get("episode_id") or
            not isinstance(control.get("condition_id"), str) or
            not isinstance(treatment.get("condition_id"), str) or
            control.get("applied_slots") != [] or
            treatment.get("applied_slots") != treatment.get("configured_slots") or
            control.get("objective_route", {}).get("route_id") !=
                treatment.get("objective_route", {}).get("route_id")):
        raise ValueError("direct-pursuit paired records differ in exact episode/provenance binding")
    slots = control["configured_slots"]
    for record in (control, treatment):
        exposure = record.get("physics_exposure", {})
        if (exposure.get("censored") is not False or
                record.get("objective_route", {}).get("censored") is not False or
                type(exposure.get("active_physics_ticks")) is not int or
                exposure["active_physics_ticks"] <= 0 or
                not math.isfinite(float(exposure.get("active_physics_seconds", math.nan))) or
                len(record.get("contact_pairs", ())) != len(slots)):
            raise ValueError("direct-pursuit pair has missing or censored simulator exposure")
        contacts = record.get("contact_events")
        if not isinstance(contacts, list):
            raise ValueError("direct-pursuit pair has no event-level contact stream")
        for event in contacts:
            if (event.get("source") != "simulator_robot_pair_sat_contact_solver" or
                    event.get("robot_slot_ids", [None])[0] != 3 or
                    event.get("robot_slot_ids", [None, None])[1] not in slots or
                    type(event.get("physics_tick")) is not int or
                    not math.isfinite(float(event.get("physics_time_s", math.nan)))):
                raise ValueError("direct-pursuit contact event lacks source, pair, or time identity")
        pair_event_counts = {tuple(pair["robot_slot_ids"]): 0
                             for pair in record["contact_pairs"]}
        previous_event_tick = 0
        for event in contacts:
            pair = tuple(event["robot_slot_ids"])
            if (pair not in pair_event_counts or event["physics_tick"] < previous_event_tick
                    or event["physics_tick"] > exposure["active_physics_ticks"]
                    or event["physics_time_s"] > exposure["active_physics_seconds"]):
                raise ValueError("direct-pursuit contact event falls outside ordered simulator exposure")
            pair_event_counts[pair] += 1
            previous_event_tick = event["physics_tick"]
        if any(pair_event_counts[tuple(pair["robot_slot_ids"])] !=
               pair["distinct_contact_substep_count"] for pair in record["contact_pairs"]):
            raise ValueError("event-level direct-pursuit contacts do not reconcile to pair counters")
        numerator = sum(pair["distinct_contact_substep_count"]
                        for pair in record["contact_pairs"])
        denominator = (exposure["active_physics_seconds"] * len(slots))
        observed_rate = numerator / denominator
        if (record.get("robot_contact_rate_numerator_distinct_substeps") != numerator or
                not math.isclose(record.get("robot_contact_rate", math.nan), observed_rate,
                    rel_tol=1e-12, abs_tol=1e-12) or
                not math.isclose(record.get("robot_contact_rate_denominator_pair_seconds", math.nan),
                    denominator, rel_tol=1e-12, abs_tol=1e-12) or
                record.get("intercepted") is not (numerator > 0)):
            raise ValueError("direct-pursuit contact/interception aggregates do not reconcile to events")
        if (record.get("interception_episode_numerator") != int(numerator > 0) or
                record.get("interception_eligible_episode_denominator") != 1):
            raise ValueError("episode-level interception incidence does not reconcile to event criterion")
    control_time, control_censored = _time_at_route_fraction(control)
    treatment_time, treatment_censored = _time_at_route_fraction(treatment)
    if control_censored or treatment_censored or control_time is None or treatment_time is None:
        raise ValueError("direct-pursuit paired progress is censored before the frozen objective threshold")
    return {"schema": DIRECT_PURSUIT_TIME_LOSS_SCHEMA,
        "scenario_id": treatment["scenario_id"], "category": treatment["category"],
        "qualification_case": treatment["qualification_case"],
        "split": treatment["split"], "seed": treatment["seed"],
        "policy_id": treatment["policy_id"], "factor_id": treatment["factor_id"],
        "source_identity_sha256": treatment["source_identity_sha256"],
        "execution_device": treatment["execution_device"],
        "policy_checkpoint_sha256": treatment["policy_checkpoint_sha256"],
        "checkpoint_bundle_sha256": treatment["checkpoint_bundle_sha256"],
        "assignment_identity": treatment["assignment_identity"],
        "control_episode_id": control["episode_id"],
        "treatment_episode_id": treatment["episode_id"],
        "pair_binding_id": treatment["pair_binding_id"],
        "control_condition_id": control["condition_id"],
        "treatment_condition_id": treatment["condition_id"],
        "interception_episode_numerator": treatment["interception_episode_numerator"],
        "interception_eligible_episode_denominator":
            treatment["interception_eligible_episode_denominator"],
        "treatment_contact_event_numerator_distinct_substeps":
            treatment["robot_contact_rate_numerator_distinct_substeps"],
        "treatment_contact_event_denominator_pair_seconds":
            treatment["robot_contact_rate_denominator_pair_seconds"],
        "route_id": treatment["objective_route"]["route_id"],
        "threshold_fraction": DIRECT_PURSUIT_PROGRESS_THRESHOLD_FRACTION,
        "control_crossing_time_s": control_time,
        "treatment_crossing_time_s": treatment_time,
        "time_lost_to_opponents_s": treatment_time - control_time,
        "time_loss_reference_semantics": ("additional focal delay from direct pursuit relative to the registered category-script opponent in the same assigned pressure slot"),
        "progress_samples_interval_ticks": DIRECT_PURSUIT_PROGRESS_SAMPLE_TICKS,
        "control_progress_samples_sha256": _sha(
            control["objective_route"]["progress_samples"]),
        "treatment_progress_samples_sha256": _sha(
            treatment["objective_route"]["progress_samples"]),
        "control_censored": False, "treatment_censored": False}
