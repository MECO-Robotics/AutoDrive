# Phase 4 direct-pursuit executable gate

## Goal

Advance the `opponent.direct_pursuit` requirement from planned-only to a
candidate executable validation cell by binding one exact held-out paired
cell to the sensed-state pursuit policy and producing all three registry
outcomes from simulator-authoritative records. Keep qualification disabled
until the required end-to-end replay is inspected and the campaign compiler
accepts the evidence contract.

The plan is deliberately one requirement (`direct_pursuit`), one policy
(`direct_pursuit`), one pressure-category case at a time, all five tracks and
the existing D0/D1/D2 controls, and the frozen validation seeds. It does not
expand to trench interception, the full opponent pool, or final qualification.

## Current evidence and gap

Implementation now samples the simulator's pair-specific SAT contact counters
after every evaluator physics step. Each counter increment is emitted with its
slot pair, physics tick, and physics time; per-pair summaries are reconciled to
that event stream. The evaluator also records slot-3 displacement every physics
12 physics ticks along a frozen start-to-field-end-center segment. Threshold
crossing time is linearly interpolated between adjacent timestamped samples. The
paired time-loss
measure is the treatment minus no-factor crossing time at 50% of that same
segment. Here, the no-factor reference keeps the registered category-script
opponent in the assigned pressure slot; the result measures the additional
delay from replacing that slot with direct pursuit. Either arm failing to cross
is censored and rejected.

The frozen interception criterion is episode-level physical encounter: at least
one SAT contact between focal slot 3 and an assigned pursuer in a fully observed
episode. Its numerator is the number of such episodes and its denominator is
the number of uncensored eligible paired episodes. This incidence measure is
kept separate from contact frequency, which counts every distinct contact
substep over assigned pair-seconds. This operational criterion measures direct
physical encounter with a pursuer; it does not claim that contact alone proves
a score was prevented.

The code now captures these records and computes paired outcomes, but no
representative replay has been run or reviewed. The campaign planner therefore
continues to report every direct-pursuit cell as non-executable until the
required held-out simulator and current-dashboard evidence is accepted. The
optional `--direct-pursuit-review-evidence` compiler input uses
`meco-phase4-direct-pursuit-review-evidence-bundle-v1`; each row must bind the
exact protocol/case/seed and assignment digests, checkpoint bundle, source,
device, all uncensored policy/seed pairs and outcome aggregates, plus a reviewed
fresh replay whose backend and frontend are verified current. Missing or
mismatched fields leave the cell non-executable. Qualification remains false
even when that candidate gate is satisfied.

The two matrix launches may have different arm-specific `condition_id` values.
They therefore share a stable `pair_binding_id` derived from the exact protocol,
case, seed list, schedule, checkpoint bundle, source, and execution device; each
outcome keeps its own `condition_id` and the reducer preserves both IDs.

For each case, the evidence row uses
`meco-phase4-direct-pursuit-review-evidence-v1` and carries the exact configured
pressure slots, schedule and seed-manifest digests, paired-outcome artifact
digest, and a `replay_review` with two replay records (`no_factor` and
`factor_treatment`) for one seed attested as present in that manifest. Each replay
record binds all eleven policy IDs, assignment/config digest, checkpoint bundle,
source, device, applied slots, replay ID and hash, and current frontend asset
manifest hash. The case-level review must also carry the current backend source
hash, healthy endpoint, reviewer identity, uncensored 11-by-100 policy/seed pair
coverage, and reconciled outcome denominators.

- `frc_defense/meco_phase4_opponents.py` already defines `direct_pursuit` and
  `opponent_velocity_command`; the command is based on `own_xy` and a
  validity-masked sensed target.
- `frc_defense/curriculum_scenarios.py::_phase4_opponent_commands` consumes
  `phase4_opponent_policy_id`, but assignment/runtime metadata is constructed
  only when `phase4_opponent_schedule` is supplied. `make_curriculum_env`
  currently requires a schedule only for `trench_interceptor`.
- `frc_defense/meco_validation_campaign_runner.py::run_phase4_opponent_cell`
  already dispatches a protocol-bound policy schedule through the paired
  matrix runner. The campaign planner currently creates these assignments for
  `reactive_defenders_and_opponent_pools`, not for the `direct_pursuit`
  requirement.
- `frc_defense/meco_track_training.py::evaluate_meco` validates an opponent
  schedule, passes it into the environment, and records
  `phase4_opponent_policy`; its opponent outcomes are intentionally null.
- `frc_defense/meco_phase4_runtime_adapter.py::PHASE4_REQUIREMENT_BINDINGS`
  lists no runtime event or authoritative signals for direct pursuit, and
  `frc_defense/meco_phase4_campaign_plan.py` therefore cannot mark those cells
  executable. The runtime status names three missing outcomes:
  `interception_rate`, `time_lost_to_opponents_s`, and `robot_contact_rate`.
- Existing `safety_robot_contact_rate` in `meco_track_training.py` counts
  focal slot-3 contact ticks with any robot. It is not automatically an
  authoritative direct-pursuer contact measure. Do not relabel it without
  proving pair identity and contact causality from simulator records.

## Scoped implementation

### 1. Bind the assignment to the exact held-out cell

Update `frc_defense/meco_phase4_campaign_plan.py` to emit a
`direct_pursuit` schedule assignment only for the registry's exact
`opponent.direct_pursuit` selector domain, and update
`frc_defense/meco_validation_campaign_runner.py::run_phase4_opponent_cell` to
accept that requirement while retaining the existing validation-only
protocol, case, seed, checkpoint, source, device, and pair checks. In
`frc_defense/meco_phase4_opponents.py`, keep the registered pressure-slot
mapping and policy config digest as the source of assignment identity.

Pass the exact schedule and seed through the existing matrix CLI,
`frc_defense/meco_track_training.py::evaluate_meco`, and
`frc_defense/curriculum_scenarios.py::make_curriculum_env`. Require the
schedule for `direct_pursuit`, enforce exact scenario/category/qualification
case/seed binding, and capture configured slots, slots that actually received
commands, and command-call counts. The policy must continue to consume only
the controller-visible, validity-masked sensed observation. No truth-state
fallback is allowed.

The no-factor pair must retain the registered category opponent. The
factor-treatment pair replaces only the designated pressure slot with direct
pursuit; all other participants and conditions stay fixed. Preserve exact
track/control/category/seed pairing and source/checkpoint/device provenance.

### 2. Add source-grounded outcomes

Trace the simulator's robot-pair contact and state-transition records from
`frc_defense/tensor_sim.py` and the collision solver through
`frc_defense/meco_track_training.py`. Add direct-pursuit records bound to the
episode ID, source identity, policy ID/config digest, assigned slot(s), and
paired condition:

- `robot_contact_rate`: count focal slot-3 contacts with the assigned pursuit
  slot from simulator-confirmed pair-contact summaries, divided by a declared
  exposure denominator. Retain numerator, denominator, pair IDs, time bounds,
  and censoring. Do not use an aggregate all-robot contact rate.
- `interception_rate`: define an interception as a simulator-observed pursuit
  contact that meets a frozen encounter criterion. Store each event with
  simulator tick/time and participant IDs; derive rate over eligible episodes
  using an explicit denominator. If the simulator exposes only accumulated
  substep summaries without enough event timing/identity, this signal remains
  unavailable and the gate fails closed.
- `time_lost_to_opponents_s`: compute from simulator-timestamped focal progress
  along a fixed, episode-bound objective/route under direct-pursuit exposure,
  using a predeclared no-opponent reference or same-seed paired baseline.
  Preserve raw progress intervals, elapsed physics time, route/objective
  identity, and censoring. Do not infer it from contact, selected-goal
  endpoint, or aggregate match time alone.

The implementer must first confirm that the current simulator exposes the
required event-level identity and timing. If it does not, scope the smallest
simulator-side record extension before attempting to mark any signal
authoritative. Never substitute a proxy under a registry-required name.

### 3. Promote only after all gates pass

Update `frc_defense/meco_phase4_runtime_adapter.py` with the exact assignment
event and signal sources only after the records above exist. Update the
campaign planner's executability rule so a cell is executable only when the
exact schedule applies and all three outcomes have authoritative sources.
Update the pair validator in `meco_validation_campaign_runner.py` to reject
missing, censored, mismatched, or incomplete outcome rows and to reconcile
aggregates against event records. Keep qualification eligibility false until
independent held-out analysis and all other promotion criteria pass.

## Required validation and evidence

1. Static/contract checks must cover exact requirement domain, slot and seed
   binding, sensed-state-only policy inputs, pair identity, denominator,
   censoring, aggregate-to-event reconciliation, and rejection of null or
   proxy outcomes. These checks do not establish simulator behavior.
2. Run one representative end-to-end simulator evaluation of the paired
   direct-pursuit case with a reproducible validation seed. Inspect both the
   simulator record and a fresh replay. Record scenario, category, seed,
   policy/config digest, assigned slot, actual applied command slots, and
   outcome records.
3. Before replay review, restart dashboard services against the current
   checkout; verify relevant services are active, their endpoints respond,
   current frontend asset hashes match the checkout, and replay provenance
   names the current source identity. Mark any older replay outdated.
4. Only after those checks pass may the campaign compiler report a
   direct-pursuit cell as executable. A generated schedule, static tests, or
   successful process exit is insufficient by itself.

## Dependencies and stop conditions

- Frozen Phase 4 registry, protocol, validation seed partition, and the
  existing paired five-track plus D0/D1/D2 matrix.
- Simulator pair-contact identity/timestamps and a defensible progress-based
  time-loss measurement. Missing simulator evidence blocks executability;
  it does not justify weakening the required signal definitions.
- Current dashboard/replay pipeline for fresh evidence and review.
- No GPU training lease, training process, or campaign-wide qualification run
  is required to implement this gate. Do not interrupt the live training run.

## Explicit nonclaims

This plan proves nothing about policy behavior, contact accuracy, measured
outcomes, simulator validity, qualification, promotion, deployment, or GPU
performance. Even if this direct-pursuit cell becomes executable and its
paired run succeeds, the other ten Phase 4 requirements and the rest of the
MECO evaluation plan remain open.
