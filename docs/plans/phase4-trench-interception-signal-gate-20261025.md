# Phase 4 trench-interception signal gate

## Goal and scope

Plan one implementation tranche for `opponent.trench_interception`: bind one
exact held-out validation cell to the existing sensed-state `trench_interceptor`
schedule, capture simulator-grounded interception and trench-crossing outcomes,
and add a fail-closed replay/dashboard review gate. This is a candidate cell
only. It must not make the Phase 4 requirement executable or qualified until
the registered requirement's full 14-category and design-axis coverage has
authoritative evidence. The direct-pursuit runtime and outcome path remains a
separate factor and must not be reused or relabeled as trench evidence.

Use the first active, reactive trench case already present in the frozen
validation protocol:

| Axis | Frozen value |
| --- | --- |
| Scenario | `offense_vs_scripted_pressure` |
| Category | `active/trench/level_4_reactive` |
| Layout | `2026_rebuilt_shifted_structures_v1` |
| Defender family | `orbiting_interceptor` |
| Start condition | `offset_upper_lane` |
| Sensor profile | `noisy` |
| Split | `validation` |
| Seeds | the exact 100 protocol seeds, bound by their digest |
| Policy | `trench_interceptor` |
| Factor | `opponent.trench_interception` |
| Assigned pressure slot | slot 0 only |

The control arm runs the registered level-4 reactive category script in slot 0;
the treatment replaces exactly slot 0 with `trench_interceptor`. There is no
other configured pressure slot for level 4. Both arms use the same condition
case, seed manifest, checkpoints, source identity, and device binding. Preserve
the policy's existing controller-visible sensed-state input contract; evaluator
truth state may be used only by the outcome recorder.

The paired manifest must cover the eleven existing policy/control IDs
`A1_P0`, `A2_P0`, `A3_P0`, `A1`, `A2`, `A3`, `C1`, `C2`, `D0`, `D1`, and
`D2`, each on all 100 seeds in both arms: 1,100 policy/seed rows per arm and
2,200 total. Require exact key coverage and one-to-one control/treatment pairs;
do not treat the eight-policy generic matrix as sufficient.

## Current code evidence and gap

- `meco_phase4_opponents.py` already freezes the 14-category trench domain,
  maps level 0 to no pressure slot, levels 1–4 to slot 0, and level 5 to slots
  0/1, and validates the protocol/case/seed/policy schedule.
- `curriculum_scenarios.py` accepts that exact schedule and applies the
  `trench_interceptor`; evaluator records expose schedule identity and runtime
  applied slots. This is assignment evidence only.
- `meco_phase4_runtime_adapter.py` still marks trench stimulus unsupported and
  has no authoritative sources for `interception_rate`,
  `alternate_crossing_success_rate`, or `trench_crossing_time_s`.
- The campaign planner's direct-pursuit review mechanism is factor-specific.
  Do not reuse its outcome schema, metric names, review artifacts, or gate as
  trench evidence. Add trench-specific assignment and evidence identities.
- `field.py` defines the trench collision geometry and opening; the evaluator
  can observe focal slot-3 truth pose and physics time independently of the
  opponent policy's sensed inputs. Verify those sources and their timestamp
  semantics before implementation claims them authoritative.

The runtime status document is a registry-wide status report. Preserve its
semantics: a planned cell or successful command assignment is not an executed
or qualified episode; `stimulus_supported` and `authoritative_signals` may be
promoted only after exact assignment and record validators exist. Even after
the candidate replay gate passes, this single cell does not close the
14-category requirement.

## Required outcome contract

Bind every outcome record to episode ID, exact protocol and case identity,
factor, policy ID/config digest, assignment and schedule digests, configured
and actually applied slots, seed, paired condition, source identity,
checkpoint identity, device, and physics clock. Reconcile aggregates from the
event/interval rows; reject missing, duplicated, mismatched, non-finite, or
proxy-derived values.

1. **Interception rate.** Define an interception episode as at least one
   simulator-confirmed collision-solver contact between focal slot 3 and the
   assigned trench-interceptor slot 0, with participant IDs and physics
   substep/timestamp provenance. Count episodes with any such event over all
   fully observed treatment episodes. Retain pair-contact events, exposure
   bounds, and denominator. Do not use distance-to-target or an aggregate
   contact metric. If the collision system cannot provide pair identity and
   time, keep the signal unavailable and scope the smallest simulator record
   extension first.
2. **Alternate crossing success.** At reset, freeze the initial executable
   route's designated trench opening from the actual selected route and the
   pinned field geometry. From simulator-truth slot-3 pose samples, detect
   complete crossing of the trench obstacle band through a different
   collision-free opening, with continuous segment-versus-opening checks and
   physics timestamps. The numerator is episodes completing that alternate
   crossing by the fixed scenario horizon; the denominator is all episodes
   with valid full-horizon pose coverage. A normal full-horizon episode with
   no crossing is a failure in this fixed-window estimand. Invalid clocks,
   missing pose frames, environment reset, or source/goal/geometry changes
   censor the episode and are retained with explicit reason; report censor
   counts and never silently remove them. Do not equate an alternate planned
   route or a route-goal endpoint with a physical crossing.
3. **Trench crossing time.** Record continuous-time interpolated entry and
   exit across the trench obstacle band from truth pose and pinned geometry;
   crossing time is exit minus entry for a complete physical crossing. For
   non-crossers observed through the fixed horizon, retain a right-censored
   interval at the horizon and report restricted mean crossing time through
   that horizon with its event/censor counts. Invalid observation/clock or
   reset censors the row as invalid, separately from an ordinary horizon
   censor. Do not report a completion-only mean as the population crossing
   time.

### Denominators, pair validity, and promotion gate

The assignment denominator is exactly 1,100 policy/seed episodes per arm
(eleven policy IDs times 100 seeds). For each arm and metric, publish the
assigned count (1,100), full-horizon-valid count, invalid/censored count by
reason, and metric denominator. Interception incidence and alternate crossing
success use the full-horizon-valid count in that arm; a fully observed
non-crosser remains in the alternate-crossing denominator as a zero. RMST uses
the full-horizon-valid count in that arm. Report these per-arm denominators
even when they are equal.

Define a valid policy/seed pair as the same exact policy ID and seed present
once in each arm, with both rows matching the frozen case, schedule/source,
checkpoint, device and slot bindings, and both records carrying continuous
finite full-horizon physics-clock and focal-pose coverage plus complete contact
event coverage. A normal episode reaching the registered horizon without
crossing is valid (crossing success zero; crossing time censored at the
horizon). A missing/duplicate row, invalid or discontinuous clock, missing
pose/contact coverage, early reset/termination, or provenance mismatch makes
that arm row invalid and its policy/seed pair invalid. Keep invalid rows and
reasons in the artifacts; never drop them and silently shrink the paired
denominator. Per-arm metric rates may be described over their full-horizon-
valid rows, but paired deltas include only exact valid pairs and must report
their count.

The minimum promotion validity is all 1,100 valid pairs for every required
outcome and exactly 1,100 valid rows in each arm, with zero invalid pairs and
zero unaccounted assignments. Any invalid row or pair fails the candidate
executability/review gate; it does not get repaired by excluding the seed,
substituting another seed, imputing an outcome, or weakening the denominator.
The campaign may retain diagnostic aggregates with their explicit smaller
descriptive denominator, but such aggregates cannot satisfy the gate. This is
a validity rule, not a policy-performance threshold: valid non-crossings and
valid no-contact episodes are outcomes, not invalid data.

For each full-horizon-valid episode, define the time-to-crossing duration from
scenario reset as `d_i = t_exit_i - t_reset_i` when a complete physical
crossing occurs by horizon `H`; otherwise set `d_i = H` and mark that episode
right-censored at `H`. The fixed-horizon restricted mean is the explicit
arithmetic `RMST(H) = (sum(d_i for crossing events) + sum(H for horizon-censored
episodes)) / N_full_horizon_valid`. Thus every valid episode contributes once;
completion-only averaging is prohibited. Invalid rows are outside
`N_full_horizon_valid`, are reported separately, and force the promotion gate
to fail. Publish arm-specific RMST/event/censor counts and same-policy,
same-seed paired differences only across the 1,100 valid pairs.

Reconcile each aggregate to its row-level outcomes. Keep physical interception
incidence separate from number of contact substeps. A missed horizon crossing
is a valid fixed-window zero and contributes `H` as a time censor; it is not
evidence that the robot never could cross.

## Implementation tranche

1. Add a trench-specific outcome recorder, schedule/binding contract, and
   evaluator hooks. Record the reset-bound selected opening, applied slot
   commands, truth-pose samples and solver-confirmed slot-3/slot-0 contacts.
   Keep the treatment/control assignment exact and validation-only.
2. Add a trench-specific matrix pairing/coverage validator for the 11-by-100
   policy/seed assignments, row-level censoring, contact reconciliation,
   crossing geometry, fixed-window success, and restricted-mean-time
   arithmetic. Reject incomplete pairs or incorrect source/config/slot
   provenance.
3. Wire `opponent.trench_interception` to its own assignment and outcome
   binding in the runtime adapter and campaign planner. The candidate cell may
   become executable only when exact schedule application, all three
   authoritative outcome validators, exact pair coverage, and a review bundle
   are accepted. Keep `qualification_eligible=false` and the registry-wide
   requirement incomplete. Add a separate tranche-level review status so one
   case cannot imply full 14-category coverage.
4. Produce a provenance bundle containing plan/source/registry/protocol
   digests, exact case and seed-assignment digest, policy/config and schedule
   digests, all 11 policy IDs, 100 seeds per arm, applied slot evidence,
   matched outcomes with censoring, and the reviewable replay pair. Replays
   must represent the same seed under both control and treatment. Bind replay
   IDs and hashes, backend source identity, checkpoint bundle, device,
   frontend asset manifest hash, and reviewer identity.

## Required validation and gates

- Contract checks must exercise exact case and slot binding; validation-only
  schedule enforcement; sensed-state-only policy inputs; all eleven policies
  times 100 seeds in both arms; contact pair identity and clock; physical
  trench-plane crossing through legal opening geometry; alternate-vs-initial
  route classification; full-horizon non-crossing and invalid-record
  censoring; restricted-mean calculation; aggregate/event reconciliation; and
  rejection of null, proxy, duplicate, or mismatched outcome rows.
- Run a representative paired end-to-end simulator evaluation using one
  reproducible seed from the exact manifest, inspect the simulator records and
  both fresh replays, and report scenario, seed, policy/config, assigned and
  applied slot, physical contact result, crossing result, and censor status.
- Before generating or reviewing replay, restart the dashboard services on
  the current checkout. Verify relevant services are active, backend endpoints
  respond, served frontend asset hashes match checkout, and fresh replay
  provenance binds the current source. Mark replays from earlier code as
  outdated.
- Do not promote the cell until the evidence bundle is independently
  reviewed. Do not mark the whole trench requirement executable or qualified
  until all 14 frozen categories and registered axes have complete validated
  evidence and the campaign's remaining qualification gates pass.

No simulator, replay, GPU, training, or dashboard action is part of this plan
pass. This plan makes no claim that the existing schedule has been applied in
simulation, that any signal has been measured, or that the policy or cell is
qualified.
