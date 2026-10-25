# Phase 4 legal-team-start executable evidence gate

## Planner pass

Advance only the registry requirement `multiple_legal_team_start_states` by
promoting its already captured legal-start validation and per-start-state
outcome records into authoritative signals, then requiring exact held-out
paired evidence and a reviewed fresh replay before a cell becomes executable.
This is a code-and-evidence-contract tranche; it does not claim that any start
profile has been validated in a representative simulator run. Keep
qualification disabled.

The unit of work and evidence is **one registered profile as one candidate
cell**. Each candidate cell's `outcome_by_start_state` compares only the
scenario-defined no-factor start against that exact cell's assigned profile
(currently `team_start_set_a`, `team_start_set_b`, or `team_start_set_c`). It
must not aggregate the three profile cells or imply that one profile's evidence
covers another. All three registered profile cells need their own complete
evidence before the `multiple_legal_team_start_states` requirement can be
called complete. This tranche can open a candidate cell; it cannot complete
the whole requirement by itself.

This slice is deliberately separate from `opponent.trench_interception`
(slot-0 policy schedule, 100 seeds, 11 policies) and
`opponent.direct_pursuit`. It must not edit either opponent assignment,
signal, or evidence path. It also must not expand the start-profile requirement
to additional policies, profiles, scenarios, or seed partitions beyond the
immutable registry/protocol domain.

## Current evidence and gap

- `frc_defense/meco_phase4_context.py::apply_full_team_start_profile` applies
  the registered six-robot pose set after reset, zeros physical velocities,
  and checks the actual simulator pose tensor before the first observation.
  `full_team_start_profile_legality` checks finite poses, field bounds, robot
  envelope overlap, and static collision boxes. The runtime stores each
  world's Boolean legality and a rate; an illegal application raises.
- `frc_defense/meco_track_training.py::evaluate_meco` records the bound factor,
  profile ID/name/config digest, assignment digest, pre-observation application
  flag, world count, per-world legality, legal rate, and the per-episode
  `success`, `scores`, `useful_throughput`, and `completed_cycles` metrics.
- `frc_defense/meco_validation_campaign_runner.py::_validate_phase4_start_profile_pair`
  validates exact source/checkpoint/device and policy/category/seed coverage,
  no-factor versus treatment application, all-world legality, metric
  reconciliation, and paired outcomes by registered start state. The result
  remains `qualification_eligible: false`.
- `frc_defense/meco_phase4_campaign_plan.py` creates protocol-bound assignments
  but reports them planned-only. In
  `frc_defense/meco_phase4_runtime_adapter.py`, neither
  `legal_start_validation_rate` nor `outcome_by_start_state` is authoritative,
  so the compiler correctly keeps every such cell non-executable.
- No current source-bound held-out start-profile report or reviewed fresh
  replay is accepted. Runtime wiring and static contract coverage do not prove
  that the start poses behave as intended in simulation.

At planning time the user's r4 supervisor is live in the separate
`AutoDrive-meco-final-plan-integration` checkout (PID 2750253). Its current
status heartbeat is `2026-10-25T06:29:09Z`, cycle 2, one completed cycle;
A1 is running at update 125 on `cuda:0`, and A2/A3/C1/C2 have completed 200
updates. Preserve the run and both WX9100 leases. This tranche needs no GPU,
simulator, or training work.

## Bounded implementation slice

1. **Define the authoritative source contract per profile cell.** Update the
   start-state requirement binding only after auditing the existing producer
   and reducer. Name the captured event fields as the signal sources. Retain
   each row's exact profile ID, policy/category/seed key, `validated_pose_world_count`,
   and ordered per-world legality booleans (with stable world IDs or indices).
   Define the descriptive legal-start rate exactly as
   `sum(legal_world_booleans) / sum(validated_pose_world_count)` over the
   complete treatment rows for this one profile cell. Recompute it from rows;
   reject a caller-supplied value that disagrees. This ratio is a diagnostic
   aggregate, not the cell gate: candidate executability requires every
   expected world in every expected treatment row to be legal, so an average
   of 1.0 cannot conceal missing or illegal worlds.
2. **Seal outcome-by-start-state evidence for the single candidate cell.**
   Preserve the exact registered profile ID as the treatment stratum. In that
   cell's result, `outcome_by_start_state.states` must contain exactly
   `scenario_defined` (no-factor) and that assigned profile ID (treatment),
   with complete policy/category/seed coverage and per-policy metrics
   reconciled to the two source reports. Reconcile paired deltas against the
   exact same-policy/category/seed rows. Reject missing, duplicate, censored,
   non-finite, or mismatched rows. Do not combine profile strata in this
   artifact or infer a winning start state or qualification from it.
3. **Add an exact replay-review gate per profile assignment.** Consume one
   requirement-specific evidence row for one exact candidate cell, binding
   registry/protocol digests, exact validation case and seed-assignment digest,
   that profile's assignment/config digests, both condition reports and their
   exact policy/checkpoint/source/device bindings, per-cell outcome-artifact
   digest, and one paired replay seed present in the frozen validation
   manifest. The two replay records must bind the scenario/category/case,
   assigned profile, condition, current source, checkpoint bundle, device,
   replay IDs and hashes. Require reviewer acceptance plus current dashboard
   backend, endpoints, frontend asset manifest, and fresh current-checkout
   replay provenance. A stale or mismatched item keeps that profile cell
   closed.
4. **Open only the evidenced candidate cell.** Let the compiler mark one
   exact `multiple_legal_team_start_states` profile cell executable only when
   both required signals are authoritative for that profile and its reviewed
   evidence row accepts. Keep `executed` false until a genuine paired campaign
   is run and `qualification_eligible` false regardless. The requirement is
   complete only after all three registered profile cells independently meet
   their full evidence gate. No other requirement status changes in this
   tranche.

The implementation remains offline and must not dispatch simulator work,
acquire devices, or alter any campaign manifest/checkpoint/output. After the
live r4 lease is released, the required initial runtime acceptance is one
candidate profile cell with a reproducible validation seed in both arms,
followed by per-profile source-bound pair reduction and replay inspection. That
acceptance validates only that candidate cell; a full requirement claim still
needs independent complete evidence for all three profile cells across their
exact protocol cases, policies, categories, and seeds.

## Review criteria

- The assigned pose set is applied after reset and before the first policy
  observation, and the checked tensor is the simulator's actual pose state.
- Per-profile legal-start numerator and denominator reconcile exactly to
  retained world IDs/indices and booleans; the reported ratio is recomputed as
  sum of legal worlds divided by sum of world counts. Any missing/illegal world
  fails the candidate gate even if the diagnostic ratio rounds to 1.0.
- One profile cell's `outcome_by_start_state` has exactly two strata:
  `scenario_defined` and its assigned profile. Metrics and paired deltas
  reconcile to individual records and exact frozen policy/category/seed
  coverage. No cross-profile aggregation is accepted.
- Each evidence row binds one immutable protocol, registry, profile
  assignment, checkpoint/source/device, replay seed, replay artifact, backend,
  and frontend. Evidence for one registered profile cannot satisfy another.
- Old-code replay, stale dashboard, mismatched condition, incomplete pair,
  proxy metric, or missing reviewer acceptance leaves the cell non-executable.
- Qualification remains false. No direct-pursuit or trench-interception code,
  schedule, signal, or gate is modified.

## Required validation and evidence after implementation

1. Run focused offline contract checks for digest binding, per-world legality
   reconciliation, exact pair coverage, stratum/delta reconciliation, and
   fail-closed replay evidence. These checks alone do not establish simulator
   behavior.
2. When r4 releases devices, run a paired, representative end-to-end simulator
   evaluation for one chosen candidate profile cell on a reproducible
   validation seed. Inspect the record for actual pose application before
   observation, each retained world ID/index and legality value, policy rows,
   and outcome metrics; retain profile ID, scenario, seed, source/checkpoint/
   device, and pair identifiers. Do not extrapolate this evidence to either
   of the other two registered profiles.
3. Restart the dashboard services on the implementation checkout; verify
   service health, relevant endpoints, and frontend asset hashes. Generate a
   fresh paired replay from that checkout, inspect both arms, and bind both
   replay hashes and the current dashboard evidence to the review bundle.
   Mark any older replay outdated.
4. Only then provide that profile's evidence to the compiler's reviewed-
   evidence gate. This opens one candidate executable cell, not a qualification
   or promotion result. Repeat the full process independently for the other
   two registered profiles before describing the requirement as complete.

## Explicit nonclaims and remaining gates

This plan establishes no measured start-state outcome, simulator validity,
qualification, promotion, architecture win, safety claim, deployment readiness,
or GPU performance. Other Phase 4 requirements remain open, including trench
interception, reactive opponent pools, teammate behavior, sensor degradation,
abrupt policy change, feints, contact recovery, stationary blockage, and pose
mislocalization. Full matched five-track evaluation, Phase 2 warm starts,
Phase 0 calibration, Phase 5 target-device profiling, and final qualification
remain separate gates in `docs/meco-full-plan-remaining-gates.md`.
