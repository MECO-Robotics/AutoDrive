# Phase 4 focal-pose mislocalization wiring and evidence plan

## Pass 1 finding

**Planning base:** `origin/main` at `77245a02a1f18e35d4bfc569eb00cad4f334aa16`.

The earlier status row saying validation reset/observation application is
unwired is stale at this base. The current source already implements the
registered assignment, reset-time application, truth preservation, and
evaluator diagnostics. This plan records the exact contract to preserve and
closes only the remaining evidence gap. Pass 2 should make no duplicate runtime
implementation; it may make a narrow correction only if review identifies a
specific violation of the acceptance criteria below.

## Current code anchors

- `frc_defense/meco_phase4_context.py:215-260` compiles a paired no-offset or
  registered-offset assignment bound to full-match validation, protocol,
  qualification case, categories, and seeds. The two arms share a
  `pair_assignment_sha256` and have individual assignment digests.
- `frc_defense/meco_phase4_context.py:263-275` reconstructs and validates the
  exact assignment. `:279-310` applies the offset to the estimated focal pose,
  copies it to policy pose, checks physical truth remains unchanged, and marks
  application as before policy observation.
- `frc_defense/curriculum_scenarios.py:475-485` validates that the assignment
  covers this scenario, validation split, category, and seed, then attaches it
  to the environment. `:777-782` synchronizes policy pose and applies the
  assignment from the scenario refresh path. The reset path calls refresh at
  `:1015-1042` after scenario/start setup and before returning the initial
  observation at `:1075`.
- `frc_defense/meco_track_training.py:3460-3464` captures evaluator diagnostics
  for the assigned episode; `frc_defense/meco_phase4_context.py:313-325`
  records simulator-truth pose, policy estimate, and initial pose error.
- `frc_defense/meco_validation_campaign_runner.py:439-474` constructs and
  validates both paired arms. `frc_defense/meco_ablation_matrix.py:1157-1168`
  and `:1244` bind the assignment to the condition/report path.
- Existing offline coverage is in
  `tests/test_meco_focal_pose_boundary.py:40-108`, including exact pair
  binding, truth invariance, evaluator truth/estimate diagnostics, reset
  idempotence, and both matrix arms. Observation-transform invariants also
  appear in `tests/test_meco_phase4_context.py:296-320`.

## Intended acceptance

The offline contract must retain all of the following:

1. Construct one no-offset control and one registered-offset treatment for
   each exact validation case/category/seed set. Require the same protocol,
   source case, category list, seed list, paired offset identity, and pair
   digest; reject altered or unpaired arms and any non-validation/final-test
   assignment.
2. Apply the estimate only after scenario reset and before the first policy
   observation. The no-offset arm copies the physical pose into the policy
   estimate; the treatment applies only the registered translation/heading
   offset. Repeated reset application must be relative to truth and must not
   accumulate error.
3. Preserve simulator truth and unrelated physical state. Policy and estimate
   pose must not alias physical pose. Emit an application event with the pair
   and assignment digests, arm, registered offset, pre-observation marker, and
   truth-unchanged result.
4. Keep truth-versus-estimate values in evaluator diagnostics, not in the
   runtime control event. Bind those diagnostics and assignment provenance to
   the evaluated episode/report. Keep relocalization time, pose-error
   recovery, safety contact, and qualification unavailable until authoritative
   runtime signals exist.
5. Keep the code-level checks metadata/fake-state contracts. They do not
   establish behavior in the simulator.

At this planning base, existing tests cover most of these invariants, so Pass 2
should identify a concrete uncovered contract before changing code. Do not
expand this tranche into estimator correction, later relocalization logic,
other Phase 4 factors, model training, or qualification.

## Pass 2 deliverable: correct the stale focal-pose status row

Pass 2 must update only the focal-pose row in
`docs/MECO_PHASE4_RUNTIME_STATUS.md` to reflect current source. Replace the
stale statement that validation reset/observation application is unwired and
that no runtime event exists. The corrected row must say:

- A protocol-bound paired no-offset/registered-offset assignment exists and is
  validated against its full-match validation case, categories, seeds, and
  protocol.
- The assignment is applied after final scenario reset and before perception
  and the first policy observation. Simulator truth is unchanged.
- Evaluator records contain initial simulator truth, policy estimate, and
  initial pose error. They do not establish later relocalization or recovery.
- End-to-end simulator/replay evidence is not yet accepted; relocalization
  time, pose-error recovery, and safety-contact outcomes remain unavailable;
  qualification stays disabled.

Bind the status row's runtime-event text to the implementation contract:
`phase4_focal_pose_assignment` and
`records[].phase4_focal_pose.applied_before_policy_observation`, as declared in
`frc_defense/meco_phase4_runtime_adapter.py:198-206`. Cite the reset,
application, and diagnostic anchors listed above in an adjacent note or
provenance reference. Do not change any other requirement row or imply that
focal-pose qualification is executable.

**Pass 2 acceptance:** inspect the final diff to confirm it changes only the
focal-pose status wording and directly supporting citation; compare every claim
to the source anchors in this plan; run no simulator/device workload. Submit
the exact commit for independent Pass 3 review before considering the status
correction accepted. Do not edit the status document before this plan amendment
has received independent review.

## Deferred runtime evidence gate

After the active r4 lease releases and the required device/simulator capacity is
available, run a reproducible full-match validation comparison for the paired
no-offset and registered-offset arms, using the frozen validation protocol
and its assigned seeds. Inspect the fresh replay for the applied estimate,
unchanged physical truth, initial truth-versus-estimate record, and policy
behavior. Restart the dashboard stack on the exact reviewed checkout and
verify both backend endpoints and frontend assets before replay review. Record
the scenario, seeds, replay identifiers, source/checkpoint/protocol digests,
device identity, and results. Any test of simulated game behavior must be
accompanied by this representative end-to-end simulator evaluation.

No simulator, GPU/device, training, production data, r4 artifact, replay, or
dashboard operation is part of this plan-writing tranche. No behavior or
qualification claim follows from offline contract acceptance alone.
