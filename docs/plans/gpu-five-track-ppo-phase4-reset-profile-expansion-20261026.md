# Pass 1 superseding plan: five-track PPO, selected-row reset, and Phase 4 profile residency

**Status:** proposed scope expansion after Pass 2 stopped before edits; no implementation or runtime evidence.
**Planning base:** origin/codex/gpu-five-track-ppo-reset-residency-expansion-plan-20261026 at e7e426c556b80b7eda8bcefae014bf3c29c9129d.
**Supersedes in full:** docs/plans/gpu-five-track-ppo-selected-row-reset-residency-expansion-20261026.md, SHA-256 ad90181efcdafa141b6f1006ac7017a53e9b17204c3045d8556a7f011d338fb5, and all predecessor rollout/candidate/PPO plans it incorporates.
**Requirements:** docs/plans/whole-compute-gpu-only-meco-plan-20261025.md, steps 4–8; docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md, §2, §3 and §5 Phase 3; repository AGENTS.md simulator/dashboard validation rules.

## New Pass 2 stop evidence

The previous selected-row reset scope is incomplete. In curriculum_scenarios.py, reset_curriculum_scenario calls contact_exposure_profile and later apply_full_team_start_profile through meco_phase4_context.py. That module is outside the authorized target list.

Source audit found two concrete conflicts with selected-row GPU reset:

- apply_full_team_start_profile writes env.sim.pose for every world and zeros every world's velocity. It then computes per-world legality but converts the result to legal.detach().cpu().tolist(), calls legal.sum().item(), and stores Python validation count/rate fields. A selected-row reset would therefore change unselected state and move a reset report/decision result to the host.
- contact_exposure_profile derives separation via a device tensor max().item(), broadcasts one pose table to every world, then calls validate_full_team_start_profile. Validation branches on tensor-wide predicates in Python and formats host error detail. curriculum_scenarios currently copies the returned poses into the whole pose tensor.

These helpers are part of the live reset chain. No PPO production edit may proceed until their mask behavior, state ownership, legality reporting and contact-profile validation are explicitly reviewed.

## Objective and inherited requirements

This document fully supersedes the selected-row reset plan and inherits all its exact requirements, targets, tests, three-pass gates and stop conditions, plus every five-track PPO batching, candidate ordering, route-query, terminal/status, replay, GAE, optimizer, RNG and checkpoint requirement in its predecessors. No inherited invariant is waived.

Extend the selected-row reset contract through Phase 4 start-profile and contact-exposure helpers. A device reset mask must cause only selected worlds' robot poses, velocities, legality state, exposure geometry, and associated reset metadata to change. Unselected worlds retain all tensor values and future RNG/action behavior. Numeric legality remains a device tensor used for fail-closed reset/update control; Python lists/rates/counts are passive report projections only and must not be materialized during live reset or influence action eligibility, candidate membership, PPO mutation, qualification or continuation.

## Exact source trace and state ownership

1. curriculum_scenarios.py applies contact_exposure_profile(env, contact_geometry_id), copies returned poses to env.sim.pose, zeros all velocity, and later invokes apply_full_team_start_profile(env, runtime_profile_name). It initializes phase4_start_profile_applied, phase4_start_profile_legality, phase4_start_profile_validation_rate and phase4_start_profile_validated_world_count; these must become fixed-shape row state for live execution, with immutable profile identity kept static.
2. meco_phase4_context.py owns registered profile names/config, construction of full-team poses, contact geometry constants, field dimensions/radii, legality predicates, post-application legality reporting, and contact exposure poses. Separate static assignment validation and numeric per-world legality evaluation. Preserve registry hashes and assignments; make both pose construction helpers accept a device mask or produce full-shape proposals that are committed only through that mask.
3. full_team_start_profile_legality evaluates bounds, pairwise chassis overlap and static collision boxes. Its output must be one bool per world, remain on device, and align with selected rows. Static collision box configuration may be host metadata; no data-dependent legality scalar may be inspected by Python.
4. contact_exposure_profile creates a six-slot pose proposal for crossing_contact, side_contact or recovery_contact. Its separation currently reduces a radii tensor to a host scalar. Replace that with equivalent per-world device arithmetic and preserve exact pose/clearance semantics. Do not call the whole-batch throwing validator on the live masked path.
5. validate_full_team_start_profile and profile assignment validators have offline/configuration and test/report uses as well as possible reset use. Keep static schema/registry validation where it is; route live tensor legality into a typed device status and only format detailed errors/report records at the passive artifact boundary.
6. The existing tests/test_meco_phase4_context.py asserts Python report fields after full-batch application and geometry fixtures. Adapt reset-specific expectations to the new typed device report and mixed-mask invariants without removing the established frozen pose, legal-boundary, overlap, obstacle or registered-contact tests.
7. meco_track_training.py reads the legality list/count/rate for evaluation reporting. Update only its passive serialization call sites so report values are projected after the relevant device-gated boundary; they cannot enter a live policy/training decision. Preserve schema meaning and identity or version any report schema change.
8. Preserve the previous reset inventory: curriculum_scenarios, architecture_scenarios, tensor_3v3 reset coordinator, tensor_physics, 3v3 state owner mixins, fuel physics/runtime/sleep, curriculum_metrics, tensor_defense_reset and its gamepiece/observation delegates. If tracing either profile helper finds another mutable row-state owner/helper, stop before editing it and return for new Pass 1 scope.

## Additional authorized targets

All targets from the superseded selected-row reset plan remain authorized and mandatory. This expansion adds:

| Target | Authorized work | Boundary |
|---|---|---|
| frc_defense/meco_phase4_context.py | Add selected-row profile application and contact exposure proposal APIs. Compute legality per world as device tensors; commit pose/velocity only under a validated selected mask; preserve profile coordinates, slot ordering, physical clearance and registry/config digests. Separate device legality/status from passive Python report formatting. | Keep assignment/config validation static; no live .item(), .tolist(), .cpu(), host reduction, whole-batch write, host legality exception branch or report field used in training decisions. No changes to teammate/focal-pose runtime semantics outside reset-owned helpers. |
| frc_defense/curriculum_scenarios.py | Call mask-aware contact/start profile APIs; keep contact profile and start profile ordering identical; preserve selected-row velocity reset and phase-four flags. Store legality as device tensor/status plus static schema; do not overwrite unselected flags. | No change to contact geometry, curriculum scenario placement, sensor exposure or qualification contract. |
| tests/test_meco_phase4_context.py | Retain current registered pose, legality, obstacle and contact geometry coverage; add all-true/all-false/single/mixed masks, unselected pose/velocity equality, per-world radii/clearance checks, device legality output and report-only projection checks. | No test may treat a Python report projection as the live legality decision. |
| tests/test_meco_curriculum_device_reset.py | Add reset ordering and mixed masks where contact exposure and full-team start profiles are each configured, including selected legality/failure status and unchanged unselected report state. | Keep scenario reset parity and full owner inventory assertions. |
| tests/test_meco_phase4_runtime_adapter_contract.py and tests/test_phase4_pose_policy_state.py | Preserve profile assignment identity, static registry/config hashes, truth-versus-observation boundary, and report schema while verifying the reset path does not host-materialize legality. | No changes to policy pose estimation or observation semantics. |
| frc_defense/meco_track_training.py | Move legality list/count/rate conversion to passive report projection after device-gated update/reset boundary; update report schema identity only if shape/type changes. | These fields remain diagnostic and cannot affect training, reset, qualification, PPO eligibility or campaign control. |

No other profile helper is authorized. In particular, changes to contact_exposure_actions, contact proxy recorder, focal pose assignment application, phase-four outcome scoring, native kernels, physics/observation semantics or campaign evaluation are excluded unless a new Pass 1 names exact targets.

## Profile and contact-exposure tensor contract

1. All pose proposals use fixed [world, 6, 3] layout and device dtype. Masked application writes pose and velocity only at selected rows. All-false changes no state; all-true matches legacy full-profile placement modulo the new declared report serialization schema.
2. Preserve robot slot order, field scaling, registered profile configurations, geometry headings, contact-pair clearance and static-obstacle envelope. Preserve precedence: curriculum placement and configured lane offsets, then contact exposure pose/velocity reset, then optional full-team profile override and its velocity reset, then stationary blockage overlay. Do not reorder or combine these stages in a way that changes the final pose.
3. The contact geometry separation is computed per world using that world's robot radii, not a batch-wide maximum copied to host. This is required for selected-world correctness. If radii are provably static/shared, the proof must be source-based and still avoid device-to-host scalar conversion.
4. Compute legality on device after the proposed/applied selected poses. Legality combines bounds, chassis pairwise separation and static collision-box overlap exactly as before. A selected illegal pose sets a typed device reset-failure bit and gates the whole PPO update/reset transition to a no-op; do not branch in Python on the bool tensor, continue with an illegal pose, or invoke the old host diagnostic exception as a fallback.
5. Separate per-world legality from reporting. phase4_start_profile_legality becomes a device bool tensor; validated-world count and legal count/rate are device reductions. Host formatting to a list/count/rate is allowed only after the decision has been fixed and the report is passive. Reports must identify schema, mask and selected-world coverage; they may not enter action/policy logic.
6. All-false reset must not compute a host-visible report or invalidate global state. Mixed mask validates and reports selected worlds only; unselected legality/report values remain unchanged. All-true report compatibility may be produced at a later passive boundary.
7. Registry/profile names, immutable protocol assignment dictionaries, config SHA-256 values, scenario IDs and static box geometry remain host metadata. They cannot encode dynamic selected pose values or substitute for per-world legality checks.
8. Profile status and legal result are included in reset owner manifest, episode-generation identity, replay digest/schema, checkpoint schema, and source/protocol identity. A schema mismatch or bad profile assignment fails closed before PPO mutation.

## Tests, stop conditions and gates

In addition to every test in the superseded plan:

- Test full-team profile all-true parity against the frozen reference, then verify all-false is a strict no-op and mixed masks change only selected poses, velocities, legality tensors and report state.
- Test each contact geometry with per-world radii, exact slot/order/heading semantics, legal bounds/clearance, and nonselected-row preservation. Verify no host reduction is required to derive separation.
- Inject selected invalid profile/contact cases and prove device failure status prevents all partial PPO/optimizer mutation while unselected rows remain unchanged.
- Verify report projection matches the established list/count/rate meaning after the fixed decision boundary but is absent from live reset/control.
- Retain all prior Phase 4 assignment digest, paired-profile, legality boundary, obstacle and contact exposure regression cases.

All tests that check actual simulated robot contact or other game behavior require the AGENTS-mandated representative seeded end-to-end simulator evaluation with fresh replay inspection. Dashboard backend and frontend must be restarted/verified from the exact checkout first. Such evaluation is only permitted after Pass 3 and verified release of both WX9100 leases.

Stop before any edit if a profile helper mutates all rows, reads a legality/radii tensor into Python, formats dynamic legal values before the passive boundary, cannot validate selected rows without host branch, changes pose order/clearance/profile precedence, loses an unselected report value, adds a new helper/module, modifies contact actions/physics, or makes legality/report fields influence a training decision. Also retain every stop condition in the predecessor: no unlisted reset owner, no future RNG change for unselected rows, no host terminal/reset control, no PPO/candidate/route semantics changes and no CPU numeric decisions.

## Sequential review gates

1. Pass 1 independent reviewer approves this exact artifact/head/hash, profile helper call graph, state ownership, static-versus-device legality boundary, report-only semantics, full source/test target list and inherited all-five/reset invariants. No implementation or runtime operation.
2. Pass 2 starts only after approval in a fresh HDD worktree from latest main with approved lineage. Publish updated reset owner inventory and schemas before edits. Commit and push incrementally. No tests, imports, compilation, GPU, simulator, dashboard, replay, training or campaign activity while the two WX9100 leases are held. Any newly discovered helper returns to Pass 1 before editing it.
3. Pass 3 is independent exact-head read-only review of every change, profile/status/reset path, per-world isolation, report projection boundary, RNG/checkpoint continuation, no-host-sync trace and prior five-track PPO/candidate/route invariants. New target or semantics means Pass 1 again.
4. Only after Pass 3 approval, explicit user stop and verified release of both WX9100 leases may focused device tests run on attested WX9100s. Simulation behavior tests also require the seeded end-to-end evaluation and current dashboard/fresh replay provenance required by AGENTS.md.
5. This plan does not resume or modify a campaign. Future training still requires existing phase gates and explicit user authorization.

No implementation, tests, imports, runtime, GPU, simulator, dashboard, replay, training, fitting, evaluation or campaign operation was performed to create this plan. This document does not assert that Phase 4 profiles or selected-row reset are GPU-resident.
