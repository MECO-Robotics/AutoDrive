# Phase 2 effective-freeze binding for five-track warm starts

## Pass 1 plan

**Planning base:** `fd9c0d3ee` (`main`) at plan creation. The main checkout has an unrelated local HIP kernel edit; this plan is written in an isolated worktree and does not touch that edit.

**Objective:** make the existing multi-case Phase 2 data and A1/A2/A3/C1/C2 warm-start bundle bind to the effective Phase 0 freeze, partition, and approved estimand, then use those exact artifacts in a fresh five-track manifest. This is an implementation and execution plan, not evidence that approval, data, weights, training, or qualification exists.

**Source requirements:** `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md` §§2, 5 Phase 0/2/3, 6–8; `docs/plans/phase2-warmstart-production-execution-20261025.md`; `docs/plans/phase0-baseline-heldout-calibration-20261025.md`; `docs/plans/phase0-resumable-paired-d0-d2-calibration-20261025.md`; `docs/MECO_SUPERVISED_WARMSTART.md`; and `AGENTS.md` simulation-validation instructions.

## Current interfaces and exact gap

- `frc_defense/meco_phase2_multicase.py::describe_multicase` and `validate_multicase_descriptor` bind the clean source commit/tree, runtime source identity, qualification protocol bytes, ordered train cases/seeds, model/prediction settings, candidate K, and indexed WX9100 inventory requirement. The descriptor currently has no effective Phase 0 materialization, freeze digest, partition digest, or approved estimand binding.
- `collect_case_complete_set` delegates each shard to `meco_phase2_complete_set`; `validate_case_receipt` checks exact descriptor/case/source/protocol/device and the trajectory, U+B and P dataset/sidecar bytes. `combine_target_shards`, `fit_multicase_warm_starts`, and `validate_multicase_warm_start_bundle` combine, fit, and revalidate the all-case corpus and all five weights. Their shared provenance presently bottoms out at the qualification protocol and source/runtime identity rather than the effective freeze/partition.
- The A joint fitter in `frc_defense/meco_supervised.py::warm_start_plan_a_joint_ubp_v1` consumes legal-node U+B plus separate oracle-v3 P data. C1/C2 use the composite counterfactual fitter over the same declared trajectory corpus. The current Phase 2 complete-set manifest path is exported through `export_training_manifest_warm_starts` / `export_training_manifest_binding` and validated by `meco_training_manifest.py` and `meco_supervisor.py`.
- `meco_training_manifest.py` already calls `validate_effective_phase0_bundle` when `phase0_materialization` is provided, but this training-time check does not back-bind the Phase 2 descriptor, case receipts, combined data, or fitted artifacts to that bundle. This is the integration gap; do not build a parallel collector, label schema, or complete-set coordinator.
- The live r4 run is diagnostic and cannot consume these artifacts retroactively. No production Phase 2 case corpus or five warm-starts are established by this plan.

## Bounded implementation scope

Implement one versioned, fail-closed binding path through the existing multi-case APIs:

1. Add a required effective-Phase-0 materialization input to a new production descriptor schema (or a strictly versioned extension). On describe and every revalidation, call `validate_effective_phase0_bundle` against the same clean source root. Bind the exact bundle/materialization file bytes and canonical effective freeze, partition, and estimand digests. Reject a missing, legacy, stale, source-mismatched, or internally inconsistent bundle. Preserve older descriptors only as explicitly legacy diagnostic inputs; they cannot produce a production-ready bundle.
2. Thread that exact binding through each shard descriptor and case receipt, each `trajectory`, `ub`, and `p` dataset sidecar, the combined family manifests, C1/C2 and A1/A2/A3 artifact payloads, validator receipts, final five-track bundle manifest, and training-manifest export. Revalidation must recompute current file digests and compare each layer to the same descriptor binding. Do not accept caller-supplied digest strings without reopening the bound files and invoking the authoritative validator.
3. Keep the existing target consumers and validators. A1/A2/A3 must each fit from the same source-bound U+B data and separate independently labeled P-route data. U is the declared bounded sampled legal-node gross-return target; B is only a feasible supported behavior preference from valid comparable records or a separately declared human annotation source bound to its authority digest; P is only the independent oracle-v3 executed-route target. Never derive U/B/P from a teacher action, total match score, missing/censored outcome, or route-cost demonstration treated as optimal.
4. C1/C2 must fit from the same source-bound candidate rows, candidate IDs/order/masks, route-cost and safety constraints, and repeated matched gross-horizon returns/pair labels. Preserve repeat-level values, paired seeds, tie/variance/censoring information, and abstain on unsupported or unstable comparisons. Keep gross-return versus route-cost semantics explicit and shared with D1; do not subtract route cost twice.
5. Have `validate_multicase_warm_start_bundle` reopen and revalidate the effective Phase 0 bundle, exact ordered case closure, combined sidecars, all five track-specific weights, and every receipt before export. Keep `phase2_ready`, `phase2_complete`, and `qualification_eligible` false unless separate authoritative validators establish their precise gates. The training manifest must preserve the descriptor/bundle/freeze/partition/estimand digests alongside the exact five artifact paths and hashes.

## Fairness and source constraints

- Use one clean reviewed source tree, exact frozen protocol bytes, and the same declared train cases/seeds, partition, scenario/opponent allocation, and evaluation budget for all five tracks. Do not admit validation, final-test, or calibration tuples into collection or fitting.
- Require complete declared train-case and seed closure for the requested bundle, or label it partial and reject any full-protocol/readiness claim. Do not pool attempts with different descriptors, source trees, protocol bytes, runtime contracts, devices, or partitions.
- C1/C2 use the same deterministic candidate generator and exact candidate table, ordering, feasibility/safety masks, K, route estimates, input information, matched rollout seeds, and reward horizon. D1 must use that same candidate table and route/risk estimates. Record candidate recall as a separate future evaluation requirement; the ranker cannot select omitted candidates.
- A1/A2/A3 share the same legal support, node/behavior sampling declaration, U/B/P target definitions, source corpus, and P oracle. Report per-track/per-case target coverage and denominators. P0 or U+B-only weights are diagnostic, not the required full five-track Phase 2 warm-start set.
- Record actual indexed device identity under the lease (physical WX9100 model, BDF, vendor/device, HIP UUID, inventory digest), not only `cuda:N`. Host-side serialization may use CPU; simulator physics/routing and model fitting must pass the project's GPU-only gates without silent CPU fallback.

## Required contract coverage

After implementation, focused metadata/contract tests must establish:

- absent, altered, stale, wrong-source, wrong-partition, or unapproved Phase 0 materialization is rejected at descriptor, receipt, combined dataset, artifact, bundle, and training-manifest boundaries;
- edits to dataset, sidecar, receipt, descriptor, bundle, or any of five weights bytes fail digest revalidation; mixed attempts and source/device/runtime identities fail;
- omitted/duplicated/reordered cases, seeds, target families, legal support, candidate rows, repeats, P branches, or track artifacts fail; validation/final-test seeds fail even when rows otherwise look valid;
- A artifacts cannot pass the full U+B+P gate with missing, censored, unsupported, p0, or non-oracle P targets; C1/C2 reject mismatched candidate tables, unpaired repeats/seeds, manufactured ties/preferences, and unsupported gross-return targets;
- the exact five-track path/hash set and freeze/partition/estimand binding survive bundle export and supervisor initialization metadata; all readiness and qualification claims remain false absent their independent gates.

Tests that exercise simulated game behavior require a representative seeded end-to-end simulator evaluation and inspected fresh replay. Before generating/reviewing that replay, restart and verify the dashboard on the current checkout, including backend endpoints and frontend asset identity, per `AGENTS.md`. Do not count metadata tests or prefix-replay integrity alone as behavior validation.

## Runtime and launch gates

Do not collect, fit, run simulator behavior evaluation, or launch the fresh campaign until all applicable gates pass:

1. An immutable, human-authored estimand approval identifies the outcome/use, included and excluded mechanics, exact scorer/target semantics, source, and digests. Effective runtime/config capture and disjoint train/calibration/validation/final-test partition are materialized and independently validated. Do not fabricate approvals, captured config, partition rows, or labels.
2. The exact new implementation head is committed/pushed and independently Pass 3 reviewed. Use a clean checkout at that reviewed head and a new ZFS-backed attempt namespace. Bind full exact train closure or explicitly report partial coverage.
3. The user manually stops r4 or it completes naturally; verify all worker PIDs are gone and both WX9100 leases are free. Do not signal/reconfigure r4, reuse its outputs, or run competing jobs.
4. Under an exclusive lease, attest both required physical WX9100 identities and the collection device; verify pinned inventory digest and Plan A HIP route parity/profile on the selected card. No P600 and no CPU simulator/routing/fitting fallback.
5. Collect and validate C candidate corpus and A U+B/P targets under the same frozen source/protocol/partition and declared train split. Audit label coverage and validator receipts, seal the exact five artifacts, and re-open the bundle with the production validator.
6. Generate a fresh isolated five-track manifest from current reviewed settings, validate the exact artifact hashes and provenance, launch only after gates 1–4 still hold, and retain each worker's weights-only initialization record with track/hash/source/protocol/case-set/freeze/partition binding.
7. Run a separate seeded end-to-end simulator evaluation of the changed behavior and inspect its fresh replay after verifying current dashboard backend and frontend. Retain scenario, seed, replay ID, source/checkpoint digests, and result. This does not itself establish held-out success or qualification.

## Explicit unknowns and non-claims

- The exact human-approved estimand, effective capture bytes, and production freeze/partition artifacts have not been supplied or validated here.
- Confirm whether Plan A behavior preferences will use the existing replay-derived feasible pair labels, an authority-bound human annotation source, or both; do not silently substitute one for another.
- Full train-case closure and the eventual per-case data volumes must be rederived from the reviewed protocol and effective partition at execution time; the earlier one-case pickup slice is not full coverage.
- The independent P oracle path, current HIP route parity/profile, physical device release, and resulting label sufficiency remain runtime questions.
- This plan does not claim source-bound production data, annotations, fitted weights, warm-start consumption, GPU/device validation, simulator behavior, replay inspection, model benefit, qualification, or promotion.

## Three-pass workflow and delivery

1. **Planner:** this artifact fixes the integration scope and acceptance conditions.
2. **Implementer:** work in an isolated worktree from the approved plan head, implement only the Phase 0 binding in the existing multi-case APIs, and commit/push coherent checkpoints. Keep data/runtime execution separate and gated.
3. **Independent reviewer:** inspect the exact pushed implementation head and adversarially check each provenance edge, schema transition, rejection case, and claim boundary. Code approval does not approve data, device, simulator, or training outcomes.

Integrate only the exact approved code head. Update implementation provenance from observed implementation/reviewer/test evidence; do not describe the Phase 2 objective as complete until the production corpus, five weights, fresh supervisor consumption, and required simulator evidence actually exist.
