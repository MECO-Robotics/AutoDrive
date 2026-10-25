# Phase 2 multi-case five-track warm-start

## Planner pass

**Objective:** extend the reviewed one-case Phase 2 pipeline so Plan A1/A2/A3 and Plan C1/C2 can be warm-started from a single explicitly declared, source-bound collection spanning multiple frozen training cases. Preserve each case as an immutable shard, then produce a validated combined training corpus and five-track bundle. This is an offline planning artifact; it authorizes no simulator, GPU, device, training, or replay work.

**Planning base:** `main` and `origin/main` at `6936feeb4b2843d8964a8db5b0fd5d76ee1a4f03`; `qualifications/meco/protocol.json` SHA-256 `9314a1db838ead3e32db4ab8ed5a96f6944592fcedc162b2fe572c608afef35f`. The protocol contains 157 distinct train `(scenario_id, category)` cases with 100 seeds each. The root checkout has one unrelated modified generated HIP file; this plan is committed from an isolated HDD-pool worktree.

## Source requirements

- Source plan §2 requires all five model tracks and common observation/scenario conditions, while preserving the different Plan A pose outputs and Plan C candidate outputs.
- §5 Phase 0 requires fixed protocol partitions, frozen source/configuration and disjoint held-out seeds. Phase 2 requires Plan A legal-node goal/routing and feasible-behavior targets, plus Plan C candidate gross-return or pairwise labels from matched rollouts that account for simulator variability. Data from validation and final-test partitions must never enter training or fitting.
- §§6–7 require paired comparisons on held-out scenarios and explicit candidate-recall analysis. A single easy case cannot support a generalization or comparative claim.
- §8 requires the five architectures to be trained from their appropriate targets under a common protocol and source frontier.
- Project `AGENTS.md` requires a representative end-to-end simulator evaluation and inspected fresh replay for any future validation run that checks simulated game behavior.

## Exact current code gap

At the planning base, `frc_defense/meco_phase2_campaign.py::describe_campaign` and `execute_campaign` bind one exact scenario/category/seed list. `frc_defense/meco_phase2_complete_set.py::execute_campaign` then seals C1/C2 artifacts and A1/A2/A3 U+B+P artifacts for that one descriptor. The one-case runtime handoff in `docs/plans/phase2-five-track-runtime-plan-20261025.md` correctly calls its initial 100-seed `pickup_simple/single` bundle a pipeline-enabling slice and says follow-on train cases must use additional source-bound attempts; it expressly forbids implicit pooling of incompatible bundles.

No current API creates a frozen multi-case descriptor, resumes a set of independently validated case shards, produces a deduplicated combined C corpus plus aligned A U/B and independent P corpora, or proves per-case coverage for a shared five-track warm-start. Running the one-case coordinator repeatedly would leave five-track artifacts tied to individual dataset hashes; it does not create a training artifact whose provenance covers a declared multi-case set. A single-case bundle therefore remains a diagnostic slice, not broad Plan A/C training readiness.

This is distinct from the in-flight Phase 4 reactive-pool outcome bridge: the proposed work is confined to Phase 2 dataset orchestration and source/protocol coverage, and must not modify Phase 4 schedules, opponent events, reducers, or outcome records. It also does not duplicate the existing one-case orchestration contract.

## Implementation boundary

Add a multi-case orchestration layer around the existing single-case descriptor, collector, fitter, validators, and complete-set bundle. Prefer a new narrow module such as `frc_defense/meco_phase2_multicase.py`; retain the existing one-case API and artifact schemas unless a minimal explicit versioned extension is required.

The descriptor must bind:

- clean source commit/tree/source identity and the exact raw qualification-protocol bytes/hash;
- an ordered, unique set of exact protocol `train` case keys and their complete train seed lists, with explicit requested coverage closure;
- common prediction mode/contract, candidate K, action/horizon limits, A settings, indexed WX9100 identity requirement, and disjoint immutable output root;
- a deterministic descriptor digest and per-case shard destinations.

Execution should process case shards serially under the existing exclusive device lease unless the adapter contract proves safe parallel use of disjoint leases. For each shard, revalidate the descriptor and immutable receipts before adopting it. A retry of an invalid or interrupted shard uses a fresh attempt directory; no journal/status boolean may substitute for checking the bytes and validators. After all declared cases pass, build schema-aware combined inputs in deterministic case/seed/decision order and reject duplicate or missing case-seed identities. Keep C repeated-rollout labels and their repeat evidence intact. Keep Plan A U/B replay support and independent P-route oracle data cross-bound to the same exact trajectory source shards. Human-authored behavior annotations remain separately signed and bound; never infer them from a C label, an expert action, or aggregate return.

Produce the five warm-start artifacts by reusing the existing C fitter/validators and joint A U+B+P fitter/validators on the combined corpora. Publish one atomic bundle only after all five validators reconstruct the same case-set, protocol, source, device, prediction, and dataset bindings. The final manifest must preserve per-case counts and case-level coverage, not only aggregate totals.

## Acceptance criteria

1. **Descriptor and partition safety:** offline descriptor creation is deterministic, performs no simulator/GPU/device work and writes no campaign outputs. Reject unknown cases, duplicate cases, altered seed lists, validation/test seeds, partial case seed sets when full-case coverage is requested, source drift, path aliasing, and overlapping outputs. Bind exact protocol raw-file SHA and train-case list.
2. **Shard integrity and restart:** injected-adapter coverage proves complete, resumable collection and fitting across multiple cases; completed shards are adopted only after recomputing descriptor, source, protocol, dataset/sidecar, journal, and validator evidence. Tampered, partial, duplicate, missing, or contradictory shards fail closed and retry only into fresh immutable attempt paths.
3. **Correct target pooling:** combined Plan C rows preserve repeated matched candidate outcomes, common-random-number repeat evidence, candidate/support digests, and gross-return/pairwise semantics. Combined Plan A U/B rows preserve replay-bound legal support, feasible behavior labels and censoring/completeness; independent P-route rows preserve executed-route evidence, scale, oracle-integrator contract and exact shared source bytes. Reject cross-case identity collision and any U/B↔P source/protocol/split/device/prediction mismatch.
4. **Per-case coverage:** validators report expected/observed seed and row counts for every declared case and each track target family. A high aggregate count cannot hide a missing case or target. A selected subset is explicitly marked `partial_train_coverage`; only the full declared train-case closure may claim closure across the current frozen train protocol. No coverage field implies held-out generalization or qualification.
5. **Five-track bundle:** all A1/A2/A3/C1/C2 artifacts are reconstructed by existing validators and bind one multi-case descriptor, case-set digest, combined dataset hashes, frozen protocol, source identity and declared device. Reject missing/swapped tracks, stale artifact bytes, mixed attempts, mismatched case sets, unbound annotations, changed model/input semantics and fabricated receipts. Export to `meco_training_manifest.describe_campaign` without dropping a track; keep runtime/native HIP attestation and `phase2_complete` conservative.
6. **Offline validation:** focused injected-adapter tests cover descriptor safety, one and multiple cases, all expected shard interruption/retry points, every dataset and receipt mutation, complete-set rejection cases, case-level coverage arithmetic, and manifest forwarding. No test runs simulator behavior, GPU fitting or a device lease. Later behavior validation must run representative seeded simulator scenarios and inspect a fresh current-dashboard replay under `AGENTS.md`.
7. **Scope protection:** no changes to frozen qualification partitions, label semantics, model architectures, the active r4 process/checkpoints, Phase 4 event/reducer code, or root user-modified HIP file.

## Dependencies and sequencing

1. Complete independent review and integration of the currently pending single-case HIP-routing enforcement (`codex/phase2-runtime-plan-20261025`, implementation commit `054a21ea0fce42100a8c5eb5993d2e5c9e0bf53b`) before treating the one-case collector/fitter as the shard primitive. That branch changes default replay/support to require HIP routing and forces C replay through the same path; this plan must reuse that code and must not duplicate its changes.
2. Reuse the existing reviewed one-case five-track complete-set coordinator and one-case runtime plan as the only per-shard implementation. Do not modify the separate Phase 4 reactive-pool outcome bridge.
3. Implement and independently review this multi-case planner/implementation/reviewer tranche as code-only. Do not start collection while r4 owns the two WX9100 leases.
4. After the active user-requested run ends naturally or is explicitly stopped, a separate runtime campaign must verify both physical WX9100 identities under lease, collect only the declared train case set, inspect all shard and combined validators, verify warm-start consumption in a fresh manifest, and run a fresh seeded simulator acceptance evaluation. Dashboard restart/API/frontend-currentness and fresh replay inspection are mandatory for that behavior evaluation.
5. Complete Phase 0 calibration, matched held-out A/B evaluation, Phase 4 evidence, and Phase 5 deployment gates remain separate. This tranche cannot qualify or promote a model.

## Reviewer focus

Review the exact integrated code for split leakage, incomplete case closure, duplicate-seed or decision identities, nondeterministic row order, mixed shard attempts, aggregate-count masking, loss of repeated-label variance evidence, Plan A P or behavior-target provenance drift, GPU fallback, symlink/path alias attacks, and claims that equate training coverage with held-out or qualification readiness. Require evidence that the multi-case artifact is consumed by all five tracks from the same frozen source and protocol. A complete code review still does not establish simulator behavior or empirical learning benefit.
