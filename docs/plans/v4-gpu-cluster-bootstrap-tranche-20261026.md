# V4 GPU cluster-bootstrap tranche

**Status:** Pass 1 implementation plan only. Static source review only was performed for this plan. No code, tests, fixtures, calibration, runtime, device, or campaign state is changed or authorized here.

**Planning base:** `53f3d9c9a49a7e12bf65b5e1fa2797f4464e63be`.

## Objective and narrow scope

Move the v4 qualification evaluator's category-level statistical calculations and pass/fail comparisons from Python/host computation to the explicitly selected GPU. Preserve the current paired category-level cluster-bootstrap estimand and report meaning. Do not redesign the curriculum gate, add a different bootstrap estimand, change sample coverage, or implement unrelated candidate/route/training residency work in this tranche.

This is one bounded first tranche of the larger whole-compute GPU work. It does not by itself make training, candidate generation/scoring, route/risk, simulator, D0 calibration, or matched-reference variance calibration GPU-resident. Those paths remain blocking prerequisites to any later calibration or qualification campaign under the user's GPU-only requirement.

## Static source findings

The relevant current source surfaces are:

- `frc_defense/meco_qualification.py::_cluster_bootstrap_ci` (around line 3600): uses Python loops and a host xorshift PRNG to resample clusters and episodes, computes means with Python `sum`, and sorts host lists for percentile limits.
- `frc_defense/meco_qualification.py::consume_validation_round` (around lines 3691-3853): constructs category rows, computes candidate/reference means and category success counts, creates paired `(layout_id, defender_family)` clusters, calls the bootstrap for paired useful-throughput deltas and candidate-reference / candidate-D0 safety deltas, compares CI bounds to frozen margins, then passes failures to `evaluate_promotion`.
- `frc_defense/meco_qualification.py::run_validation_round` final-test branch (around lines 4100-4160): computes category success/throughput summaries and calls the same cluster bootstrap for paired throughput and safety summaries.
- `frc_defense/training_curriculum.py::_failures` / `evaluate_promotion`: the current promotion path performs category episode/success-rate comparisons in Python. To meet “no CPU decision path,” threshold predicates consumed for a v4 qualification round must be computed on the selected GPU; the state machine may perform report-ID checks, protocol/state transitions, and persistence as orchestration, but it must not redo a numeric pass/fail comparison on the host.
- `frc_defense/meco_qualification.py::_paired_seed_cluster_bootstrap_ci` and `_paired_control_summary` (around lines 2681-2800): separate paired calibration estimands used in calibration evidence. They are not the category-level v4 validation bootstrap described below and are explicitly outside this first tranche. They must be ported in a later GPU tranche before running GPU-only D0/matched-reference statistical calibration or qualification.
- `_bootstrap_ci` (around line 3578) is not called in this module at the planning base; do not expand this tranche to unrelated or dead helpers without separate call-site evidence.

## Estimand and invariants to preserve

For each required category, the v4 evaluator currently:

1. Validates complete frozen case/seed evidence for candidate, selected paired reference, and D0.
2. Pairs candidate/reference rows on the exact same episode seeds and groups them by `(layout_id, defender_family)`.
3. Resamples the same number of complete clusters with replacement. For every selected cluster, it resamples that cluster's episodes with replacement using the original episode count. It then computes one simple mean over all sampled episode deltas in that replicate. The replicate may contain a variable total number of episodes when cluster sizes differ; it is not a macro-average of cluster means.
4. Uses the deterministic seed derived from `sha256(f"{evaluation_id}/{category}")[:16]` for throughput, and the existing XOR-derived seeds for candidate-reference and candidate-D0 safety intervals. It returns percentile endpoints at the current indices `int(.025 * (replicates - 1))` and `int(.975 * (replicates - 1))` after sorting replicate means.
5. Gates category-level useful-throughput by the frozen lower-CI margin and category-level safety by the frozen upper-CI noninferiority margins against both the paired reference and D0. Promotion still requires the existing category-level success threshold, complete coverage, and three consecutive protocol-bound rounds.

The GPU implementation must preserve the cluster keys, paired row mapping, cluster/episode resampling hierarchy, variable replicate denominator, replicate count, percentile selection rule, per-category aggregation, safety comparisons, and gate margins. It must not substitute independent episode resampling, unpaired candidate/reference draws, equal-weight cluster averaging, per-stratum CI gates, or a different bootstrap target.

## Bounded implementation sequence after Pass 1 approval

Each item is a separate reviewable step. This plan does not authorize its implementation or runtime steps.

1. **Keep r4 uninterrupted and isolated.** Continue the current diagnostic campaign without modifying its checkout, source, manifest, checkpoints, processes, leases, device assignment, output tree, or status. Develop only in an isolated HDD worktree. While r4 owns the GPUs, perform static code review only: do not run tests, generate fixtures, profile, calibrate, or launch any workload.
2. **Prepare a device-native statistics interface.** In `meco_qualification.py`, introduce a small internal GPU result structure for category counts/rates, paired-throughput CIs, safety CIs, and their threshold verdicts. Input case-row schema validation and immutable report I/O may remain host-side. Convert validated numeric metrics to tensors on the explicit indexed HIP device, then do all aggregation, paired deltas, resampling, replicate means, sorting/quantile selection, and threshold comparisons on that device. Materialize only completed scalar/report fields after the algorithmic decision is final.
3. **Use deterministic device sampling with unchanged estimand.** Replace host xorshift sampling in `_cluster_bootstrap_ci` with an explicitly seeded device generator/index implementation. Generate all cluster and within-cluster sample indices on the selected device. Bind generator kind, seed derivation, device/runtime, cluster order, replicate count, and quantile index rule in a versioned bootstrap implementation identifier. Exact CI values can differ from the old host PRNG's specific draws, but the statistical estimand and sampling hierarchy must remain unchanged. Determinism is required for the same source, runtime, device, input rows, evaluation ID, and category.
4. **Move all v4 numerical verdicts off host.** In `consume_validation_round`, compute category successes, category-aggregate success rate, useful-throughput means/differences, CI endpoints, safety deltas/CIs, and all frozen-margin comparisons on the same device as the validation run. Keep `GateDefinition` and the frozen v4 protocol as the only sources of thresholds. Pass the resulting typed, report-bound GPU verdict into the existing promotion state transition; refactor the numerical comparisons in `training_curriculum._failures` so a v4 round cannot recompute category thresholds on CPU. Structural validation, duplicate-ID handling, streak mutation, and atomic file writes remain host orchestration only.
5. **Reuse the device result in isolated final-test summaries.** Route `run_validation_round(split="test")` category/safety summaries through the same device-native statistical helper. Final-test output remains isolated and cannot advance promotion state. Do not broaden to paired-control calibration summaries in this tranche.
6. **Version and bind the statistical implementation.** Introduce a new immutable qualification-protocol schema/revision for the GPU bootstrap rather than silently extending or reinterpreting the existing v4 bytes. Add a required bootstrap implementation/version identity to protocol validation, reports, confidence-interval evidence, and evidence digest; include the selected-device backend/runtime identity as required by the frozen protocol. A protocol/report produced by the old host bootstrap must not validate under the new implementation, and vice versa. Keep old readers for historical reports where needed, but never admit old statistical evidence into a new promotion streak. Do not change gate thresholds or claim old CI reports are comparable without a reviewed versioned migration decision.
7. **Wait for device access before any verification.** Continue static review only until the user manually stops r4 and all processes/device leases are confirmed released. Then use the separately approved GPU verification tranche to run deterministic fixtures, device-residency checks, profiling, and representative validation calculations on one explicitly leased WX9100. Do not share that device with training or another evaluator; device use must follow the two-WX9100 exclusive lease plan.
8. **Freeze the final source before calibration.** The current `curriculum_training.source_identity()` hashes Python/C++/CUDA/HIP source/config files under `frc_defense` plus runtime environment and native extension identity. Editing `meco_qualification.py` or `training_curriculum.py` changes this source identity. Finish and verify this tranche and any other source changes required by the whole-compute GPU plan before freezing source/runtime. Only after the final source identity is frozen may a separate calibration tranche regenerate source-bound D0/reference variance evidence, review margins, and produce a new immutable v4 protocol. Any later source/runtime/statistical implementation change requires new binding and recalibration; old provisional v1 or old-source v4 calibration is not reusable.

## Deferred deterministic fixture strategy

No fixture generation or test execution occurs while r4 uses the GPUs. After r4 is manually stopped and leases are released, the implementation tranche should use:

- fixed, immutable input rows covering multiple `(layout_id, defender_family)` clusters with unequal cluster sizes, identical candidate/reference seeds, and complete D0/reference/candidate safety values;
- analytic degenerate cases with known outcomes (all deltas zero, constant paired deltas, and a one-cluster case) to validate pairing, numerator/denominator handling, and quantile bounds without a CPU runtime oracle;
- repeated same-device runs with the same evaluation ID and protocol to verify byte-identical sampled-index digests, CI outputs, and gate verdicts;
- a reviewed frozen fixture output/digest produced by the approved GPU implementation, plus permutation/order checks that preserve cluster-key ordering and deterministic output.

No CPU bootstrap implementation may be used as a live fallback or runtime oracle. If a comparison against historical host outputs is required for review, treat the already-existing immutable historical outputs only as explanatory evidence; do not execute new CPU statistical calculations. Any observed numeric differences caused by the new device PRNG must be documented and reviewed against the preserved estimand before calibration margins are frozen.

## Acceptance evidence for this tranche

- Static review shows every v4 promotion/final-test bootstrap call uses the GPU helper and no numeric path falls back to `_cluster_bootstrap_ci`'s host loops.
- Input validation proves exact candidate/reference pairing, complete category case/seed coverage, cluster identity, finite metrics, and per-category margin selection before device computation.
- All category aggregation, resampling, means, sorting, percentile selection, and gate comparisons are performed on the explicitly selected indexed HIP device. CPU-only, unavailable, wrong, or mixed device inputs fail closed before producing a decision.
- The GPU result preserves the stated cluster/episode resampling estimator and produces deterministic results for the same bound implementation/device/runtime/input tuple.
- The promotion state machine remains the existing authority for report uniqueness and consecutive-round state. It consumes a bound device-computed verdict and does not redo numeric threshold decisions on CPU.
- Reports and evidence digests bind bootstrap implementation ID, source identity, protocol ID, runtime/backend/device, evaluation ID, exact case/seed table, CI outputs, and final gate verdict. Old implementation outputs are rejected under the new protocol identity.
- After all source changes are complete, a separate follow-on refreshes source-bound calibration and v4 protocol. This tranche alone does not authorize or establish qualification readiness because other training, candidate, route/risk, simulator, and calibration compute paths remain outside scope.

## Stop conditions

Stop before device work if r4 has not been manually stopped and released; if the exact indexed WX9100 identity/lease is unavailable; if a host calculation can affect the numerical result or promotion verdict; if pairing, cluster keys, seeds, margins, or bootstrap implementation identity is ambiguous; if source/runtime changes after freeze; or if someone attempts to reuse old-source calibration/report outputs without a reviewed binding migration. Do not touch r4, use a CPU fallback, run a test/fixture/profile while r4 is active, or treat this statistical tranche as closure of the whole-compute GPU requirement.

## Pass 1 review criteria

Confirm the plan is limited to v4 category-level validation/final-test statistics; preserves the existing paired cluster bootstrap estimand and category-level gate; moves all numerical thresholds and CI decisions off host; cleanly excludes the separate calibration bootstrap for a later tranche; defers deterministic fixtures/device verification until r4 releases the GPUs; and binds implementation/source/runtime changes before any new calibration or v4 protocol. Approval authorizes planning only.
