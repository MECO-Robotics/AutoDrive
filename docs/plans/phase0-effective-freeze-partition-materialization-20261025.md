# Phase 0 effective source/runtime freeze and four-way partition materialization plan

**Status:** Pass 1 planner proposal only. This plan creates no source freeze, partition, approval, dataset, calibration result, simulator run, or qualification evidence.
**Planning base:** `05d74abdc` (`codex/phase0-multimodel-predictor-plan-20261025`).
**Source plan:** `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md` (bind the actual bytes at implementation time).
**Parent contracts:** `docs/plans/phase0-baseline-heldout-calibration-20261025.md`; `docs/plans/phase0-resumable-paired-d0-d2-calibration-20261025.md`; `docs/plans/phase0-multimodel-predictor-integration-20261025.md`.
**Scope:** implement a reproducible handoff that captures the effective source/runtime configuration and materializes immutable freeze and disjoint four-way partition artifacts consumed by paired calibration and all five track manifests. Preserve existing schemas and historical artifacts; extend contracts only for demonstrated gaps.

## Goal and status boundary

The existing `meco_baseline_freeze` module can compile a freeze from caller-supplied fields, source references, runtime identity, and eight configuration-content byte strings. Its partition compiler binds a freeze digest and rejects repeated seed/family/layout/start-axis assignments, but it does not collect values from an effective campaign, prove where supplied configuration bytes came from, define the complete tuple universe, or prove assignment closure for every eligible scenario tuple and every model track. The provenance summary at `docs/implementation-provenance/phase0-baseline-freeze-contracts-20261025.md` confirms that no production freeze or partition was generated.

Implement the missing materialization path so that two distinct immutable artifacts can be generated from auditable evidence:

1. a clean source/runtime freeze that records the exact source tree, dependency/runtime identity, actual effective values for all eight baseline fields, and the exact bytes/digests of the eight required logical configuration artifacts;
2. a four-way train/calibration/validation/final-test partition bound to that freeze, with a canonical closed tuple registry, five model-track assignments, axis-level isolation, tuple-level non-overlap, and explicit workflow access rules.

The freeze and split are system-generated from captured evidence and preregistered assignments. They are not human estimand approval. The separate human-authored simulated-score estimand approval remains mandatory before any outcome-bearing calibration data is collected. No capture process may synthesize, pre-populate, or mark that approval accepted.

Implementation and offline contract work can proceed while diagnostic r4 is running, provided they do not alter or inspect its private data, stop/reconfigure it, acquire a device lease, launch a simulator/trainer, or compete for the WX9100s. Production freeze generation must use the final clean source/runtime intended for new work; the older r4 checkout/checkpoint cannot satisfy that binding. Any later simulator/calibration execution remains closed until the user manually stops r4 and both WX9100 devices are independently observed released. Metadata collection/device inventory must not be presented as a workload or proof that the devices are available.

## Existing contract audit and gaps this tranche closes

Current implementation in `frc_defense/meco_baseline_freeze.py`:

- `capture_source_identity()` requires the repository root, clean checkout, exact commit/tree, and source-plan digest.
- `compile_baseline_freeze()` accepts caller-supplied field values, selectors, source-file hashes, runtime identity, and bytes for exactly eight logical configuration IDs. It hashes these inputs but cannot establish that values were resolved from a real environment or that configuration bytes are those consumed by the campaign.
- `validate_baseline_freeze()` rechecks the source and supplied configuration bytes but takes no effective environment snapshot and cannot check a provenance chain from resolved values to the running evaluator.
- `compile_partition_contract()` validates four named partitions, axis duplicates/overlap, and fixed workflow uses. It does not bind a tuple-registry digest, include/exclude closure, per-track model assignment table, or source of the predeclared partition allocation. Axis-only disjointness does not prove that all registered tuples were assigned exactly once.
- Paired calibration planning requires freeze and partition digests but currently has no materializer that supplies them. It is therefore correctly unable to open collection yet.

The implementation must retain the pure existing compilers/readers for compatibility and add a narrow capture/materialize layer. Do not relax `capture_source_identity`, mutate historical schemas in place, or treat a successful hash of unverified caller input as evidence of effective values.

## Required immutable artifacts

### A. Capture receipt and effective configuration bundle

Before compiling, produce a versioned, immutable capture receipt that binds:

- source checkout root identity; exact commit and tree; clean status; source-plan path and SHA-256; and hashes for every authoritative source path used by `AUTHORITATIVE_FIELD_SOURCES`;
- exact resolved effective values for robot dynamics, field geometry, legal connectivity, opponent controllers/schedule, action cadence, sensor/noise profiles, scoring/observation semantics, and safety-event semantics;
- the selector/field path used to read each value, its runtime owner, and a machine-readable derivation/input reference so a reviewer can reproduce resolution; reject defaults or a value that only describes intended settings;
- canonical content bytes and SHA-256 for exactly `campaign_assignment`, `evaluation_protocol`, `drivetrain_physics`, `gamepiece_rules`, `field_layout`, `scenario_registry`, `opponent_schedule`, and `sensor_profile`;
- complete runtime identity: simulator ID/version/build digest, Python/Torch/backend and backend version, dependency lock digest, and explicit device model/identity/index for the eventual campaign runtime. Missing, ambiguous, unindexed or auto-selected identities fail closed;
- the exact capture tool/schema version, capture time in UTC, and a canonical receipt digest. Capture time is descriptive and must not substitute for content identity.

The capture tool must consume the actual effective configuration snapshot emitted/serialized by the selected current-source campaign factory/evaluator after all defaults, command-line overrides, scenario assignments, and model-specific settings have been resolved. It must not reconstruct effective values by re-reading source defaults or by applying a second independent configuration parser. If the current evaluator has no stable effective snapshot/export seam, the implementer must add one and bind the producer and consumer to the same normalized object; do not guess. For each value, preserve enough exact structure to distinguish absent, explicitly null, defaulted, overridden, and inherited state. Reject silent coercion, non-finite numbers, unknown fields where the schema is closed, and noncanonical serialization.

Configuration bytes may originate outside the checkout, but the receipt must name stable logical IDs and content bytes/digests rather than relying on filesystem paths. Evidence must show those exact bytes were inputs to the effective snapshot. A path or filename alone is not proof. Secret values must not be copied into the artifact; use a predeclared secret identifier and digest only where a secret affects behavior, and fail closed if equivalence cannot be verified safely.

### B. Baseline freeze

Compile the existing `meco-baseline-freeze-v2` contract from the validated capture receipt. The freeze must bind the capture receipt digest and include the full normalized effective values/configuration digests and source/runtime identities. If the current schema cannot bind the receipt digest and effective snapshot provenance without ambiguity, introduce a compatible `v3` schema and keep v2 validation intact. The freeze remains `qualification_eligible: false`; it records simulator score semantics exactly and cannot claim official match points.

Validation must re-read the exact source tree and config contents from immutable capture inputs, verify every listed authoritative source digest, recompute the canonical freeze digest, and reject source/config/runtime/capture drift. A byte change creates a new artifact identity; never overwrite or relabel an existing freeze ID.

### C. Partition registry and split artifact

Create a separate immutable preregistration input describing the complete candidate tuple universe and deterministic assignment. Include every evaluation axis needed by the source plan and calibration schedule: scenario/case ID, category/stratum, defender family, layout, start condition, sensor profile, seed, repeat/replicate, and applicable evaluation phase. Bind the exact registry bytes/digest, exact five model IDs (`A1`, `A2`, `A3`, `C1`, `C2`), and the predeclared allocation rule or human-authored assignment table. The allocation must be fixed before any outcome/calibration results are inspected.

Materialize all four partitions (`train`, `calibration`, `validation`, `final_test`) with these invariants:

- every eligible registered tuple is assigned exactly once; excluded tuples have explicit reason codes and cannot disappear from closure checks;
- no duplicate tuple and no tuple overlap across partitions;
- no shared seeds, defender families, layouts, or start conditions across partitions, using the conservative axis-level isolation contract already present;
- all five model tracks receive the same eligible tuple assignments and evaluation budgets; model-specific artifacts can bind their track ID but cannot redefine partition membership;
- calibration is the only partition permitted for deterministic comparator/reference selection and margin review; validation is for validation/model selection only; final-test is reserved to final evaluation and has no training, calibration, reference-selection, or margin-review read path;
- final-test access is represented by a single-use ledger/authorization contract compatible with the later qualification protocol; materializing the split does not consume or reveal final-test outcomes;
- the split binds the exact freeze digest, tuple-registry digest, assignment-table digest, and canonical partition digest. Validation recomputes all of them from inputs and rejects a stale freeze or any partition mutation.

The current `meco-heldout-partitions-v1` compiler may remain the axis-level base. Add a compatible versioned extension or adjacent strict tuple-closure contract; do not silently reinterpret v1. If the source plan's predeclared split is under-specified for one or more axes, do not invent a favorable split. Record the missing preregistration decision, prepare an explicit assignment proposal as a separate human-review input, and keep partition materialization/execution closed until the assignment is fixed before outcomes.

## Implementation sequence and planned changes

1. **Audit effective-value production path.** Trace `meco_track_training.evaluate_meco` → `make_curriculum_env` → `CurriculumEnv`/`TensorThreeVsThreeEnv`, scenario registries, candidate/predictor configuration, runtime initialization, and five model manifests at the selected implementation base. Identify the exact resolved objects actually consumed. Add a source map and exact selector tests; do not launch a simulator or inspect live r4 state.
2. **Add effective snapshot/export seam.** Emit a normalized immutable snapshot from the same configuration objects passed into the evaluator/environment, including provenance for overrides/defaults and all eight logical config payloads. Keep this side-effect free and independent of device acquisition. Snapshot creation must not claim that a simulation ran.
3. **Add receipt and freeze materializer.** Validate receipt completeness and provenance, call the existing freeze compiler (or compatible v3), and atomically create immutable artifacts with content-addressed IDs. Refuse dirty source, wrong plan digest, incomplete fields, missing config bytes, runtime mismatch, missing device identity, old r4 source/checkpoint, and writes over existing IDs. Provide a pure verifier for the full evidence chain.
4. **Add partition preregistration and closure materializer.** Define strict schemas for tuple registry, split assignment, five-track coverage, axis and tuple isolation, allowed uses, and final-test single-use handoff. Bind the materialized partition to the exact freeze digest. Ensure the later paired calibration coordinator consumes this partition's calibration tuple set only.
5. **Integrate gates.** Require exact freeze and partition digest validation in every new five-track training manifest and every paired D0/D1/D2 calibration manifest. Legacy r4 runs, legacy D0 artifacts and old checkpoints remain readable as diagnostic history but cannot be relabeled or admitted as current-source evidence. Make preflight errors name the missing receipt/config/partition invariant without creating partial campaign files.
6. **Incremental commits.** Commit coherent changes: (a) capture snapshot and receipt contract; (b) freeze materializer/verifier; (c) tuple registry/partition materializer; (d) training/calibration manifest intake integration; (e) focused offline tests and status/provenance record. Push each coherent checkpoint to the dedicated implementation branch. No generated production freeze, split, campaign, or approval artifact is committed in this code tranche.

Expected paths (final exact paths may be adjusted in reviewed implementation):

- `frc_defense/meco_baseline_freeze.py` and a narrowly scoped capture module;
- current MECO evaluator/environment factory for effective snapshot export;
- `frc_defense/meco_qualification.py`, training manifest validation, and the paired calibration CLI for digest intake;
- focused tests for strict schema, value provenance, source/config drift, closure/leakage, five-track symmetry, immutable writes, legacy rejection, and final-test access;
- `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md`, `docs/MECO_THREE_PASS_PROVENANCE.json`, and a provenance note after implementation review/integration only.

## Human decisions and execution gates

Keep these gates distinct in interfaces, stored status, and reports:

1. **System-generated capture/freeze/split:** produced by deterministic tools from the final clean source/runtime and exact predeclared inputs. It proves identity and partition mechanics, not outcome validity or human approval.
2. **Human estimand approval:** a human-authored immutable `meco-simulated-score-estimand-review-v1` artifact must approve the precise limited simulated score definition and bind source/rules digests. The capture/materializer must only read and validate an existing approval; it must never author it, default it to accepted, or conflate it with source freeze or split approval. If absent, outcome-bearing collection is blocked.
3. **User manual stop and device release:** the active old r4 process remains untouched. The user must manually stop it; then independently verify both WX9100 devices are released. A stale service/lease/status file or metadata snapshot is not enough. No calibration/training/simulator workload starts before this gate.
4. **Runtime evidence:** once authorized by those independent gates, run the exact registered calibration workload and generate new current-source replay evidence. Follow `AGENTS.md`: every behavior test has a representative seeded end-to-end simulator evaluation and inspected record/replay; restart current dashboard services after code changes and verify backend, assets, APIs, and fresh replay provenance. Device/workload evidence cannot be inferred from offline compiler tests.

Capture tooling and offline contract tests may be developed now. Materializing production freeze/split artifacts should use the clean current-source campaign runtime; if this requires acquiring a device or launching simulation while r4 remains active, defer that materialization until after the manual stop/release gate. Never use r4's historical process/configuration/checkpoint as a shortcut to a current-source freeze.

## Offline checks and simulator validation

Implement focused offline checks for:

- deterministic canonical bytes/digests and idempotent read-only validation;
- exact equality of effective snapshot, serialized config payloads and field selectors; missing, extra, stale, default-only, caller-forged or path-only provenance rejection;
- source-plan, commit/tree, source-file, dependency-lock, runtime/backend/device identity drift rejection;
- clean-tree enforcement and rejection of old r4 source/checkpoint provenance;
- complete registry closure, exact once-only assignment, tuple and each-axis non-overlap, all-five-track membership, fixed budgets, allowed-use matrix, and final-test isolation/single-use;
- immutable create semantics: same bytes can be reused, changed bytes at an existing ID fail, and partial writes are not accepted;
- no freeze/split compiler path can create or imply human estimand approval, calibration data, a qualification protocol, or eligibility;
- paired calibration and training intake require matching freeze, partition, registry and protocol digests and reject legacy/unbound evidence.

These tests are metadata/contract tests only. They must not instantiate the simulator, create evaluation rows, touch campaign data, or acquire accelerators. If any implementation test is expanded to check simulated robot, scoring, collection, shooting, or other game behavior, it must be accompanied by the seeded end-to-end simulator evaluation and inspected replay required by `AGENTS.md`; when unavailable because r4 owns the devices, defer the behavior test and state that runtime evidence remains open. Do not use a unit test to claim effective production values or a completed freeze.

## Acceptance criteria

- A reviewed materializer consumes the same resolved effective configuration object used by current-source evaluator/environment construction; it cannot silently rebuild values from defaults or accept unverifiable caller assertions.
- The immutable freeze binds clean source commit/tree and plan digest, dependency/runtime/device identity, all authoritative field source hashes and observed values, eight exact configuration payloads, capture receipt digest, and a recomputable canonical digest.
- The immutable partition binds the freeze and complete tuple registry; it assigns every eligible tuple exactly once, enforces four-way tuple and conservative axis disjointness, includes all five model tracks identically, enforces allowed-use/final-test boundaries, and is reproducibly verifiable.
- New five-track manifests and paired calibration artifacts reject absent/stale/mismatched freeze/split digests and legacy r4 provenance. Historical records are preserved and remain diagnostic only.
- Human estimand approval is an independent required input to outcome collection and is never generated by system tooling. User manual training stop and verified two-WX9100 release remain separate prerequisites to runtime work.
- Incremental coherent implementation commits are pushed and exact-head independent review covers the branch. Any simulator behavior, dashboard, replay, device, training, approval, calibration or qualification evidence not actually completed is explicitly left open.

## Out of scope

This plan does not collect or fabricate effective campaign values, generate production freeze/partition artifacts, approve an estimand, split or inspect private r4 data, stop/restart training, acquire devices, run simulation/calibration/training, generate a replay, choose margins/comparators, create qualification-v5, expose final-test outcomes, or claim any track is qualified. These remain separate implementation, human-decision, and runtime gates.
