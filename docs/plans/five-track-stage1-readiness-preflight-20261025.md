# Pass 1 plan: five-track Stage 1 readiness preflight

**Status:** Planning artifact only; no implementation or qualification evidence is claimed.
**Base:** `origin/main` at `160911abe1cf4644628053dae054e363b7fb010b`.
**Source specification:** `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`.
**Stage 1 gate:** `docs/plans/five-track-stage1-pickup-qualification-20261026.md` from approved planner commit `020aaa1283e4e9b407e95b943f999cdd658d3463` (currently on its own branch, not main).
**Scope:** Add a metadata-only readiness preflight for a future, matched `pickup_simple` Stage 1 qualification bundle for A1, A2, A3, C1, and C2. It must report blockers clearly and fail closed. It does not launch work or make a qualification decision.

## Why this is the next code tranche

The source plan requires all five model tracks to be evaluated under matched conditions, with an explicit frozen comparator, D0 safety floor, source-bound evidence, and independently visible track outcomes. Main already contains the v5 GPU-bootstrap protocol validator and promotion consumer in `frc_defense/meco_qualification.py`, but there is no single read-only command to establish that all prerequisites and five track manifests are coherent before a future qualification campaign is launched. Existing protocol instructions still show a v4 output filename, while the Stage 1 planner requires `meco-qualification-v5-gpu-bootstrap` with implementation ID `meco-v4-cluster-episode-device-percentile-v1`.

The latest authoritative live evidence was read from user service `meco-five-continuous-20261025.service` and its campaign files: the service was active/running (PID 2750253); `supervisor-status.json` reported `training`, cycle 7, 6 completed cycles, 30 completed chunks, no error, and `stop_requested=false`; `queue-status.json` recorded `evaluation_split=diagnostic`, null evaluation binding, and the provisional `meco-qualification-v1` protocol. The campaign manifest marks it not qualification eligible. These artifacts remain diagnostic ancestry and cannot be upgraded into Stage 1 results.

This tranche can be implemented and reviewed in a separate worktree while the live campaign continues. Its preflight is read-only with respect to campaign data and must not initialize a simulator, load checkpoints, acquire a GPU, or touch the active campaign. It removes a future launch ambiguity; it does not remove the Stage 1 planner's requirement to wait for the user to stop training and release both WX9100 leases before any calibration or qualification workload.

## Exact implementation paths

1. `frc_defense/meco_qualification.py`
   - Add a `preflight-stage1` CLI operation that consumes explicit paths to the calibrated protocol, a validated Phase 0 evidence index, a qualification campaign manifest, and exactly five per-track checkpoint manifests (`A1`, `A2`, `A3`, `C1`, `C2`). The precise JSON fields should reuse existing canonical validators and schemas rather than introduce parallel definitions.
   - Validate the protocol with the existing `_validate_v2_protocol` contract for target `pickup_simple`; require the exact v5 GPU-bootstrap version and implementation ID. Reject v1-v4, wrong scenario, an unselected/invalid D1/D2 reference, malformed protocol digest, and source identity disagreement.
   - Verify the Phase 0 index refers to immutable, digest-valid effective freeze/partition evidence, the approved score-estimand review, complete D0 and matched-reference calibration bundles, and a reviewed margin decision bound to those exact digests. This command verifies supplied artifacts; it does not create or bless human approvals.
   - Require one and only one manifest for each of the five policy IDs. Validate exact source/config/protocol/scenario/seed-partition bindings, unique checkpoint byte digests, five-track matched campaign ancestry, and the declared one-comparator-plus-D0-floor policy. A stale diagnostic campaign manifest, including the active r4 manifest, must be rejected as a qualification input.
   - Emit a deterministic JSON preflight report with `ready` or `blocked`, each check's evidence path/digest, and stable machine-readable blocker codes. `ready` means only metadata is internally consistent; it must not mean checkpoints were loaded, reports passed, the campaign was executed, or any track qualified. Do not mutate curriculum state or qualification streaks.
   - Implement this as a metadata-only path: read JSON and hash referenced immutable files only. Do not deserialize `.pt` checkpoints, initialize PyTorch/ROCm/CUDA, import simulator/runtime adapters, acquire device leases, modify supplied files, or write anywhere under a campaign root. The caller supplies an output destination outside the active campaign; refuse an existing non-identical output.

2. `docs/MECO_QUALIFICATION.md`
   - Document the metadata-only command and its exact limits.
   - Correct the Stage 1 promotion instructions so the required protocol is explicitly v5 GPU-bootstrap with implementation ID `meco-v4-cluster-episode-device-percentile-v1`; retain v1-v4 as historical/diagnostic only for this gate.
   - State that preflight readiness is not evaluation, qualification, or permission to use GPUs while the current training campaign is active.

3. `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md`
   - Record only the code capability after it is integrated and reviewed. Keep the campaign, Phase 0, simulator, five track qualification, matched outcomes, dashboard/replay, and whole-plan status open until their own evidence exists.

## Required behavior and invariants

- All five tracks are mandatory and individually represented. A missing, duplicated, or mismatched track blocks readiness; aggregation may not hide a track.
- Phase 0 prerequisites and the predeclared comparison contract are checked before any held-out outcome input is accepted. The selected promotion comparator is exactly one of `D1/current`, `D2/cv`, or `D2/multi_model_v1`; D0 is separately the safety floor.
- The Stage 1 target is the registered `pickup_simple` scenario with exact current version and normal curriculum episode budget. This preflight does not change score, FUEL lifecycle, safety thresholds, confidence rules, sample counts, seed partition, source identity, or protocol semantics.
- Validation and final-test partitions remain disjoint and are checked against the approved immutable partition evidence. Final-test outcomes must not be read or consumed by readiness preflight.
- Diagnostic reports, queue snapshots, training status, or a 100/100 update count are never accepted as Stage 1 pass evidence.
- Preflight does not advance `training_curriculum.CurriculumState`, create a qualification round, consume validation, alter campaign assignments, or produce a verdict for any model.
- Live training remains uninterrupted and untouched. No concurrent simulator/calibration/training/evaluation work, GPU lease, dashboard restart, or campaign write is allowed as part of this tranche.

## Acceptance checks for Pass 2 and Pass 3

1. Independent Pass 3 review is against the exact implementation head and verifies every invariant and path above; no claim is made from a moving branch name.
2. Static review confirms v1-v4 and the live diagnostic r4 manifest cannot produce `ready`, and a v5 manifest missing any single Phase 0 artifact or any one of A1/A2/A3/C1/C2 reports a stable blocker and stays blocked.
3. Static review confirms an internally consistent metadata set can at most produce `ready: true` with the explicit `metadata_consistent_only` scope label; no checkpoint load, model execution, outcome, curriculum-state, or promotion claim can arise from the command.
4. Static review confirms malformed digests, duplicate policy manifests, source/config/protocol mismatches, overlap in split seeds, a changed scenario version, and a non-frozen comparator all fail closed.
5. The CLI documentation states all required inputs, output schema, side-effect boundary, and examples of blocked and metadata-ready outcomes without presenting example artifacts as real evidence.
6. No tests, simulator, training, calibration, GPU, checkpoint deserialization, dashboard, or replay operation is run in this tranche. If a later implementation change modifies simulated behavior, follow project `AGENTS.md` and run its required seeded simulator evaluation and current-dashboard replay workflow in a separately authorized post-training window.

## Explicit deferrals

- Do not stop, restart, reconfigure, resume, migrate, or otherwise touch the active continuous training service, its checkpoints, source checkout, manifests, leases, or outputs.
- Do not perform Phase 0 freeze/partition generation, human score-estimand approval, D0 or matched-reference calibration, margin review, or protocol creation. Those require their separately approved artifacts and must wait for the user-requested training stop and released GPUs before compute.
- Do not evaluate candidates or controls, launch validation/final test, run the simulator, produce/review replay, change dashboard code/service, fit warm starts, promote a model, or claim Stage 1 qualification.
- Do not inspect `.pt` data or assert that current checkpoints are compatible. After training stops, perform the full compatibility audit required by the approved Stage 1 plan; if any checkpoint mismatches, treat the five-track set uniformly and do not mix old and new tracks.
- Do not implement broad Phase 2 warm-start generation, remaining Phase 4 executable stimuli, Phase 5 deployment profiling, the full A/B matrix, or five-track training curriculum changes in this tranche. They remain separate source-plan gates.

## Risks and mitigations

- **Risk: metadata readiness is mistaken for qualification.** Use a distinct `metadata_consistent_only` scope label, never update curriculum state, and state that actual reports and replay review remain mandatory.
- **Risk: format duplication drifts from existing validators.** Reuse the canonical protocol, calibration, manifest, and report validators; Pass 3 checks for duplicated gate constants or looser alternate rules.
- **Risk: active diagnostic artifacts are accidentally admitted.** Require qualification split/protocol/source bindings and reject the r4 manifest's diagnostic split, v1 protocol, and null evaluation binding explicitly.
- **Risk: a code-only preflight is mistaken for progress on model performance.** The status update records only the preflight capability. It does not change five-track Stage 1 completion status.
- **Risk: implementation disrupts the running service indirectly.** Work only in an isolated worktree and keep all runtime/data paths read-only; review the diff for service units, launch scripts, manifests, and external campaign paths.

## Next sequence

1. Pass 2 implements only the listed metadata preflight and documentation changes on a dedicated branch from this exact base.
2. Pass 3 reviews the exact candidate commit and the preflight's fail-closed/side-effect guarantees; corrections get a new exact-head review.
3. Integrate only after Pass 3 approval, preserving any user-owned dirty files.
4. Keep the live r4 campaign running until the user manually stops it. Once stopped and both WX9100 leases are confirmed free, follow the separate approved Stage 1 plan's Phase 0 prerequisites, checkpoint/source audit, matched evaluation, replay review, and dashboard currentness checks. This preflight alone does not authorize that execution.
