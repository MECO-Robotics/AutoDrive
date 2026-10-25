# Phase 2: resumable five-track warm-start orchestration

## Objective and boundary

Add one offline-described, explicitly executed orchestration campaign that produces a **single fail-closed complete-set seal** for Plan A1/A2/A3 and Plan C1/C2. Reuse the existing matched-prefix C1/C2 repeated-label campaign and Plan A U+B+P bundle orchestration. Do not change label semantics, collectors, model architectures, or the frozen qualification protocol. The campaign remains train-split-only and never qualifies a policy or reads final-test cases.

“Resumable” means a restart can revalidate and adopt a completed stage, then continue to the next stage. If an interrupted stage has no fully verifiable output, retry it in a new immutable attempt directory; never overwrite, guess completion from a status flag, or reuse unvalidated partial data. A retry may repeat simulator collection. This bounded restart behavior is safer than resuming partial optimizer state.

## Source dependencies

- `frc_defense/meco_phase2_campaign.py`: `describe_campaign`, `validate_descriptor`, `execute_campaign`; fixed train/CV, K=16, horizon/action limits, repeated candidate labels; campaign report and dataset manifest.
- `frc_defense/meco_counterfactual.py`: existing `collect_counterfactual_v2` (C candidate outcomes) and `collect_plan_a_legal_nodes_v1` / `collect_plan_a_p_routes_v3` (A U/B and independent P-route oracle).
- `frc_defense/meco_supervised.py`: `warm_start_counterfactual_v2`, `validate_counterfactual_warm_start_artifact`, `warm_start_plan_a_joint_ubp_v1`, `validate_plan_a_joint_ubp_artifact`.
- `frc_defense/meco_plan_a_ubp.py`: `orchestrate_plan_a_ubp`; atomic complete A1-A3 bundle, bound to C's source dataset, protocol, source identity, split and seeds.
- `frc_defense/meco_training_manifest.py`: `_warm_start` and `describe_campaign` warm-start metadata contract; use the five resulting artifact paths to produce the next campaign manifest, without claiming readiness until existing runtime attestation says so.
- New module: `frc_defense/meco_phase2_five_track.py` (offline descriptor/validator plus explicit executor). Add focused tests in `tests/test_meco_phase2_five_track.py`; extend `tests/test_meco_phase2_campaign.py`, `tests/test_meco_plan_a_ubp_orchestration.py`, and `tests/test_meco_training_manifest.py` only for integration contracts.

## Required flow and persisted contract

1. Descriptor binds clean source commit/tree digest, frozen qualification protocol ID and raw-file SHA-256, one exact train scenario/category/seed list, indexed device, and disjoint output root. It derives the C descriptor and fixed A settings; descriptor creation performs no simulation, GPU work, or output creation.
2. Execute C1/C2 collection/fitting/validation once. Persist an atomic orchestrator journal after each stage with descriptor digest, stage attempt ID, input/output paths and hashes, validation summary and status. On restart, rederive descriptor; independently validate existing C report, dataset+sidecar, and both C artifacts before adopting that stage. If incomplete/invalid, fail closed or start a fresh numbered C attempt; never treat `campaign-report.json` status alone as evidence.
3. Feed the exact validated C dataset and same protocol to `orchestrate_plan_a_ubp`; bind all three A artifacts and its bundle report to those bytes. On restart, validate and adopt a complete A bundle; otherwise retry in a fresh A attempt, preserving prior attempts.
4. Atomically publish `five-track-bundle.json` only after all five track validators pass. It records exact track-to-artifact path/hash, per-artifact schema/track, shared dataset and dataset-manifest hashes, protocol ID/SHA, `split=train`, scenario/category/seeds, source identities, campaign descriptor/journal hashes, and five validator summaries. Validate the seal by reconstructing those claims from current files and rerunning artifact validation; fail closed on partial sets, changed bytes, mismatched provenance, duplicate/overlapping paths, symlinks, or final-test contamination. The seal explicitly sets `qualification_eligible=false` and does not itself set runtime `phase2_ready=true`.
5. Provide a deterministic export/mapping from a valid seal to the five `warm_start_artifacts` entries accepted by `meco_training_manifest.describe_campaign`; verify the produced manifest retains every artifact digest and common protocol/source binding. Do not silently drop an artifact or accept a mixture of campaigns.

## Acceptance contract

- Offline unit tests prove descriptor determinism and no-write/no-lease behavior; exact train seed/protocol/source binding; refusal of test/validation seeds and unsafe/overlapping paths.
- Injected-adapter tests exercise uninterrupted flow and restart after: C complete/A absent; A complete/seal absent; interrupted C; interrupted A; tampered dataset, sidecar, report, checkpoint, journal, and seal. Only fully revalidated stages are adopted; retries use fresh attempt paths and do not mutate prior artifacts.
- Complete-set tests reject each missing/duplicate track, swapped A/C artifact, changed artifact bytes, wrong protocol ID or SHA, different dataset, different seeds/source identity, and fabricated status/journal claims. Valid seal produces exactly five manifest entries with their hashes; manifest validation preserves their bindings and readiness remains conservative.
- Verify collector/fitter calls use existing adapters with expected parameters: one shared C matched-prefix dataset; A U/B and independent P-route targets from that same dataset; exact frozen train seeds and protocol; indexed exclusive device lease; no final-test input.
- Regression suites: `tests/test_meco_phase2_campaign.py`, `tests/test_meco_plan_a_ubp_orchestration.py`, `tests/test_meco_training_manifest.py`, plus the new focused suite. Tests must use injected adapters and must not run simulator/GPU work.

## Validation split, risks, and gates

**Code-only acceptance:** the above adapter-based tests establish orchestration, provenance, crash/retry, and manifest-binding behavior. They do not establish game behavior or that the learned targets improve policies. Do not relabel the bundle as qualified.

**Runtime acceptance after the active campaign releases the required WX9100 lease:** run the frozen train-only collection and fits on the declared device, inspect all generated dataset/bundle manifests and validator summaries, then launch a fresh seeded simulator smoke evaluation using a manifest built from the five sealed artifacts. Retain scenario, seed, run/replay ID, artifact/seal hashes and result. Before generating/reviewing its dashboard replay, restart current dashboard services, verify backend endpoints and frontend asset hashes match this checkout, and inspect a fresh replay per `AGENTS.md`. Held-out comparison remains a separate frozen-protocol gate; no final-test data may enter warm-start production.

Main risks: repeated simulator collection can be expensive after an interrupted C stage; GPU lease contention must leave the live campaign untouched; A and C artifact formats may expose provenance fields under different names, so the seal must normalize from validator outputs rather than trust report text; the current manifest launcher’s warm-start handoff must be checked before claiming those artifacts are consumed by training. This tranche should not alter or stop any active training process.
