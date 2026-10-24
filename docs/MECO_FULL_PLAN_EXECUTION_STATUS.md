# MECO learning tracks plan: execution status

**Snapshot:** 2026-10-24 03:51 UTC
**Plan:** [`MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`](MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md)
**Plan SHA-256:** `718ef9d76f5525bf8e4759d3a0a46790a3c978e2b146efec6436c89dbbb49b10`
**Implementation candidate:** `codex/meco-full-plan-integration` at `0e133780a0fadfb9015b84719e75291be89370e0`
**Candidate source identity:** `8f3d68c00e2beb52d0d188feaf40785bd79a91f04c5ac4c652a05379aa3d045a`
**Live campaign source:** `codex/meco-c1-replay-integrated` at `d7b8dd4b1c1814c2a368ded7604b3c887571fdcc`

This ledger separates code that exists from evidence that demonstrates runtime behavior. A passing contract test, completed optimizer update, or review of a scoped patch does not establish a simulator result, qualification, or deployment readiness.

## Live training campaign

The continuous campaign is isolated in its own checkout. The supervisor process was observed active at PID `1178932` at the snapshot, invocation `3ccef8583e0843bc92e2dbddbde6569c`, configured with `--devices cuda:0,cuda:1 --evaluation-device cuda:1 --cycles 0 --seed 20261024 --prediction-mode cv --evaluation-prediction-mode cv --phase4-train-exposure`. It writes to `/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive-meco-c1-replay-integrated/checkpoints/meco-five` and `.../evaluations/meco-five`. At heartbeat `2026-10-24T03:51:04Z`, A3 had 98/100 updates and C2 remained queued; A1, A2, and C1 had each completed 100/100. An immutable compact observation is recorded in [`MECO_CAMPAIGN_SNAPSHOT_20261024T035104Z.json`](MECO_CAMPAIGN_SNAPSHOT_20261024T035104Z.json), SHA-256 `921a4919aaf24fae61c7138d695a9fb4077242db64c6bdc9bb0d41bfe3e88efa`. It is a point-in-time record, not a live status feed.

At the status heartbeat `2026-10-24T03:51:04Z`:

| Track | Device | Training updates | State |
|---|---|---:|---|
| A1 | `cuda:0` | 100/100 | First chunk complete |
| A2 | `cuda:1` | 100/100 | First chunk complete |
| A3 | `cuda:0` | 98/100 | Running |
| C1 | `cuda:1` | 100/100 | First chunk complete |
| C2 | `cuda:0` | 0/100 | Queued |

The supervisor reports `completed_cycles: 0`, `completed_training_chunks: 3`, and `qualification_protocol_id: null`. No held-out evaluation has completed. These update counts are optimizer progress, not a 100-case evaluation or a pass. The campaign has no warm-start artifacts (`warm_start_artifact_sha256: {}`); it began from scratch. Training currently runs against source identity `c56cc4001441e10e72b191025a97141033907aed40f42c0f3b61f7bacab3e8e2`, while the integrated candidate has source identity `8f3d68c00e2beb52d0d188feaf40785bd79a91f04c5ac4c652a05379aa3d045a`.

The implementation candidate is a separate checkout and changes source-hashed Python files. Checkpoints from the live `d7b8dd4b1` campaign therefore do not satisfy strict source identity for the candidate. Do not evaluate or promote those checkpoints as candidate-source artifacts. No candidate-source evaluation or replay ID is available; `20261024` is the active training seed, not an evaluation seed. The candidate dashboard service has not been restarted or verified against its current backend and frontend after these code changes, and no fresh learned-policy replay has been generated or inspected.

## Completion ledger by plan section

| Section | Implemented | Evidence still required |
|---|---|---|
| **1. Decision and hypothesis** | The repository represents three Plan A spatial models and two Plan C candidate rankers, with deterministic comparisons and a shared safety boundary. | No matched, held-out result establishes that a learned architecture beats D0 or the matched candidate baseline. No winner or promotion is supported. |
| **2. Five models** | A1 U-Net+GRU, A2 structured SE(2) U-Net+regional attention+GRU, A3 hierarchical SE(2) GNN+regional attention+GRU, C1 candidate MLP+GRU, and C2 map encoder/candidate cross-attention+GRU are implemented. Parameter inventory and adapter shape checks exist. The inventory reports 171,717 parameters for A1, 122,311 for A2, 118,215 for A3, 49,026 for C1, and 124,162 for C2. | Counts and synthetic/adapter checks do not establish checkpoint-specific exported graphs, memory, Hailo compatibility, latency, or simulator performance. The current five-track frontier is incomplete. |
| **3. Deterministic and predictive controls** | D0/D1/D2 evaluation paths and prediction-mode controls exist. The schema-4 paired matrix analyzer validates protocol-bound reports and can calculate deterministic paired cluster-bootstrap intervals. | No current matrix report or interval exists for the requested architecture, control, prediction, penalty, candidate-recall, and fallback comparisons. D3 is not established. |
| **4. Shared observation and planner contract** | Shared raster/scalar inputs, candidate masks/order, stale-output fallback, legal action support, Plan A penalty provenance, evaluation-to-replay record binding, and generation-based replay publication are implemented. The async broker has deterministic fake-adapter contract coverage, including replacement, freshness, faults, and close; raised inference exceptions no longer kill its worker. No-route replay validation checks complete contiguous action coverage. | No fresh five-track learned-policy replay has been generated and inspected. Broker contract tests do not establish GPU-load freshness or simulator fallback. The new replay tooling does not prove P-shaped routes, robot behavior, collection/scoring, fallback behavior, or representative end-to-end safety. Global planning remains XY AD* with swept SE(2) checks, not full global SE(2). |
| **5. Training strategy** | Phase 0 protocol/configuration, Phase 1 interfaces, stochastic PPO support, Phase 4 train-split exposure scheduling, and the resumable two-device supervisor exist. Phase 2 collectors/loaders/fitters exist. A human-authored Plan A goal annotation materializer now packages explicit sidecar labels with source/legal/protocol/authority hashes; it invents no labels and marks replay validation as required. | No human-authored expert annotations, independent U oracle, generated annotation dataset, or warm-start artifact is available. The current campaign does not consume warm starts. Baseline variance calibration, a matched evaluation frontier, Phase 4 factor outcome runs, and candidate-source training remain outstanding. |
| **6. Metrics and decision gates** | Metrics, qualification protocol, confidence interval, calibration, and promotion interfaces exist. Plan A PPO penalty semantics use one versioned contract shared by training protocol hashing, reports, and dashboard validation. | Qualification remains provisional pending baseline variance review. There is no completed calibration bundle, held-out report, confidence interval on match outcomes, or primary success gate. The live supervisor's qualification protocol ID is null. |
| **7. A/B experiments** | Matrix runs record paired policy rows and bind inputs to schema-4 reports. The new summarizer enforces seed/category/stratum and policy completeness and calculates paired bootstrap intervals; Phase 4 factor completeness is checked. | The analyzer has no experiment results to summarize. The listed A/B experiments, K=8/16/32 candidate recall, P=0 ablation, matched prediction and inference-disable comparisons have not been completed as a current results bundle. |
| **8. Implementation boundaries and order** | Candidate generation, model adapters, controls, evaluation, replay capture, provenance, deployment interfaces, and the three-pass dashboard are present in the implementation candidate. | A frozen baseline/fixture evidence package, complete matched five-track bundle, representative replay, qualification, and promotion evidence remain absent. |
| **9. Unresolved design decisions** | The code exposes candidate K, prediction mode, graph support, penalty mode, profiling, and replay/provenance fields needed to measure several open questions. | Legal graph/bin adequacy, candidate recall, route evaluation cost, prediction calibration/occlusion, held-out generalization, Hailo support, latency, and simulator shortcut resistance remain measurement questions. No result should be inferred from current code or synthetic tests. |

The annotation materializer publishes staged dataset and manifest files with rollback for caught staging/replacement/fsync failures and rejects input/output path aliases. Its two-file rename sequence is not crash-atomic; the loader rejects a mixed pair when the dataset hash differs, and still validates the other provenance fields. No annotation data has been materialized.

## Three-pass evidence and validation

The dashboard's structured evidence validator is at `docs/MECO_THREE_PASS_PROVENANCE.json`. It validates hashes, paths, role records, commit formats, reviewer verdicts, and per-tranche limits, and the UI separates workflow pass status from agent liveness. The manifest covers eight selected tranches and is explicitly classified as `scoped-transcript-summaries`: original reviewer/planner reports for earlier tranches were not persisted. This is not whole-plan approval, original-transcript evidence, runtime validation, or proof that any agent is currently active.

On the integrated candidate at `0e133780a`, the focused suites for curriculum and MECO dashboards, replay integrity, async inference broker, transition history, evaluation binding, deployment contracts, Phase 4 exposure, penalty provenance, paired CI analysis, three-pass provenance, and Plan A annotation materialization passed **156 tests and 5 mutation subtests**. Static Python compilation, JavaScript syntax checks, protocol/provenance JSON parsing, and `git diff --check` passed. The curriculum dashboard fixtures now use the active curriculum version (`2`) instead of a stale hard-coded version (`1`). These are unit/contract checks; no test in this validation run exercised end-to-end simulated game behavior, so no simulator evaluation was run.

**Simulator validation:** not run for this candidate. Scenario, seed, and replay identifier: none.
**Dashboard currentness:** not verified for this candidate; no fresh replay was generated or reviewed.
**Qualification/deployment:** no qualified model, Hailo result, or deployment claim.

## Next evidence gates

1. Obtain human-authored Plan A goal annotations and an independent value oracle; materialize only explicitly supplied labels.
2. Train artifacts under the exact source identity intended for evaluation. The active `d7b8dd4b1` campaign remains separate and uninterrupted.
3. Complete baseline variance calibration and freeze the matched held-out protocol.
4. Run the full paired five-track/control/ablation matrix, summarize paired intervals, and inspect a fresh seeded simulator replay after restarting and verifying the candidate dashboard backend and frontend.
5. Run the Phase 4 adversarial/generalization scenarios and Phase 5 export, Hailo, memory, and end-to-end latency measurements where the actual toolchain/device is available.
6. Apply the predeclared qualification and promotion gates only after the required outcome and safety evidence exists.
