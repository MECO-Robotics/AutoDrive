# MECO learning tracks plan: execution status

> **Current campaign update — 2026-10-24 04:28:14 UTC.** A fresh continuous
> run is active from the current checkout `main` at `2ecc067b780a693088ec0992df70560806485628`,
> source identity `31e5f85370c3ad83d0b68392de59ec3b525828d2045e86b98aedf309cd8f1350`.
> Supervisor `5e8f956474594bf8b69417b39b4692fa` writes to this checkout's
> `checkpoints/meco-five` and `evaluations/meco-five`, uses `cuda:0,cuda:1`,
> evaluation device `cuda:1`, `--cycles 0`, and protocol
> `dec8964454769b01fa4d79fe5f1cb7919013d94b5076e3a99a20ba5a4181f6f2`.
> At the recorded heartbeat, A1 and A2 were alive at 2/100 updates on the two
> WX9100s; A3, C1, and C2 were queued; no chunk or evaluation had completed.
> This confirms launch only, not completed optimizer progress, evaluation, or
> qualification. The evaluation split is diagnostic, so this run does not by
> itself produce held-out qualification evidence. The current service on port
> 8765 was restarted and verified against this checkout: `/meco`, the status
> API, and both dashboard JavaScript assets returned HTTP 200; served asset
> hashes matched the checkout. Its data root now points to this checkout.
> The point-in-time evidence is [`MECO_CAMPAIGN_SNAPSHOT_20261024T042814Z.json`](MECO_CAMPAIGN_SNAPSHOT_20261024T042814Z.json),
> SHA-256 `3f98af6d40d16719aabdafb134fa7538649c6d7ff90d39d6a0e9753bbf7d94f5`.
> The source identity is recomputed from both live workers' `run.json` source
> manifests and matches the supervisor record; the native HIP extension's
> initialization state changes the result of a later fresh identity call.
> The earlier campaign's subsequent C2 update-20 failure and terminal state
> are hash-pinned in this snapshot. Runtime JSON and process handles remain
> authoritative for later status.

**Historical analysis snapshot:** 2026-10-24 03:53 UTC
**Plan:** [`MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`](MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md)
**Plan SHA-256:** `718ef9d76f5525bf8e4759d3a0a46790a3c978e2b146efec6436c89dbbb49b10`
**Current implementation checkout:** `main` at `2ecc067b780a693088ec0992df70560806485628`
**Prior implementation candidate at the historical snapshot:** `codex/meco-full-plan-integration` at `0e133780a0fadfb9015b84719e75291be89370e0`
**Prior campaign checkout:** `codex/meco-c1-replay-integrated` at `d7b8dd4b1c1814c2a368ded7604b3c887571fdcc`

This ledger separates code that exists from evidence that demonstrates runtime behavior. A passing contract test, completed optimizer update, or review of a scoped patch does not establish a simulator result, qualification, or deployment readiness.

## Previous campaign (historical)

The earlier continuous campaign was isolated in another checkout and failed at C2 update 20. The snapshot below records its state at heartbeat `2026-10-24T03:53:45Z`; A1, A2, A3, and C1 completed one 100-update chunk, while C2 failed before completing that chunk. An immutable compact observation is recorded in [`MECO_CAMPAIGN_SNAPSHOT_20261024T035345Z.json`](MECO_CAMPAIGN_SNAPSHOT_20261024T035345Z.json), SHA-256 `b13df6632b64ba5cf8f872c7c5b8fec45141762e9445ed49265ee7aed17ad524`. It is a historical point-in-time record, not a live status feed.

At the status heartbeat `2026-10-24T03:53:45Z`:

| Track | Device | Training updates | State |
|---|---|---:|---|
| A1 | `cuda:0` | 100/100 | First chunk complete |
| A2 | `cuda:1` | 100/100 | First chunk complete |
| A3 | `cuda:0` | 100/100 | First chunk complete |
| C1 | `cuda:1` | 100/100 | First chunk complete |
| C2 | `cuda:0` | 1/100 | Running |

At that heartbeat, the supervisor reported `completed_cycles: 0`, `completed_training_chunks: 4`, and `qualification_protocol_id: null`. No held-out evaluation had completed. These update counts are optimizer progress, not a 100-case evaluation or a pass. The campaign had no warm-start artifacts (`warm_start_artifact_sha256: {}`); it began from scratch. At that historical snapshot, training used source identity `c56cc4001441e10e72b191025a97141033907aed40f42c0f3b61f7bacab3e8e2`, while the separate candidate then had source identity `8f3d68c00e2beb52d0d188feaf40785bd79a91f04c5ac4c652a05379aa3d045a`.

The old campaign's checkpoints do not satisfy strict source identity for the current checkout and must not be used for current-source evaluation or promotion. No current learned-policy evaluation or replay ID is available; `20261024` is a training seed, not evidence of an evaluation. Dashboard code and assets are now verified current for the active run, but no fresh learned-policy replay has yet been generated or inspected.

## Completion ledger by plan section

| Section | Implemented | Evidence still required |
|---|---|---|
| **1. Decision and hypothesis** | The repository represents three Plan A spatial models and two Plan C candidate rankers, with deterministic comparisons and a shared safety boundary. | No matched, held-out result establishes that a learned architecture beats D0 or the matched candidate baseline. No winner or promotion is supported. |
| **2. Five models** | A1 U-Net+GRU, A2 structured SE(2) U-Net+regional attention+GRU, A3 hierarchical SE(2) GNN+regional attention+GRU, C1 candidate MLP+GRU, and C2 map encoder/candidate cross-attention+GRU are implemented. Parameter inventory and adapter shape checks exist. The inventory reports 171,717 parameters for A1, 122,311 for A2, 118,215 for A3, 49,026 for C1, and 124,162 for C2. | Counts and synthetic/adapter checks do not establish checkpoint-specific exported graphs, memory, Hailo compatibility, latency, or simulator performance. The current five-track frontier is incomplete. |
| **3. Deterministic and predictive controls** | D0/D1/D2 evaluation paths and prediction-mode controls exist. The schema-4 paired matrix analyzer validates protocol-bound reports and can calculate deterministic paired cluster-bootstrap intervals. | No current matrix report or interval exists for the requested architecture, control, prediction, penalty, candidate-recall, and fallback comparisons. D3 is not established. |
| **4. Shared observation and planner contract** | Shared raster/scalar inputs, candidate masks/order, stale-output fallback, legal action support, Plan A penalty provenance, evaluation-to-replay record binding, and generation-based replay publication are implemented. The async broker has deterministic fake-adapter contract coverage, including replacement, freshness, faults, and close; raised inference exceptions no longer kill its worker. No-route replay validation checks complete contiguous action coverage. | No fresh five-track learned-policy replay has been generated and inspected. Broker contract tests do not establish GPU-load freshness or simulator fallback. The new replay tooling does not prove P-shaped routes, robot behavior, collection/scoring, fallback behavior, or representative end-to-end safety. Global planning remains XY AD* with swept SE(2) checks, not full global SE(2). |
| **5. Training strategy** | Phase 0 protocol/configuration, Phase 1 interfaces, stochastic PPO support, Phase 4 train-split exposure scheduling, and the resumable two-device supervisor exist. Phase 2 collectors/loaders/fitters exist. A human-authored Plan A goal annotation materializer now packages explicit sidecar labels with source/legal/protocol/authority hashes; it invents no labels and marks replay validation as required. | No human-authored expert annotations, independent U oracle, generated annotation dataset, or warm-start artifact is available. The current campaign does not consume warm starts. Baseline variance calibration, a matched evaluation frontier, Phase 4 factor outcome runs, and candidate-source training remain outstanding. |
| **6. Metrics and decision gates** | Metrics, qualification protocol, confidence interval, calibration, and promotion interfaces exist. Plan A PPO penalty semantics use one versioned contract shared by training protocol hashing, reports, and dashboard validation. | Qualification remains provisional pending baseline variance review. There is no completed calibration bundle, held-out report, confidence interval on match outcomes, or primary success gate. The active supervisor records a protocol ID, but its evaluation split is diagnostic and has no bound evaluation protocol. |
| **7. A/B experiments** | Matrix runs record paired policy rows and bind inputs to schema-4 reports. The new summarizer enforces seed/category/stratum and policy completeness and calculates paired bootstrap intervals; Phase 4 factor completeness is checked. | The analyzer has no experiment results to summarize. The listed A/B experiments, K=8/16/32 candidate recall, P=0 ablation, matched prediction and inference-disable comparisons have not been completed as a current results bundle. |
| **8. Implementation boundaries and order** | Candidate generation, model adapters, controls, evaluation, replay capture, provenance, deployment interfaces, and the three-pass dashboard are present in the current checkout. | A frozen baseline/fixture evidence package, complete matched five-track bundle, representative replay, qualification, and promotion evidence remain absent. |
| **9. Unresolved design decisions** | The code exposes candidate K, prediction mode, graph support, penalty mode, profiling, and replay/provenance fields needed to measure several open questions. | Legal graph/bin adequacy, candidate recall, route evaluation cost, prediction calibration/occlusion, held-out generalization, Hailo support, latency, and simulator shortcut resistance remain measurement questions. No result should be inferred from current code or synthetic tests. |

The annotation materializer publishes staged dataset and manifest files with rollback for caught staging/replacement/fsync failures and rejects input/output path aliases. Its two-file rename sequence is not crash-atomic; the loader rejects a mixed pair when the dataset hash differs, and still validates the other provenance fields. No annotation data has been materialized.

## Three-pass evidence and validation

The dashboard's structured evidence validator is at `docs/MECO_THREE_PASS_PROVENANCE.json`. It validates hashes, paths, role records, commit formats, reviewer verdicts, and per-tranche limits, and the UI separates workflow pass status from agent liveness. The manifest covers eight selected tranches and is explicitly classified as `scoped-transcript-summaries`: original reviewer/planner reports for earlier tranches were not persisted. This is not whole-plan approval, original-transcript evidence, runtime validation, or proof that any agent is currently active.

On the integrated candidate at `0e133780a`, the focused suites for curriculum and MECO dashboards, replay integrity, async inference broker, transition history, evaluation binding, deployment contracts, Phase 4 exposure, penalty provenance, paired CI analysis, three-pass provenance, and Plan A annotation materialization passed **156 tests and 5 mutation subtests**. Static Python compilation, JavaScript syntax checks, protocol/provenance JSON parsing, and `git diff --check` passed. The curriculum dashboard fixtures now use the active curriculum version (`2`) instead of a stale hard-coded version (`1`). These are unit/contract checks; no test in this validation run exercised end-to-end simulated game behavior, so no simulator evaluation was run.

**Simulator validation:** not run for this candidate. Scenario, seed, and replay identifier: none.
**Dashboard currentness:** the current checkout's backend, frontend assets, and live data root were verified at 04:25 UTC as recorded in the pinned campaign snapshot; no fresh five-track learned-policy replay has been generated or reviewed.
**Qualification/deployment:** no qualified model, Hailo result, or deployment claim.

## Next evidence gates

1. Obtain human-authored Plan A goal annotations and an independent value oracle; materialize only explicitly supplied labels.
2. Continue the active current-checkout campaign and verify all five checkpoint manifests bind to one source identity. The earlier `d7b8dd4b1` campaign failed and remains historical.
3. Complete baseline variance calibration and freeze the matched held-out protocol.
4. Run the full paired five-track/control/ablation matrix and summarize paired intervals. Generate and inspect a fresh seeded simulator replay; restart and verify the dashboard again after any code changes before replay review.
5. Run the Phase 4 adversarial/generalization scenarios and Phase 5 export, Hailo, memory, and end-to-end latency measurements where the actual toolchain/device is available.
6. Apply the predeclared qualification and promotion gates only after the required outcome and safety evidence exists.
