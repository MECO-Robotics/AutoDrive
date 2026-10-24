# MECO learning tracks plan: execution status

> **Current campaign update — 2026-10-24 04:49:02 UTC.** A fresh continuous
> run was launched from `main` at `2ecc067b780a693088ec0992df70560806485628`; current HEAD
> `45ec06796` follows documentation and calibration-manifest commits only,
> source identity `31e5f85370c3ad83d0b68392de59ec3b525828d2045e86b98aedf309cd8f1350`.
> Supervisor `5e8f956474594bf8b69417b39b4692fa` writes to this checkout's
> `checkpoints/meco-five` and `evaluations/meco-five`, uses `cuda:0,cuda:1`,
> evaluation device `cuda:1`, `--cycles 0`, and protocol
> `dec8964454769b01fa4d79fe5f1cb7919013d94b5076e3a99a20ba5a4181f6f2`.
> At the recorded heartbeat, A1 and A2 were alive at 30/100 and 27/100 updates
> on the two WX9100s; A3, C1, and C2 were queued; no chunk or evaluation had
> completed. D0 shard indices 0 and 1 each completed 150 entries on `cuda:1`;
> index 2 had 139/150 entries on `cuda:0`, and index 3 had just been launched
> on `cuda:1`. The two-shard result is partial calibration progress, not a
> merged calibration report or qualification.
> This confirms launch only, not completed optimizer progress, evaluation, or
> qualification. The evaluation split is diagnostic, so this run does not by
> itself produce held-out qualification evidence. At capture, the dashboard
> process on port 8765 served this checkout's data root; `/meco`, the status
> API, and both JavaScript assets returned HTTP 200, and served asset hashes
> matched the checkout. The named systemd unit was absent, so process ownership
> and listeners were verified directly. The point-in-time evidence is
> [`MECO_CAMPAIGN_SNAPSHOT_20261024T0448Z.json`](MECO_CAMPAIGN_SNAPSHOT_20261024T0448Z.json),
> SHA-256 `02247f858a2276750cf2c4352f703de33b8827f96b719d87847b08a6a10bbe78`.
> The source identity is recomputed from both live workers' `run.json` source
> manifests and matches the supervisor record; the native HIP extension's
> initialization state changes the result of a later fresh identity call.
> The earlier campaign's subsequent C2 update-20 failure and terminal state
> are hash-pinned in this snapshot. Runtime JSON and process handles remain
> authoritative for later status.

> **D0 calibration milestone — 2026-10-24.** The merged `pickup_simple` D0
> calibration report is `qualifications/meco/d0-calibration-pickup_simple.json`
> (SHA-256 `eaf39d5df985c854b157f4abf9fc48a66517cd4864d4486006d0681dda7f76a6`,
> calibration ID `1be58a36b643f3ed48fc661e0d7454007b8f0f5fb2eb5f9a9bf380c9c4c48030`);
> its bundle is `qualifications/meco/d0-calibration-pickup_simple-bundle.json`
> (SHA-256 `0ba1f570c352d7167ab94c4f84f22706865c6e5da38afdc77a160a2c85087b5a`).
> Scoped planner, implementer, and reviewer evidence covers eight 150-entry shards,
> exact manifest/artifact digests, source identity
> `31e5f85370c3ad83d0b68392de59ec3b525828d2045e86b98aedf309cd8f1350`, normalized
> HIP/WX9100 runtime, and 1,200 records across four strata (100 unique seeds ×
> repetitions 0, 1, 2), with required safety metrics and source binding verified.
> This is D0 baseline calibration only; it does not qualify a learned policy.
> This milestone does not record live training progress; consult the active
> campaign runtime for current status. Target v2 still needs explicit
> operational margins and review; learned-policy held-out report, CI, and primary
> gate are absent. No margin approval or held-out result is asserted.
>
> **Dashboard/training point-in-time capture — 2026-10-24 05:19:49 UTC.** Snapshot
> [`MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T051949Z.json`](MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T051949Z.json),
> SHA-256 `81991d6d429bcb7300e5ccffe967d51e9ecae5f22f01a9fc29190df717566006`. It records the serving checkout, dashboard/API responses and asset
> equality, plus two active training tracks and three queued. Reads were not atomic;
> this is historical evidence, and current status must be read from runtime.

> **Live status refresh — 2026-10-24 06:02:43 UTC.** Snapshot
> [`MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T060243Z.json`](MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T060243Z.json),
> SHA-256 `c41fbdca73b18c77d09c71c6a2709a0a4f466cd7e2d23461dff8c7c04fdaa51b`.
> The supervisor is healthy on `main` at source identity
> `31e5f85370c3ad83d0b68392de59ec3b525828d2045e86b98aedf309cd8f1350`:
> A1 and A2 completed 100/100 updates, A3 is 29/100, C1 is 41/100, and C2
> remains queued. A3 and C1 are confirmed on separate WX9100 GPUs. All four
> dashboard endpoints returned HTTP 200 and the served frontend asset matches
> this checkout. The split is diagnostic and has no held-out protocol binding;
> these are training updates, not qualification results. Reads were
> non-atomic; the snapshot pins each captured runtime file.

> **Live status refresh — 2026-10-24 06:23:13 UTC.** Snapshot
> [`MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T062313Z.json`](MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T062313Z.json),
> SHA-256 `f4f2b4e30faa315333ca1d1783834a50bfb6fb0f6b5283eb186854506ab54b82`.
> The pinned supervisor is still training from `main` at source identity
> `31e5f85370c3ad83d0b68392de59ec3b525828d2045e86b98aedf309cd8f1350`:
> A1 and A2 completed 100/100 updates, A3 is 54/100, C1 is 85/100, and C2 is
> queued. The active run reports two completed training chunks and no error.
> Dashboard page, MECO API, curriculum API, and JavaScript asset all returned
> HTTP 200; the served JavaScript hash matches this checkout. Evaluation remains
> diagnostic with no bound held-out protocol, so update counts are not scores
> or passes. Reads were non-atomic; the snapshot pins the captured runtime
> files and response bodies.

> **Training interruption and recovery — 2026-10-24 06:26:48 UTC.** Snapshot
> [`MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T062648Z.json`](MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T062648Z.json),
> SHA-256 `09ec0c981ceffec1800bf6142f842021857d82708270ac96da1ad5631dfe77b9`.
> A merge command was accidentally run in the live checkout while preparing the
> isolated integration candidate. The supervisor's source guard stopped A3 and
> C1 at their last saved checkpoints (56 and 88 updates) and aborted queued C2;
> A1/A2 remained complete at 100. I reverted the source change and confirmed the
> saved checkpoint file hashes match the restored source files and protocol. The
> same continuous two-device supervisor configuration was relaunched as PID
> `1523650`; at capture it had resumed A3 at 56 and C1 at 88, with C2 queued and
> no supervisor error. This recovery is recorded explicitly; the run was not
> uninterrupted. Dashboard endpoints returned HTTP 200 and the served asset
> matched the checkout. Reads were non-atomic; see pinned files and response
> digests in the snapshot.

> **Hardware availability check — 2026-10-24 06:32 UTC.** `lspci` reports the
> two AMD/ATI Vega 10 XT WX9100 devices used for training. Neither `hailortcli`
> nor a `/dev/hailo*` device is available on this host, so Hailo compilation,
> accelerator execution, latency, and memory evidence remain unavailable here.
> At the 06:33 supervisor heartbeat the campaign was healthy: A1/A2/C1 were at
> 100 updates, A3 at 65, and C2 queued behind A3 on `cuda:0`. The paired
> diagnostic evaluation had not started; these updates are not qualification.

> **Phase 5 production-input parity tranche — 2026-10-24.** Isolated branch
> `codex/meco-production-input-parity`, commit
> `b098d725b7c2c2da90ddd6b228989bfc27d949ac`, passed scoped reviewer re-review.
> It adds opt-in bounded production-request capture and binds the exact captured
> tensors, checkpoint/protocol/source provenance, ONNX graph, and parity report;
> capture loading preflights size limits and validates request metadata. No tests,
> simulator, or model workloads were run for this tranche. Integration candidate
> `codex/meco-production-input-parity-integration` at `eebd4ae57b9f226dade8bac99a6734596381759e`
> merges the approved branch into main `4d93b2f97`; a second scoped review
> approved the clean merge. It is ready to fast-forward after training is manually
> stopped and the final worker/checkpoint source-binding audit passes. The
> representative seeded simulator capture/replay and dashboard restart/currentness
> check remain outstanding.

**Historical analysis snapshot:** 2026-10-24 03:53 UTC
**Plan:** [`MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`](MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md)
**Plan SHA-256:** `718ef9d76f5525bf8e4759d3a0a46790a3c978e2b146efec6436c89dbbb49b10`
**Current implementation checkout at the historical analysis snapshot:** `main` at `2ecc067b780a693088ec0992df70560806485628`
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
| **6. Metrics and decision gates** | Metrics, qualification protocol, confidence interval, calibration, and promotion interfaces exist. Plan A PPO penalty semantics use one versioned contract shared by training protocol hashing, reports, and dashboard validation. | The D0 baseline calibration report and bundle are complete for `pickup_simple` (see milestone above); this is not learned-policy qualification. Target v2 still requires explicit operational margins and review. The learned-policy held-out report, confidence interval on match outcomes, and primary success gate are absent. The active supervisor records a protocol ID, but its evaluation split is diagnostic and has no bound evaluation protocol. |
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
