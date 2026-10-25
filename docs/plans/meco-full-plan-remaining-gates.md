# MECO evaluation plan: remaining implementation and evidence gates

Audit basis: source plan `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md` §§1–9; current status `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md` completion ledger and next evidence gates; source at main `bb2498de795217129bf7ec4130a21a6dde55e6c2`. This is a scoped implementation plan, not a completion or qualification claim.

## Current campaign boundary

The read-only r4 supervisor snapshot at `AutoDriveCampaigns/meco-five-continuous-20261025-r4/supervisor-status.json` (SHA-256 `1f8ad79f56e960e5b4bdc19f179b7532fddf885455bec3867ce53fcaba528d8f`) has heartbeat `2026-10-25T04:09:32.852960Z`, cycle 1, zero complete cycles/evaluations, four lanes at 100/100, and C2 at 86/100. This digest makes the external status snapshot used for this audit reproducible. It is `diagnostic`, its evaluation binding/protocol ID is null, its exposure is only `pickup_simple/cluster`, and its warm-start hash map is empty. Leave this user-requested uninterrupted run and its checkpoints alone. It cannot be retroactively made a matched held-out or Phase 2 run. Any future run must wait for its device leases to be released, use a newly validated manifest and output root, and preserve r4 only as unqualified ancestry.

**No new model architecture is needed before running five lanes:** A1/A2/A3/C1/C2 code and the two-device supervisor exist, and r4 is already exercising all five. A complete Plan 2 warm start is a prerequisite to claiming the next campaign is warm-started, not a technical prerequisite to launching a fresh diagnostic run. No fresh campaign may start while r4 owns the WX9100 pair.

## Requirement-by-requirement status

| Source plan requirement | Current state | Remaining work/evidence |
|---|---|---|
| §1 research decision | Five architectures and controls are represented. | No matched held-out outcomes, confidence intervals, or winner. Decide only from §6 gates. |
| §2 A1/A2/A3/C1/C2 | Model constructors, adapters and parameter inventory exist. | No source-bound Phase 2 datasets or fitted warm starts. Runtime candidate recall, Plan A P-route effect and A/C seeded behavior remain unverified. |
| §3 D0/D1/D2 controls | Evaluation paths and prediction modes exist; D3 is optional/unestablished. | Calibration is only recorded for `pickup_simple`; no complete matched control matrix or result report. Keep D1 on exactly the same candidate set, route estimates, feasibility and risk as C1/C2. |
| §4 shared interface/safety | Versioned async inference, masks, freshness, fallback and replay provenance contracts exist. | No fresh learned-policy replay was reviewed; fake-adapter tests do not establish simulator safety/fallback or loaded-GPU freshness. Global planning remains XY AD* plus swept SE(2) checks, not global SE(2). |
| §5 Phase 0 baseline/protocol | Qualification, calibration and manifest machinery exists. | Checked-in protocol is provisional v1; no reviewed v4 operational margins or complete D0/matched-reference calibration bundle. Freeze train/validation/test partitions and baseline before comparative runs. |
| §5 Phase 1 interfaces | Scripted/model interfaces and D1 candidate path exist. | Required behavior validation still needs seeded end-to-end simulator scenarios with inspected replay; contract tests alone do not pass it. |
| §5 Phase 2 supervised warm start | C repeated-label audit/collector/fitter and A U/B/P collectors/fitter/validators exist. | No audited/validated C corpus or fitted warm starts are available; A replay-bound U/B data and an independent P oracle dataset also remain unavailable. Build and validate one resumable shared-source campaign and complete-set seal (see tranche 1). |
| §5 Phase 3 controlled RL | PPO and executed-action/log-probability support exist. | r4 has no completed matched cycle/evaluation and has no warm starts. Run fresh matched five-track training under one source/config frontier; prove each policy's executed distribution, reward/risk and checkpoint ancestry. |
| §5 Phase 4 generalization | Factor registry, schedule plumbing and some stimuli/outcome instrumentation exist. | `MECO_PHASE4_RUNTIME_STATUS.md` reports zero executable qualification cells. Wire the remaining registered stimuli and authoritative outcome signals, then run held-out policy families, starts, teammate, perception, abrupt-change and contact scenarios. Diagnostic wiring is not evidence. |
| §5 Phase 5 deployment | Export/profile/parity scripts and synthetic CPU ONNX graph tests exist. | No trained-checkpoint production-capture parity, Hailo compiler/operator/device result, memory fit, or end-to-end route-to-actuation latency. Actual toolchain/device access is a later external gate. |
| §6 decision gates | Calibration, paired-CI, metric, promotion and final-test ledger code exists. | No learned-policy held-out report, operational margin review, safety outcome bundle or primary gate result; no architecture qualifies today. |
| §7 A/B experiments | Matrix runner/analyzer and paired bootstrap code exist. | None of the listed comparisons has a current complete results bundle: A1/C2; C1/C2; A1/A2/A3; C1/D1; D1/D2; matched prediction; P=0; K=8/16/32 recall; inference-disable fallback. |
| §8 implementation order | Most interfaces, controls, manifest/supervisor and replay plumbing exist. | Complete source-bound warm-start path, then matched simulator evidence and the experiment matrix; do not treat scoped implementation summaries as whole-plan evidence. |
| §9 open design questions | Configurable K/prediction/P modes and profiling hooks exist. | Measure legal graph/bin adequacy, candidate recall, route cost, uncertainty/occlusion calibration, shortcut resistance and target-hardware support; code presence cannot answer these. |

## Prioritized scoped tranches

### 1. Phase 2 five-track source campaign and warm-start binding — highest code leverage

Implement a stage-journaled orchestrator around existing `frc_defense/meco_phase2_campaign.py::execute_campaign` (C1/C2 repeated-label dataset + weights) and `frc_defense/meco_plan_a_ubp.py::orchestrate_plan_a_ubp` (A1/A2/A3 U+B plus independent P labels from that exact dataset). Bind exact train seeds, qualification protocol ID and bytes, source identity, dataset/manifest digests, five artifact digests and validator summaries in one atomic complete-set seal. Restart must revalidate completed stages; interrupted invalid stages retry in fresh attempt directories; never overwrite/reuse unvalidated partial outputs. Export exactly five paths/hashes to `meco_training_manifest.describe_campaign`.

**Integration blocker to resolve in this tranche:** manifest launch in `frc_defense/meco_supervisor.py::main` validates `manifest["warm_starts"]` but the manifest-mode `MecoSupervisor(...)` call currently omits `warm_start_artifacts`; only ordinary CLI mode forwards that argument. Pass the exact five manifest paths after hash verification and test that missing/partial/mutated sets fail before lease acquisition. Otherwise a future manifest may describe artifacts without workers consuming them.

**Acceptance:** adapter-injected tests for uninterrupted/restart-after-each-stage, dataset/artifact/journal tampering, missing/swapped tracks, train/test seed leakage, protocol/source mismatch and manifest export; existing `tests/test_meco_phase2_campaign.py`, `tests/test_meco_plan_a_ubp_orchestration.py`, `tests/test_meco_training_manifest.py`, and new orchestration/supervisor-launch tests pass. No simulator is needed for code acceptance; real dataset collection/fitting is a separate GPU/simulator gate after r4 exits.

### 2. Clean next-campaign launch gate after r4

Rebuild and validate a fresh manifest against the chosen clean integrated source, two WX9100 inventory, qualification protocol, evaluation protocol/schedule, five complete warm starts if available, and new output roots. Preserve separate training-qualification and evaluation protocol IDs. Never resume r4 under changed source or upgrade its diagnostic records. At launch, record all five source/config/protocol/warm-start hashes and verify exact checkpoint ancestry. This is a runtime/device gate, not a code-only pass.

### 3. Seeded Phase 1/A-C runtime acceptance before broad training claims

Run small reproducible simulator cases that exercise Plan A P-route costs (P=0 vs bound P), Plan C feasible candidate ranking against same-table D1, candidate K/recall, action remasking/preemption, disabled/late/faulted inference deterministic fallback, and representative pickup/score/ferry behavior. Inspect the simulator record and action traces, not just aggregate status. Retain scenario, seed, run/replay ID and source/checkpoint hashes.

For each simulation behavior test, follow project `AGENTS.md`: restart dashboard backend after source changes; verify backend APIs and current frontend asset hashes; generate a fresh replay from that process and inspect it. Old-code/cached replays do not satisfy this gate.

### 4. Phase 0 calibration and frozen validation protocol

Run complete D0 and matched-reference variance calibration over the intended scenario closure, validate merged immutable bundles, have operational margins reviewed, and derive the target-scoped promotable v4 protocol. Preserve independent training, validation, and final-test assignments. The current `pickup_simple` calibration alone is insufficient; provisional v1 is non-promotable. Use `frc_defense/meco_qualification.py`, `scripts/meco_d0_calibration_campaign.py`, and `docs/MECO_QUALIFICATION.md`.

### 5. Matched five-track/control/ablation evaluation

Once all five checkpoints share the same source frontier, run `frc_defense/meco_validation_campaign_runner.py` / `scripts/run_meco_validation_campaign.py` and `frc_defense/meco_ablation_matrix.py` / `scripts/run_meco_ablation_matrix.py` with exact paired seeds, D0/D1/D2, matched prediction, A penalty P0, C K=8/16/32 and oracle-recall checks, and inference-disable controls. Reduce via `scripts/summarize_meco_ablation_matrix.py`; require exact policy/case completeness, safety metrics, calibrated improvement margins and paired confidence intervals. No result exists today.

### 6. Phase 4 evidence completion

Use `frc_defense/meco_phase4_campaign_plan.py`, `meco_phase4_runtime_adapter.py`, `meco_phase4_opponents.py`, evaluator and matrix reducers to close every registry row in `docs/MECO_PHASE4_RUNTIME_STATUS.md`. Before qualification, each factor needs an executable stimulus and authoritative required signal (e.g. blockage/interception/recovery, defender-family outcome, legal start and teammate outcome, sensor/relocalization reaction, contact-caused recovery). A planned or realized stimulus without the required outcome remains ineligible. Run paired held-out policy-family scenarios and inspect fresh records/replays.

### 7. Phase 5 deployment and final qualification

Export actual trained checkpoints, verify captured production-input ONNX parity, profile full feature→route→inference→arbitration timing and memory, and exercise actual Hailo compiler/device when available. Synthetic scorer profiling is not deployment evidence. Only after calibration, complete held-out matched outcomes, Phase 4 signals, safety and latency gates should the selected candidate consume the protocol-scoped final-test ledger and be considered for promotion.

## Run/validation ordering and stop criteria

- Keep r4 uninterrupted. Do not acquire its GPUs or alter its run/checkpoints.
- Code-only tranches use offline adapters/mocks; they do not start a campaign or satisfy simulator behavior requirements.
- First real compute after r4 is released: Phase 2 shared dataset + warm starts, then seeded behavior smoke checks. If warm starts are unavailable, a fresh diagnostic run may proceed only with that explicit limitation; do not call it Phase 2 complete.
- Pause comparative interpretation if source/config/protocol hashes diverge, any track lacks the matched seed frontier, a replay fails provenance/currentness checks, or a required factor signal is unavailable.
- No promotion unless the predeclared calibrated primary gate and safety/reliability gates pass; reserve final-test seeds for a single authorized evaluation round.

## Evidence references

- Plan details: `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md` §§2–9.
- Current completion ledger and stated next gates: `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md` §§“Completion ledger by plan section” and “Next evidence gates”.
- Phase 4 executable-cell status: `docs/MECO_PHASE4_RUNTIME_STATUS.md`.
- Calibration/protocol workflow: `docs/MECO_QUALIFICATION.md`.
- Deployment evidence limits: `docs/MECO_DEPLOYMENT_PROFILE.md`.
- Next-run manifest/evidence state: `docs/MECO_NEXT_FIVE_TRACK_TRAINING_HANDOFF.md` (revalidate its pinned source before launch).
