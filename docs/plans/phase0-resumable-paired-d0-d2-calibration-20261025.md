# Phase 0 resumable paired D0/D1/D2 calibration implementation plan

**Status:** planner proposal only. No code, manifests, shards, simulator runs, training, or device use is performed by this plan.
**Planning base:** `2a13d84ec44a8d28e774af6fa0c48451556ffe82`.
**Source plan:** `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md` (SHA-256 `718ef9d76f5525bf8e4759d3a0a46790a3c978e2b146efec6436c89dbbb49b10`).
**Phase 0 plan:** `docs/plans/phase0-baseline-heldout-calibration-20261025.md` (SHA-256 `95e1b9185509259af952fab7fc956c12974d50aacf43525302275f859f917d49`).

## Goal and prerequisites

Implement one compatible, immutable, resumable campaign format for matched deterministic-control calibration. Extend the existing D0 shard coordinator and report contracts in `frc_defense/meco_qualification.py` and `scripts/meco_d0_calibration_campaign.py`; do not create a parallel incompatible format. Cover each control in the frozen preregistration: D0, D1/current, D2/cv, and D2/multi_model_v1 when that variant is preregistered. For each eligible control, execute the exact same frozen calibration work units and preserve matched pairing by tuple, seed, and repetition. Bind every result to the authoritative simulated-score ledger and reconstructable replay contract from Pass 1.

Implementation and offline contract work may proceed now. **Collection and execution remain closed** until all three prerequisites hold:

1. Pass 1's clean source/runtime freeze and disjoint four-way partition have been generated from actual effective configuration values, and the exact freeze and partition digests are available.
2. A human has approved the limited simulated outcome estimand using the canonical, source/rules-bound approval artifact required by `frc_defense/meco_simulated_score_ledger.py`. No such artifact is currently present. If the human-approved scope adds currently excluded outcomes, revise and review the scorer rules/ledger completeness contract before collecting any rows.
3. The user has manually stopped the diagnostic r4 run, and both WX9100 devices have actually been released. A stale checkout, lease/status file, or reported run state does not satisfy device release. Do not stop or alter r4 on the user's behalf.

This pass creates calibration infrastructure, not an executed calibration, accepted margin, qualification protocol, training dataset, or qualification result.

## Source audit: current contracts and concrete gaps

- `frc_defense/meco_qualification.py::_d0_calibration_shard_manifests`, `_validate_d0_shard_manifest`, `run_d0_calibration_shard`, `_validate_d0_shard_artifact`, and `_merged_d0_calibration_report` implement immutable D0-only manifests, deterministic disjoint assignment, resumable partial artifacts, and strict merge. The present manifest binds scenario-specific schedule/protocol, source identity, indexed runtime device, and runtime identity, but does not bind Pass 1's generated baseline freeze/partition artifact digests, human estimand approval, score rules/ledger version, or replay-bound score event digest. D0 rows currently validate throughput and safety metrics, not the primary simulated match outcome.
- `scripts/meco_d0_calibration_campaign.py` orchestrates full 12-scenario D0 planning, resume, and merge. Its `plan_campaign` defaults to `device="cuda:0"` and creates manifests through the D0-only contract. `describe_campaign` is explicitly offline arithmetic, has no serialized seeds or source/runtime identity, and is not executable evidence. Its interface is not yet a paired multi-control campaign.
- `frc_defense/meco_qualification.py::run_matched_reference_variance_calibration` evaluates one scenario sequentially in a single process for one variant, paired internally with D0. `MATCHED_REFERENCE_VARIANTS` supports D1/current, D2/cv, and D2/multi_model_v1; it currently emits throughput and three safety rates. It has no immutable campaign coordinator, shards, resume, or strict multi-variant merge, and its records do not require score-ledger/replay binding.
- `frc_defense/meco_simulated_score_ledger.py` defines `meco-simulated-score-ledger-v1`, rules `meco_simulated_fuel_score_v1`, outcome `simulated_match_score`, source/rules-bound human approval, and replay reconstruction/record validation helpers. Its current limited rules count active HUB FUEL score awards; they do not represent official match points. Current evaluated outcome collection remains blocked pending human approval and seeded replay/E2E evidence.
- Current runtime identity helpers accept indexed CUDA/HIP accelerator identities, but they do not prove that two selected indices are the two approved WX9100s for this campaign. A new calibration plan must bind an explicit index per worker to verified device model and stable device identity, and fail closed on CPU, P600, unindexed/automatic selection, or mixed runtime.
- The existing D0 stored shards and `pickup_simple` qualification artifacts predate these bindings. They are historical evidence only and must not be upgraded or merged into the new outcome calibration by assumption.

## Implementation scope

### 1. Canonical campaign and full frozen schedule

Extend the existing coordinator with a paired-control campaign manifest whose canonical digest covers:

- all source, runtime, and protocol identities required by the Phase 0 freeze, including exact source commit/tree and dependency/runtime lock;
- exact baseline-freeze artifact and partition artifact paths, schemas, and byte digests;
- scorer ledger schema, rules ID and digest, event-stream/replay contract version, human-approved estimand artifact digest, and outcome name/unit;
- scenario registry closure and each exact case tuple: scenario, category, stratum, layout, defender family, start condition, sensor profile, seed, and repetition;
- the predeclared control set and exact predictor identity/configuration for each included variant;
- candidate K, route cost, action cadence, and all other D1/D2 matched candidate-generator and safety-floor settings;
- per-worker explicit accelerator index, verified WX9100 model, stable device UUID/identity, backend, framework/runtime versions, and normalized runtime identity.

Derive the workload by enumerating the current frozen calibration tuple set and repetition/seed schedule, then make disjoint deterministic shard assignments over that complete tuple/control matrix. Require the same tuple, seed, repetition, source freeze, event/scorer semantics, and matched D0 counterpart for each selected D1/D2 control. Keep all seed partition and no-overlap guarantees from `protocol_cases`; derive completeness from the frozen schedule, never a remembered row count. Do not create a final-test or held-out work unit in this calibration campaign.

Use a versioned compatible extension of the existing D0 manifest/artifact/report types. Preserve historical readers and artifacts; either add explicit new schema versions or an unambiguous campaign-type version, and reject old D0-only artifacts where the new bindings are required.

### 2. Immutable shards, resume, and failure handling

For each shard, store an immutable manifest and atomic progress/final artifact. The progress artifact must bind the exact manifest digest and contain only unique completed work-unit keys. Resume accepts existing bytes only after validating every campaign, schedule, freeze, partition, source, scorer, approval, runtime/device, control, and row identity field. A different byte for an existing immutable ID, duplicate/overlapping unit, missing completion, changed source/runtime, or changed approval/scoring binding is a hard error; never overwrite or relabel it.

Run a worker only on its explicitly assigned accelerator. At launch, independently verify the requested index resolves to an allowed WX9100 and the runtime identity equals the manifest. Do not use `cuda` auto-selection, CPU fallback, silent retry on another device, or wait indefinitely while holding a lease. Preserve per-shard device identity in all final records and the merged bundle.

### 3. Outcome, replay, and safety rows

Each evaluated case must produce an evaluation row bound to a fresh replay that contains the ordered score event stream and source/rules/approval digests. Reconstruct the limited `simulated_match_score` from replay events using the Pass 1 API and require exact agreement among reconstructed ledger, replay summary, evaluation row, terminal score counts, and shard record. Preserve event-stream digest, ledger digest, rules/schema identifiers, approval digest, replay ID/provenance, and source identity at each layer.

Report the primary simulated outcome separately from useful throughput, raw FUEL counts, and any other secondary metric. Preserve all three required safety rates: robot contact, field contact, and wall contact. Require finite values and exact safety schema/source binding for every policy row. Where D0 is the safety-floor control, preserve the existing semantics and compare its exact paired tuple rows; never pool or substitute another seed/repetition.

No artifact may name the outcome “official match points.” With current rules, tower awards, fouls, disqualification, ranking points, end-game bonus, and advancement points remain excluded and official points remain null.

### 4. Strict merge and paired-control summaries

Merge only one exact manifest and complete artifact per expected shard, control variant, and worker assignment. Validate disjoint/full work-unit coverage, full registered scenario closure, repeated-seed coverage, exact D0/control pairing, and identical freeze/partition/source/scorer/approval/runtime/protocol identities. Reject missing/extra controls, unregistered variants, duplicate or overlapping records, altered artifacts, replay/scorer disagreement, missing safety values, wrong device/model/backend, partition leakage, partial scenario closure, or any identity drift.

Produce immutable per-control bundles and a merged summary bound by stable digests. Summarize paired simulated-score differences by predeclared control and stratum, plus between-seed and within-seed variance, confidence intervals using a predeclared method, and all three safety-rate comparisons. Keep throughput and FUEL counts secondary. Report each control variant separately; do not choose the best observed comparator after inspecting outcomes. Margin selection remains a separate, human-authored, digest-bound review step after complete bundles are independently reviewed.

## Planned paths

- `frc_defense/meco_qualification.py` — compatible paired manifest, row/artifact validation, resumable runner, strict merge, and summary contracts.
- `scripts/meco_d0_calibration_campaign.py` — extend campaign planning/resume/merge orchestration to the frozen control set while retaining legacy CLI behavior for old D0 diagnostic use.
- `frc_defense/meco_simulated_score_ledger.py` — use the accepted Pass 1 replay/score verification API; change it only if the new artifacts reveal a specific contract gap, with scorer changes separately reviewed.
- `tests/test_meco_qualification.py` and/or focused new `tests/test_meco_paired_calibration_campaign.py` — offline schema, binding, immutable-resume, mutation, coverage, pairing, merge, and runtime identity contracts.
- `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md` and `docs/MECO_THREE_PASS_PROVENANCE.json` — after independently reviewed integration, report code support and all remaining execution gates without claiming completed calibration.

## Required offline contract coverage

Plan implementation tests for canonical manifest/artifact digests; exact tuple/repetition completeness and cross-control pairing; all preregistered control variants; source/freeze/partition/approval/scorer/replay/runtime/device binding; immutable resume and stable retry; and rejection of missing/extra/duplicate/overlapping work, altered bytes, wrong variant, partial scenario closure, leaked partition tuples, changed score events, replay/evaluation/terminal-score disagreement, malformed safety metrics, CPU/P600/unindexed device selection, mixed backend/version, wrong WX9100 identity, and missing human approval. These are planned checks only; this planning task does not run them.

If implementation tests exercise simulated game/scorer behavior, `AGENTS.md` additionally requires a representative seeded end-to-end simulator evaluation that exercises the changed path and inspection of its record or replay. Before reviewing a fresh dashboard replay, restart the serving dashboard services on the integrated checkout and verify current backend, frontend assets, APIs, and replay provenance. Report test names, simulator scenario/seed/result, replay ID, and dashboard currentness, or explicitly state each gate is unperformed. Unit tests alone cannot establish simulator behavior.

## Acceptance criteria

- One compatible, resumable campaign contract covers D0 and every predeclared eligible D1/D2 control over the exact frozen tuple × seed × repetition schedule.
- Baseline freeze, partition, source/protocol/runtime, scorer rules, human-approved estimand, replay, predictor/candidate settings, all safety metrics, and explicit WX9100 identity are digest-bound end to end.
- Shard resume is immutable and exact; merge rejects any drift, missing/duplicate/overlapping work, excluded partition tuple, or replay/outcome disagreement.
- Primary outcome and safety metrics remain distinct from throughput/FUEL counts; no official-score or qualification claim is introduced.
- Offline checks and any required simulator/current-dashboard evidence are reported separately. Before execution, the human estimand approval exists and the user has manually stopped r4 with both WX9100 devices released. No CPU/P600 or automatic device selection is permitted.
- The integrated status/provenance update describes implementation and evidence accurately, leaves margin approval and Phase 0 completion open until their independent gates pass, and does not reuse legacy pickup-simple or D0-only artifacts as complete evidence.

## Out of scope

This tranche does not produce the effective freeze or partition, approve the estimand, run calibration/simulation/training, access devices, generate replay evidence, choose operational margins, create qualification-v5, touch final-test data, or claim qualification. Those are separately gated follow-on work.

