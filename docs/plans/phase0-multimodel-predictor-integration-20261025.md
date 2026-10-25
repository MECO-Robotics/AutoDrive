# Phase 0 D2 multi-model predictor integration plan

## Objective

Integrate the already specified `D2/multi_model_v1` predictor into the current `main` source tree so the five-track evaluator and paired calibration can select and identify it deterministically. Preserve current D1/D2 defaults, existing checkpoint identities, the approved qualification protocol, and the live training run. This is implementation planning only: no predictor code, simulation, GPU, dashboard, or training changes are made by this plan.

This tranche makes the predictor executable and provenance-bindable. It does not claim predictor calibration against real held-out opponent trajectories, qualification, improved match outcomes, or completion of the broader learning-track plan.

## Authoritative source audit

The audited candidate implementation is in local branch `codex/meco-d2-multimodel-v1`, worktree `autodrive-meco-d2-multimodel-v1`:

- `b32597c26e87a2523d2929566cec4b100a6571ab` (`Add bounded multimodel opponent prediction mode`) defines and wires the predictor. Its exact patch changes `frc_defense/meco_tracks.py`, `frc_defense/meco_candidates.py`, `frc_defense/meco_track_training.py`, `tests/test_meco_multimodel_prediction.py`, `docs/MECO_PREDICTION_MODE.md`, and `docs/LEARNING_TRACKS.md`.
- `0b41d4db69cf1711e7a64f550051f26807528fb1` (`Add paired multimodel control matrix condition`) adds per-control mode selection, the matched D1/current vs D2/multi_model_v1 matrix condition, and its reporting/tests. It changes `frc_defense/meco_ablation_matrix.py`, `frc_defense/meco_track_training.py`, `tests/test_meco_multimodel_prediction.py`, and `docs/MECO_ABLATION_MATRIX.md`.
- The feature commits' common source base is `b23b9197a0553b7ac74266d8d44d0b070758e721`. That is not current `main`; the candidate branch also contains large unrelated historical changes. Do not merge that branch or cherry-pick its broad history. Port the two feature commits' semantics selectively onto current `main` at implementation time and preserve later mainline behavior.

The candidate's predictor identity is `meco_opponent_kinematics_mm_v1`, mode `multi_model_v1`, with four deterministic hypotheses: constant velocity 0.40, left turn 0.15, right turn 0.15, and exponentially damped velocity 0.30. It caps speed at 3 m/s, valid track age at 0.5 s, prediction horizon at 2.5 s, turn-rate magnitude at 0.6–1.5 rad/s (using observed angular velocity when available, otherwise 0.6), and damping at 0.8/s. Turn directions remain separate hypotheses. Candidate/node risk is the weighted expected proximity penalty across hypothesis endpoints. Stale or invalid motion falls back to finite observed pose; missing/nonfinite pose contributes no prediction risk. This is a deterministic kinematic diagnostic, not an intent, route-feasibility, collision, or calibrated uncertainty model.

The second candidate source, local branch `codex/meco-warmstart-readiness-preflight`, contains:

- `0e7b60506554d55e80aeb239853170f1308a2f79` (`Add five-track warm-start readiness preflight`), adding `frc_defense/meco_warmstart_preflight.py`, `tests/test_meco_warmstart_preflight.py`, and documentation in `docs/MECO_SUPERVISED_WARMSTART.md`.
- `13782e136a81223a36bfd5ab605b1469a6d735f2` (`Correct warm-start preflight prediction mode docs`), correcting the same warm-start documentation.

The preflight reads `PREDICTION_MODES` and `prediction_contract()` dynamically and verifies the same mode/contract on bundle, artifact, dataset manifest, and established warm-start validation. It has no special-case multi-model algorithm. The source branch is also historical and divergent from current `main`; do not merge it wholesale. Port the preflight as a separate, reviewable dependent change only if the implementation audit confirms it remains absent and it is needed to validate five-track warm-start artifacts. Its primary dependencies are the current `meco_tracks` prediction contract and current warm-start artifact validator, not the old source branch's unrelated changes.

## Current-main integration points and dependencies

The implementation agent must re-audit these paths at the exact implementation base before editing, since current `main` may have moved:

- `frc_defense/meco_tracks.py`: `PREDICTION_MODES`, `prediction_contract`, `opponent_forecast`, and `opponent_risk_penalty` are the canonical scalar contract.
- `frc_defense/meco_candidates.py`: route-feasible candidate construction and per-route-ETA forecasts serve C1/C2 and D1/D2.
- `frc_defense/meco_track_training.py`: learned A/C prediction-mode resolution, D1/D2 control mode resolution, Plan A node-risk path, checkpoint/run/evaluation provenance, and CLI behavior must use the canonical contract.
- `frc_defense/meco_ablation_matrix.py`: condition identity, per-control overrides, paired D1/current vs D2/multi_model_v1 execution, and contrast provenance.
- `frc_defense/meco_qualification.py`: `MATCHED_REFERENCE_VARIANTS` already names `D2/multi_model_v1`, but `_matched_reference_identity()` cannot produce the contract until `prediction_contract("multi_model_v1")` exists. The paired calibration manifest must bind the full canonical predictor contract and its canonical digest, not only the mode string or policy label.
- Existing `docs/MECO_PREDICTION_MODE.md`, `docs/MECO_ABLATION_MATRIX.md`, `docs/LEARNING_TRACKS.md`, tests, and the approved paired calibration plan must be reconciled. Do not rewrite the approved plan to hide the existing D2 variant.

The calibration worker should consume the predictor identity emitted from `MATCHED_REFERENCE_VARIANTS` and bind the complete canonical contract (model ID, ordered hypotheses/weights, bounds, horizon semantics, fallback/risk semantics) in each control identity and campaign manifest. Rows, shard artifacts, and merge summaries must compare the exact contract and digest. A label-only `multi_model_v1` match is insufficient. Do not collect calibration outcomes until the existing source freeze, partition, score-estimand approval, manual training stop, and device-release gates all pass.

## Planned implementation sequence

1. **Rebase and inventory.** Start a new implementation worktree from current `origin/main`. Compare relevant current files against the audited feature patches and record the base SHA. Do not merge candidate branches. Resolve whether the warm-start preflight is still absent and whether its current APIs can be ported independently.
2. **Canonical scalar contract.** Add the constants, ordered hypothesis contract, deterministic forecast, fallback, and weighted risk semantics to `meco_tracks.py`. Keep `current`/`cv` defaults and serialized legacy contracts byte-compatible. Test finite/nonfinite input, speed/age/horizon bounds, deterministic output, turn-rate clamping, and normalized weights.
3. **Decision-path parity.** Wire candidate route ETA prediction and expected risk through candidate-based C/D1/D2 scoring. Wire Plan A per-node ETA evaluation to the same mixture semantics; validate its tensor implementation against the scalar `opponent_forecast` + `opponent_risk_penalty` result over representative edge/boundary inputs within a declared float tolerance. Keep stale, missing, and masked-node behavior consistent. Record mode and full predictor contract in evaluation/checkpoint provenance.
4. **Five-track and control selection.** Add opt-in learned A/C selection via the existing prediction-mode interface. Add per-control override support so a matched condition can run D1/current and D2/multi_model_v1 on identical seeds. Add the explicit matrix condition and ensure condition IDs, policy manifests, records, and contrasts all carry the predictor contract. Existing default D1/current and D2/cv condition must remain unchanged.
5. **Qualification and calibration binding.** Make the existing `D2/multi_model_v1` matched reference resolvable by the canonical prediction contract. Extend the paired calibration implementation from `docs/plans/phase0-resumable-paired-d0-d2-calibration-20261025.md` to freeze the exact predictor contract and digest in manifest, shard, row, resume validation, and merge output. Preserve D0 pairing, frozen tuples/seeds/repetitions, source/runtime/score-ledger/replay bindings, and explicit WX9100 device identity. Unsupported or contract-drifted modes fail closed.
6. **Warm-start readiness compatibility.** If the readiness preflight remains absent on the current base, selectively port `0e7b60506554d55e80aeb239853170f1308a2f79` and its focused tests/docs as a separate commit after the predictor contract exists. Add a test that a `multi_model_v1` five-track bundle is accepted only when bundle, each artifact, every declared dataset manifest, source identity, and qualification protocol bind the identical canonical contract. Do not port unrelated candidate history.
7. **Docs and provenance.** Update predictor and ablation docs, full-plan status, and three-pass provenance after code review. State explicitly that the predictor is implemented but not calibrated/qualified until seeded simulation and held-out paired calibration pass. Record source commits and the port's own commit IDs distinctly.

## Contract and test coverage required

Use focused offline tests before any behavioral evaluation:

- Scalar contract: exact model/version/constants, normalized ordered weights, determinism, speed/age/horizon/turn-rate bounds, damped/turn endpoint equations, expected-penalty mixture, and invalid/stale fallback behavior.
- Cross-path parity: candidate route ETA and Plan A node ETA use equivalent scalar mixture penalties; test zero and capped ETA, each hypothesis contribution, invalid pose/velocity, stale age, no valid Plan A nodes, and masked/unreachable nodes.
- CLI/checkpoint/evaluation: opt-in learned mode, unchanged defaults and old checkpoint identity, exact control override validation, and full prediction contract in provenance.
- Matrix: explicit D1/current + D2/multi_model_v1 condition, same seed tuple, mode-specific control contracts, deterministic condition identity, and correct D1-vs-D2 contrast provenance.
- Qualification: `D2/multi_model_v1` resolves only to the exact registered canonical contract and rejects altered model ID, weights/order, bounds, semantics, or contract digest.
- Calibration: same complete frozen case tuple/seed/repetition pairing across controls, full predictor contract/digest at every artifact layer, and rejection of missing, extra, altered, or drifted contract data on resume/merge.
- Warm-start preflight, if included: same canonical contract equality across bundle/artifact/all dataset manifests and established artifact validator.

Pure contract tests must not launch simulator/GPU work. Any test run that checks simulated robot/scoring/collection behavior requires a representative seeded end-to-end simulator evaluation, with the changed path exercised and its simulation record/replay inspected. Follow `AGENTS.md`: restart dashboard services after runtime code changes; verify active services and successful API responses; verify served backend/frontend match the current checkout; generate a fresh replay after restart; inspect replay provenance; and report the scenario, seed, replay identifier, and results. Since training is currently instructed to continue uninterrupted and owns the requested GPUs, do not stop it, contend for its devices, or substitute a competing simulator workload. Keep the implementation in progress and defer device-dependent/e2e gates until the user manually stops training and the devices are actually released.

## Incremental commits and push checkpoints

Keep each checkpoint reviewable and push the branch after it is committed:

1. predictor contract + pure scalar tests;
2. candidate/Plan A parity + evaluation provenance + focused tests;
3. matrix and qualification/calibration contract binding + focused tests;
4. optional separate warm-start readiness port + focused tests;
5. docs/status/provenance reconciliation.

Do not squash away these checkpoints. Run the independent Pass 2 review against the exact pushed implementation head before integration. The reviewer must verify feature semantics against the audited commits, current-main compatibility, preservation of defaults, contract equality through calibration, test coverage, and that no live training/device state was altered. Dashboard/simulator checks are required after reviewed integration when the runtime is available.

## Acceptance criteria

- `multi_model_v1` is available through one canonical versioned contract to learned A/C, candidate-based C/D1/D2, Plan A, the matrix runner, qualification identity, paired calibration, and (if included) warm-start readiness.
- Scalar and tensor/route risk agree within the declared tolerance; no prediction path silently uses a different horizon, weights, fallback, or penalty.
- D1/current vs D2/multi_model_v1 is an explicit, reproducible paired matrix condition with exact seed pairing and contract-bound provenance; existing defaults remain unchanged.
- The paired calibration manifest and every derived artifact bind the entire predictor contract and its digest along with the already required freeze/partition/outcome/replay/runtime/device identities.
- Current-main history is preserved; only audited feature semantics are selectively ported from the divergent candidate branches.
- Offline tests and mandatory simulator/dashboard gates are reported separately. No calibration, held-out performance, qualification, or training claim is made without its evidence.
- The implementation branch is incrementally committed/pushed and reviewed at its exact head. Training is untouched during this tranche.
