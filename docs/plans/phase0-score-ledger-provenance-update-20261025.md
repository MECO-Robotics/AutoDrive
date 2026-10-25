# Phase 0 score-ledger provenance and status update plan

**Status:** planner artifact only; no documentation integration has been performed.
**Planning base:** `457c469a5d2e00c56449aaeae0773d3cb6a689ff` (integrated score-ledger code).
**Reviewed implementation candidate:** `c4f8a591b018e2208dcb3f3b60863155aac6929d`.
**Approved score-ledger implementation plan:** `docs/plans/phase0-score-ledger-replay-implementation-20261025.md`, SHA-256 `3a553986ad9849e8b482292f19d2b7936e9a050ecc6bc2c7cbde6e9ba3e3672c`.

## Purpose and boundaries

Update the full-plan execution status and summary-only three-pass provenance to record what the integrated MECO simulated score ledger supports, while keeping its evidence limits explicit. The record must not imply the official match score, qualification readiness, successful simulated scoring, or completed Phase 0. This is a documentation tranche; do not change scorer code or dashboard data and do not run tests, simulator workloads, training, or device/service operations as part of this plan.

## Source-grounded implementation facts

The integrated source defines `meco-simulated-score-ledger-v1`, rules ID `meco_simulated_fuel_score_v1`, estimand schema `meco-simulated-score-estimand-review-v1`, and outcome name `simulated_match_score` in `frc_defense/meco_simulated_score_ledger.py`. The rules call the unit `simulated_score_unit`; only active HUB FUEL score awards are included. Tower awards, foul penalties, disqualification, ranking points, end-game bonus, and advancement points are excluded, and official points remain null. Therefore the quantity is a deliberately limited simulated outcome, not official match points.

The MECO scorer-path fixtures are in `tests/test_meco_score_ledger_scorer_path.py`; both `vectorized=True` and `False` parameter cases are currently guarded by a skip requiring seeded simulator E2E evaluation and fresh dashboard-currentness evidence. Pure ledger contract coverage is in `tests/test_meco_simulated_score_ledger.py`. The planner must not report that file as run or passed unless a specific implementer report supplies that evidence. Even reported pure-contract results do not satisfy the skipped scorer-path, simulator, replay, or dashboard gates.

No approved human estimand artifact is present in the repository. Collection and interpretation of this outcome remain blocked until a human approves the limited estimand and supplies a source/rules-bound artifact. If approval includes any currently excluded outcome, ledger completeness and rules revision must be addressed before collection; do not silently label the current ledger complete for that scope.

## Documentation changes

1. Add a dated, concise entry at the top of `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md` recording integration `457c469a5d2e00c56449aaeae0773d3cb6a689ff`, reviewed candidate `c4f8a591b018e2208dcb3f3b60863155aac6929d`, and the exact API/schema identifiers above. State separately: code support is integrated; pure contract test source exists (report execution only when evidenced); the two scorer-path cases are skipped; human estimand approval is absent; no seeded simulator evaluation, replay inspection, current-dashboard check, runtime evidence, or device evidence was produced. Keep Phase 0 incomplete and qualification false.
2. Update the relevant Phase 0/scoring status row or paragraph in the existing status document to show code support present while outcome evidence and approval gates remain open. Preserve historical entries and unrelated status rows.
3. Add `docs/implementation-provenance/phase0-simulated-score-ledger-replay.md` as a compact evidence summary. Include planner, implementer, and independent reviewer evidence; identify exact reviewed candidate separately from integration SHA; distinguish reported pure test evidence from skipped scorer fixtures; and state the human-approval, simulator/replay, dashboard-currentness, runtime/device, and qualification limits. Do not manufacture source transcript hashes; hash the materialized summary bytes for the dashboard row.
4. Append one row to `docs/MECO_THREE_PASS_PROVENANCE.json` with unique requirement ID `phase0-simulated-score-ledger-replay`. Follow schema `meco-three-pass-provenance-v2`: planner summary artifact, implementation commit `457c469a5d2e00c56449aaeae0773d3cb6a689ff`, feature/reviewed candidate `c4f8a591b018e2208dcb3f3b60863155aac6929d`, exact implementation paths, reviewer approved commit set to the reviewed candidate (not the integration merge), and a limits string explicitly naming all outstanding evidence gates. Use the same summary path and exact raw-file SHA-256 in the planner and reviewer artifact fields. The existing ledger has 84 rows; this addition makes 85.
5. Update top-level `scope` and `artifact_origin` text only as needed to describe this scoped summary and its evidence boundaries. Preserve the other 84 tranche records byte-for-byte in content and do not alter the source plan SHA (`718ef9d76f5525bf8e4759d3a0a46790a3c978e2b146efec6436c89dbbb49b10`) or provenance schema.
6. Compute SHA-256 over the final raw bytes of `docs/MECO_THREE_PASS_PROVENANCE.json` and update `frc_defense/dashboard_api.py::_EXPECTED_THREE_PASS_PROVENANCE_SHA256` to that exact value. Leave API behavior/schema unchanged. Ensure provenance path validation accepts the new summary path and implementation paths.

### Exact implementation paths to record

- `frc_defense/_tensor_3v3_gamepieces.py`
- `frc_defense/gamepiece_actions.py`
- `frc_defense/meco_simulated_score_ledger.py`
- `frc_defense/meco_track_training.py`
- `tests/test_meco_score_ledger_scorer_path.py`
- `tests/test_meco_simulated_score_ledger.py`

These paths describe the integrated code change; they do not mean every listed test ran. The scoring path is MECO's `TensorThreeVsThreeGamepieceMixin` through `gamepiece_actions.update_fuel_actions()`, not the separate TensorDefenseEnv fused kernels.

## Acceptance criteria

- The status page and provenance row agree on integration SHA, reviewed candidate SHA, outcome/rules/schema identifiers, and limits.
- The ledger has exactly one new unique tranche ID, preserves all prior entries and the source-plan digest, and has 85 rows.
- The provenance JSON raw-byte digest exactly matches the API pin; the existing summary-only API contract remains unchanged.
- Human estimand approval is an explicit prerequisite; excluded outcomes are not represented as covered.
- No text claims scorer-path tests passed, a simulator run or replay exists, the dashboard was verified current, runtime/device evidence exists, or Phase 0/qualification is complete unless new, specific evidence is supplied and reviewed.
- Documentation changes are reviewed at an exact candidate head, committed incrementally, and pushed. The later integration must use the documented dashboard currentness gate after documentation changes; this planning task itself performs no service restart or endpoint check.

## Execution and review sequence

1. Implement only the three documentation changes and digest pin described above; record exact evidence for each claimed test result, or state it was not verified.
2. Check the final diff and JSON validity without running tests or workloads in this tranche. Record the new summary's raw-byte SHA and provenance JSON digest.
3. Commit the status/provenance work incrementally, request an independent review against the exact candidate head, resolve findings, then commit/push the reviewed result. Report the exact final head and clearly list any evidence that remains unavailable.

