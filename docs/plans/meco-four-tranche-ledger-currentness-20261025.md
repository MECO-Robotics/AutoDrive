# MECO four-tranche ledger and dashboard-currentness plan

**Version:** 1  
**Status:** Planner scope only; no implementation or service operation  
**Base:** `c55141876f2d6ff2ab1102294da79c222a742219` (`origin/main`)  
**Plan date:** 2026-10-25

## Objective

Register four already-merged implementation tranches that are absent from the
80-entry `meco-three-pass-provenance-v2` ledger, refresh its dashboard-pinned
digest, and verify the dashboard serves the resulting checkout and assets.
This is a provenance/currentness update. It does not complete Slice B, assert
match behavior, create performance evidence, or authorize training.

## Current state and source pins

- `origin/main` is `c55141876f2d6ff2ab1102294da79c222a742219`. The local main
worktree had one unrelated generated modification at
`frc_defense/csrc/tensor_collision_multi_hip_kernel.hip`; leave it untouched.
- `docs/MECO_THREE_PASS_PROVENANCE.json` has 80 rows and its SHA-256 matches
`frc_defense/dashboard_api.py::_EXPECTED_THREE_PASS_PROVENANCE_SHA256` at
`76f2badfb5e7daa8f5a811dd2b788dcdc76f1db75b81dfef2d2e32d2067a01ad`.
The plan digest pin is
`718ef9d76f5525bf8e4759d3a0a46790a3c978e2b146efec6436c89dbbb49b10`; preserve
it.
- The source documentation and campaign-pass history do not yet persist exact
planner/reviewer artifact rows for the four changes below. The parent task
confirms they are approved. Before recording an `approved-summary` ledger row,
persist a concise summary from the actual scoped planner/reviewer records, use
the exact approved feature HEAD as `approved_commit`, and hash that summary.
If the original exact-head approval record cannot be retrieved, do not invent
one or label the entry approved; report the missing artifact as a blocker.

## Four missing ledger entries

For every listed source file, the SHA-256 below is the file content at this
plan base. The implementation record must confirm the same hash before adding
the row. Changed-path lists must be exact and safe relative paths accepted by
the existing dashboard validator.

### 1. Dashboard curriculum queue run-directory handling

- **Feature commits:**
  `ea6f039ced9c1d249fea7059ec2afc94ef3c27e7` (read `run_directory` safely),
  `8c9b6bdf80ca7b97a392c88e8b41742607b80182` (preserve safe absolute legacy
  queue paths).
- **Integration:** `ff5c862b183d71c733e0094b186fb8fbf9042904`.
- **Approved feature HEAD to verify from review record:**
  `8c9b6bdf80ca7b97a392c88e8b41742607b80182`.
- **Implementation paths and base hashes:**
  `scripts/report_curriculum_progress.py` —
  `7a5aacc4f89478479fb186621b68d9d1e0df9766780862799bbd8802d5cca952`;
  `tests/test_dashboard_curriculum.py` —
  `c4831a3faf28ee5afb390e35751cbf6de5653667dc537f33db5b5c265916fc19`.
- **Summary artifact:** add
  `docs/implementation-provenance/dashboard-queue-run-directory-20261025.md`
  from the actual planner and reviewer evidence. Capture the safe containment
  behavior and the legacy path compatibility limit; make no current-service
  claim until the dashboard check below.

### 2. REBUILT official Slice A scoring contract

- **Feature commits:**
  `515a714b2d86f47cd8688465e5edf1c093114d01` (pin TU22 rules snapshot),
  `a90369928098185983ba0ff204a8a90e53066844` (pure scoring contract),
  `13cebc4603b42b58538680c12d9d382e13cafc8e` (active-lineup scope),
  `e9240c18b2c7ba2a5402461b5385056ef36ed619` (bind TOWER awards to active
  alliance slots).
- **Integration:** `f17a1c7c89da00425abc60e0b5b28ee20498e704`.
- **Approved feature HEAD to verify from review record:**
  `e9240c18b2c7ba2a5402461b5385056ef36ed619`.
- **Implementation paths and base hashes:**
  `frc_defense/rebuilt_match_scoring.py` —
  `e0245096277b75e494b37d476a008afa0b0837b70da2016897dc1342fb7d3dda`;
  `qualifications/meco/rebuilt-rules-TU22.json` —
  `18cf92041290e8fa1df15d7e967fe5e107118a6d443fe2b6526c1c05b73f47a8`;
  `tests/test_rebuilt_match_scoring.py` —
  `be5ea76b19854e352ad6912792b9dd08cb3a5ca3c3d4e79952e663ab1d5bfe92`.
- **Planner source:**
  `docs/plans/rebuilt-match-points-ledger-20261025.md` —
  `05f11b1838643258ac57f56734e1ee33aacd511230e59dcef88942cc1713d1c9`.
- **Summary artifact:** add
  `docs/implementation-provenance/rebuilt-score-slice-a-20261025.md` from
  the actual planner and reviewer records. State that this is the rules-bound
  pure official-ledger contract; it is not wired to complete official
  six-robot/sensor/Tower/referee evidence and produces no current qualification
  evidence.

### 3. Six-slot simulated proxy scope correction

- **Feature commit:**
  `01543a6f81451c182cd8834ca67b17b52800f1f2` (label the proxy as six-slot 2D
  simulation, not two-robot official scope).
- **Integration:** `9d89720e3b99ab08074402e60b8c65433546c7b7`.
- **Approved feature HEAD to verify from review record:**
  `01543a6f81451c182cd8834ca67b17b52800f1f2`.
- **Implementation paths and base hashes:**
  `frc_defense/rebuilt_match_scoring.py` —
  `e0245096277b75e494b37d476a008afa0b0837b70da2016897dc1342fb7d3dda`;
  `tests/test_rebuilt_match_scoring.py` —
  `be5ea76b19854e352ad6912792b9dd08cb3a5ca3c3d4e79952e663ab1d5bfe92`.
- **Planner source:**
  `docs/plans/rebuilt-active-hub-slice-b-implementation-20261025.md` —
  `51c7e0342d276fef1f8d70abf11e88b6d079212742d46ad94e494a7432b38815`.
- **Summary artifact:** add
  `docs/implementation-provenance/rebuilt-proxy-scope-six-slot-20261025.md`
  from the actual scope/review records. Explicitly retain
  `simulation_only=true` and `qualification_eligible=false`; six simulator
  slots do not establish six official active robots, sensor-verified scoring,
  complete game context, or official match points.

### 4. Bounded large-evaluation dashboard report sidecars

- **Feature commits:**
  `36b0832b0d72c3b724c7ab51b54751b9afa95e76` (bind summaries to terminal
  evaluation status),
  `4714bab0b7bbae89c780893ef428c57d0fe87b7e` (bounded summaries),
  `68fa7b945ae55e1ffc743c4ffb50908ffe271e15` (bound concurrent writers),
  `e0895dbc29cb3c661302c5eb33e14090ac927cee` (reject incomplete/negative
  metrics),
  `0fbefa10bff2efa1e40196d20759bba0a090813e` (accept partial legacy bout
  summaries).
- **Integration:** `c55141876f2d6ff2ab1102294da79c222a742219`.
- **Approved feature HEAD to verify from review record:**
  `0fbefa10bff2efa1e40196d20759bba0a090813e`.
- **Implementation paths and base hashes:**
  `frc_defense/dashboard_api.py` —
  `048ff48c82521f59a9936976ba89ab2f06f9fbf01ae137dad15b3c8fda875be2`;
  `frc_defense/meco_dashboard.py` —
  `7eee6a15015b5517224eefd2b3e61597371d942e7eec88e8ea09687265ac82b6`;
  `frc_defense/meco_supervisor.py` —
  `d6e6bfe5a54493683b0103cf7035a1846cff50683af841d1323988cb36dcde24`;
  `frc_defense/meco_track_training.py` —
  `2c94a4c717d28737f3ff89fd129fc6a5528fd8bc83de2b61a2c3a150ab6debdc`;
  `tests/test_dashboard_curriculum.py` —
  `c4831a3faf28ee5afb390e35751cbf6de5653667dc537f33db5b5c265916fc19`;
  `tests/test_meco_dashboard.py` —
  `898fe6a2a9463e93fa7cc4f0bf6651d3dae8ae21ae89419b7fd56da145407cc6`.
- **Planner source:**
  `docs/plans/dashboard-current-error-remediation-20261025.md` —
  `adde46e4d4e5952b947ca6521bc118ddf693d5a6f191a752917913210bfb046f`.
- **Summary artifact:** add
  `docs/implementation-provenance/dashboard-large-report-sidecar-20261025.md`
  from actual planner/reviewer evidence. State limits to compact diagnostic
  summaries and legacy-field projection; do not claim report contents are
  full raw reports, or infer model performance/qualification.

## Ledger and currentness procedure

1. In an isolated HDD worktree at this exact base, verify all feature and merge
   commits are ancestors of `HEAD`. Recompute each file hash listed above;
   if any differs, stop and update the artifact against the observed commit
   instead of silently trusting this plan's pins.
2. Persist the four scoped summary artifacts from actual planner/reviewer
   evidence. Each artifact must clearly separate planner scope, implementer
   commits/paths, exact reviewer approved HEAD/verdict, checks performed, and
   nonclaims. Compute and record each artifact's SHA-256. Use
   `kind: scoped-transcript-summary`, planner status
   `transcript-summary`, reviewer status `review-summary`; do not put the same
   file hash in a role unless that file contains evidence for that role.
3. Add exactly four unique requirement rows to
   `docs/MECO_THREE_PASS_PROVENANCE.json`, with exact 40-character commit IDs,
   existing safe relative changed paths, reviewer verdict details, and honest
   per-row limits. Use each approved feature HEAD as `reviewer.approved_commit`
   and include the integration commit separately in the implementer object.
   The final ledger should contain 84 rows. Do not register Slice B as complete
   or reviewed unless its separate implementation is finished and reviewed.
4. Preserve schema `meco-three-pass-provenance-v2`, status
   `scoped-transcript-summaries`, evidence class `scoped_transcript_summary`,
   and `_EXPECTED_PLAN_SHA256`. Recompute the digest over the exact final
   manifest bytes and update only
   `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in `dashboard_api.py`.
5. Restart the dashboard services on the updated checkout so backend modules
   and frontend files are current. Record service PID/invocation, working
   directory, UTC time, checkout/source identity, and endpoint responses.
   Verify `/api/meco-status` reports
   `three_pass_provenance.validation_status == "validated_summary_only"`,
   all four new IDs, and no `untrusted_reason`; verify `/api/curriculum-status`
   shows the current queue projection. Fetch `/`, `/meco`, and all referenced
   JS/CSS assets, and compare served bytes/hashes with the same checkout.

## Acceptance and limits

- The v2 validator accepts all 84 rows; every summary-artifact digest matches
  its repository file; every commit is a full lowercase 40-hex ID; every
  implementation path exists, is unique within its row, and is safe.
- The manifest digest equals the dashboard pin, while the source-plan digest
  pin is unchanged. The API validates summary-only provenance and renders the
  four new rows.
- Dashboard endpoint and frontend checks are from the same post-restart
  checkout. They establish dashboard currentness only; no simulator replay is
  relevant to this metadata-only update.
- Official Slice A is a pure scoring/validation contract, not complete official
  match evidence. The six-slot proxy correction remains simulation-only and
  nonqualifying. The large-report sidecar is a bounded dashboard projection.
  The queue fix establishes path/currentness handling, not training liveness.
- No training launch, live r4 service change, simulator, device/GPU/HIP probe,
  replay, official match result, qualification, promotion, or deployment claim.
  Slice B remains a distinct work item and its live implementation must not be
  overwritten or relabeled by these ledger rows.

## Explicit downstream blockers

The ledger update does not complete the source evaluation plan or enable the
next five-track run. Remaining gates include finishing/reviewing Slice B and
its seeded end-to-end simulator evidence; complete official match evidence
requires separate roster, sensor-equivalent score windows, Tower/referee
inputs, and Slice D. Plan A labels, validated warm-start artifacts, source-bound
baseline/calibration bundles, operator-reviewed margins, matched held-out
five-track outcomes, reviewed fresh replays, and target-device/Hailo evidence
remain separate requirements. Preserve the active r4 source and do not start a
new campaign until the user-authorized device/run boundary allows it.
