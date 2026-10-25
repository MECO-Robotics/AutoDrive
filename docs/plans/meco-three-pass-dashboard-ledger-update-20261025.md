# MECO three-pass dashboard ledger update plan

**Version:** 2  
**Status:** Planner scope, not implementation or runtime evidence  
**Base:** `464239c4875362a9c27c50c13e17fac30e09fae6` (`main`)  
**Plan date:** 2026-10-25

## Objective

Register the reviewed Phase 4 start-profile executable-gate implementation and
Plan A HIP parity/profile harness in the summary-only v2 three-pass ledger,
refresh the dashboard-pinned manifest digest, then verify the dashboard serves
this checkout and its frontend assets. Record only the actual scoped planner,
implementation, review, and validation evidence. Make no simulator,
qualification, hardware-execution, or deployment claim.

## Verified inputs

- Current integration base is `464239c4875362a9c27c50c13e17fac30e09fae6`.
- Phase 4 implementation commit
  `b580d13d139e65a43568b4384a27bb99c783388a` is an ancestor of the base;
  integration is present in `b012c24779b50c9fda56f2e0bfb3c7769be4f89a`.
  Its scoped plan is `docs/plans/phase4-start-profile-executable-gate-20261025.md`.
  Independent exact-head review approved replay-to-report row linkage,
  evaluation identity, same-policy pairing, source/checkpoint/device bindings,
  and dashboard evidence reconciliation at `b580d13d1`.
- Plan A implementation commit
  `ca4604e019826cb6b2a010f84872d0eb15f83d16` and corrected documentation
  commit `d4703a810271adf73250f86bd8540cf7d3552399` are ancestors of the base.
  The corrected scoped summary is
  `docs/implementation-provenance/plan-a-hip-parity-profile-harness-20261025.md`.
  Its independent implementation review approved exact candidate `ca4604e01`.
- The four Plan A implementation/evidence files are present and byte-identical
  to the approved candidate versions at `ca4604e01`:
  `frc_defense/meco_plan_a_hip_fixture_graph.py`,
  `frc_defense/meco_plan_a_hip_parity_harness.py`,
  `scripts/verify_plan_a_hip_parity.py`, and
  `tests/fixtures/plan_a_hip/manifest.json`. The corrected Plan A provenance
  summary is byte-identical to its version at `d4703a810`.
- The manifest currently contains 74 entries and hashes to
  `76f2badfb5e7daa8f5a811dd2b788dcdc76f1db75b81dfef2d2e32d2067a01ad`, which
  matches `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in
  `frc_defense/dashboard_api.py`. Neither requested tranche is registered yet.

## Bounded implementation sequence

1. Work in a dedicated HDD-backed worktree at the verified base. Do not alter
   the live training source, supervisor state, or unrelated generated files.
2. Add a concise persisted Phase 4 transcript-summary artifact at
   `docs/implementation-provenance/phase4-start-profile-executable-gate-20261025.md`.
   Summarize the actual planner scope and independent exact-head review; do not
   invent or infer transcript details. Use its exact SHA-256 in the planner and
   reviewer artifact objects with the schema-required kinds and statuses.
3. Append exactly two entries to `docs/MECO_THREE_PASS_PROVENANCE.json`:
   - `phase4-start-profile-executable-gate`: planner summary artifact above;
     implementer candidate `b580d13d139e65a43568b4384a27bb99c783388a`,
     integration commit `b012c24779b50c9fda56f2e0bfb3c7769be4f89a`, and the
     ten paths changed relative to the integration commit's first parent;
     reviewer exact approval at `b580d13d1`.
   - `plan-a-hip-parity-profile-harness`: planner and reviewer evidence from
     the corrected
     `docs/implementation-provenance/plan-a-hip-parity-profile-harness-20261025.md`;
     implementer candidate `ca4604e019826cb6b2a010f84872d0eb15f83d16`,
     integration commit `464239c4875362a9c27c50c13e17fac30e09fae6`, and the
     four byte-matched paths listed above; reviewer approval remains explicitly
     for exact candidate `ca4604e01`.
4. Preserve schema `meco-three-pass-provenance-v2` and evidence class
   `scoped_transcript_summary`. Each role artifact must be a repository-local
   regular file with its exact SHA-256, kind `scoped-transcript-summary`, and
   the validator-required role status. Each entry needs a unique kebab-case
   requirement ID, nonempty title and limits, exact 40-character commit IDs,
   unique safe relative implementation paths present in this checkout, and a
   reviewer verdict detail. Refresh summary-only scope/limits and
   artifact-origin text without implying whole-plan approval.
5. Hash the final manifest bytes and update only
   `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in `frc_defense/dashboard_api.py`.
   Preserve `_EXPECTED_PLAN_SHA256`; do not change the evaluation plan.
6. Deploy this integrated checkout to the dashboard services and restart the
   relevant services so Python and static assets reload. Capture service
   health and checkout/source identity; verify `/api/meco-status` reports
   `three_pass_provenance.validation_status == "validated_summary_only"`,
   includes both entries, and has no `untrusted_reason`; verify the frontend
   asset manifest and served asset hashes match this checkout. Do not generate
   a replay for this metadata-only tranche.

## Acceptance criteria

- The two provenance entries cite exact full hashes and distinguish reviewed
  candidate commits from their integration commits. All artifact digests and
  paths validate against the integrated checkout. Plan A's four implementation
  blobs match the approved candidate exactly.
- The dashboard-pinned digest equals SHA-256 of the final manifest bytes, and
  manifest validation is not `untrusted`.
- After service restart, the backend endpoint and frontend assets are verified
  current for the same integration checkout. Retain timestamp, service
  identity, source identity, endpoint result, and asset-manifest/hash evidence
  in the implementation record.
- No simulator, GPU/HIP, training, replay, qualification, or deployment result
  is asserted. The Plan A production dispatch gate remains closed.

## Limits and risks

- Plan A review approved exact candidate `ca4604e01`. Its byte-identical
  implementation paths are integrated at `464239c48`; do not describe the
  integration commit itself as independently reviewed unless a separate
  exact-head review establishes that.
- The Plan A harness is an offline fixture/device-candidate utility. No offline
  CLI, HIP import/compile, WX9100 lease acquisition, parity/profile, simulator,
  replay, training, or deployment was run in the reviewed implementation pass.
  Device verification remains gated on exclusive WX9100 leases; normal Plan A
  dispatch remains on the CPU oracle.
- The Phase 4 tranche establishes reviewed executable-evidence contracts only.
  It does not establish profile outcomes, fresh paired simulator replay,
  runtime dashboard review, qualification, promotion, or an architecture win.
- Dashboard currentness is a post-change operational check and must be recorded
  separately from code/ledger review. Cached replay or asset responses are not
  currentness evidence.

## Out of scope

No training interruption or resume, simulator/evaluation workload, test run,
GPU/HIP or hardware probe, replay generation, dashboard product redesign, or
qualification decision. This plan scopes only the provenance registration and
dashboard currentness check in an isolated checkout.
