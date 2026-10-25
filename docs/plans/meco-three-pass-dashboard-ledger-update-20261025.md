# MECO three-pass dashboard ledger update plan

**Version:** 1  
**Status:** Planner scope, not implementation or runtime evidence  
**Base:** `b012c24779b50c9fda56f2e0bfb3c7769be4f89a` (`main`)  
**Plan date:** 2026-10-25

## Objective

Register the reviewed Phase 4 start-profile executable-gate implementation and
Plan A HIP parity/profile harness in the summary-only v2 three-pass ledger,
then refresh the dashboard-pinned manifest digest and verify the dashboard
serves that checkout and its frontend assets. Keep all evidence limited to the
actual scoped planner, implementation, review, and validation records. Do not
claim simulator outcomes, qualification, hardware execution, or deployment.

## Frozen inputs and dependency

- The integration work starts at `b012c24779b50c9fda56f2e0bfb3c7769be4f89a`,
  which contains the Phase 4 implementation merge. Its approved implementation
  head is `b580d13d139e65a43568b4384a27bb99c783388a`. The scoped plan is
  `docs/plans/phase4-start-profile-executable-gate-20261025.md`; an independent
  exact-head review approved replay-to-report row linkage, evaluation identity,
  same-policy pairing, source/checkpoint/device bindings, and dashboard
  evidence reconciliation at `b580d13d1`.
- Plan A implementation head
  `ca4604e019826cb6b2a010f84872d0eb15f83d16` and documentation correction
  `d4703a810271adf73250f86bd8540cf7d3552399` are independently approved, but
  that branch is based at `48df1c7d03d72b2645e36070f34f91bf9a4b92ba` and is
  not an ancestor of this base. The corrected documentation commit descends
  from `9f7d48df3713a8fddaa263e914eb011ede8549be`.
- **Integration dependency:** port/cherry-pick only the reviewed Plan A
  implementation and documentation changes onto this base, reconciling shared
  docs with the newer Phase 4 state. Do not merge the stale Plan A branch as a
  whole: its tree predates this Phase 4 integration and its full diff against
  this base removes Phase 4 changes. Review the resulting integration tree for
  unintended drift and record both the approved candidate SHA and the new
  integrated SHA. If the port changes the implementation beyond the reviewed
  patch, obtain fresh exact-head review before registering it as approved.
- Keep the integration in an isolated HDD-backed worktree. Do not check out or
  merge it into the active training checkout. It changes repository source
  identity; active checkpoints and live training source remain untouched.

## Bounded implementation sequence

1. Port the Plan A candidate chain in its reviewed order: planner record
   `8dbbfee696b74a5fc4e0b071ccc364dd8db84571`, planner approval
   `5e9667b7a99bf7dc2ab8feea57de6e0e903422cf`, implementation commits
   `c17db4ad4ec0b4b6f7d15ddccd25018e8aeb7e9a`,
   `c758b6178c8634f9ea0a116f3421b41331832779`,
   `e9a1e09e45a2851c56f76abd610e426403cf5c0a`, and
   `ca4604e019826cb6b2a010f84872d0eb15f83d16`, then the reviewed summary
   `9f7d48df3713a8fddaa263e914eb011ede8549be` and correction
   `d4703a810271adf73250f86bd8540cf7d3552399`. Preserve the corrected record
   that fixture outputs were generated using the CPU oracle; describe this as
   fixture authoring, not as a test run. Resolve documentation overlap against
   the current Phase 4 files without discarding newer content.
2. Add a concise persisted Phase 4 transcript-summary artifact at
   `docs/implementation-provenance/phase4-start-profile-executable-gate-20261025.md`.
   It must summarize the actual planner scope and independent exact-head review;
   include no invented or inferred transcript details. Use it for the planner
   and reviewer artifact objects with the required artifact kinds/statuses and
   its exact SHA-256.
3. Append exactly two unique entries to
   `docs/MECO_THREE_PASS_PROVENANCE.json`:
   - `phase4-start-profile-executable-gate`: planner summary artifact above;
     implementer candidate `b580d13d139e65a43568b4384a27bb99c783388a`,
     integration commit, and the ten Phase 4 paths changed relative to the
     first parent of `b012c2477`; reviewer exact approval at `b580d13d1`.
   - `plan-a-hip-parity-profile-harness`: planner and reviewer evidence from
     the corrected `docs/implementation-provenance/plan-a-hip-parity-profile-harness-20261025.md`;
     implementation candidate `ca4604e019826cb6b2a010f84872d0eb15f83d16`,
     integration commit, and the four implementation paths from that
     candidate chain: `frc_defense/meco_plan_a_hip_fixture_graph.py`,
     `frc_defense/meco_plan_a_hip_parity_harness.py`,
     `scripts/verify_plan_a_hip_parity.py`,
     `tests/fixtures/plan_a_hip/manifest.json`, plus any additional code path
     changed by the reviewed chain if present after reconciliation. Reviewer
     approval remains explicitly for exact candidate `ca4604e01`; do not imply
     the rebased tree was reviewed unless separately reviewed.
4. Keep the manifest schema and evidence class at
   `meco-three-pass-provenance-v2` / `scoped_transcript_summary`. Every role
   artifact must be a repository-local regular file with its exact SHA-256,
   `kind: scoped-transcript-summary`, and the validator-required role status.
   Each entry needs unique kebab-case `requirement_id`, nonempty title and
   limits, exact 40-character commit IDs, unique safe relative implementation
   paths that exist in the integrated checkout, and a reviewer verdict detail.
   Refresh the manifest's summary-only scope/limits and artifact-origin text
   without converting transcript summaries into whole-plan approval.
5. Hash the exact final manifest bytes and update only
   `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in `frc_defense/dashboard_api.py`.
   Preserve `_EXPECTED_PLAN_SHA256`; do not change the evaluation plan.
6. Deploy this integration checkout to the dashboard services and restart the
   relevant services so Python and static assets reload. Capture currentness
   evidence: service health and checkout/source identity; successful
   `/api/meco-status` response whose `three_pass_provenance` is
   `validated_summary_only`; and frontend asset manifest plus served asset
   hashes equal to the integrated checkout. Do not generate a replay for this
   metadata-only tranche.

## Acceptance criteria

- The integration history retains the Phase 4 code from `b012c2477` and the
  approved Plan A candidate behavior, with no unrelated source changes.
- Both ledger entries use exact full hashes and accurately distinguish the
  approved candidate from its integration commit. All artifact and path hashes
  validate against files in the integrated checkout. The currentness response
  is summary-only and includes the expected two entries.
- The dashboard-pinned digest equals SHA-256 of the final manifest bytes;
  dashboard validation is not `untrusted`.
- After service restart, backend endpoint and frontend assets are verified
  current for the same integration checkout. Retain timestamp, service
  identity, source identity, endpoint result, and asset-manifest/hash evidence
  in the final implementation record.
- No simulator, GPU/HIP, training, replay, qualification, or deployment result
  is asserted. The Plan A production dispatch gate remains closed.

## Limits and risks

- The Plan A branch is stale relative to Phase 4; wholesale merge can remove
  Phase 4 implementation and documentation. Selective integration and diff
  review are mandatory.
- The Plan A reviewer approved `ca4604e01`, not an integration commit on top of
  `b012c2477`. A changed integration result needs a fresh exact-head review
  before its row can say `approved-summary` for that resulting implementation.
- The Plan A harness is an offline fixture/device-candidate utility. No
  offline CLI, HIP import/compile, WX9100 lease acquisition, parity/profile,
  simulator, replay, training, or deployment was run in the reviewed
  implementation pass. Device verification remains gated on exclusive WX9100
  leases; normal Plan A dispatch remains on the CPU oracle.
- The Phase 4 tranche establishes reviewed executable-evidence contracts only.
  No profile outcome, fresh paired simulator replay, runtime dashboard review,
  qualification, promotion, or architecture win is established by ledger
  registration.
- Dashboard currentness is a post-change operational check and must be recorded
  separately from the code/ledger review. Existing cached replays or asset
  responses are not currentness evidence.

## Out of scope

No training interruption or resume, simulator/evaluation workload, test run,
GPU/HIP or hardware probe, replay generation, dashboard product redesign, or
qualification decision. This plan authorizes scoped integration and dashboard
provenance registration in an isolated checkout only.
