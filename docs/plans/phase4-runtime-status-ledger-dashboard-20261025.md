# Phase 4 runtime-status provenance and dashboard currentness plan

**Status:** Pass 1 plan only; no implementation, service operation, or runtime evidence
**Base:** `origin/main` at `384c9a2baa9a97a27271cc6ed26fb316f01be8ed`
**Date:** 2026-10-25

## Objective and boundaries

Register one accurate summary-only three-pass provenance entry for the two
integrated Phase 4 runtime-status documentation corrections, refresh the
dashboard-pinned ledger SHA, and update the top-level execution status with
their precise code-wiring and evidence limits. After the implementation has
passed exact-head review and is integrated, restart the dashboard from that
current checkout and verify the APIs, ledger, and all frontend assets.

This tranche records documentation and provenance. It does not add runtime
behavior or establish an end-to-end simulator run, fresh replay, behavior
outcome, qualification, training result, GPU/device result, or deployment
evidence. The runtime-status document says qualification remains disabled and
reports zero executable qualification cells. Preserve those limits.

Keep implementation/review and post-merge service operation as separate gates:

1. **Implementation/review:** update the summary ledger, its API digest pin,
   focused offline contract coverage, and `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md`;
   obtain exact-head independent review before integration.
2. **Post-merge operation:** only after integration, restart
   `frc-defense-dashboard.service` from the integrated current checkout, verify
   both APIs and all frontend assets against that checkout, and retain actual
   service/response/hash evidence in a separate currentness note. Do not
   regenerate a replay or interpret this metadata check as simulator evidence.

## Exact source records to attribute

The current `docs/MECO_THREE_PASS_PROVENANCE.json` has 87 entries and SHA-256
`aa036f2ae4a61317f090b9249b4915f02dabd56f63e640570866b72d64f498e6`; this
matches `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in
`frc_defense/dashboard_api.py`. Preserve all 87 existing entry objects, order,
and attribution. Add exactly one unique tranche; the dashboard contract test
must then observe 88 entries.

A read-only service check at planning time found the dashboard active on the
main checkout (PID `3503505`, started `12:59:42`) while
`/api/curriculum-status` still reported 87 tranches and the earlier
`phase0-effective-v3-paired-calibration` as latest. This predates the current
status-correction merges and is not post-merge currentness evidence; the
post-merge restart and checks below are required.

The first status correction is covered by:

- Plan `docs/plans/phase4-focal-pose-mislocalization-wiring-20261025.md`,
  SHA-256 `e563af9b2b68498e088bc40ff6ce0d2777cc495299e07b30c8333d7877745a7e`.
- Pass 2 status-document commit `2357ae4a9661c7cb6d6c038a72881a55fbdcfbec`;
  Pass 3 approved that exact candidate. Its integration commit is
  `252eb3d65e27a0d96352cf2abb83265384ce1d8d`.

The second status correction is covered by:

- Plan `docs/plans/phase4-runtime-status-corrections-20261025.md`, SHA-256
  `ebe11578f52c1753e670c70bc9b0d10159a3938b352f5b58a201d41d8ffb0f98`.
- The amended Pass 3-approved plan head is
  `efbe2bb33cc29cb611de654179fedc5f91497a57`. Pass 2 status-document commit
  `3ed5c86e5108db24eb1588afc890d1cf379cad39` was approved at that exact head
  of the status change; its integration commit is
  `384c9a2baa9a97a27271cc6ed26fb316f01be8ed`.

The corrected runtime-status document is
`docs/MECO_PHASE4_RUNTIME_STATUS.md`. Its current rows describe code paths and
explicitly retain the limits: focal-pose relocalization/recovery/safety
outcomes and held-out E2E replay evidence are unavailable; stationary-blockage
registry outcomes remain non-authoritative/unavailable; direct-pursuit and
trench rows remain fail-closed pending accepted held-out simulator/replay/
dashboard review; legal-start candidate execution remains fail-closed pending
accepted source-bound report/replay review. Qualification remains disabled.
Do not convert any source implementation record into a claimed simulator
result. Do not rewrite or broaden that runtime-status document in this
provenance tranche.

## Planned implementation

1. Work in a clean isolated worktree based on the stated `origin/main` base;
   preserve the unrelated user-modified root-checkout HIP file.
2. Create one concise, repository-local implementation/review summary under
   `docs/implementation-provenance/`. It must identify both plan paths and
   hashes, both Pass 2 status heads, the exact Pass 3-approved heads, the two
   integration commits, and the bounded facts/limits above. The summary must
   distinguish these prior approved status changes from the new ledger/API/doc
   update being implemented. State only evidence actually reported by the
   implementer/reviewer for the latter; do not infer test or service results.
3. Append exactly one `meco-three-pass-provenance-v2`
   `scoped_transcript_summary` entry, with a unique ID such as
   `phase4-runtime-status-ledger-dashboard`. Use this plan as the planner
   artifact and the implementation/review summary as the reviewer artifact,
   with exact path, kind, status, and SHA-256 values. Record the two prior
   Pass 2 heads, their exact Pass 3 approvals, and integration commits in the
   summary and explicit distinct commit fields in the new implementer object.
   The new entry's reviewer `approved_commit` must identify the exact reviewed
   ledger-update candidate, not a prior status-document commit or an
   integration merge. The new entry must say summary-only and must not imply
   whole-plan approval.
4. Preserve every existing ledger object and order. Update top-level scope,
   artifact-origin, and limits only enough to record this one additional
   scoped documentation/provenance summary. Keep the existing schema,
   `plan_path`, `plan_sha256`, status, and evidence class unchanged.
5. Update only `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in
   `frc_defense/dashboard_api.py` to the SHA-256 of the final ledger bytes.
   Leave `_EXPECTED_PLAN_SHA256` unchanged.
6. Add focused offline metadata/contract coverage. Assert the new entry names
   both prior plans, exact Pass 2 and approved Pass 3 heads, and both
   integration commits; assert 88 valid tranches and the exact ledger/API
   digest match; and assert the previous 87 entry objects remain equal and in
   order. Keep existing mutation/rejection coverage intact. These checks do
   not test simulator behavior.
7. Add a concise status-only section to
   `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md`, citing the runtime-status document
   and the two approved plans/heads. Say the status rows summarize wired code
   paths only; no fresh E2E simulator evaluation/replay or held-out dashboard
   review was performed for these corrections; outcome claims and
   qualification remain unavailable/disabled. Do not claim the dashboard
   currently serves the new ledger before the separate post-merge check.
8. Commit the coherent implementation in reviewable increments, run only the
   scoped offline contract tests and static diff checks authorized by the
   implementation plan, and request independent exact-head review before
   integration. No dashboard service, API, simulator, replay, training,
   private-campaign-data, GPU, or device operation occurs in this gate.

## Post-merge dashboard operation and acceptance

After the reviewed candidate is integrated, perform this separate operational
check against the exact current checkout:

- Restart `frc-defense-dashboard.service`; record UTC timestamp, active state,
  PID, invocation ID, service working directory, checkout commit, and source
  identity. Verify the service is active and the endpoint responses succeed.
- Query `/api/meco-status` and `/api/curriculum-status`. Both must return HTTP
  200. The current ledger must report `validated_summary_only`, 88 tranches,
  the new final requirement ID, no `untrusted_reason`, and the exact ledger
  digest pinned in `dashboard_api.py`. Report unrelated data-read errors
  separately from provenance validation state.
- Fetch `/`, `/meco`, `/assets/dashboard.css`, `/assets/dashboard.js`,
  `/assets/meco.css`, `/assets/meco.js`, `/assets/field_rendering.js`, and
  `/assets/training_rendering.js`. For each route record HTTP status and
  compare the served body SHA-256 byte-for-byte with the corresponding file
  from the same checkout.
- Save these service, API, ledger, and eight-asset results in a separate
  currentness evidence note linked from the top-level execution status. If
  this evidence note is committed after the service check, state its commit
  separately; it does not change application code or the ledger digest.

Acceptance requires an independently approved exact candidate before merge,
preservation of the original 87 entries, a valid 88-entry ledger and matching
API pin, status-only execution-plan wording with all outcome/qualification
limits intact, and successful post-merge API/asset verification from the
current checkout. Report implementation tests separately from service checks.

## Explicit exclusions

No runtime source behavior changes, Phase 4 campaign execution, simulator or
replay generation/review, training, campaign/private artifact access, GPU/HIP
or device use, qualification, promotion, or dashboard UI redesign. This plan
does not claim the corrected status rows have been exercised in a new E2E
evaluation.
