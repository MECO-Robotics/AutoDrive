# Plan C route-device residency provenance and dashboard plan

**Version:** 1
**Status:** Pass 1 plan only; no ledger, source, test, service, or training changes
**Base:** `dd6ca02af38d998eec8f956645de88065c19f7b0` (`origin/main`)
**Plan date:** 2026-10-25

## Objective

After the `meco-plan-c-route-device-residency` implementation candidate receives its first exact-head Pass 3 approval, register that scoped three-pass record in `docs/MECO_THREE_PASS_PROVENANCE.json`, update the dashboard API's ledger digest pin, and verify the integrated dashboard serves the resulting ledger and all eight frontend assets from the same checkout. This registration is necessary for the dashboard to reflect the three-pass work.

Use the candidate → first review → finalization → second review → integration → currentness sequence below. The candidate entry remains explicitly pending and fail-closed until its exact SHA is approved. The final ledger records only that first-reviewed candidate as the approved implementation head. Finalizer, integration, and second-review evidence belongs in a separate currentness artifact. Preserve every original ledger entry, value, and order.

This is provenance and serving-currentness work only. It does not validate route behavior, complete GPU-only execution, establish a simulator result, or qualify a policy. No replay or dashboard-currentness claim exists until post-merge checks are completed and recorded.

## Audited base and constraints

- Base is `dd6ca02af38d998eec8f956645de88065c19f7b0`. At this base the dashboard API pins the ledger bytes with `_EXPECTED_THREE_PASS_PROVENANCE_SHA256`; the source-plan pin `_EXPECTED_PLAN_SHA256` is separate and must remain unchanged.
- The reviewed Plan C implementation plan is `docs/plans/gpu-resident-route-decision-slice-20261025.md`, SHA-256 `8c6c67ffb2934fa3a4ce40b54c5cc2f2e2cb4877b5eedac9c31a2936a6879b5a`. The final ledger planner artifact must bind this exact document and digest; do not use a different plan branch or mutable path content.
- The manifest uses schema `meco-three-pass-provenance-v2`, status `scoped-transcript-summaries`, and top-level evidence class `scoped_transcript_summary`. The dashboard validator only accepts completed planner summaries, implementation records, and approved reviewer summaries. A pending candidate must therefore not be represented as approved or reported as validated.
- The base manifest has 88 rows and the current focused test asserts that count. The finalized manifest is expected to have 89 rows after appending exactly one new row; confirm this by parsed comparison during implementation.
- Existing contract coverage is in `tests/test_meco_three_pass_provenance.py`; its current-manifest test asserts the existing row count and exact digest pin. Update those assertions only as part of the implementation sequence and retain the mutation/rejection coverage.
- The main worktree had a pre-existing unrelated modification in `frc_defense/csrc/tensor_collision_multi_hip_kernel.hip` when this plan was prepared. Leave it untouched; implement from an isolated worktree.
- Do not alter `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`, its SHA pin, simulator or training code, active training, checkpoints, or dashboard service as part of this tranche.

## Required sequence

### 1. Candidate checkpoint and first exact-head Pass 3

Create the implementation candidate in an isolated worktree. It should contain the scoped planner summary, one new ledger row with a pending reviewer placeholder (no approved reviewer artifact yet), focused contract-test updates, a status-only update in `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md`, and the ledger digest-pin update required to make the candidate bytes internally consistent.

The new row must use requirement ID `meco-plan-c-route-device-residency`. Bind the planner to the exact approved Plan C device-residency plan artifact and SHA. Bind implementer commit/path metadata to the actual candidate and its exact changed source/test paths. Do not invent commit IDs, changed paths, reviewer records, or behavioral evidence; fill these from the actual implementation and review artifacts.

At this stage the candidate row must be explicitly pending and fail-closed: mark the reviewer approval as pending, omit `reviewer.approved_commit`, and ensure the dashboard validator cannot treat it as an approved summary. Preserve all previous entries, objects, and order exactly. This pending candidate is a review artifact, not a current dashboard manifest; do not claim that `/api/curriculum-status` validates or serves it.

Push the candidate and request independent Pass 3 review of its exact full SHA. The reviewer checks the three-pass summary's scope, source-plan and implementation bindings, tests, candidate row's pending state, exact preservation of prior rows, digest consistency, and status wording. Do not run or report the final acceptance tests as passing before review.

### 2. Post-review finalization checkpoint and second exact-head Pass 3

Only after the first Pass 3 approves the exact candidate SHA, create a separate finalization commit. In this finalizer:

1. Change the new row from pending to approved summary, set `reviewer.approved_commit` to the exact first-reviewed candidate SHA, and persist the approved reviewer artifact/path/hash from that first review. Set the candidate reference in the implementer record to the same approved candidate SHA where the established schema requires it.
2. Keep the finalizer commit SHA, eventual main integration SHA, and second reviewer verdict out of the digest-pinned ledger. Record those in the separate post-merge currentness evidence note.
3. Recompute the manifest SHA-256 from its final bytes and update only `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in `frc_defense/dashboard_api.py`. Leave `_EXPECTED_PLAN_SHA256` unchanged.
4. Preserve every pre-existing ledger entry byte/value-for-value and in its original order. Append exactly one new row. Preserve schema, status, evidence class, plan path, and plan SHA. Change top-level scope, artifact-origin, and limits only as required to describe this single scoped tranche.
5. Update the status-only execution-status section to cite the exact approved candidate and reviewer evidence. State that the entry records scoped implementation/review evidence only. Do not claim behavior, replay, current service state, GPU-only completion, qualification, or plan completion.

Run focused offline provenance tests and diff checks against this finalized state. Tests must establish: the final manifest validates as summary-only; the new row's `reviewer.approved_commit` equals the first-reviewed candidate SHA; the approved reviewer artifact exists and hashes correctly; the ledger digest equals the API pin; the previous entries are unchanged and in order; and a pending candidate state is rejected. Keep existing negative validator tests. These checks do not validate route behavior, simulator execution, GPU residency, or dashboard service currentness.

Push the finalization as a distinct checkpoint and request a second independent exact-head Pass 3 review. The second reviewer checks the finalizer's full SHA, the stored first candidate approval, the exact ledger/API digest, preservation of prior entries, focused test evidence, and that no finalizer/integration/second-verdict claim leaked into the ledger. Do not integrate until this second review approves the exact finalization head.

### 3. Integration and currentness evidence

After second-review approval, integrate the finalization through the approved project workflow. Then restart the dashboard service from the exact integrated checkout so backend modules are reloaded. Record UTC timestamp, service name, active state, PID, invocation ID, working directory, integrated commit, and source identity.

Verify `/api/curriculum-status` and `/api/meco-status` both return HTTP 200. In the curriculum response, verify the three-pass provenance result is `validated_summary_only`, the new requirement ID is present exactly once, the expected total row count is correct, and there is no `untrusted_reason`. Verify the manifest SHA-256 equals `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in the same checkout. Report any unrelated MECO data-read errors separately from ledger validation.

Fetch and hash all eight frontend routes after restart; each must return successfully and match the corresponding file byte-for-byte from the same checkout:

| Route | Checkout file |
|---|---|
| `/` | `frc_defense/dashboard_static/index.html` |
| `/meco` | `frc_defense/dashboard_static/meco.html` |
| `/assets/dashboard.css` | `frc_defense/dashboard_static/dashboard.css` |
| `/assets/dashboard.js` | `frc_defense/dashboard_static/dashboard.js` |
| `/assets/meco.css` | `frc_defense/dashboard_static/meco.css` |
| `/assets/meco.js` | `frc_defense/dashboard_static/meco.js` |
| `/assets/field_rendering.js` | `frc_defense/dashboard_static/field_rendering.js` |
| `/assets/training_rendering.js` | `frc_defense/dashboard_static/training_rendering.js` |

Persist these results in a separate currentness evidence note. Include the exact first-reviewed candidate SHA, finalizer SHA, integrated main SHA, second Pass 3 verdict/artifact, service identity, API responses, ledger and API-pin digest, and all eight local/served asset hashes. Link the note from the status document. If writing the note requires a later documentation commit after the service check, identify that commit separately. Do not add finalizer, integration, or second-review data to the ledger.

Only after all post-merge checks pass may the status document say the integrated dashboard currently serves the updated three-pass ledger and assets. If the service or any endpoint/asset check is unavailable or mismatched, report currentness as incomplete; do not reuse pre-update or stale replay evidence.

## Focused implementation files and validation

Expected ledger/dashboard/test touchpoints are:

- `docs/MECO_THREE_PASS_PROVENANCE.json` — append one summary-only entry, preserving the existing entries exactly and in order.
- `frc_defense/dashboard_api.py` — update only `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` after finalization.
- `tests/test_meco_three_pass_provenance.py` — extend focused assertions for the new entry, pending rejection, exact candidate approval binding, row preservation, count, and digest pin.
- `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md` — status-only summary; currentness wording remains pending until proven.
- One scoped implementation summary and one separate post-merge currentness evidence note under `docs/implementation-provenance/`.

The implementer must record the exact focused test command and results at the finalized candidate. No broad simulator or training test is authorized by this provenance tranche. Do not generate or claim a replay for this metadata-only work. Per repository instructions, any future test that checks simulated game behavior must be accompanied by representative seeded end-to-end simulation, inspected fresh replay, and dashboard currentness evidence; this ledger update alone supplies none of those results.

## Acceptance and nonclaims

Acceptance requires all of the following:

- First Pass 3 approves the pending candidate at its exact full SHA.
- Finalization pins only that first-reviewed candidate SHA and the approved first reviewer artifact in the ledger row; all original rows remain unchanged and ordered.
- Focused provenance tests pass on the finalization state; the API digest pin matches the exact final ledger bytes.
- Second Pass 3 approves the exact finalization commit before integration.
- Post-integration dashboard service, both API endpoints, ledger validation, and all eight frontend asset hashes are verified from the same current checkout and recorded separately.

This process registers one scoped three-pass record. It does not establish that Plan C route behavior was tested in simulation, that the GPU path meets the full GPU-only request, that a replay is current or reviewed, that training occurred, or that any track is qualified. No replay/currentness claim is made before the post-merge service and asset checks are complete.
