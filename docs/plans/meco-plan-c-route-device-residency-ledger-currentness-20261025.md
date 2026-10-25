# Plan C route-device residency provenance and dashboard plan

**Version:** 1
**Status:** Pass 1 plan only; no ledger, source, test, service, or training changes
**Base:** `dd6ca02af38d998eec8f956645de88065c19f7b0` (`origin/main`)
**Plan date:** 2026-10-25

## Objective

After the `meco-plan-c-route-device-residency` code candidate receives its exact-head Pass 3 approval, register that scoped three-pass record in `docs/MECO_THREE_PASS_PROVENANCE.json`, update the dashboard API's ledger digest pin, and verify the integrated dashboard serves the resulting ledger and all eight frontend assets from the same checkout. This registration is necessary for the dashboard to reflect the three-pass work.

Use three distinct checkpoint types: **C** is the implementation/code candidate, reviewed first for code correctness; **D** is the separate provenance-ledger candidate, reviewed for ledger correctness while its own reviewer state remains pending; **E** is the post-review finalizer that records D's approval in the ledger. A second Pass 3 then reviews E. The ledger row must never contain its own checkpoint SHA: D's row references C and C's approved code-review artifact, but has no `candidate_commit` or `reviewer.approved_commit`; E sets both `implementer.candidate_commit` and `reviewer.approved_commit` to D. The reviewer artifact in E is the approved review of D, not the code review of C. Keep E, main integration, and the second-review verdict outside the digest-pinned ledger. Preserve every original ledger entry, value, and order.

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

### 1. Checkpoint C: implementation candidate and code review

The implementation team produces code candidate **C** in an isolated worktree. Obtain an independent exact-head Pass 3 review of C. That review concerns the Plan C route-device implementation, its tests, and its implementation evidence. It is not a review of the eventual ledger candidate or finalizer.

Record C's exact full SHA, the code-review artifact path and SHA-256, the approved verdict, and C's exact changed source/test paths in the subsequent row under explicitly named implementer fields, for example `code_candidate_commit`, `code_review_artifact`, `code_review_verdict`, and `code_candidate_paths`. The established `implementer.paths` field must identify the implementation paths changed by C. D's provenance bookkeeping paths are recorded in D's separate review artifact, not misattributed to C. The code-review artifact must be the persisted Pass 3 approval of C. Do not conflate it with the later review artifact for D.

### 2. Checkpoint D: pending ledger candidate and ledger review

Create separate provenance checkpoint **D** after C is approved. D adds the scoped planner summary, a single new row, focused provenance tests, a status-only update in `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md`, and the digest pin matching D's ledger bytes. The row uses requirement ID `meco-plan-c-route-device-residency`; it binds the planner to the exact approved Plan C plan artifact and SHA. Its implementer fields bind C's exact implementation paths and code candidate SHA, including C's approved code-review artifact path/hash/verdict. Record D's bookkeeping changed files in its separate review evidence, but do not include D's own SHA in its content.

In D, the row's ledger-review fields remain explicitly pending: no `reviewer.approved_commit` and no approved reviewer artifact for D. The row is fail-closed and D must not be represented as a valid/servable approved manifest. The digest pin may match D's pending ledger bytes, but the dashboard validator must reject the row as unapproved. Preserve all 88 prior entries, objects, values, and order exactly. Push D and request an independent exact-head Pass 3 review of D for ledger correctness: scope, role binding, code-review evidence for C, pending ledger-review state, digest consistency, prior-row preservation, tests, and status wording. Do not claim that the dashboard validates or serves D.

### 3. Checkpoint E: post-review finalization and second review

Only after D receives its exact-head ledger-review approval, create finalization checkpoint **E**. In E, change the new row from pending to approved summary and set both `implementer.candidate_commit` and `reviewer.approved_commit` to D's exact SHA. Persist D's approved reviewer artifact path/hash/verdict in the `reviewer` object. Keep C's code review separately and explicitly identified in the implementer fields; the two approvals must not be conflated. D has no self-SHA field in its content; E may cite D because E is a different commit.

Keep E's own SHA, eventual main integration SHA, and the second-review verdict/artifact out of the ledger. Recompute the manifest SHA-256 from final bytes and update only `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in `frc_defense/dashboard_api.py`; leave `_EXPECTED_PLAN_SHA256` unchanged. Preserve schema, status, evidence class, plan path and SHA. Append exactly one row and preserve all previous entries, values, and order. Change top-level scope, artifact-origin, and limits only as needed to describe this scoped registration. Update execution status with summary-only wording and no behavior/replay/current service/GPU-only/qualification/full-plan claims.

Run focused offline provenance tests and diff checks on E. Tests must establish: the final manifest validates as summary-only; `implementer.code_candidate_commit` equals C and its code-review artifact/verdict are valid; `implementer.candidate_commit` and `reviewer.approved_commit` both equal D; D's reviewer artifact exists and hashes correctly; the ledger digest equals the API pin; the previous 88 entries are unchanged and in order; and a D-like pending row with no `reviewer.approved_commit` is rejected. Keep existing negative validator tests. These checks do not validate route behavior, simulator execution, GPU residency, or dashboard service currentness.

Push E as a distinct checkpoint and request a second independent exact-head Pass 3 review. The second reviewer checks E's full SHA, the stored C code approval, D's exact pending-candidate approval, the D reviewer artifact, exact ledger/API digest, preservation of prior entries, focused test evidence, and that E/integration/second-verdict data are absent from the ledger. Do not integrate until this second review approves E. Store the second-review verdict and artifact outside the ledger.

### 3. Integration and currentness evidence

After second-review approval, integrate the finalization through the approved project workflow. Then restart the dashboard service from the exact integrated checkout so backend modules are reloaded. Record UTC timestamp, service name, active state, PID, invocation ID, working directory, integrated commit, and source identity.

Verify `/api/curriculum-status` and `/api/meco-status` both return HTTP 200. In the curriculum response, verify `three_pass_provenance.validation_status == "validated_summary_only"` and `three_pass_provenance.status == "scoped-transcript-summaries"`; also verify the new requirement ID is present exactly once, the expected total row count is 89, and there is no `untrusted_reason`. Verify the manifest SHA-256 equals `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in the same checkout. Report any unrelated MECO data-read errors separately from ledger validation.

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

Persist these results in a separate currentness evidence note. Include C and its code-review artifact/verdict; D and its ledger-review artifact/verdict; E's finalizer SHA; integrated main SHA; second Pass 3 verdict/artifact; service identity; both API responses including the exact provenance field paths; ledger/API-pin digest; and all eight local/served asset hashes. Link the note from the status document. If writing the note requires a later documentation commit after the service check, identify that commit separately. Do not add E, integration, or second-review data to the ledger.

Only after all post-merge checks pass may the status document say the integrated dashboard currently serves the updated three-pass ledger and assets. If the service or any endpoint/asset check is unavailable or mismatched, report currentness as incomplete; do not reuse pre-update or stale replay evidence.

## Focused implementation files and validation

Expected ledger/dashboard/test touchpoints are:

- `docs/MECO_THREE_PASS_PROVENANCE.json` — append one summary-only entry, preserving the existing entries exactly and in order.
- `frc_defense/dashboard_api.py` — update only `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` after finalization.
- `tests/test_meco_three_pass_provenance.py` — extend focused assertions for C's distinct code-review binding, D pending rejection, E's exact D approval binding, row preservation, count, and digest pin.
- `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md` — status-only summary; currentness wording remains pending until proven.
- One scoped implementation summary and one separate post-merge currentness evidence note under `docs/implementation-provenance/`.

The implementer must record the exact focused test command and results at E. No broad simulator or training test is authorized by this provenance tranche. Do not generate or claim a replay for this metadata-only work. Per repository instructions, any future test that checks simulated game behavior must be accompanied by representative seeded end-to-end simulation, inspected fresh replay, and dashboard currentness evidence; this ledger update alone supplies none of those results.

## Acceptance and nonclaims

Acceptance requires all of the following:

- First Pass 3 approves code candidate C and persists its code-review artifact.
- Ledger candidate D references C and its approved code-review artifact, has no self-SHA field, and remains pending/fail-closed until a distinct exact-head ledger review approves D.
- Finalizer E records `implementer.candidate_commit == D` and `reviewer.approved_commit == D`, along with D's ledger-review artifact; C's code review remains clearly distinguished in the implementer fields.
- Finalization leaves E, main integration, and the second-review verdict outside the ledger; all original rows remain unchanged and ordered.
- Focused provenance tests pass on the finalization state; the API digest pin matches the exact final ledger bytes.
- Second Pass 3 approves exact finalizer E before integration; the verdict is recorded only in separate currentness evidence.
- Post-integration dashboard service, both API endpoints, ledger validation, and all eight frontend asset hashes are verified from the same current checkout and recorded separately.

This process registers one scoped three-pass record. It does not establish that Plan C route behavior was tested in simulation, that the GPU path meets the full GPU-only request, that a replay is current or reviewed, that training occurred, or that any track is qualified. No replay/currentness claim is made before the post-merge service and asset checks are complete.
