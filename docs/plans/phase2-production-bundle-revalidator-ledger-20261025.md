# Phase 2 production bundle revalidator: provenance ledger update

**Status:** planner scope only; no ledger or dashboard change
**Planning base:** `main` / `origin/main` at `79e2d228cde22932b8cf437337be8c1de7859637`
**Existing provenance:** v2 summary-only ledger, 78 rows, SHA-256 `670b4420288cc15d94e7b685a4467a84a868669820dafc080da17dc95adaa472`; `frc_defense/dashboard_api.py` pins the same digest.

## Objective

Register the reviewed Phase 2 production bundle revalidator as exactly one new v2 `scoped_transcript_summary` row. Preserve all current rows and their order/value, update the dashboard ledger digest pin, and retain post-integration evidence that both dashboard APIs validate 79 rows and all eight frontend assets match the same checkout after service restart.

This is a ledger and dashboard-currentness tranche. It does not establish production dataset generation, warm-start readiness, training, simulator behavior, GPU/device evidence, qualification, deployment, or overall plan completion.

## Verified source and review facts

- Planner source for this revalidator: `docs/plans/phase2-warmstart-production-execution-20261025.md`, committed as `eed7a2f1d7a35ec9b89b49c547837ef492b6749c`, SHA-256 `04322980aa10eca3b858f3abfc3cfd5fa49ad653a41ca302a58fac587259037b`.
- Implementation candidate/source commit: `3a0d17c001dc35c05659d70180850555258be711` (`Revalidate multi-case warm-start bundles on manifest use`).
- Integrated and independently approved exact head: `79e2d228cde22932b8cf437337be8c1de7859637`. The integration has the implementation candidate as first parent and the then-current main as second parent; keep candidate and integration attribution separate.
- Independent review approved exact head `79e2d228cde22932b8cf437337be8c1de7859637`. The reviewer reports 68 focused tests passed, `git diff --check origin/main...HEAD` clean, no simulator/device work, and verified that the revalidator checks descriptor/current source/protocol/device bindings, per-case receipts and pooled sidecars, re-derives the three combined datasets, and reruns production validation for all five artifacts. Readiness and qualification remain false and runtime errors are not broadly swallowed. Report these as scoped review evidence only; they were not rerun for this metadata pass.
- The implementation changed exactly four source/test paths from the prior coordinator integration. Their SHA-256 values at both candidate and current integration are:

  | Implementation path | SHA-256 |
  |---|---|
  | `frc_defense/meco_phase2_multicase.py` | `b736cb46fab62eb9982c5450847511c1858e1aa628b5f4f6aa12c733a094fb9b` |
  | `frc_defense/meco_training_manifest.py` | `e5276b2dce232c5dbcfa4b17aa9d2c66ecfccbf780c2384f6331e319b5377609` |
  | `tests/test_meco_phase2_multicase.py` | `476b9216c4850302d29dd1fdeacdc17ac1e86f7442f04c7e49fcf817ae415d2f` |
  | `tests/test_meco_training_manifest.py` | `69a6676b87aa6ef7d8f8d29dc886eb06e328a8565ff030e41aa3b074ed6ba99c` |

- The implementation/reviewer summary is not yet present. Create `docs/implementation-provenance/phase2-production-bundle-revalidator-20261025.md` during the ledger implementation pass and include the four path hashes above. That summary is the fifth persisted path/hash in this metadata tranche; do not put it in the ledger's `implementer.paths` list as if it were a source file changed by the revalidator. Compute its actual SHA-256 after writing and bind it in the reviewer evidence object. Do not invent reviewer identity, tests, runtime outcomes, or detailed findings.
- Existing currentness evidence for the prior 78-row ledger is `docs/implementation-provenance/phase2-multicase-dashboard-currentness-20261025.md`. Its service response and asset hashes are historical baseline, not proof of the 79-row post-update state.

## Implementation and integration sequence

1. Use a clean isolated HDD-backed worktree at `79e2d228...`. Keep the root checkout's unrelated user-modified HIP source untouched. Do not inspect, stop, or otherwise change active training processes, checkpoints, or device leases for this metadata-only pass.
2. Add the concise implementation/review summary named above. Identify the planner artifact/commit/hash, implementation candidate `3a0d17c...`, integration/reviewer-approved head `79e2d228...`, the reviewer-reported 68 tests and diff check, the four exact implementation path hashes, and the no-simulator/no-device limits. Make no data, readiness, qualification, or deployment claims.
3. Append exactly one row, with a unique ID such as `phase2-production-bundle-revalidator`, to `docs/MECO_THREE_PASS_PROVENANCE.json`. Retain schema `meco-three-pass-provenance-v2`, status `scoped-transcript-summaries`, evidence class `scoped_transcript_summary`, the existing source-plan path and SHA, and all 78 prior rows in their existing order and unchanged values. Bind the planner role to the existing production-execution plan and exact SHA; bind implementer candidate and integration commits separately and list exactly the four source/test paths; bind reviewer approval to exact commit `79e2d228...`, verdict `approved`, and the persisted reviewer summary path/hash. Include the four file hashes in the summary artifact; preserve any other top-level summary metadata unless a precise row-count/scope update is required.
4. Recompute the final ledger SHA-256 and change only `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in `frc_defense/dashboard_api.py`. Leave `_EXPECTED_PLAN_SHA256` unchanged. Before committing, verify the new manifest validates structurally, the pin equals the exact ledger bytes, the count is 79, and comparing parsed old/new ledgers proves that each of the prior 78 rows remains unchanged and in order.
5. Keep commits incremental and push each committed stage as a separate commit: reviewer/implementation summary; one-row ledger append; digest-pin update; then post-integration currentness evidence. Do not squash or amend away the auditable increments. Review the final ledger/pin diff before integrating to `main`.
6. After integration, restart `frc-defense-dashboard.service` from the integrated checkout so Python modules reload. Record UTC time, service active state, PID/invocation, working directory, integrated commit and source identity. Request both `/api/meco-status` and `/api/curriculum-status`; each must return HTTP 200. Require `validated_summary_only`, exactly 79 tranches, the new row present, no `untrusted_reason`, and the API ledger digest equal to the final ledger SHA-256. Report unrelated data-read errors separately from ledger validation.
7. After restart, fetch all eight frontend routes and compare each served body's bytes/hash to its file in the same local checkout: `/` (`frc_defense/dashboard_static/index.html`), `/meco` (`meco.html`), `/assets/dashboard.css`, `/assets/dashboard.js`, `/assets/meco.css`, `/assets/meco.js`, `/assets/field_rendering.js`, and `/assets/training_rendering.js`. Record HTTP status and local/served SHA-256 per route in a concise currentness note. Recompute after the update; prior 78-row hash evidence is not reusable as post-change proof.
8. No replay is needed or authorized for this metadata-only change. Do not run tests, simulation, training, GPU/device checks, or qualification work in this tranche.

## Acceptance criteria

- Exactly one v2 scoped summary row is appended, accurately distinguishing feature candidate `3a0d17c...` from integration and independently approved exact head `79e2d228...`.
- The reviewer record reports only the verified 68 focused tests, clean diff check, reviewed validation areas, and absence of simulator/device work. The ledger row lists the four source/test implementation files; the persisted summary artifact is separately hash-bound as reviewer evidence.
- All four implementation path hashes match both candidate and integrated checkout. The implementation/reviewer summary has its own actual fifth path/hash and reports bounded claims.
- All previous 78 row values and order are preserved, the final API pin equals the ledger byte digest, and the pinned source plan digest is unchanged.
- Following integration and service restart, both status endpoints prove 79 validated summary rows from the integrated checkout and every one of the eight frontend bodies matches that checkout.
- No data generation, warm-start readiness, training outcome, simulator/replay result, GPU/device evidence, qualification, promotion, or full-plan completion is claimed.

## Limits

The implementation's 68 tests and reviewer approval cover software contract behavior only. Revalidating a manifest at consumption time does not prove that any production datasets or weights exist, that the five tracks have been trained from them, or that learned policies improve match outcomes. Dashboard currentness verifies serving only.
