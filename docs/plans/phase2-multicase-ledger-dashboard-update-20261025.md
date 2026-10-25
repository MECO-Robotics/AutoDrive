# Phase 2 multi-case coordinator: provenance ledger and dashboard update

**Status:** planner scope only; no ledger change or dashboard/runtime verification
**Planning base:** `main` / `origin/main` at `d23b4d5e791c3eb37f5d8e48fd5a69c8e22c3dae`
**Existing provenance:** schema `meco-three-pass-provenance-v2`, 77 tranches, SHA-256 `1998ee1cffb6c657c0667dc55c1ec7d9d9fb51f9f5ade3397a5e88050387c532`; API pin matches at this base.

## Objective

Add exactly one accurate `scoped_transcript_summary` row for the reviewed Phase 2 multi-case five-track warm-start coordinator. Preserve the existing 77 rows and their attribution exactly, update only the ledger's scope/artifact-origin/limits as needed plus the new row, and change only `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in `frc_defense/dashboard_api.py`. After integrating the ledger update, restart the dashboard from that checkout and retain evidence that the backend validates all 78 summaries and that all eight served frontend assets match the same checkout.

This is metadata and dashboard-currentness work only. It does not establish a generated dataset, warm-start artifact, training readiness, simulator behavior, GPU execution, qualification, deployment, or completion of the source plan.

## Verified inputs and exact attribution

- Planner commit: `39040927ea05be978268a1488a3ba142e5ae9c28`.
- Planner artifact: `docs/plans/phase2-multi-case-five-track-warmstart-20261025.md`, SHA-256 `ebfb38a05d1bd9ca443c29f3b7af08401ea6dedaaaf58e68382480696daecf54`.
- Reviewed feature head: `8a7c077606f241a24b3438344c3f95f83e71c593`.
- Integration candidate/head: `d23b4d5e791c3eb37f5d8e48fd5a69c8e22c3dae`. Its second parent is `eed7a2f1d7a35ec9b89b49c547837ef492b6749c`; this is the separately pushed production-execution planner and does not replace the multi-case coordinator's feature attribution.
- Independent reviewer approved exact merge candidate `d23b4d5e791c3eb37f5d8e48fd5a69c8e22c3dae` for fast-forward integration. The reviewer reported 67 focused Phase 2 and manifest tests passed in 7.45 seconds and a clean `git diff --check`. The review checked positive shard adoption, malformed receipt/row retry, and `RuntimeError` propagation. No simulator, GPU, or device work was run.
- The five implementation paths changed from planner base `39040927e` to reviewed feature head `8a7c077` are exactly:

  | Path | SHA-256 at candidate and integration |
  |---|---|
  | `frc_defense/meco_phase2_multicase.py` | `663d8e8b99e391c3e7dd7c5733f96bf6a5e3e5f6f185471a92103a6642d63426` |
  | `frc_defense/meco_supervised.py` | `d54bf3817a3edf34f011927b8ee850598188a7276dde087a209d61c6d7b226e3` |
  | `frc_defense/meco_training_manifest.py` | `f769bc4d218fc0c63660110e1f81146837aff4c602de0875ea5d278d80eeb4bd` |
  | `tests/test_meco_phase2_multicase.py` | `1fc75188fdb8a29a5fc7023d742d8d065917077ddd465b1c2a2a8903b661da09` |
  | `tests/test_meco_training_manifest.py` | `cf066b257fded94d45ce28f5999e0e9bc92e3d6ae7963d4b6b095b19adfdd155` |

  The implementation summary must list these hashes as evidence and verify that each final integration blob remains byte-identical. Exclude unrelated Phase 4 files from this row. The reviewer approval is for the merge candidate above; do not imply that every historical intermediate commit was separately reviewed.
- The independent reviewer verdict is available in the scoped review record, not a persisted repository artifact. The ledger implementation pass must materialize a concise reviewer summary at `docs/implementation-provenance/phase2-multicase-five-track-warmstart-20261025.md` from the exact facts above, then compute its real SHA-256 and use that value in the reviewer artifact reference. Do not invent reviewer identity, findings, test coverage, or runtime evidence.
- Existing post-ledger-update baseline note `docs/implementation-provenance/meco-three-pass-ledger-dashboard-currentness-20261025.md` records the prior 77-row digest and eight frontend routes/hashes. Those are baseline evidence only; the 78-row dashboard check must be collected after the new ledger and digest are integrated.

## Required isolated implementation sequence

1. Use a clean isolated HDD-backed worktree based on the verified integration head. Preserve the unrelated user-modified HIP file in the root checkout. Do not alter training processes, output roots, leases, or campaign state.
2. Add the concise implementation/reviewer summary above. It must identify planner commit/artifact and hash, feature candidate `8a7c077...`, integration commit `d23b4d5...`, exact independent approval of `d23b4d5...`, reviewer-reported 67 tests / 7.45 seconds and diff check, all five changed paths and their SHA-256 values, plus the explicit no-simulator/no-GPU/no-device limits. Keep planner and reviewer artifact kinds/statuses distinct in the ledger row while they reference this summary only if the ledger contract permits; otherwise use the planner plan artifact for planner role and the new summary for reviewer role.
3. Append exactly one tranche with a unique requirement ID such as `phase2-multicase-five-track-warmstart-coordinator`. Use the v2 schema. Record planner commit `39040927...` and its exact plan artifact/hash; implementer feature candidate `8a7c077...`, integration commit `d23b4d5...`, and exactly the five scoped paths; reviewer `approved_commit` `d23b4d5...`, verdict `approved`, exact reviewer-summary path/hash, and bounded verdict detail. Ensure all implementation hashes are present in the persisted summary and correspond to the paths listed in the ledger row.
4. Preserve all prior 77 tranche JSON objects and their attribution unchanged. Preserve `plan_path`, `plan_sha256`, schema `meco-three-pass-provenance-v2`, status `scoped-transcript-summaries`, and evidence class `scoped_transcript_summary`. Update top-level scope, `artifact_origin`, and limits only as necessary to say that one additional scoped coordinator record is included; do not imply all Phase 2 warm-start work is complete.
5. Recompute the SHA-256 of the final ledger bytes, then update only `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in `frc_defense/dashboard_api.py`. Do not alter the evaluation plan digest or unrelated dashboard behavior.
6. Keep commits incremental and push each committed stage: provenance summary; ledger append; digest pin; then post-integration dashboard evidence note. Do not amend away earlier pushed stages. Review the exact final ledger and API pin before integration and verify a structural comparison proves that the original 77 rows are byte/value-identical and in the same order.
7. After integration, restart `frc-defense-dashboard.service` from the exact integrated checkout so the backend reloads. Record UTC timestamp, active service state/PID/invocation, cwd, checkout commit and source identity. Query both `/api/meco-status` and `/api/curriculum-status`; require HTTP 200, `validated_summary_only`, exactly 78 tranches, new final requirement ID present, no `untrusted_reason`, and the API's manifest digest equal to the final file SHA-256. Report any unrelated data-read errors separately without hiding ledger-validation state.
8. Fetch and hash each of the eight frontend routes after the restart, comparing served bytes to the local files from that checkout: `/` (`dashboard_static/index.html`), `/meco` (`meco.html`), `/assets/dashboard.css`, `/assets/dashboard.js`, `/assets/meco.css`, `/assets/meco.js`, `/assets/field_rendering.js`, and `/assets/training_rendering.js`. Save each local and served SHA-256 and HTTP status in the post-update evidence note. Recompute them after integration rather than assuming the previous 77-row hashes remain current.

## Acceptance criteria

- The new row is structurally valid v2 summary-only provenance and accurately separates planner commit, reviewed feature head, integration candidate, and the exact merge candidate independently approved by the reviewer.
- The reviewer summary gives only supported review claims: 67 focused tests in 7.45 seconds, clean diff check, positive shard adoption, malformed receipt/row retry, RuntimeError propagation, and no simulator/GPU/device run.
- The implementation summary contains all five exact candidate/integration file hashes. The row's implementation path list contains exactly those five scoped files and no unrelated Phase 4 files.
- The prior 77 ledger rows, order, and values are preserved. The final API digest pin exactly equals the final ledger file SHA-256; `_EXPECTED_PLAN_SHA256` remains unchanged.
- Post-integration dashboard evidence proves both APIs validate 78 rows from the integrated checkout and all eight frontend served bodies match that checkout after restart.
- No simulator replay is generated for this metadata-only change. No data-generation, warm-start-readiness, training, qualification, GPU, deployment, or whole-project completion claim is added.

## Explicit limits

The 67 reported tests validate scoped software contracts only. This provenance entry does not prove that the coordinator has produced any data or weights, that the five approaches were warm-started, or that the warm starts improve behavior. No simulator or replay is part of this ledger/dashboard task.
