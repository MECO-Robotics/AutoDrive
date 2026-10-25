# Plan to repair the Phase 0 baseline provenance binding

**Status:** planner artifact only; no ledger, code, or service changes have been made.
**Planning base:** `69a67037bb8e51031c0ac9ef5914e7bf82fa7ae4`.
**Observed failure:** `/api/curriculum-status` reports `untrusted_reason=phase0-baseline-heldout-contracts: planner summary artifact is invalid`.

## Objective and scope

Restore trust in the existing `phase0-baseline-heldout-contracts` provenance row with the smallest correction: update only its stale planner-artifact binding and planner/merge attribution to the currently reviewed Phase 0 plan. Preserve the implementer evidence, reviewer artifact and approval, substantive limits, and the separate 85th simulated score-ledger tranche. The repair must not broaden the baseline claim or imply calibration, simulator, qualification, or runtime completion.

This task changes only the Phase 0 row in `docs/MECO_THREE_PASS_PROVENANCE.json`, the raw-byte digest pin in `frc_defense/dashboard_api.py`, and the minimum provenance/status documentation needed to explain the repair. It does not modify the Phase 0 plan, scorer implementation, training, devices, or service state during planning.

## Verified current evidence

- The row's planner artifact points to `docs/plans/phase0-baseline-heldout-calibration-20261025.md` but pins obsolete SHA-256 `504484dd9c38446b92faac2837dd56fb0f583643f8bc3dc70ecf21d09afb6fa4`.
- The current bytes at that same path have SHA-256 `95e1b9185509259af952fab7fc956c12974d50aacf43525302275f859f917d49`.
- The current plan text is the reviewed replacement authored at `a5aebfb034e0329ed66f097ea38d8cf4e93723ee`; its merge/mainline attribution is `81dcf7ab668c6cb40d1e2e9c83648f2894d7c361`. The old planner commit recorded in the row, `f39ab80986e0bbb1db1adc880fa99bc7d7df169c`, refers to an earlier revision and must not remain the sole attribution for the current artifact.
- The reviewer summary artifact remains valid at `docs/implementation-provenance/phase0-baseline-freeze-contracts-20261025.md`, SHA-256 `95e6a5a1d15d4eaf606d2bee16c7cf6a039a8d5a6ef94147a4b19cf8f5b25c78`; its reviewed implementation candidate remains `16aecdc0c9427cdd45c6f7a6f16f1bb07462ca1d`.
- The row currently describes offline code contracts only and explicitly says no production baseline/partition freeze, runtime campaign, calibration, margin review, qualification protocol, qualification, or promotion. Preserve this text and its implementer/reviewer contents.
- Current dashboard validator checks the manifest raw-byte SHA against `_EXPECTED_THREE_PASS_PROVENANCE_SHA256`, checks each planner artifact's exact path and SHA, and returns `validated_summary_only` only after validating all rows. The provenance ledger currently has 85 rows, including `phase0-simulated-score-ledger-replay`; that row must remain intact.

## Required repair

1. In the existing Phase 0 row, set planner `artifact.sha256` to `95e1b9185509259af952fab7fc956c12974d50aacf43525302275f859f917d49`. Keep its path, artifact kind, planner status, reviewer summary binding, and all substantive evidence/limits unchanged.
2. Correct the planner lineage so it names the current artifact's source commit `a5aebfb034e0329ed66f097ea38d8cf4e93723ee` and its mainline merge attribution `81dcf7ab668c6cb40d1e2e9c83648f2894d7c361`. Represent source and merge separately (for example `commit` and `merge_commit`) so neither is misreported as the other. Remove the stale `f39ab80986e0bbb1db1adc880fa99bc7d7df169c` planner attribution unless retained clearly as historical context rather than current-plan origin.
3. Make no changes to the Phase 0 reviewer artifact, approved implementation candidate, implementer commits/paths, row limits, other tranche rows, or the score-ledger row. In particular, the manifest remains 85 rows and preserves `phase0-simulated-score-ledger-replay` exactly.
4. Recompute the SHA-256 over the final raw bytes of `docs/MECO_THREE_PASS_PROVENANCE.json` and update only `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` to the resulting value. Do not change validator behavior, manifest schema, or source-plan digest.
5. Add a short dated status note stating that the API previously rejected this stale planner digest and that this scoped metadata correction restores the binding. Make clear that it does not add baseline, partition, calibration, simulator, runtime, or qualification evidence.

## Acceptance and validation gates

- Independent reviewer inspects the exact candidate diff and confirms that only the stale planner binding/attribution and necessary digest/status documentation changed; confirms the reviewer/implementer evidence, all substantive limits, and score-ledger row are preserved.
- Static inspection confirms the planner artifact hash equals the current plan file hash, source and merge commits are labeled accurately, the reviewer summary hash still matches, the ledger still contains 85 unique rows, and the score-ledger row is unchanged.
- Before claiming dashboard recovery, integrate the reviewed candidate, restart the dashboard service so it serves the current checkout, and verify `/api/curriculum-status` returns HTTP 200 with `three_pass_provenance.status == "validated_summary_only"`, 85 rows, and no `untrusted_reason`. Confirm current frontend assets and backend are served from that checkout as required by `AGENTS.md`.
- Report the service/currentness check and exact endpoint result. If restart or endpoint validation cannot be completed, state that API recovery is unverified; a corrected hash alone is not dashboard validation.
- This work does not require tests, simulator workloads, replay generation, training, or device access. Do not claim any of those results.

## Execution sequence

1. Apply the minimal ledger metadata correction and digest pin update, then add the dated status note.
2. Inspect the JSON structurally and review the exact diff; do not run simulator or training workloads.
3. Commit the metadata repair and digest update incrementally, request an independent exact-head review, resolve any findings, then integrate and perform the required dashboard restart/currentness check.
4. Report the exact integrated head, API status and row count, currentness evidence, and unchanged outcome-evidence limits.

