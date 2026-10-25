# Plan A fixed-capacity integration and dashboard ledger

**Pass:** 1 — planner scope  
**Status:** proposed; no integration, ledger write, service restart, or runtime validation is authorized by this plan artifact alone.  
**Current main:** `2cddbcaf10a72a62e3cc0feb3fa83410a151eaea`  
**Reviewed source candidate:** `e116941c439020b8a4bf7447972d9229190f1fbe` on `codex/plan-a-device-decision-20261026`  
**Pass 3 verdict:** approved exact candidate `e116941c439020b8a4bf7447972d9229190f1fbe`, against candidate parent `b9489cfa615a236db2cbf1a65402ee06ed515804`.

## Objective

Integrate the exact-head-reviewed fixed-capacity Plan A graph and device-residency increment into the current main frontier, record its scoped three-pass evidence in the summary-only v2 provenance ledger, update the API digest pin in a separately reviewed finalizer commit, and verify the dashboard serves the resulting checkout and frontend assets.

This tranche records code/review/currentness evidence only. It does not establish HIP execution, WX9100 parity or profiling, simulator behavior, campaign stage passage, deployment readiness, qualification, or completion of the full learning-track plan.

## Inputs and boundaries

- Work only in the isolated HDD-backed worktree rooted at this plan's checkout. Current main is dirty only in the primary checkout at `frc_defense/csrc/tensor_collision_multi_hip_kernel.hip`; preserve that file byte-for-byte and do not carry it into the integration worktree.
- Merge the full reviewed candidate branch, whose merge base is `f981c45891da924cb4529035fe6bd98fe551bc97`, into current main. Keep current-main Plan C and dashboard provenance commits. Resolve conflicts by preserving the current main behavior and applying only the reviewed Plan A increment; the integrated merge is a new candidate and is not covered by source-candidate review.
- The live r4 training supervisor currently owns both WX9100 device leases. Do not stop, reconfigure, inspect devices through workloads, or launch competing GPU/simulator tasks. The campaign is diagnostic (`evaluation_binding` is null), and its scheduler records exposure rather than promotion. Dashboard metadata checks may run only after the required finalizer and service restart.
- The source candidate's implementation report records `py_compile` and `git diff --check` passing. No pytest, HIP compilation/import/profile, GPU work, simulator run, replay, or dashboard currentness check was performed. Retain these exact limits.

## Plan amendment: correct inherited Plan C current-state summaries

The current-main repository already contains authoritative later Plan C evidence that is not reflected in the current ledger summary. The evidence is recorded in `docs/implementation-provenance/plan-c-device-resident-selection-currentness-20261025.md` and `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md`. Before implementing the Plan A ledger entry, correct exactly these four existing summary strings in the Plan A ledger candidate: top-level `scope`, top-level `artifact_origin`, the final Plan C item in top-level `limits`, and `tranches[requirement_id=meco-plan-c-route-device-residency].limits`.

Use `docs/implementation-provenance/plan-c-device-resident-selection-currentness-20261025.md` and the matching Plan C section in `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md` as the evidence. They record finalizer E `0ce905a77cd5d7ae92dea6917999cf034892a6ea` passing its second exact-head Pass 3 review, integration at `8e0ad89aad4721d9f58bcfb50e27d8394f2f0fa0`, and post-restart dashboard verification at `2026-10-25 16:04 UTC`: both APIs returned HTTP 200, the ledger validated as summary-only with the Plan C entry present once, and all eight frontend routes matched the integrated checkout.

The four corrected strings must say that C and D retain their separate historical approvals, E passed the second exact-head review and was integrated, and dashboard currentness was verified as recorded in that evidence note. Keep the existing GPU/HIP parity and profiling, simulator/replay, runtime, training, qualification, and whole-system GPU-only limitations. Do not change any tranche's `planner`, `implementer`, or `reviewer` objects, candidate/approval SHAs, verdict details, artifact references, or any other existing ledger field. In particular, retain the historical D reviewer verdict that says its approval is distinct from E; it remains accurate. Do not add a duplicate Plan C tranche.

This amendment changes the required review sequence: obtain a fresh Pass 1 approval of this amended plan before any ledger edit; then the corrected ledger candidate must receive a fresh exact-head Pass 3 review. The already completed Plan A code-only integration review remains a separate historical gate and is not reopened by this summary correction. No ledger or implementation change is authorized until the amended plan receives Pass 1 approval.

## Bounded sequence

1. **Fresh Pass 1 plan review:** independently review the amended exact plan artifact and confirm its candidate/base SHAs, protected user change, four-string Plan C summary correction boundary, integration review, finalizer review, and dashboard verification requirements. Do not edit the ledger or implementation before approval.
2. **Pass 2 integration candidate:** merge the reviewed source branch into this current-main worktree. Inspect the complete diff from `2cddbcaf`, confirm current Plan C files are retained and the protected user file remains untouched in the primary checkout, then commit and push the code-only integration candidate. This candidate is not yet represented in the ledger.
3. Obtain exact-head Pass 3 review of that code-only integration candidate against its actual parent. It must review the integrated tree, not infer approval from source-candidate review. Any requested changes become a new integration candidate and get a new exact-head review.
4. After integration review approval, add one ledger entry with a concise repository-local planner/reviewer summary. Record the exact planner artifact, reviewed source candidate SHA, separately reviewed code-only integration SHA, ledger implementation commit/paths, source-candidate and integration verdicts, and scoped offline check evidence. Correct only the four inherited Plan C current-state summary strings specified above; preserve all other fields and attributions. The ledger commit must refer to the already-reviewed code-only integration SHA; do not create a self-referential commit field. Preserve schema `meco-three-pass-provenance-v2`, `scoped_transcript_summary`, every existing entry, and accurate top-level scope/limits. Commit and push this ledger candidate, then obtain a fresh exact-head Pass 3 review of the corrected ledger candidate.
5. After ledger-candidate approval, make a separate finalizer commit that updates only `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in `frc_defense/dashboard_api.py` to the byte digest of the final ledger. Preserve `_EXPECTED_PLAN_SHA256`. Push and obtain a separate exact-head review of this finalizer.
6. Once code integration, ledger candidate, and finalizer reviews approve, restart the dashboard service(s) that serve this checkout. Verify service identity/health; `/api/meco-status` and `/api/curriculum-status` return HTTP 200; and inspect `/api/curriculum-status.three_pass_provenance` specifically. Require its `status` to be `scoped-transcript-summaries`, its `validation_status` to be `validated_summary_only`, no `untrusted_reason`, and the new entry ID/count in the response. Independently compute SHA-256 over the local `docs/MECO_THREE_PASS_PROVENANCE.json` bytes and confirm equality with `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in the checked-out `frc_defense/dashboard_api.py`; do not imply the endpoint exposes the manifest digest if it does not. Verify all eight frontend asset routes match the local checkout byte-for-byte. Record timestamp, checkout SHA, service identity, endpoint status/ledger fields, local digest and source pin, and local/served asset hashes in a currentness evidence note.
7. Do not generate a replay as part of this metadata/source integration. A later behavior test must be paired with the seeded end-to-end simulator evaluation, replay inspection, and dashboard-currentness gates in `AGENTS.md`, after the device lease is available.

## Acceptance criteria

- The integrated candidate contains the complete reviewed Plan A source increment and retains newer Plan C/main provenance without unrelated changes.
- The primary checkout's pre-existing HIP edit remains byte-identical.
- The ledger contains exactly one new accurately attributed scoped entry; only the four specified current-state summary strings are corrected using base-main evidence, all historical attributions/approval facts and other existing fields are preserved, and the plan digest pin is preserved.
- Fresh Pass 1 approves this amended plan before ledger or implementation edits; fresh exact-head Pass 3 approves the corrected ledger candidate.
- Pass 1 approves this plan; exact-head Pass 3 separately approves the code-only integration candidate, ledger candidate, and API pin finalizer.
- The post-restart dashboard evidence proves the backend and all eight served assets correspond to the final checkout; it separately proves API ledger validation/entry visibility and the local ledger-byte digest equals the source pin.
- Claims remain limited to source integration, static checks, exact-head scoped review, ledger validity, and dashboard currentness. No runtime/qualification or first-stage training passage is implied.

## Open work outside this tranche

The user’s broader objective remains active: complete the learning-track plan through the curriculum milestone and keep all five tracks training. The live campaign currently shows `pickup_simple` exposure 4/10, cycle 5, diagnostic split, and no evaluation binding; `promotion_status` is explicitly not evaluated by the exposure scheduler. The Plan A HIP dispatch/profile gate is still closed, and no runtime memory admission, simulator/replay, Phase 2 warm-start handoff, matched held-out evaluation, or qualification evidence is established by this integration.
