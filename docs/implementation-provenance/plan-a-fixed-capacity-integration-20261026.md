# Plan A fixed-capacity integration provenance

This is a scoped transcript summary for the Plan A source integration and its ledger record. It records repository evidence and exact-head verdicts; it is not a runtime qualification report.

## Planner and code candidates

- Pass 1 plan: `docs/plans/plan-a-fixed-capacity-integration-20261026.md`, planner commit `c9c0ffe327593474d9122d028a950a0592c7c38e`.
- Reviewed source candidate: `e116941c439020b8a4bf7447972d9229190f1fbe`, based on `b9489cfa615a236db2cbf1a65402ee06ed515804`. Its independent exact-head Pass 3 verdict was approved.
- Integrated code candidate: `305b8d9fef3db7bf2ee8cd98c6740c55b8f7da95`, with parents `c9c0ffe327593474d9122d028a950a0592c7c38e` and `e116941c439020b8a4bf7447972d9229190f1fbe`. An independent exact-head Pass 3 review approved this integrated candidate. That verdict was communicated as a review message; no standalone reviewer artifact was persisted.

The integration brings the reviewed fixed-capacity Plan A graph and device-residency changes into the current-main frontier while preserving current Plan C and provenance. The source-candidate approval does not substitute for the separate integrated-candidate verdict.

## Plan C summary correction evidence

The corrected ledger summaries use [Plan C dashboard currentness evidence](plan-c-device-resident-selection-currentness-20261025.md) and its matching section in `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md`. Those records state that code candidate C and ledger candidate D retain separate historical approvals, finalizer E passed its second exact-head review and was integrated at `8e0ad89aad4721d9f58bcfb50e27d8394f2f0fa0`, and dashboard currentness was verified after restart at 2026-10-25 16:04 UTC. Both APIs returned HTTP 200; the curriculum endpoint validated the 89-row summary-only ledger with one Plan C entry and no `untrusted_reason`; all eight frontend routes matched checkout assets byte-for-byte. The `/api/meco-status` response also recorded five unrelated errors in older campaign reports, detailed in the evidence note.

The ledger correction changes only the four Plan C summary strings authorized by the amended Pass 1 plan and refreshes this plan artifact's hash in the existing Plan A entry. No historical Plan C attribution, verdict, candidate SHA, approval SHA, or other ledger field was changed. The corrected ledger candidate is pending a fresh exact-head Pass 3 review.

## Evidence limits

The source candidate reported `py_compile` and `git diff --check` only. No pytest, HIP compilation/import/profile, GPU work, simulator run, replay, or training validation was performed. The Plan C currentness evidence above does not validate the later Plan A ledger candidate or final API digest pin; the corrected ledger candidate needs its own exact-head review, and Plan A API pin finalization and dashboard currentness verification remain pending. This record makes no HIP/WX9100 parity or profiling, runtime memory admission, simulator behavior, replay, campaign passage, training, deployment, qualification, or whole-plan completion claim.
