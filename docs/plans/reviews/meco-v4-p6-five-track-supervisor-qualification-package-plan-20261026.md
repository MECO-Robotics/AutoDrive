# Independent Pass 1 review: P6 five-track supervisor and qualification plan

- **Verdict:** REVISE
- **Review date:** 2026-10-26
- **Reviewer identity:** `plan_a_owner`
- **Reviewer task:** independent exact-head Pass 1 review assigned by `/root/closure_wave1_reviewer`
- **Planner:** `/root/closure_wave1_reviewer`
- **Reviewed commit:** `364c0a31fcb80aab9af31d7a09006eea45009b12`
- **Reviewed file:** `docs/plans/meco-v4-p6-five-track-supervisor-qualification-package-plan-20261026.md`
- **Git blob:** `d2f170ebc8be503773aa972ece37eaad54e64be3`
- **File SHA-256:** `f0577c402e271b1189334c065114296a2b437161eb34ca1e9ad55bb87347deda`

## Review scope and evidence

The exact plan pins the accepted overall-plan amendment `8efec9378d1912f0839d7c6cd8e88467fceb1d5a` and its ACCEPT record `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`. It distinguishes accepted P3/P4/P5 plan artifacts from still-pending implementation/API heads, and requires P1–P5 candidate API source dispositions before P6 source-only authoring.

The staged gate is otherwise coherent: lines 82–90 permit isolated allowlisted source/test-file authoring and static SOURCE/API review only after accepted dependencies and P0a decisions; line 86 explicitly bars builds, test execution, simulation/numerical work, calibration, evaluation, qualification, dashboard, device/GPU use, training, checkpoint changes and integration before G0. Lines 92–100 require manual user stop of PID `2750253`, independent release confirmation for both WX9100 leases, maintainer authorization, post-G0 contract-anchor integration, runtime evidence and final exact-head Pass 3 ACCEPT before integration. The campaign no-contact boundary is explicit. P0b follows frozen/reviewed P1–P6 source; P6 qualification follows P0b, avoiding a source/calibration cycle. Training continues until manual user stop, and evaluation reads immutable snapshots without mutating trainer state.

## Required correction

The plan contains a blocking contradiction between its dependency table and staged authoring gate. Lines 22 and 26–28 say P6 requires exact accepted implementation/API **and integration** heads for P3/P4/P5 before P6 Pass 2 source work starts. Line 30 and lines 84–90 instead say P6 may author source after candidate source/API heads receive SOURCE/API ACCEPT, with integration and runtime use prohibited until post-G0 final acceptance. The first requirement prevents the source-only work that the second gate authorizes and incorrectly requires integration before G0.

Revise the table introduction and its “Required before P6 Pass 2” cells to distinguish two phases: before-G0 P6 source authoring requires exact candidate API heads with SOURCE/API ACCEPT, but does not require integration; post-G0 validation/final ACCEPT and reviewed integration require the exact final P3/P4/P5 heads. Refresh the table pins to the current package-plan amendment heads, or explicitly record that those amendments remain pending and block source authoring until accepted. Preserve the current clear statement that plan ACCEPT does not accept implementation/API code.

## Review limits

This REVISE applies only to the exact commit and file hash above. It does not reject the P6 scope or grant implementation/runtime authority. No tests, source edits, builds, runtime, simulator, device/GPU, dashboard, campaign or training actions were performed.
