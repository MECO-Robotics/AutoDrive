# Independent Pass 3 review: Plan C V4 owner response

- **Verdict:** REVISE
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** `/root/closure_wave1_reviewer`, independent exact-head Pass 3 review
- **Response author:** `/root/plan_c_owner` (delegated Plan C technical owner)
- **Reviewed commit:** `0d8be1b664106b3744d62fe2f78b5f5191963d58`
- **Reviewed file:** `docs/plans/v4-owner-responses/plan-c.md`
- **Reviewed file SHA-256:** `8f7a8f9cf4b12a89679e34fd245d8be345a7f1f5c931a5b84952ac1cab0066c7`
- **Pinned proposal:** V4 commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256 `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- **Pinned matrix:** commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256 `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.
- **Pinned owner-closure plan:** commit `4bef4525acf8022f3b38edaafe0793d13ef7a0a2`, SHA-256 `b0a72285269189f383d25e3463a258fa97a692bdee7674792e213eb0ec8657b7`.
- **Supersedes:** the prior ACCEPT record at this path for response commit `78b98e47e8689d9c231104b02928f1fb4d7bc9f8`; that record does not review this corrected response head.

## Review scope and evidence

Reviewed only the corrected Plan C row against the pinned V4 proposal, owner matrix, closure plan, accepted K-capped model-view amendment, prior response/review, and cited candidate source. The new response usefully resolves the `K=16` profile choice, spells out a row-local mask/rank projection, and proposes an explicit descriptor-code registry. Its owner disposition appropriately remains **PENDING** on the cross-row route-sink join.

However, the descriptor registry's offset-row ordering is inconsistent with authoritative source and would bind wrong variant/action metadata to source-global rows. The response says rows 512–535 are arranged so “each consecutive four rows use one action, variants 1-6 in source offset order.” In `frc_defense/meco_candidates.py:1310-1313`, the loop is variant-outer and action-inner: rows 512–515 are variant 1 with actions 0–3; 516–519 are variant 2 with actions 0–3; and so on through variant 6 at rows 532–535. This is a concrete descriptor/identity mapping error, not merely the still-pending Route join. Correct the row dictionary and any related descriptor examples before another exact-head review.

The exact response commit and blob SHA-256 were independently rechecked. No tests or runtime actions were performed. This finding is limited to the Plan C response; it does not decide another owner row, approve the V4 ABI, imply human maintainer approval, or authorize implementation. The previous review record is superseded by this record under the closure plan's same-path corrected-head rule.
