# Independent Pass 1 review: P6 dual-protocol package-plan amendment

- **Verdict:** REVISE (supersedes the ACCEPT for prior P6 plan head `ea51dd8c84ab4e59bf056fae8c47facaee0c1fbd` as review of this amendment)
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Original P6 planner:** `/root/closure_wave1_reviewer`; amendment planner: `plan_a_owner`
- **Reviewed amendment commit:** `de974e053c796f06a3f8e8312a9cb176f95c2025`
- **Reviewed file:** `docs/plans/meco-v4-p6-five-track-supervisor-qualification-package-plan-20261026.md`
- **Git blob:** `0b400ff10fdb0570cc33389b5d251accf5a86ddb`
- **Plan file SHA-256:** `64ff84f6fa7b63c911659b77608315b8948ecc9c166bbee27898eb80ee52d786`
- **Accepted overall plan:** commit `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; ACCEPT record commit `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`
- **Prior accepted P6 plan:** commit `ea51dd8c84ab4e59bf056fae8c47facaee0c1fbd`, file SHA-256 `60cb8e6126cea8d307a96caef6b3099e5db5e43efe41170df0cb4e3237462a5e`; ACCEPT record commit `f97a9ed025b892e75669b95eec4785d15f05afff`, record SHA-256 `8bee77e4ce40dbc3be0058f61f08f992f539c8ed2a57945802263ae0eecc9710`
- **P0 dual-protocol candidate:** commit `55e788f19d676e65a2150debd99d25aaa16f7716`, file SHA-256 `8f959afd2071970afcf2f94bf0879104dd7bcdd17ddcbd1587682d2940c00ef3`; REVISE record commit `91179c53b88a75933c9ce7a00510f6bc9daf01cb`, record SHA-256 `c34c0b05ff0fab64810cc14f7e97aa058ed2306ef7cc8b49c574a78b2c5d8004`

## Review scope and evidence

The amendment addresses the P0 dual-protocol dependency: it keeps `pickup_simple` Stage 1 and `full_match_offense` primary score evaluation proposal-only pending P0a and authorized human decisions; defines separate role-bound protocol records, evidence/calibration paths, and Phase 0 cross-binding; assigns schema, runner, API, and tests to P6; preserves three-round Stage 1 semantics; and maintains the acyclic P0a → P1–P6 source freeze → P0b → P6 runtime order. It preserves the manual-stop and dual-WX9100 lease-release gates, explicit authorization, post-G0 contract bridge, pre-G0 no-execution boundary, and live-campaign no-contact boundary.

### Finding: P1 ACCEPT record digest is stale

In the dependency table, the accepted current P1 plan is correctly pinned to `606ba5adce04afd84bfb372260aebde8fd4fcec6`, file SHA-256 `7dc5be35d3826a6047653021e97a941d6872a73a6997ba768e95753f693b0653`, and review commit `77816edbd1923c018d4af774b2f1b3bae0f415b8`. But the table gives that record SHA-256 as `0680a57e5ff0427b20a0a5bd86c224085583b241b3f57bcb18bde69cab0f158f`, which belongs to the earlier P1 plan at commit `53686a254318484f9c6bc3496ecc1c461572c53a`. The exact review record for current P1 head `606ba5...` has SHA-256 `3dbadfb7243d891fdacf993962722b90e245120442c039a15a48def97f0ca193`. Replace the stale digest and verify the full dependency table's review-record hashes before resubmission.

This stale record hash breaks the exact accepted dependency chain. The dual-protocol scope and staged gates otherwise pass; no scope or protocol-role changes are requested.

## Authority limits

This REVISE applies only to the exact P6 amendment commit/blob/hash above. It does not authorize source/test changes, P0a decisions, P0b calibration, G0 execution, runtime, device, dashboard, campaign, training, or qualification. No tests or operational actions were performed.
