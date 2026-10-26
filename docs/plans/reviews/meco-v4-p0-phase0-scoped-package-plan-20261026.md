# Independent Pass 1 review: P0 dual-protocol artifact-allowlist amendment

- **Verdict:** REVISE (supersedes ACCEPT for prior P0 plan head `72ee8fce1d4a955d100b97d47b58528374faddec`)
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Planner:** `plan_a_owner`; amendment author: `p0a_implementer`
- **Reviewed amendment commit:** `55e788f19d676e65a2150debd99d25aaa16f7716`
- **Reviewed file:** `docs/plans/meco-v4-p0-phase0-scoped-package-plan-20261026.md`
- **Git blob:** `13b69eefb9aae20cdbd2c23eaec0a71ec7898282`
- **Plan file SHA-256:** `8f959afd2071970afcf2f94bf0879104dd7bcdd17ddcbd1587682d2940c00ef3`
- **Accepted overall plan:** commit `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; ACCEPT review commit `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`
- **Prior accepted P0 amendment:** commit `72ee8fce1d4a955d100b97d47b58528374faddec`, file SHA-256 `16c1fbcd881bd77adf178c8d8bf5f17bd3cf3e988cde767fbbc7731c8ff088d9`; ACCEPT review commit `cb02ef8fd74ab4b65c4b0f86a1796eaad616e77d`, record SHA-256 `a8c5164730d62bd04c7a134e0d395a81ce8f76c1088db6e5f5a8e2b5bd62701d`
- **Accepted P6 package plan inspected:** commit `ea51dd8c84ab4e59bf056fae8c47facaee0c1fbd`, file SHA-256 `60cb8e6126cea8d307a96caef6b3099e5db5e43efe41170df0cb4e3237462a5e`; ACCEPT review commit `f97a9ed025b892e75669b95eec4785d15f05afff`, record SHA-256 `8bee77e4ce40dbc3be0058f61f08f992f539c8ed2a57945802263ae0eecc9710`

## Review scope and evidence

The amendment clearly identifies two separately hashed v5 protocol records: Stage 1 qualification for `pickup_simple` and score evaluation for `full_match_offense`. It assigns distinct protocol and calibration evidence paths, has the Phase 0 artifact cross-bind both protocol digests and their roles, and keeps qualification schema/runner implementation and tests owned by P6. P0 remains documentation-only in Pass 2; P0b outputs are generated outside Git. The prior source-authoring and execution gates, G0, manual-stop and dual-lease conditions, post-G0 bridge, campaign no-contact boundary, and P0/P6 sequencing remain unchanged.

### Finding: the dual-protocol dependency is not covered by the accepted P6 plan pin

The amendment requires the “accepted finite allowlist and API” for P6 to support two protocol records, their Phase 0 cross-binding, and separate calibration inputs/evidence paths, and says P6 must resolve validation/runner behavior before P0b. However, the accepted P6 plan at `ea51dd8c84ab4e59bf056fae8c47facaee0c1fbd` still pins the prior P0 package plan (`13dbd7f8...`) and describes P0b as emitting a target-scoped protocol artifact. Its accepted scope does not pin this new dual-protocol contract or require the P6 schema/runner to validate both records and the cross-binding.

This changes a P6-owned API/schema dependency beyond the exact accepted P6 plan. Before P0 Pass 1 ACCEPT, update the P6 plan (or a scoped P6 amendment) to pin this exact P0 successor and explicitly include both protocol identities, their independent calibration/evidence paths, the Phase 0 digest cross-binding, and schema/runner validation requirements. Obtain independent exact-head acceptance for that P6 scope, then update this P0 plan's dependency pin to the accepted P6 head and record. The existing G0/source-authoring restrictions remain necessary but do not resolve the missing accepted interface contract.

## Authority limits

This REVISE applies only to the exact P0 amendment commit/blob/hash above. It does not authorize P6 source/test changes, P0b generation, training, or any runtime action. No campaign, tests, simulator, dashboard, device, or runtime was inspected or operated.
