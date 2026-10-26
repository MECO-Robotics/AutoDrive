# Plan C owner response for PPO transition ABI V4

Date: 2026-10-26

## Attribution and pinned artifacts

- Owner role: `plan_c_owner` (Plan C source-row axis, candidate/model feature
  contract and source identity).
- Delegated task identity: `/root/plan_c_owner`.
- Response author: delegated technical decision owner `/root/plan_c_owner`.
- V4 proposal: commit
  `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256
  `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- Owner-adoption matrix: commit
  `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256
  `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.

## Disposition: PENDING

The source evidence corroborates the proposed Plan C capacities and feature
widths for the named profile, but does not settle all identity, map, padding,
and selector semantics required for adoption. This response therefore does not
adopt the V4 proposal or supply a replacement.

## Evidence and scoped findings

1. **Full source axis.** The device candidate builder constructs eight base
   action rows, a fixed-capacity expansion over the sensed target buffer,
   twenty-four approach-offset rows, one continuation row, then oracle rows.
   The relevant construction is in
   [`frc_defense/meco_candidates.py`](../../../frc_defense/meco_candidates.py):
   base rows at lines 1291-1313, reserved continuation at 1323-1381, and
   oracle rows at 1421-1449. With profile `E=504`, this supports the proposed
   `G_C = 8 + 504 + 24 + 1 = 537` policy-source axis, excluding the separate
   oracle pool. The expansion rows are fixed-capacity; ineligible/padding
   entries remain present and are masked (lines 1221-1240, 641-644, 719-725).
   This is source evidence for the proposed shape, conditional on the separate
   environment owner adopting the `E=504` profile.
2. **Feature width and order.** The source registry has fourteen ordered
   `SAFE_ACTION_FEATURES` followed by six ordered `EXTRA_FEATURES`, yielding
   raw width 20; the model input is raw features concatenated with the missing
   mask, yielding width 40 ([`frc_defense/meco_tracks.py`](../../../frc_defense/meco_tracks.py)
   lines 772-782 and [`frc_defense/meco_candidates.py`](../../../frc_defense/meco_candidates.py)
   lines 241-243, 777-783, 851-855). This corroborates V4's width and order
   recommendation at the current source base. It does not make the schema
   immutable or establish a cross-version feature registry.
3. **K and continuation behavior.** The candidate builder accepts `K` in
   `{8,16,32}` (meco_candidates.py lines 1244-1269). Its device policy rows
   rank base actions first, then eligible sensed-target rows by distance, then
   offsets; ordinary eligibility is `selection_rank < candidate_k`, and the
   continuation row is appended separately to the candidate policy view
   (lines 788-835). Thus the current device path supports K ordinary rows plus
   a separately represented continuation selection. The trainer's configured
   default and the current source behavior do not alone establish the V4
   `ordinary_row_map[16]` and `continuation_row_map[1]` encoding or prove the
   exact accepted K/K+1 artifact referenced by the matrix.
4. **Masks and ranks.** Source has a fixed-spec eligibility mask for expansion
   rows, `route_available`, an ordinary `selection_rank`, and a separate
   continuation commitment validity gate (meco_candidates.py lines 661-725,
   788-835). The proposal's serialized fields `candidate_valid`,
   `candidate_model_input_mask`, `candidate_selection_rank`, and
   `candidate_policy_eligible` are plausible representations, but their
   distinct normative meanings, especially for invalid ordinary rows versus
   inactive oracle/padding entries, are not fully specified by those source
   tensors. See V4 proposal lines 69-87 and 140-155.
5. **Identity.** Runtime candidate labels are strings derived from action,
   target track, and approach variant; duplicate/padded labels receive a row
   suffix (meco_candidates.py lines 575-600). Current candidate records retain
   `candidate_index` as an int64 ordinal and align route query order to
   candidate order (lines 133-180, 224-227, 700-709). V4 proposes an int64
   `candidate_identity` and a selected source-row field, but neither the source
   nor V4 binds the stable integer encoding, collision behavior, versioning,
   or exact relationship between dynamic string identity, source-row ordinal,
   policy local index and route sink rows.
6. **Padding and sentinels.** Source uses several local sentinels (`-1` for
   absent track identity and large rank values for inactive rows) but these
   are implementation details at different axes (meco_candidates.py lines
   583-599, 642-644, 796-818). V4 specifies zero values for mask-false floats
   and provides fixed array shapes, but does not provide a complete Plan C
   integer sentinel table, padding identity rule, or normative map behavior
   for every unused K/source slot (V4 proposal lines 114-120, 147-155).

## Exact unresolved fields

Before this row can adopt V4, a versioned contract/reconciliation must specify:

- the exact accepted K/K+1 artifact and its immutable commit/hash;
- the mapping from each of the K ordinary view indices to the immutable
  537-row source axis, including deterministic behavior when an ordinary
  candidate is invalid or absent;
- the continuation-map encoding and its relationship to `continue_index`,
  candidate policy eligibility, and the selected source-row identity;
- canonical identity encoding from source candidate IDs to `candidate_identity`
  and command `source_global_row`, including duplicate, padding and version
  behavior;
- exact distinctions among candidate validity, model-input validity, policy
  eligibility and selection rank for padded/inactive rows; and
- integer sentinels and mask-false values for every Plan C field, with explicit
  padding semantics for the source axis and K-wide ordinary map.

## Scope and authority

This is only the `plan_c_owner` technical decision. It does not decide another
owner's matrix row, claim human maintainer approval, establish cross-owner
adoption or ABI publication, or authorize implementation. No tests, imports,
compilation, runtime/device, simulator, dashboard, checkpoint, campaign,
training or evaluation action was performed.
