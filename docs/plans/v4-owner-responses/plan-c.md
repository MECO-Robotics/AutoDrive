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

I resolve the source-axis, feature-width/order, K ordinary view, continuation
map, and mask/rank mechanics below against the accepted Plan C K-capped model
view contract and current source. I do not adopt V4 yet: the production profile
choice `K=16` and the canonical integer candidate-identity dictionary/code
binding remain explicit owner decisions. V4 and the accepted K-capped plan are
planning artifacts; neither publishes that identity dictionary or makes the
profile normative.

## Evidence and scoped findings

1. **Full source axis and capacity.** The builder starts with eight base rows,
   appends the sensed-target expansion, adds 24 approach-offset rows, reserves
   one continuation row, and appends the separate oracle pool ([`frc_defense/meco_candidates.py`](../../../frc_defense/meco_candidates.py)
   lines 1291-1313, 1323-1381, 1421-1449). For the proposed `E=504` profile,
   the Plan C policy axis is `G_C=8+504+24+1=537`; the oracle pool is not part
   of that axis. Expansion rows are fixed capacity and masked when ineligible
   or absent. V4 § Profile and §§69-87 specify those capacities, while the
   integration sequence says the environment owner adopted E=504 but asks
   Plan C/PPO owners to accept K=16, feature ordering, continuation mapping,
   and the 537-row binding (lines 137-139). The 537 value is therefore
   conditional on that pinned profile, not a general source ceiling.
2. **Feature width and order.** `SAFE_ACTION_FEATURES` has fourteen ordered
   fields and `EXTRA_FEATURES` six; source concatenates these as raw width 20,
   then concatenates the missing mask for model width 40 ([`frc_defense/meco_tracks.py`](../../../frc_defense/meco_tracks.py)
   lines 772-782; [`frc_defense/meco_candidates.py`](../../../frc_defense/meco_candidates.py)
   lines 770-784). V4's profile row and dynamic Plan C field group specify
   raw 20 and model 40 (V4 lines 54-58, 147-153). This establishes current
   order/width evidence, subject to a future immutable source/profile digest.
3. **K ordinary rows and continuation.** Current source accepts K in
   `{8,16,32}`; it ranks the eight base rows first, eligible sensed-target
   rows next by stable distance order, then offsets, and selects the first K
   ordinary ranks. It appends `continue_index` separately and applies a
   separate continuation validity gate (meco_candidates.py lines 788-835,
   1314-1381, 1618-1690). The accepted K-capped model-view amendment
   (`8556c82b7715499b935bc25f067f2a5d0dfac75e`, SHA-256
   `782279be4857e849f9f90d2f6e938b22b244d34b5567f562847e61b88ebf4fb9`;
   accepted plan text per `docs/plans/meco-full-plan-integration-sequence-20261026.md`
   lines 100-101) resolves the former map ambiguity: stable rank selection
   defines `ordinary_row_map[j]=selected_policy[j]` for `j∈[0,K)`, and
   `continuation_row_map[0]=continue_index`. The model receives separate
   `[1,K,40]` ordinary and continuation views; continuation occupies only
   slot 0 in its view, with K+1 selectable policy outputs. Local index `j<K`
   maps through the ordinary map; local index K maps through continuation
   map. The route amendment acceptance explicitly closes rank<K input and C2
   context masking, but integration sequence lines 100-101 says this does not
   implement the full two-view request/row-map/descriptor protocol. This is
   contract-level mapping evidence, not evidence that the full protocol is
   implemented.
4. **Masks, ranks and padding.** Source distinguishes fixed-spec target
   eligibility, route feasibility, ordinary selection rank, and continuation
   commitment validity (meco_candidates.py lines 661-725, 788-835, 1323-1381).
   The accepted K-capped amendment specifies ordinary-prefix-only C2 context,
   separate continuation query, mask-false zero feature padding, and `-1`
   row-map sentinel; V4 uses separate `candidate_valid`,
   `candidate_model_input_mask`, `candidate_selection_rank`, and
   `candidate_policy_eligible` fields (amendment § “Decision: two K-width score
   views”; V4 lines 147-153). The amendment also requires device guards for
   range, uniqueness, eligibility, sentinel/mask agreement, and continuation
   identity (amendment line 87). Thus the K-view maps, rank order and padding
   sentinel are settled at contract level. Current source still has local
   sentinels and ranks that are not a complete canonical ABI table; do not
   treat source-local values as serialization rules. The proposed full source
   table uses validity and policy/model masks as distinct columns, but the
   exact canonical projection of all source masks/ranks into those V4 columns
   has not been implemented or independently source-reviewed.
5. **Identity and consumer binding.** `_device_candidate_ids` emits dynamic
   labels based on action, target track and approach variant, uses the
   continuation label `continue:inactive` or `continue:<candidate ID>`, and
   suffixes duplicates with the source row index (meco_candidates.py lines
   575-600). `DeviceCandidateBatch` retains an ordinal `candidate_index`; the
   source row map and current string labels are not a versioned integer
   identity registry. The accepted shared ABI defines `candidate_identity_code`
   as an integer from a versioned static candidate dictionary/table and says
   Plan C `source_global_row` comes through the row maps (ABI lines 39-40,
   53). V4 also carries `candidate_identity:int64[537]` and selected
   `candidate_identity_code` (V4 lines 152, 156). Neither source nor accepted
   contract binds the exact dictionary/version, code allocation, or how dynamic
   target/continuation identity and duplicate rows bind to that code. The
   source-global row is a coordinate, not a substitute stable identity.

## Exact remaining decision gates

1. **Profile selection:** explicitly adopt K=16 for this V4 profile (as
   recommended by current trainer default and V4), or provide a versioned
   replacement K. K=8/32 source support alone does not make those profile
   choices equivalent.
2. **Candidate identity registry:** assign and pin numeric values for the
   descriptor-v2 `kind`/`variant` enum and define the exact mapping from each
   descriptor plus dynamic target/continuation identity to V4
   `candidate_identity` and command `candidate_identity_code`. Specify how
   duplicate/padding rows and version mismatch behave. Preserve the
   source-global row map as a separate coordinate; the accepted descriptor plan
   does not define the V4 int64 code allocation.
3. **Source-axis field projection:** map current source eligibility, route
   feasibility, model-input membership, selection rank, and continuation
   validity into each of V4's `candidate_valid`,
   `candidate_model_input_mask`, `candidate_selection_rank`, and
   `candidate_policy_eligible` columns for all 537 rows. The K-view contract
   defines masks for model queries and selection, but does not fully specify
   this 537-row serialized projection.

These gates affect Plan C command selection and replay identity, Route's
`policy_source_row`/candidate association, Observation/native target-handle
binding, and artifact/source digests. The environment row's E=504 adoption
sets the conditional source capacity but does not close either Plan C gate.
A change to K or identity/mask projection requires the affected Route,
Observation/native, PPO/RNG, and Artifact/manifest owners to re-evaluate their
row shapes, mappings, and digests before any unified profile can be published.

## Scope and authority

This is only the `plan_c_owner` technical decision. It does not decide another
owner's matrix row, claim human maintainer approval, establish cross-owner
adoption or ABI publication, or authorize implementation. No tests, imports,
compilation, runtime/device, simulator, dashboard, checkpoint, campaign,
training or evaluation action was performed.
