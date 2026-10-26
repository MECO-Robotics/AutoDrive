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

## Disposition: ADOPT

I adopt the V4 Plan C profile choice `K=16` and define the Plan C source-row,
descriptor-code, candidate-mask, rank, and identity mappings below, together
with the Route owner's accepted full-sink join and sentinel contract. This
adopts the exact pinned V4 proposal for the delegated Plan C row; it does not
claim implementation or global ABI publication.

## Evidence and scoped findings

1. **Full source axis and capacity.** The builder starts with eight base rows,
   appends the sensed-target expansion, adds 24 approach-offset rows, reserves
   one continuation row, and appends the separate oracle pool ([`frc_defense/meco_candidates.py`](../../../frc_defense/meco_candidates.py)
   lines 1291-1313, 1323-1381, 1421-1449). For the proposed `E=504` profile,
   the Plan C policy axis is `G_C=8+504+24+1=537`; the oracle pool is not part
   of that axis. Expansion rows are fixed capacity and masked when ineligible
   or absent. Environment/profile response `cd9276c62edb695715bf54ef44dcd7703183ee5b`
   (SHA-256 `e0cac32ea061f73b96b57d378c35784f73643ea3b2dfb6fa3ca1a65e56d47a2a`)
   adopts the V4 profile including E=504 and K=16. The 537-row axis is
   conditional on that profile, not a general source ceiling.
2. **Feature width and order.** The raw 20-column order is `action_index`,
   `sensor_track_buffer_index`, `goal_x_m`, `goal_y_m`, `heading_bin`,
   `behavior_index`, `field_length_m`, `field_width_m`, `target_distance_m`,
   `nearby_sensed_track_count`, `track_confidence`, `track_age_s`,
   `opponent_distance_m`, `route_time_s`, `legal_node_index`,
   `route_distance_m`, `risk_penalty_points`, `target_heading_rad`,
   `is_continue`, `commitment_switch_cost`; the 20 missingness indicators
   follow in the same order to form model width 40 (`frc_defense/meco_tracks.py`
   lines 772-782; `frc_defense/meco_candidates.py` lines 738-784, 851). This
   matches V4's profile and field table (V4 lines 54-58, 147-153). Bind this
   order to the Plan C descriptor schema/source digest.
3. **K ordinary rows and continuation.** Environment/profile has adopted
   `K=16` for this V4 profile. Current source also supports K in `{8,16,32}`;
   it ranks the eight base rows first, eligible sensed-target
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
4. **Mask, rank, and padding projection.** For V4's 537 source rows, preserve
   the source tensors as follows: `candidate_valid[r] = route_available[r]`
   before the K-prefix mask; `candidate_model_input_mask[r] = candidate_policy_mask[r]`
   (ordinary `selection_rank < 16`, plus the reserved continuation row);
   `candidate_selection_rank[r] = selection_rank[r]`; and
   `candidate_policy_eligible[r] = valid_mask[r]`, where source defines
   `valid_mask = route_available & candidate_policy_mask` (meco_candidates.py
   lines 710-725, 788-835, 831-835, 872-884). The effective ordinary query
   mask is `candidate_valid & candidate_model_input_mask`; the continuation
   query mask is `candidate_valid` at `continue_index` only, as required by the
   accepted two-view contract. Ordinary local slots follow the stable sorted
   rank map; continuation is separate local selector slot 16. Source selection
   ranks are preserved for ordinary rows, including its deterministic large
   rank for ineligible extras; continuation rank remains the reserved source
   sentinel `2**60`. The accepted K/K+1 amendment specifies masked finite-zero
   feature padding, `-1` for unused view-map slots, no continuation view when
   invalid, and device guards for map range/uniqueness and mask agreement
   (amendment lines 40-43, 87). Oracle rows are outside G_C and are not copied
   into any Plan C candidate column. V4 field semantics are pinned here as a
   row-local contract; they do not imply that the current source serializes
   those fields.
5. **Descriptor and identity registry.** The accepted K-capped amendment
   defines descriptor-v2 fields `kind:u16`, `variant:u16`,
   `source_identity:i64`, `action_index:i16`, `target_identity:i64`,
   `goal_ref:i32`, `route_ref:i32`, and `valid:u8`; it requires static kind and
   variant enums and device-generated dynamic identity columns (amendment
   lines 56-57, 85-89). I define schema ID `meco-c-source-row-v1` for E=504
   and this numeric dictionary:

   | `kind` code | Meaning |
   |---:|---|
   | 0 | Reserved invalid code; never assigned to a populated source row. |
   | 1 | Base action, rows 0-7; action index equals row index. |
   | 2 | Sensed target, rows 8-511; row `r` targets observer buffer slot `r-8`. |
   | 3 | Approach offset, rows 512-535; each consecutive four rows share one variant and enumerate actions 0-3. |
   | 4 | Continuation, row 536; action and goal variant come from the revalidated commitment. |

   Variant 0 means no offset; variants 1-6 map in source order to
   `(+.22,0)`, `(-.22,0)`, `(0,+.22)`, `(0,-.22)`, `(+.34,0)`, and
   `(-.34,0)` meters (meco_candidates.py lines 1306-1313). Therefore rows
   512-515 are variant 1/actions 0-3; 516-519 are variant 2/actions 0-3;
   520-523 are variant 3/actions 0-3; 524-527 are variant 4/actions 0-3;
   528-531 are variant 5/actions 0-3; and 532-535 are variant 6/actions
   0-3. The versioned
   static row dictionary maps source-global row `r` to identity code `r+1`;
   code 0 is invalid. Thus `candidate_identity[r]` and selected command
   `candidate_identity_code` equal `r+1` in the range 1-537. This identifies
   the immutable source slot; the full per-decision candidate identity is the pair
   of that code and the target handle selected through `target_identity`.
   `source_identity=r`, `goal_ref=r`, and `route_ref=r`; `target_identity` is
   the observer-local buffer slot for a live target and `-1` when no target is
   present. The full four-int handle is read from V4's `target_handle` table;
   buffer slot alone is never treated as a lifetime-safe identity. The row's
   descriptor `valid` byte mirrors `candidate_valid`. For rows 8-511, action
   index is the source's selected legal collect action; for base/offset/continue
   rows it is the current action tensor. This preserves current action, target,
   goal, route, and continuation behavior while eliminating dynamic string IDs
   from the V4 identity.
6. **Cross-row consumer binding.** The accepted shared ABI maps C
   `source_global_row` through the ordinary/continuation row maps and requires
   `candidate_identity_code` from a versioned dictionary (ABI lines 39-40, 53).
   PPO/RNG now ADOPTs those V4 digest contracts and defines `support_digest` over
   the exact family-specific support/candidate content and ordering under the
   Plan C owner's adopted contract, including profile digest, row masks, and
   code registries (response commit `602565ef99986996d2c12d06024833247dc3d395`,
   SHA-256 `219f3466741c17d9e49a4d402f0b3f8ae6f016f0738a068034a8ad312f610d75`).
   Observation/native ADOPTs V4 and defines the anonymous generation-safe
   target-handle and snapshot lifecycle used by this mapping (response commit
   `fa21a91d81b11008014925e8c62ed4c42f4c58e8`, file SHA-256
   `b43f1a57da8548088f439e65475c6d4963206638`). Route's accepted addendum
   closes the Plan C-specific join: route response commit
   `96a75df0221a8c76e8b9c5fe564bb8d336c98c80`, file SHA-256
   `29ce9a21719e2cb65dbe513565b9012c51b9c7d8e00b65a57a46a98172aaf113`,
   independently reviewed ACCEPT in review-record commit
   `7ea6d8225d3d12acf72be6bc13f1b1b1c052ad4d`, review-file SHA-256
   `0e223122f0322b07da0e011227be70ce7bb7f068caa39913ebf98113708b64a1`.
   The addendum binds policy sink row `q` to source-global row `r=q`, carries
   the same row in `policy_source_row` and `route_ref`, maps continuation to
   row 536, preserves full target handles, and specifies inactive/oracle
   sentinels. Its ACCEPT validates that exact Route response; the Route owner
   retains its own PENDING disposition for independent profile-generalization,
   learned-cost provenance, and canonical payload questions, which are outside
   this Plan C row.

## Disposition basis and scope limits

The route/source join and observation target-handle dependency that previously
held this row pending are now pinned above. This Plan C adoption remains scoped
to the exact V4 artifact and the `E=504`, `K=16`, `G_C=537` profile. It does not
resolve Route's remaining owner-row questions about generalized profiles,
learned-cost provenance, or serialized payload bytes, nor Artifact/manifest's
separately owned digest framing and checkpoint binding. Any change to row
ordering, handle reference, or code allocation requires the affected owners to
revise their mappings and support-digest registry and receive fresh review.
These external limits do not leave an unresolved Plan C source-row decision.

## Scope and authority

This is only the `plan_c_owner` technical decision. It does not decide another
owner's matrix row, claim human maintainer approval, establish cross-owner
adoption or ABI publication, or authorize implementation. No tests, imports,
compilation, runtime/device, simulator, dashboard, checkpoint, campaign,
training or evaluation action was performed.
