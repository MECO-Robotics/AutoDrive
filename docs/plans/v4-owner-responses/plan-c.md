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

I adopt the V4 Plan C profile choice `K=16` and define the Plan C source-row,
descriptor-code, candidate-mask, rank, and identity mappings below. The only
remaining Plan C adoption gate is the cross-owner route-sink join: Route has
not yet adopted the mapping from each full source-global row to its route sink
row, target handle, and inactive-row representation. The source-local mapping
is explicit here; V4 adoption remains pending until Route accepts the shared
join and sentinel contract.

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
   | 3 | Approach offset, rows 512-535; each consecutive four rows use one action, variants 1-6 in source offset order. |
   | 4 | Continuation, row 536; action and goal variant come from the revalidated commitment. |

   Variant 0 means no offset; variants 1-6 map in source order to
   `(+.22,0)`, `(-.22,0)`, `(0,+.22)`, `(0,-.22)`, `(+.34,0)`, and
   `(-.34,0)` meters (meco_candidates.py lines 1306-1313). The versioned
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
   Observation/native defines anonymous target handles with generation-safe
   observer-slot identity but remains PENDING on environment-instance issuance
   and macro trace production (response commit
   `af0f8b4ca4759441030f2625aa23ce555a06bbac`, SHA-256
   `ceb78d960ce29313dda5d3fe3f8e2f6a60258dfaeaf2ca96275e50698431b24a`).
   Route remains PENDING because its response still lacks an adopted exact join
   among full route-sink row, Plan C source-global row, ordinary/continuation
   maps, target handle, and selected command (route response commit
   `e44d5d2c815ee176d95cb87cb3a9ea0635c76dd9`, SHA-256
   `e3d3102faf8476d00516e1facafdb8850e9d50a9e98dd6bd2199005bff478995`).

## Exact remaining cross-row gate

Route owner must adopt the Plan C mapping `route_ref = source_global_row` for
policy rows 0-536, with the V4 route sink's `policy_source_row` carrying that
same source-global row. It must bind each policy query's `target_handle` and
snapshot to the descriptor/target table for that source row, preserve the
continuation at row 536, and specify inactive/oracle-row sentinels without
reusing Plan C source identities. Until that route join and sentinel contract
is accepted, Plan C's source-local mapping is defined but cross-row identity
is not closed. Any change to row ordering, handle reference, or code allocation
requires Route and PPO/RNG to update their mappings and support-digest registry;
Artifact/manifest must then bind the resulting source/profile identity and
payload length. No product choice or source evidence remains unresolved within
this Plan C row.

## Scope and authority

This is only the `plan_c_owner` technical decision. It does not decide another
owner's matrix row, claim human maintainer approval, establish cross-owner
adoption or ABI publication, or authorize implementation. No tests, imports,
compilation, runtime/device, simulator, dashboard, checkpoint, campaign,
training or evaluation action was performed.
