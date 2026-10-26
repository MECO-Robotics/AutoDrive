# Route owner response for PPO transition ABI V4

- **Owner role:** `route_owner`
- **Delegated agent/task identity:** `/root/route_owner`
- **Response author:** delegated technical owner `/root/route_owner`
- **Date:** 2026-10-26
- **Decision scope:** Plan C route query capacity/order/chunking, inactive-row semantics, policy/oracle identities, path retention, route-cost units/provenance, and selected-row association.

## Pinned inputs

- V4 proposal: commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256 `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- V4 owner-adoption matrix: commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256 `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.

## Disposition: PENDING

The source and consumers resolve the geometric meaning, broad capacity, and ordered construction of the Plan C route rows. They do not establish all of the proposed fixed ABI's code registries and cross-row joins. I keep the disposition **PENDING** specifically for those unresolved shared decisions rather than infer ABI values from Python construction details.

## Evidence and limits

- **Query capacity and ordering:** The proposal's C-only `Q_C=9,681` formula is consistent with the current fixed profile arithmetic `105 + 19E` at `E=504`: eight base actions + E sensed-target rows + 24 policy offsets + one continuation row + 18 oracle variants × (four base target slots + E sensed-target rows). The source construction order is visible at `meco_candidates.py:1291-1313,1323-1381,1421-1452`; oracle target rows are ordered by stable track-index sort before variant-major expansion at `meco_candidates.py:1427-1449`. These are current construction facts, not a guarantee that every row is active or an independent proof of a maximum for every configurable profile. V4 §“Independent Plan A and Plan C capacities” (proposal lines 82-90) correctly makes `Q_C` Plan C only and excludes Plan A graph support.
- **Chunking:** `experimental_route_evaluation.py:21-23,48-64` defines `MAX_ROUTE_QUERY_BATCH=64` and contiguous ordered global/chunk/local mapping. `meco_candidates.py:1535-1556` validates the whole query set for snapshot consistency and duplicate keys before splitting it into consecutive chunks. For 9,681 rows, the arithmetic is 151 full chunks plus one 17-row chunk, i.e. 152 chunks. Current `evaluate_device_route_chunks` concatenates chunk outputs (`experimental_route_evaluation.py:631-707`); it does not implement the proposal's fixed sink/direct-scatter serialization, which remains a proposed ABI producer requirement.
- **Path geometry and retention:** `RouteQuery.goal_xy` is an XY goal in field meters; the evaluator uses policy localization for starts and perceived robot obstacles, and explicitly limits itself to snapshot XY feasibility (not terminal heading, intake legality, or time-varying collision prediction) (`experimental_route_evaluation.py:112-122,717-738`). The route planner default is `max_points=72` (`tensor_adstar.py:257`); the device evaluator prepends the start and stores the resulting padded XY path (`experimental_route_evaluation.py:568-584`). Therefore V4 `path_xy[73,2]` is enough for that exact default profile and coordinates are meters. The producer should define `path_point_mask[i] == (i < path_length)`; V4's canonical rule then requires false-mask path coordinates to be zero, while the current device implementation pads tails with the endpoint and would need canonical serialization. A profile with another planner bound needs a new capacity/profile decision.
- **Statuses and validity:** Current route codes are `INVALID=0`, `AT_GOAL=1`, `NO_PATH=2`, `FEASIBLE=3` (`experimental_route_evaluation.py:168-171`). Active requires a valid query mask and finite/in-field inputs (`experimental_route_evaluation.py:496-528`); invalid input has zero path length/coordinates and status INVALID, while at-goal and feasible rows have paths; infeasible rows can retain diagnostic route outputs where a path exists (`experimental_route_evaluation.py:568-628`). V4 defines `query_kind=0` for inactive and a separate `query_active` bool but does not give their normative equivalence or the required values for status, `feasible`, costs, path mask, and identity. V4's global mask-false float-zero rule constrains serialization but does not decide which validity mask applies to each route field. This exact inactive encoding remains unresolved.
- **Policy/oracle identity:** Current code has ordered policy specs, a continuation row, then oracle variants and target tracks (`meco_candidates.py:1291-1313,1323-1381,1421-1452`). Current `DeviceRouteQuery` carries candidate ID, snapshot, world, slot, and intake state (`experimental_route_evaluation.py:140-148`); it has no serialized policy/oracle enum, route-variant registry, or policy source-row field. V4 proposes `query_kind`, `policy_source_row`, `oracle_identity_code`, `route_variant`, and `candidate_index`, and says oracle `policy_source_row=-1`, but it does not give normative code tables or define which canonical global ordinal `candidate_index` indexes (full query source axis versus compacted active rows). These identity meanings need an exact registry and a Plan C mapping agreement.
- **Costs and provenance:** Current route status, path distance, endpoint error, and cost outputs are device tensors; the cost units are `normalized_grid_edge_units` (`experimental_route_evaluation.py:82-109,150-166,615-628`). The contract says path coordinates/distance use meters, normalized route cost integrates raster penalties in grid-edge units, and learned values are dimensionless additive penalties per cardinal grid step (`experimental_route_evaluation.py:731-738`). Learned-map provenance binds producer, scale, raster geometry/schema, and a SHA-256 of normalized map values (`experimental_route_evaluation.py:751-785`). This supports the proposed distance, endpoint error, weighted, bump, and learned columns as float64 fields. However V4 supplies only a bundle component-presence mask (proposal lines 122-138), not a fixed learned-cost provenance identity or map digest field; source/config digest coverage of a runtime learned map is not established. The ABI needs the Plan C producer and artifact owner to resolve that boundary and absent-component encoding.
- **Selected-row association:** Device candidate construction gathers policy route values through `selected` and creates an aligned compact policy route batch (`meco_candidates.py:815-830`); candidate selection uses that canonical candidate order and validity mask (`meco_candidates.py:831-850`). The selected route goal/target are gathered through the same selected candidate path before command construction (`meco_candidates.py:852-880`). V4 supplies `candidate_index`, `policy_source_row`, `ordinary_row_map`, `continuation_row_map`, and common command `source_global_row` (proposal lines 152,155-156), but does not define a total join among the full route sink ordinal, `G_C` source row, compact K ordinary index, continuation row, and command source-global row. The sentinels for inactive/oracle rows also need to be consistent with Plan C's independently adopted maps.

These findings settle route geometry as snapshot-specific XY in meters, route cost as normalized grid-edge units, the status code meanings, the default path bound, and the formula/order at the proposed profile. They do not establish runtime capacity across arbitrary environment configurations, source/config identity of a learned map, adoption by Plan C or artifact/manifest owners, implementation correctness for a V4 serializer, or Stage 1 qualification.

## Pass 2 closure addendum: route/source joins and sentinels

This addendum resolves the Route-owner mapping left open above against the
accepted Plan C owner response at commit
`1af5d55dea0eb7f3a2f2938cd71423841bfa86b1`, SHA-256
`8ac11dfbe85aa0c03d834c0530eee2f313f06d6fdc97ef479bff9b923cefdc50`. It is an owner-level contract for the
pinned `meco-ppo-v3-3v3-f504-n1-s6-r32-t12-v1` profile, not a source/runtime
change.

### Full-sink row identity

The route sink keeps the proposed fixed order, with `Q_C=9,681` and `E=504`:

| Sink ordinal `q` | Class / source mapping | `candidate_index` | `policy_source_row` | `route_variant` |
|---:|---|---:|---:|---|
| 0–7 | Base action policy rows, same Plan C source-global row `r=q` | `q` | `q` | 0 |
| 8–511 | Sensed-target policy rows, same source-global row `r=q` | `q` | `q` | 0 |
| 512–535 | Policy approach-offset rows, same source-global row `r=q` | `q` | `q` | 1–6 in source offset order; four consecutive actions per variant |
| 536 | Reserved continuation policy row, Plan C `continue_index` | 536 | 536 | 0–6 as bound by the revalidated commitment; 0 if no offset |
| 537–9,680 | Oracle rows, 18 variants × 508 target slots, variant-major then stable target-buffer-slot order | `q` | -1 | 7–24 in the V4 source offset order |

For every query, `candidate_index=q` is the immutable **full route-sink
ordinal**, not the chunk-local index and not a compacted active-row ordinal.
This agrees with `make_device_route_chunk_map` (`experimental_route_evaluation.py:24-63`),
which constructs contiguous ordered spans and a global row vector, and with
`evaluate_device_route_chunks` (`:631-707`), which concatenates chunks in that
order. The fixed profile therefore maps q to chunk `floor(q/64)` and local row
`q mod 64`, with the final 17-row chunk at q=9,664–9,680.

For policy rows, `route_ref = policy_source_row = q = source_global_row`.
Consequently the independent Plan C maps join without a second lookup:
`ordinary_row_map[j] -> r -> route_sink[r]` for each populated ordinary slot
`j` in 0–15, and `continuation_row_map[0] -> 536 -> route_sink[536]` only when
the continuation is present and valid. A selected command's
`source_global_row=r` must be in 0–536 and resolves to exactly route sink row
`q=r`; its `candidate_identity_code` is the Plan C dictionary's `r+1`.
Padding in `ordinary_row_map` remains -1; an absent continuation map is empty
as defined by Plan C. Oracle rows never enter either Plan C view and never
produce a selected command. No route identity is compacted or renumbered when
rows are inactive.

### Kind, target identity, and invalid-row values

The existing V4 values are retained: `query_kind` 0=inactive, 1=policy,
2=oracle; route status remains 0=INVALID, 1=AT_GOAL, 2=NO_PATH, 3=FEASIBLE.
`query_active` is true exactly when `query_kind` is 1 or 2. For a policy row,
`policy_source_row` retains q even if that reserved row is inactive, so the
source join remains inspectable; for oracle and inactive oracle rows it is -1.
Inactive rows retain their stable `candidate_index`, `route_variant`, and (for
policy slots) `policy_source_row`, while their result payload is canonical:
status INVALID, `feasible=false`, zero path length, all-false path-point mask,
zero path coordinates, and zero route-cost values. An inactive row's
`query_kind` is 0 and `query_active` is false. An active NO_PATH row is distinct:
it remains kind 1 or 2, active=true, status NO_PATH, feasible=false; it may
retain a nonzero diagnostic path only where the route evaluator produced one.
For active rows `path_point_mask[i]` is exactly `i < path_length`; all masked
tail coordinates are zero in the V4 serialization.

The target handle is always the complete adopted tuple
`(world_reset_generation, observer_slot, buffer_slot, slot_generation)` from
the target table belonging to that transition's observer. For a sensed-target,
continuation, or oracle row with a live target, the route row copies the exact
four int64 values from `target_handle[buffer_slot]`; it does not reconstruct a
handle from a slot number. The handle's observer must equal the route query's
robot slot, its reset generation must equal the row snapshot's reset
generation, its buffer slot must index the same target table, and the table's
`target_valid` bit must be true. A missing target or a policy row with no target
uses `(-1,-1,-1,-1)`. Such a targetless policy query may still be active for
base action/offset routes; target-handle validity is required only when the
query's row identity requires a sensed target. A target-required row without a
live exact handle is inactive and cannot be feasible or selected. The route
query snapshot token must be the same row token that owns this target table;
world, robot slot, reset generation, and observation sequence are not inferred
from a handle.

For an oracle row, `oracle_identity_code = buffer_slot + 1` (1–504) names the
target-table slot for this snapshot and is meaningful only together with its
full target handle. For policy rows and inactive oracle slots the code is 0;
0 is the invalid/non-oracle sentinel and no live oracle uses it. Oracle target
slots are enumerated in ascending buffer-slot order within each variant,
matching the source's stable sort of target indices (`meco_candidates.py:1427-1449`).
Oracle variants 7–24 retain the source order documented at `:1432-1436`; policy
offset variants 1–6 retain the source order at `:1306-1313`. Source candidate
identity codes remain Plan C's row-code `r+1`; they are never reused as oracle
identity codes.

This is a deterministic join/sentinel decision, not evidence that the current
Python/native producers already serialize it. The response remains pending on
the independent review of this exact head and on the separately owned learned
cost-map provenance and observation-instance issuance decisions. No V4,
matrix, source, schema, runtime, device, simulator, dashboard, or campaign file
is changed or authorized here.

## Required resolution before disposition

1. **Environment/profile dependency:** profile owner must bind `E=504`, the environment's sensed-target capacity, field/raster geometry, planner resolution, and `max_points=72` (hence `P=73`) to one exact profile/config identity. Any changed E or path bound changes `Q_C` and the route byte subtotal.
2. **Plan C dependency:** The source-policy row registry and route/source joins are resolved by the Pass 2 addendum above against Plan C's exact response head. This resolution remains subject to independent exact-head review; a changed Plan C row map or code registry invalidates it and requires a replacement review.
3. **Observation/native dependency:** the observation/native owner must establish the target handle and snapshot lifecycle used by route `target_handle`; a route row must refer to the same observer/world/robot snapshot whose pose, perceived obstacles, and target coordinates produced the query. Stale target handles or mismatched snapshots must not remain route-valid. V4's four-int target handle cannot be treated as an established route-source fact because the current `RouteQuery` uses string IDs and snapshot IDs (`experimental_route_evaluation.py:112-148`).
4. **Shared inactive-row decision:** The Route-owner values for `query_active`, `query_kind`, route status, feasibility, path length, path-point mask, and identity sentinels are stated in the Pass 2 addendum. Artifact/payload owners must still bind component-presence and canonical absent-cost bytes to the adopted bundle representation.
5. **Artifact/provenance dependency:** artifact owner and route producer must choose a fixed representation for the learned-cost map identity/provenance (or declare the component absent), bind its digest to source/config/profile identity, and specify presence-bit and float-mask behavior for all five route cost fields.

## Authority statement

This response records only the delegated `route_owner` technical decision under the scoped owner-response plan. It is not human maintainer approval, does not decide another owner row, does not adopt a shared ABI, does not authorize implementation, and does not establish ABI publication or Stage 1 qualification.
