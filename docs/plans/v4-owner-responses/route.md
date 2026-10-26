# Route owner response for PPO transition ABI V4

- **Owner role:** `route_owner`
- **Delegated agent/task identity:** `/root/route_owner`
- **Response author:** delegated technical owner `/root/route_owner`
- **Date:** 2026-10-26
- **Decision scope:** Plan C route query capacity/order/chunking, inactive-row semantics, policy/oracle identities, path retention, route-cost units/provenance, and selected-row association.

## Pinned inputs

- V4 proposal: commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256 `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- V4 owner-adoption matrix: commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256 `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.

## Disposition: ADOPT

**ADOPT** the exact pinned V4 proposal for the Route-owner row, with the
profile-scoped capacities, registries, joins, sentinels, target-handle checks,
and cost provenance defined below. This is a delegated owner decision; it does
not claim that a V4 producer or serializer is implemented, or publish the
cross-owner ABI.

## Evidence and limits

- **Query capacity and ordering:** The proposal's C-only `Q_C=9,681` formula is consistent with the current fixed profile arithmetic `105 + 19E` at `E=504`: eight base actions + E sensed-target rows + 24 policy offsets + one continuation row + 18 oracle variants × (four base target slots + E sensed-target rows). The source construction order is visible at `meco_candidates.py:1291-1313,1323-1381,1421-1452`; oracle target rows are ordered by stable track-index sort before variant-major expansion at `meco_candidates.py:1427-1449`. These are current construction facts, not a guarantee that every row is active or an independent proof of a maximum for every configurable profile. V4 §“Independent Plan A and Plan C capacities” (proposal lines 82-90) correctly makes `Q_C` Plan C only and excludes Plan A graph support.
- **Chunking:** `experimental_route_evaluation.py:21-23,48-64` defines `MAX_ROUTE_QUERY_BATCH=64` and contiguous ordered global/chunk/local mapping. `meco_candidates.py:1535-1556` validates the whole query set for snapshot consistency and duplicate keys before splitting it into consecutive chunks. For 9,681 rows, the arithmetic is 151 full chunks plus one 17-row chunk, i.e. 152 chunks. Current `evaluate_device_route_chunks` concatenates chunk outputs (`experimental_route_evaluation.py:631-707`); it does not implement the proposal's fixed sink/direct-scatter serialization, which remains a proposed ABI producer requirement.
- **Path geometry and retention:** `RouteQuery.goal_xy` is an XY goal in field meters; the evaluator uses policy localization for starts and perceived robot obstacles, and explicitly limits itself to snapshot XY feasibility (not terminal heading, intake legality, or time-varying collision prediction) (`experimental_route_evaluation.py:112-122,717-738`). The route planner default is `max_points=72` (`tensor_adstar.py:257`); the device evaluator prepends the start and stores the resulting padded XY path (`experimental_route_evaluation.py:568-584`). Therefore V4 `path_xy[73,2]` is enough for that exact default profile and coordinates are meters. The producer should define `path_point_mask[i] == (i < path_length)`; V4's canonical rule then requires false-mask path coordinates to be zero, while the current device implementation pads tails with the endpoint and would need canonical serialization. A profile with another planner bound needs a new capacity/profile decision.
- **Statuses and validity:** Current route codes are `INVALID=0`, `AT_GOAL=1`, `NO_PATH=2`, `FEASIBLE=3` (`experimental_route_evaluation.py:168-171`). Active requires a valid query mask and finite/in-field inputs (`experimental_route_evaluation.py:496-528`); invalid input has zero path length/coordinates and status INVALID, while at-goal and feasible rows have paths; infeasible rows can retain diagnostic route outputs where a path exists (`experimental_route_evaluation.py:568-628`). The closure addendum adopts `query_kind=0` iff `query_active=false`, and fixes inactive result values, active NO_PATH distinction, and path-mask/tail encoding. This is an owner serialization decision; the current device implementation pads path tails with the endpoint and needs zero-tail serialization in V4.
- **Policy/oracle identity:** Current code has ordered policy specs, a continuation row, then oracle variants and target tracks (`meco_candidates.py:1291-1313,1323-1381,1421-1452`). Current `DeviceRouteQuery` carries candidate ID, snapshot, world, slot, and intake state (`experimental_route_evaluation.py:140-148`); it has no serialized policy/oracle enum or route-variant registry. The closure addendum defines the V4 code meanings and full-sink ordinal; Plan C adopts the source-row/identity registry at its pinned accepted response. This remains a fixed-profile mapping, not a claim that the current Python query object serializes those fields.
- **Costs and provenance:** Current route status, path distance, endpoint error, and cost outputs are device tensors; path distance/XY are meters and weighted, bump, and learned costs use `normalized_grid_edge_units` (`experimental_route_evaluation.py:82-109,150-166,615-628,731-738`). Learned values are dimensionless additive penalties per cardinal grid step. `_cost_map_digest` pins the producer's digest preimage: schema tag `autodrive.learned_traversal_cost_map.xy_cell_center_affine_v2`, `[nx,ny]`, 11 float64 geometry values (resolution, field bounds, origin and affine cell-center transform), then row-major float64 map values, all explicitly little-endian (`experimental_route_evaluation.py:267-290`). The result provenance string binds producer, `xy_grid_f64le_v2`, geometry, digest, `learned_cost_scale.hex()`, and `path_integral_grid_edge_v2` (`:751-785`). The route source supplies the producer label `meco_legal_heading_penalty_projection_v1` (`meco_candidates.py:709,1556`); the Plan A training path supplies the projected current P map to that candidate route pool when `p_mode="learned"` (`meco_track_training.py:1778-1786`). The adopted PPO/RNG digest rule places exact family-specific support/candidate content in `support_digest`, while Artifact adopts the exact payload/digest envelope. I therefore bind the route pool's canonical cost-provenance ASCII string, NUL-terminated once after its ordered route rows in the Plan C support-digest preimage; this carries the existing map digest and scale without adding a V4 field. The component mask is bit 0=1 because bump cost is always computed, and bit 1 is 1 exactly when a learned map is supplied; with bit 1 clear, the learned-cost column is canonical zero and provenance is `legacy_bump_cost_path_integral_v2`. For the adopted training path, Plan A's `p_mode` config determines whether its candidate-route pool includes the projected learned map; C1/C2 construct that pool without a learned map. The header mask is constant within a run's V4 batch, and the run config is already bound by its config digest. The current helper's digest reference transfers tensor values to CPU; V4 digest computation is on-device, so implementation must produce the same canonical preimage/hash on device and must not call the legacy CPU helper in the live V4 path.
- **Selection utility units:** ETA is in seconds and the route utility penalty is `ETA × 0.05` simulated score points per second (`meco_candidates.py:34,732-740,838-840`). Opponent risk remains a separate simulated-point penalty in `[0,0.75]`, linearly scaled to zero at 2.0 m separation and weighted across forecast hypotheses (`meco_prediction_contract.py:12-23`, `meco_tracks.py:1986-2007`). The adopted evaluation owner response pins these separately from gross PPO reward and counterfactual targets (commit `57fe139b9f22f6bfee4bf874119117e26aa8f84c`, response SHA-256 `4c4b06740f08b17396824626acee686d13d5ee3c0394538ec5449d244867d6d5`). These simulated utility points are not official match points and are not the grid-edge route fields above.
- **Selected-row association:** Device candidate construction gathers policy route values through `selected` and creates an aligned compact policy route batch (`meco_candidates.py:815-830`); candidate selection uses that canonical candidate order and validity mask (`meco_candidates.py:831-850`). The selected route goal/target are gathered through the same selected candidate path before command construction (`meco_candidates.py:852-880`). The closure addendum defines the total join from sink ordinal through Plan C's source-global, ordinary/continuation, and selected-command rows, including inactive/oracle sentinels. The target handle remains the full observation handle and is checked against the exact target table and snapshot that produced the route query.

These findings and the closure addendum settle route geometry as snapshot-specific XY in meters, route costs and their dynamic provenance, the status code meanings, the default path bound, and the formula/order for the adopted profile. The profile and owner inputs are pinned to the following accepted response heads: environment/profile `cd9276c62edb695715bf54ef44dcd7703183ee5b` (SHA-256 `e0cac32ea061f73b96b57d378c35784f73643ea3b2dfb6fa3ca1a65e56d47a2a`); Plan C `1af5d55dea0eb7f3a2f2938cd71423841bfa86b1` (SHA-256 `8ac11dfbe85aa0c03d834c0530eee2f313f06d6fdc97ef479bff9b923cefdc50`); observation/native `fa21a91d81b11008014925e8c62ed4c42f4c58e8` (response SHA-256 `b354e0abf467c9eb30de4ff5e76a38d0c6cbf78762a4703d8f81987a79ae7f86`); PPO/RNG `602565ef99986996d2c12d06024833247dc3d395` (SHA-256 `219f3466741c17d9e49a4d402f0b3f8ae6f016f0738a068034a8ad312f610d75`); Artifact/manifest `45dc7b4894d97fd7aa04b1f84aac3cdc207f0c9d` (SHA-256 `9b7a571bd98c6aed6700d445aea8e05a2308cdc5c1bcba9c93c961c4194d8289`); and evaluation/curriculum `57fe139b9f22f6bfee4bf874119117e26aa8f84c` (SHA-256 above). This adoption is for the pinned profile only. It does not establish implementation correctness, shared V4 publication, or Stage 1 qualification.

## Pass 2 closure addendum: joins, handles, and route costs

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
| 537–9,680 | Oracle rows, 18 variants × 508 target-capacity slots, variant-major; live buffer slots sort ascending, followed by stable invalid padding | `q` | -1 | 7–24 in the V4 source offset order |

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

The observation/native owner adopts an environment-instance UUIDv4 retained
for that environment object, world reset generation, and non-wrapping
observer-slot generations; its exact response is commit
`fa21a91d81b11008014925e8c62ed4c42f4c58e8`, SHA-256
`b354e0abf467c9eb30de4ff5e76a38d0c6cbf78762a4703d8f81987a79ae7f86`.
Accordingly the V4 route query's full snapshot token is
`(environment_instance_id, world_slot, robot_slot,
world_reset_generation, observation_sequence)` as serialized by V4, and the
route target table must be sourced from that same token. For route evaluation,
reject a target-required row if its environment, world, observing robot,
reset generation, buffer slot, slot generation, current validity, or age
timeout does not match the adopted observation contract. During later macro
ticks, continue to validate the live handle generation and current slot state;
the original route-row snapshot remains the decision provenance and is not
silently replaced by a later snapshot. Observation handles remain disjoint
from physical score-event piece identities.

For an oracle row, `oracle_identity_code = buffer_slot + 1` (1–504) names the
target-table slot for this snapshot and is meaningful only together with its
full target handle. For policy rows and inactive oracle slots the code is 0;
0 is the invalid/non-oracle sentinel and no live oracle uses it. Within each
variant, valid target buffer slots are enumerated in ascending order, matching
the source's stable sort after non-target and duplicate base slots are replaced
by the invalid sort sentinel (`meco_candidates.py:1427-1449`). Invalid padding
rows follow valid slots in stable concatenated source order; they carry
identity code 0 and the all-minus-one target handle, regardless of the
temporary safe index used by the current evaluator implementation.
Oracle variants 7–24 retain the source order documented at `:1432-1436`; policy
offset variants 1–6 retain the source order at `:1306-1313`. Source candidate
identity codes remain Plan C's row-code `r+1`; they are never reused as oracle
identity codes.

This is a deterministic join/sentinel and cost-provenance decision, not
evidence that the current Python/native producers already serialize it. The
existing route-cost provenance string is included in the canonical support
digest as specified above; it is not added as a payload field. The accepted
Artifact/manifest and PPO/RNG owner responses fix the payload digest and
family-specific support-digest roles (Artifact commit
`45dc7b4894d97fd7aa04b1f84aac3cdc207f0c9d`, file SHA-256
`9b7a571bd98c6aed6700d445aea8e05a2308cdc5c1bcba9c93c961c4194d8289`; PPO/RNG
commit `602565ef99986996d2c12d06024833247dc3d395`, file SHA-256
`219f3466741c17d9e49a4d402f0b3f8ae6f016f0738a068034a8ad312f610d75`). No
Route-owner decision remains pending. The current legacy map-digest helper
copies tensor data to CPU; the eventual V4 implementation must compute the
specified identical digest on device and preserve the user's no-CPU runtime
requirement. No V4, matrix, source, schema, runtime, device, simulator,
dashboard, or campaign file is changed or authorized here.

## Route-owner closure status

The fixed profile capacity/path bound is adopted by the environment/profile owner;
source-row order, descriptors, and masks are adopted by Plan C; target-handle
and snapshot lifecycle are adopted by observation/native; route utility units
are adopted by evaluation/curriculum; and route, support, and payload digest
roles are adopted by PPO/RNG and Artifact/manifest. This Route response pins
those exact owner heads above and defines the compatible route mapping. The
only remaining gates are external to this owner row: independent Pass 3 review
of this exact Route response head, ABI-owner reconciliation/publication, then
the separately planned implementation and qualification work. A change to
any pinned input requires a new Route review.

## Authority statement

This response records only the delegated `route_owner` technical decision under the scoped owner-response plan. It is not human maintainer approval, does not decide another owner row, does not adopt a shared ABI, does not authorize implementation, and does not establish ABI publication or Stage 1 qualification.
