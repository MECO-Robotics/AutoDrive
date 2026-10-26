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

The current route implementation supports the proposal's broad capacities and representation, but the pinned proposal does not yet define a complete, source-bound serialized identity and inactive-row contract. I cannot adopt those details by inferring ABI codes from current Python list construction or assigning meanings to fields that the proposal leaves unspecified. I am not publishing a replacement because the required replacement needs cross-owner reconciliation with Plan C's source-row identity and the artifact owner's exact serialization registry.

## Evidence and limits

- **Query capacity and ordering:** The proposal's C-only `Q_C=9,681` formula is consistent with the current fixed profile arithmetic `105 + 19E` at `E=504`. `meco_candidates.py:1291-1313` builds eight base actions, sensed-target rows, then 24 policy offsets. `meco_candidates.py:1323-1381` reserves one continuation row. `meco_candidates.py:1421-1452` adds 18 oracle offset variants across four base target slots plus the profile's additional sensed-target capacity. These are current construction facts, not a guarantee that every row is active or an independent proof of a maximum for every configurable profile. The V4 proposal correctly says this route capacity is Plan C only and must not cap Plan A's graph support.
- **Chunking:** `experimental_route_evaluation.py:21-23,48-64` defines `MAX_ROUTE_QUERY_BATCH=64` and contiguous ordered global/chunk/local mapping. `meco_candidates.py:1535-1556` validates the whole query set for snapshot consistency and duplicate keys before splitting it into consecutive chunks. For 9,681 rows, the arithmetic is 151 full chunks plus one 17-row chunk, i.e. 152 chunks. Current `evaluate_device_route_chunks` concatenates chunk outputs (`experimental_route_evaluation.py:631-707`); it does not implement the proposal's fixed sink/direct-scatter serialization, which remains a proposed ABI producer requirement.
- **Path retention:** The route planner default is `max_points=72` (`tensor_adstar.py:257`); the device evaluator prepends the explicit start point and stores the resulting padded path (`experimental_route_evaluation.py:568-584`). Thus a 73-point ABI capacity covers that default route representation. It is not established as a bound for all caller-configurable planner instances; the proposal must bind the planner/profile and define path padding and `path_point_mask` semantics together with `path_length`.
- **Statuses and inactive rows:** Current route status codes are `INVALID=0`, `AT_GOAL=1`, `NO_PATH=2`, `FEASIBLE=3` (`experimental_route_evaluation.py:168-171`); inactive numeric inputs become `INVALID` with zero path length and zero path coordinates (`experimental_route_evaluation.py:546-610`). V4 proposes `query_kind=0` for inactive and a separate `query_active` bool, but does not state the canonical relationship among `query_active`, `query_kind`, `status_code`, `feasible`, cost fields, path mask, or whether an inactive slot has a stable identity. A serializer needs one normative rule for all these fields.
- **Policy/oracle identity:** Current code has ordered policy specs, a continuation row, then oracle variants and target tracks (`meco_candidates.py:1291-1313,1323-1381,1421-1452`). Current `DeviceRouteQuery` carries candidate ID, snapshot, world, slot, and intake state (`experimental_route_evaluation.py:140-148`); it has no serialized policy/oracle enum, route-variant registry, or policy source-row field. V4 proposes `query_kind`, `policy_source_row`, `oracle_identity_code`, `route_variant`, and `candidate_index`, and says oracle `policy_source_row=-1`, but it does not give normative code tables or define which canonical global ordinal `candidate_index` indexes (full query source axis versus compacted active rows). These identity meanings need an exact registry and a Plan C mapping agreement.
- **Costs and provenance:** Current units are `normalized_grid_edge_units` (`experimental_route_evaluation.py:82-83,135,163-166`). The evaluator documents meters for path points and normalized grid-edge units for path-integrated cost, with learned costs as dimensionless additive penalties per cardinal grid step (`experimental_route_evaluation.py:731-738`). Learned-map provenance includes producer, scale, raster schema/geometry, and SHA-256 of normalized map values (`experimental_route_evaluation.py:751-785`). The device path emits distance, endpoint error, weighted cost, bump cost, and learned cost tensors (`experimental_route_evaluation.py:615-628`). This supports the proposed five columns and their broad units, but V4 needs to specify the exact source/provenance encoding in the fixed ABI: dynamic strings cannot be stored, and the bundle currently provides only a component-presence mask, not a provenance registry/digest field. Current optional component presence also differs when no learned map is supplied.
- **Selected-row association:** Current candidate construction gathers policy route fields by an explicit device `selected` index and creates a compact policy route batch (`meco_candidates.py:815-830`); candidate selection uses validity masks and the selected canonical candidate position (`meco_candidates.py:831-850`). This supports preserving a stable association, but the V4 fields need a normative join rule connecting the serialized route query ordinal, source candidate row, `candidate_index`, the Plan C ordinary/continuation maps, and the selected common command. The proposal does not fully specify that join or sentinel behavior for inactive/padded rows.

These observations are limited to the pinned source checkout and the cited route contract. They do not establish runtime capacity across arbitrary environment configurations, adoption by Plan C or artifact owners, implementation correctness for a V4 serializer, or Stage 1 qualification.

## Required resolution before disposition

1. Pin `Q_C=9,681` to the exact `E=504` profile and specify how a different profile changes `Q_C` and payload sizes.
2. Define normative query ordinal/order, `query_kind`, policy/oracle identity and variant code registries, plus exact `candidate_index` and `policy_source_row` mapping/sentinels.
3. Define inactive-row values and invariants for status, feasibility, costs, path length, path coordinates, and path-point mask.
4. Bind `P=73` to the exact planner configuration and define full path padding/mask semantics.
5. Define fixed encoding/provenance for weighted, bump, and learned cost components, including absent learned-cost behavior and source-map digest linkage.
6. Specify the join from route rows through Plan C ordinary and continuation mappings to the selected command, including inactive rows.

## Authority statement

This response records only the delegated `route_owner` technical decision under the scoped owner-response plan. It is not human maintainer approval, does not decide another owner row, does not adopt a shared ABI, does not authorize implementation, and does not establish ABI publication or Stage 1 qualification.
