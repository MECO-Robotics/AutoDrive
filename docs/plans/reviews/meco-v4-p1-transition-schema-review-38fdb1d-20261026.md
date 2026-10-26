# Independent pre-G0 SOURCE/API review: P1 transition schema declarations

- **Verdict:** SOURCE/API ACCEPT
- **Reviewed commit:** `38fdb1d04`
- **Reviewed path:** `frc_defense/meco_transition_v4.py`
- **File SHA-256:** `689b84acfd2d327c9ca4c26b081cf997acf511a0f14ed6f605a6caa91562f52f`
- **Accepted V4 proposal:** `6b4ce4a939a2cdea375bd4be6879f6f7bc73141`, digest `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`
- **Reconciled profile:** artifact commit `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`, profile SHA-256 `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`
- **P1.1 map:** `c902659d1b9079b3afcc2e4796780d97020b6726`, SHA-256 `c8c1d0594977ec456d8513293fe7ba0ca0020eba52a822e75cad34104aa36b4e`
- **P1.2 event registry:** `78b640630fde1aa9c0934463525e322ba57e027e`, SHA-256 `5e4c803a676e7a445b686d0f73e5d18e7341533119571eae8e9545d7a259d278`
- **Review mode:** static source and Git-object inspection only; no imports, tests, builds, or runtime execution.

## Evidence

The corrected declaration restores the accepted V4 field names, including `fallback` for the fixed macro trace and the unprefixed route-sink names `query_kind`, `policy_source_row`, `oracle_identity_code`, `target_handle`, `goal_xy`, `query_active`, `status_code`, `feasible`, `path_xy`, `path_length`, `distance_m`, `endpoint_error_m`, `weighted_cost_grid_edges`, `bump_cost_grid_edges`, `learned_cost_grid_edges`, `candidate_index`, and `path_point_mask`. `route_variant` already matched the accepted name. Field order, scalar types, shapes, and group boundaries match the accepted proposal.

The profile identifier, digest, capacities, track codes, 96-byte little-endian header, and 123-byte identity prefix match the reconciled profile. Static arithmetic over the declared fields gives the accepted group byte counts: 94, 30,628, 512, 149,959, 28,728, 1,983,764, 7,628,628, 99, 28, 80, 122, and 264 bytes, totaling 9,822,906 bytes per row. The 32 rows occupy 314,332,992 bytes; adding the 135,936-byte event table and 123-byte identity gives the 314,469,051-byte payload; the 96-byte header yields 314,469,147 bytes total.

The reviewed candidate adds only the allowlisted `frc_defense/meco_transition_v4.py`. It remains a declaration registry: it adds no serializer, reader, dynamic value validation, digest computation, or host numeric processing. The separate accepted P1.2 event registry and corrected P1.1 source map remain pinned above. This acceptance covers static schema declarations only; it does not accept a transition serializer/reader, event producer or consumer, P1.3/P1.4 work, runtime behavior, or integration.

No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, or lease operations were performed.
