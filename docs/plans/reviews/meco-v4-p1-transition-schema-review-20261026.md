# Independent pre-G0 SOURCE/API review: P1 transition schema declarations

- **Verdict:** REVISE
- **Reviewed commit:** `566f15e92983c72cd0f0b02ebb7e7ac8496062f6`
- **Reviewed path:** `frc_defense/meco_transition_v4.py`
- **File SHA-256:** `c90041346a05eaad862d7826a8b925263071d6148f26eecffc4d378ee292dcff`
- **Accepted V4 proposal:** `6b4ce4a939a2cdea375bd4be6879f6f7bc73141`, raw proposal/profile digest `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`
- **Accepted reconciled profile:** artifact commit `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`, profile SHA-256 `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`
- **Review mode:** static source and Git-object inspection only; no imports, tests, builds, or runtime execution.

## Findings

The candidate's profile identifiers and capacity constants match the reconciled profile. Header declarations match the adopted 96-byte little-endian header: field widths total 96 bytes, and magic, versions, track codes, digest spans, and reserved regions match the V4 proposal. Static identity fields total 123 bytes. Static arithmetic over the 120 transition field declarations yields these group byte counts, matching the accepted profile exactly: identity/schema 94; observation 30,628; recurrent state 512; Plan C candidates 149,959; observed targets 28,728; Plan A support/search 1,983,764; Plan C route sink 7,628,628; selected command 99; PPO 28; RNG 80; provenance/replay 122; macro trace 264. Total is 9,822,906 bytes per row, 314,332,992 bytes for 32 rows, 314,469,051 payload bytes including event table and static identity, and 314,469,147 bytes including the header.

The candidate is confined to the single allowlisted path `frc_defense/meco_transition_v4.py`; it declares metadata only and contains no serializer, reader, dynamic validation, digest computation, or other host numeric work. The ABI registry's fixed field names nevertheless diverge from the accepted declaration:

- The accepted macro-trace field is `fallback:bool[12]`; the candidate names it `macro_fallback`.
- The accepted route-sink group uses `query_kind`, `policy_source_row`, `oracle_identity_code`, `target_handle`, `goal_xy`, `query_active`, `status_code`, `feasible`, `path_xy`, `path_length`, `distance_m`, `endpoint_error_m`, `weighted_cost_grid_edges`, `bump_cost_grid_edges`, `learned_cost_grid_edges`, `candidate_index`, and `path_point_mask`. The candidate renames these with `route_` prefixes.

The accepted P1 plan pins field names along with types, widths, order, capacities, and registries. No ABI amendment authorizes these name changes. Restore the accepted names in the schema declaration, or obtain and review an ABI reconciliation amendment before changing them.

## Scope

This review does not accept the transition schema candidate. It confirms the profile/header/identity/shape arithmetic and allowlisted static-only scope, but field-name drift requires correction before SOURCE/API ACCEPT. No P1.3/P1.4 behavior or runtime API is reviewed or authorized here.

No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, or lease operations were performed.
