# Independent Pass 3 review: Route V4 owner response

- **Verdict:** REVISE
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/route_join_reviewer`
- **Reviewer task:** independent exact-head Pass 3 review assigned by `/root`
- **Response author:** `/root/route_owner` (delegated Route technical owner)
- **Reviewed commit:** `953ee503dbd140146b7c2914e0200dd0339d93e0`
- **Reviewed file:** `docs/plans/v4-owner-responses/route.md`
- **Reviewed file SHA-256:** `ecbd325343f9d3cabaf05e042060d8e4d2b7a6948cea2a19f9c4ad21acd19924`
- **Pinned proposal:** V4 commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256 `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- **Pinned matrix:** commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256 `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.

## Review scope and evidence

Reviewed the exact Route response against the accepted closure plan, pinned V4 proposal and owner matrix, Plan C and Observation/native owner responses, and route/candidate source. The response's core route-sink arithmetic and global ordinal mapping are supported: at E=504, the source builds 537 policy rows and 18 oracle variants over 508 target entries, giving Q_C=9,681; chunk mapping is contiguous at 64 rows, and the default retained path bound is 73 points. Policy sink ordinal q maps to source-global row r=q. The source's ordinary and continuation views use separate maps, so mapping a populated Plan C row through its map and then to sink q=r is structurally independent of descriptor kind/variant enum values. The source's oracle target ordering sorts stable track indices; with the fixed sensor-buffer representation this is consistent with valid target slots in ascending buffer-slot order, followed by invalid placeholders.

The response nevertheless pins Plan C commit `0d8be1b664106b3744d62fe2f78b5f5191963d58` as the contract for its cross-row mapping. That head's descriptor/identity mapping is known to have been superseded after a reversed mapping was identified. The available subsequent Plan C response (`78b98e47e8689d9c231104b02928f1fb4d7bc9f8`, SHA-256 `31e0c14d883063f7dca9fbe35b1d49f87ae93b4969240123a8a81b07514edc88`) returns K=16, candidate identity registry, and mask projection to explicit pending decisions. It does not currently adopt the `candidate_identity_code = r+1` registry relied on in the Route addendum. Therefore the coordinate join `route_ref = policy_source_row = source_global_row = q` can stand independently, but the full compatibility claim that a selected command's candidate identity is the Plan C dictionary's `r+1` is not established against the current Plan C response. The stale pin also makes “independent Plan C maps” ambiguous because K and those maps are conditional owner decisions in the newer response.

The Observation/native response at `af0f8b4ca4759441030f2625aa23ce555a06bbac` defines the four-int target handle lifecycle but remains pending on environment-instance issuance, macro projection and T=12 binding. Route appropriately lists those as unresolved dependencies. The response also correctly keeps learned-cost provenance and fixed sink serialization pending. No source/runtime/tests or simulator/dashboard actions were performed.

## Required revision

Update the Route addendum to pin and describe the current exact Plan C response head. Keep the source-backed full-sink ordinal to source-global-row coordinate join separate from the Plan C descriptor/candidate-identity registry. Mark `candidate_identity_code = r+1`, K=16 ordinary-map binding, and any dependent mask projection as conditional until Plan C adopts an exact registry/profile and an independent reviewer accepts that exact head. Preserve the Route disposition as PENDING and identify the re-review dependency. Then request a fresh exact-head Pass 3 review of the updated Route response.

This verdict does not imply that the route/source ordinal join itself is reversed or unusable; it means the response's current pinned cross-row identity contract is not supported by the superseding Plan C decision. The owner response's PENDING disposition is directionally appropriate, but this exact response head must be corrected before it can be accepted as a reliable closure record.

This record is only an independent technical review. It is not human maintainer approval, cross-owner adoption, ABI publication, or implementation authorization.
