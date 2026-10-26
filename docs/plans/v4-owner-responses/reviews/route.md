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

The response pins Plan C commit `0d8be1b664106b3744d62fe2f78b5f5191963d58` as the contract for its cross-row mapping. That Plan C head had the offset descriptor ordering reversed. The corrected Plan C response is now at `1af5d55dea0eb7f3a2f2938cd71423841bfa86b1`, SHA-256 `8ac11dfbe85aa0c03d834c0530eee2f313f06d6fdc97ef479bff9b923cefdc50`. It specifies four consecutive actions per variant for rows 512–535, and confirms the `r+1` row identity registry, K=16, and ordinary/continuation row maps. The Route addendum independently states the corrected four-actions-per-variant ordering and q=r full-sink coordinate join, so its mapping is semantically compatible with the corrected Plan C response; the old pin does not itself make the join incompatible. However, Route currently attributes this closure to the superseded head, whose reversed descriptor order contradicts the cited variant mapping. The response must pin the corrected Plan C head and cite it as the compatibility basis before its cross-owner closure claim can be accepted. The source-global coordinate join remains independently sound, while the candidate identity and K/map bindings are supported by the corrected Plan C head, which remains PENDING until this Route join receives acceptance.

The Observation/native response at `af0f8b4ca4759441030f2625aa23ce555a06bbac` defines the four-int target handle lifecycle but remains pending on environment-instance issuance, macro projection and T=12 binding. Route appropriately lists those as unresolved dependencies. The response also correctly keeps learned-cost provenance and fixed sink serialization pending. No source/runtime/tests or simulator/dashboard actions were performed.

## Required revision

Update the Route addendum to pin and describe the corrected exact Plan C response head `1af5d55dea0eb7f3a2f2938cd71423841bfa86b1` and its SHA-256. Preserve the source-backed full-sink ordinal to source-global-row coordinate join, and cite the corrected Plan C variant ordering, K=16 maps, and `r+1` identity registry. Keep Plan C and Route owner dispositions PENDING until both exact-head reviews are accepted. Then request a fresh exact-head Pass 3 review of the updated Route response.

This verdict does not imply that the route/source ordinal join or candidate identity mapping is semantically incompatible. The corrected Plan C response establishes compatibility; this Route response still needs to replace its stale, contradictory pin so the closure record is reviewable against the current exact head.

This record is only an independent technical review. It is not human maintainer approval, cross-owner adoption, ABI publication, or implementation authorization.
