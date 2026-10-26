# Independent Pass 3 review: Route V4 owner response

- **Verdict:** ACCEPT
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** independent exact-head Pass 3 review assigned by `/root`
- **Response author:** `/root/route_owner` (delegated Route technical owner)
- **Reviewed commit:** `499cd2bf5a4dead0975f4e14fcb6df25d7661858`
- **Reviewed file:** `docs/plans/v4-owner-responses/route.md`
- **Git blob:** `1574e2fe7059960fd74ee9c67452437be1bceb25`
- **Reviewed file SHA-256:** `f3e32aba59c9ba018e4e2d6a37bfee26b1ddb536d7e98ebf0ff1042aff481b82`
- **Pinned proposal:** V4 commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256 `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- **Pinned matrix:** commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256 `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.
- **Pinned Plan C response:** commit `1af5d55dea0eb7f3a2f2938cd71423841bfa86b1`, SHA-256 `8ac11dfbe85aa0c03d834c0530eee2f313f06d6fdc97ef479bff9b923cefdc50`.
- **Pinned Observation/native response:** commit `f6af063a025eef967be442468355eb5a73d742c9`, SHA-256 `886a177f91250cdbbb4a9c4a767c3424a1b56a0ce0033c419a570facb26582d0`, Git blob `a58f84aa1c19679d5fb34f0fbd116776a1c7216b`; its exact-head ACCEPT record is commit `ae991535600b3fc5c67dd6ea1d167a379e3a5c64`, record SHA-256 `53649f76ee8204823cf7850cc3cbed7934a7b34a8a6fa71aec8210a906567cff`.
- **Pinned PPO/RNG response:** commit `602565ef99986996d2c12d06024833247dc3d395`, SHA-256 `219f3466741c17d9e49a4d402f0b3f8ae6f016f0738a068034a8ad312f610d75`.
- **Pinned Artifact/manifest response:** commit `45dc7b4894d97fd7aa04b1f84aac3cdc207f0c9d`, SHA-256 `9b7a571bd98c6aed6700d445aea8e05a2308cdc5c1bcba9c93c961c4194d8289`.
- **Supersedes:** prior review record at this path for Route addendum commit `96a75df0221a8c76e8b9c5fe564bb8d336c98c80`; this record reviews the updated full Route owner disposition and its pinned dependencies.

## Review scope and evidence

Rechecked the exact response commit, blob, and file SHA-256 against the accepted V4 proposal, owner matrix, and owner-response closure plan. The ADOPT is limited to the Route owner row and pinned `meco-ppo-v3-3v3-f504-n1-s6-r32-t12-v1` profile. It does not claim that producers or serializers implement the contract, publish the shared ABI, authorize implementation, or establish qualification. Its explicit authority statement and these scope limits are consistent with the closure plan.

The profile capacity and source mappings are internally consistent with the cited source: `E=504`, `Q_C=9,681`, 64-row chunks yielding 151 full chunks and one 17-row chunk, policy sink rows `q=0..536`, and oracle rows `q=537..9,680`. Source construction orders the six policy offset variants outside their four actions (`meco_candidates.py:1306-1313`). Oracle target indices are stably sorted with invalid entries sent to a stable sentinel before variant-major expansion (`meco_candidates.py:1427-1449`). The response specifies stable full-sink ordinals, `q=r` for policy rows, continuation row 536, `route_ref=policy_source_row=q`, oracle `policy_source_row=-1`, and no compaction when rows are inactive. These agree with the pinned Plan C row maps and its `r+1` candidate identity dictionary. The inactive sentinels, active NO_PATH distinction, path masks and zero-tail serialization are explicitly normative owner decisions and are not misrepresented as current producer behavior.

The Observation/native dependency is now pinned to corrected response `f6af063...` and exact ACCEPT record `ae991535...`. The response's complete target-handle tuple, UUIDv4 environment identity, reset generation, observer slot, slot generation and snapshot token are used as the route contract; it prohibits deriving a handle from a buffer index or confusing observer-track identity with physical score-event identity. The addendum preserves the original decision snapshot as route provenance while requiring live handle validity during later macro ticks. This is consistent with the accepted Observation/native lifecycle and its corrected terminal/truncation rule does not alter route identity semantics.

Route geometry and status claims match the cited route evaluator: snapshot XY in meters, status codes `INVALID=0`, `AT_GOAL=1`, `NO_PATH=2`, `FEASIBLE=3`, and the default 72-point planner bound plus prepended start for `path_xy[73,2]`. The response distinguishes the evaluator's current endpoint-padded tails and concatenated chunk outputs from required V4 zero-tail and fixed-sink serialization.

The cost/provenance decision is supported by the cited source and V4 contracts. The learned-map digest preimage includes the schema tag, dimensions, 11 little-endian float64 geometry values, and row-major little-endian float64 values (`experimental_route_evaluation.py:267-290`); the provenance string includes producer, geometry, map digest, scale, and integration schema (`:751-785`). Cost units and bump/learned component-presence bits match the field table. The response binds the canonical provenance string to Plan C's family-specific `support_digest`, while the Artifact response retains the separate full-payload digest and PPO/RNG keeps `support_digest`, `input_digest`, and `transition_digest` distinct. It correctly requires future V4 device digest computation to match the canonical bytes without invoking the legacy CPU-copy helper in the live path. This is a contract decision, not implementation evidence.

No unresolved Route-owner decision remains within this pinned row/profile. The response identifies ABI-owner reconciliation/publication and implementation/qualification as subsequent external gates. Changes to pinned inputs require a fresh Route review. It makes no human maintainer approval claim and decides no other owner's row.

## Review limits

This verdict is an independent technical review of the exact Route owner response only. It is not shared ABI adoption/publication, implementation authorization, or qualification. No tests, imports, compilation, runtime/device, simulator, dashboard, checkpoint, campaign, training, or evaluation actions were performed.
