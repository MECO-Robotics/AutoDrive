# Independent Pass 3 review: Route V4 owner response

- **Verdict:** ACCEPT
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/route_join_reviewer`
- **Reviewer task:** independent exact-head Pass 3 review assigned by `/root`
- **Response author:** `/root/route_owner` (delegated Route technical owner)
- **Reviewed commit:** `96a75df0221a8c76e8b9c5fe564bb8d336c98c80`
- **Reviewed file:** `docs/plans/v4-owner-responses/route.md`
- **Reviewed file SHA-256:** `29ce9a21719e2cb65dbe513565b9012c51b9c7d8e00b65a57a46a98172aaf113`
- **Pinned Plan C response:** commit `1af5d55dea0eb7f3a2f2938cd71423841bfa86b1`, SHA-256 `8ac11dfbe85aa0c03d834c0530eee2f313f06d6fdc97ef479bff9b923cefdc50`.
- **Pinned proposal:** V4 commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256 `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- **Pinned matrix:** commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256 `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.
- **Supersedes:** REVISE review of Route commit `953ee503dbd140146b7c2914e0200dd0339d93e0` at this same path; this record reviews the corrected exact head.

## Review scope and evidence

Rechecked the exact Route response commit and blob SHA from the pushed response branch. The addendum now pins the corrected and independently accepted Plan C response. Plan C's persisted exact-head review identifies that response's offset ordering correction: rows 512–515 are variant 1/actions 0–3, successive groups of four advance the variant, and rows 532–535 are variant 6/actions 0–3. The Route table states the same source order. Plan C also defines `ordinary_row_map[j] = selected_policy[j]`, `continuation_row_map[0] = continue_index`, and row identity code `r+1`; Route's full-sink mapping `q=r` for policy rows, continuation row 536, and selected command identity therefore joins those definitions without compaction or a second identity lookup. The stale-pin issue from the prior review is corrected.

Checked the route/source mapping and sentinels against `frc_defense/meco_candidates.py` and `frc_defense/experimental_route_evaluation.py`: policy construction is eight base rows, E sensed-target rows, 24 approach rows, and one continuation row; the 18 oracle variants each use four base target entries plus E expansion entries. At E=504 this yields Q_C=9,681. The query builder sorts valid sensor track indices stably before variant-major expansion; the documented fixed target-buffer order is consistent with that construction, with inactive duplicate/base placeholders retaining their full sink ordinals. Chunk mapping is contiguous in 64-row spans, yielding 151 full chunks and one final 17-row chunk. The route response explicitly distinguishes its proposed fixed sink serialization from the existing concatenated chunk outputs.

The route status codes and current path behavior match `experimental_route_evaluation.py`; the 73-point bound follows the default `tensor_adstar.py` maximum of 72 planner points plus the prepended start. The addendum specifies inactive-row sentinels, path masks/tail zeros, target-handle checks, oracle identity codes, and selected-row joins as an owner-level ABI contract. It does not claim that current Python/native producers serialize those fields. The response keeps its owner disposition **PENDING** for profile generalization, Observation/native handle issuance/snapshot binding, artifact learned-cost provenance, and canonical payload binding. That disposition is supported; the ACCEPT validates this exact response and does not adopt V4 or resolve those other owners' gates.

No source/runtime changes, tests, device execution, simulator evaluation, dashboard action, or campaign action were performed.

This is an independent technical review only. It is not human maintainer approval, cross-owner adoption, ABI publication, implementation authorization, or Stage 1 qualification.
