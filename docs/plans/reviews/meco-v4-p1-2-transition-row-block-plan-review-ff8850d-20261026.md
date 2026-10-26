# Independent Pass 1 review: P1.2 transition-row-block codec plan

- **Verdict:** ACCEPT
- **Reviewed commit:** `ff8850d26a7311ccfe11b421299d45af43d23a8f`
- **Reviewed file:** `docs/plans/meco-v4-p1-2-transition-row-block-codec-plan-20261026.md`
- **File SHA-256:** `5db9b5cc3bd6ac12d94795d7647fa8961f4a373bfc745b904d3d18103a41c25b`
- **Review mode:** read-only static plan and contract inspection.

## Findings

The revision removes the unsupported payload-relative row-block offset. It now states that the eventual offset depends on the pending owner choice of section order and that this slice returns only the standalone row block. No event-table location or full-payload ordering is inferred.

The plan pins the accepted transition schema, event registry, and approved implementation base. Its row-block size is consistent: 32 rows × 9,822,906 bytes = 314,332,992 bytes. Traversal is concrete and bounded: declared group order, declared field order, rollout row 0 through 31, then C-order elements with the last inner axis fastest; scalars use their declared little-endian widths without padding.

The codec's claim is limited to byte layout and shape/type fidelity. It preserves float bit patterns and booleans, and does not inspect finiteness, mask-false zeroing, semantic mask correctness, or other V4 validity. It excludes host numeric inspection and hashing. The two-path allowlist is finite and excludes native HIP/C++, bundle/header/identity/event APIs, persistence, and runtime integration.

The plan preserves the accepted gates: before G0, only source and test-fixture authoring plus static review; execution and device validation require accepted G0 evidence, explicit authorization, and the reviewed contract-anchor bridge. It requires both assigned WX9100 devices with no CPU/P600 fallback or migration and retains P1.3/P1.4 as blocked. Full-bundle serialization, payload digest, envelope parsing, and artifact read/write remain blocked pending owner/maintainer adoption of section order, re-pinning, and a successor review.

Accordingly, this ACCEPT authorizes only the bounded P1.2 standalone transition-row-block source/test authoring described in this exact plan; it does not authorize imports, test execution, builds, device work, simulation, dashboard, training, integration, merge, promotion, or full-bundle implementation before their stated gates.

No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, lease, checkpoint, or artifact operations were performed.
