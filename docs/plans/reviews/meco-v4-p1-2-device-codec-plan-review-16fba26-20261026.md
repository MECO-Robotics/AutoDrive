# Independent Pass 1 review: P1.2 device codec and persistence plan

- **Verdict:** REVISE
- **Reviewed commit:** `16fba26b8875c42b11e43f25d8afe297711116ff`
- **Reviewed file:** `docs/plans/meco-v4-p1-2-device-codec-and-persistence-plan-20261026.md`
- **File SHA-256:** `157dfbd8e6b26b0e221b7a8b98e0e7a78c6c4b7e79c6c20e841815c723f84743`
- **Review mode:** read-only static plan and accepted-contract inspection.

## Supported constraints

The plan pins the accepted P1 source/API declarations and reconciled profile, confines source edits to the allowlisted transition codec file and its fixture file, treats event rows and uncomputed identity/row digest fields as opaque, and excludes producer/consumer semantics. Its no-CPU numeric/hash, no native HIP, no dependency, no P600/fallback, pre-G0 source-only, G0/runtime, two-WX9100, and P1.3/P1.4 gates are consistent with the accepted P1 plan. The revised digest scope is sound: compute only the raw proposal `profile_digest` constant and the SHA-256 over the exact payload span; preserve caller-supplied source/config/checkpoint identity bytes and input/support/transition digest fields without interpreting them. This keeps the unresolved row digest domains outside this slice.

## Required contract clarifications

1. **Inner-axis byte order:** The plan now fixes field-major order and increasing rollout-row order, but “inner axes in their listed order” does not state which axis varies fastest. Pin explicit contiguous row-major/C-order (first listed axis outermost, last listed axis fastest), or cite an accepted contract that states this exact traversal.
2. **Event-table placement:** The event table is declared an opaque region whose records retain supplied order, but the plan does not give its byte offset/placement in the payload relative to the identity prefix and transition columns. The accepted totals determine its length, not its location. Pin the exact payload section order/offset before specifying a serializer or byte-layout fixtures.
3. **Float/mask behavior:** The accepted V4 proposal fixes little-endian IEEE-754, one-byte bool encoding, valid-float finiteness, signed-zero preservation, and zero values under false masks. The plan cites the accepted profile inputs but does not say whether this codec only preserves those bit patterns or also checks those value rules. Clarify that boundary and cite the exact proposal path/commit alongside its digest. Do not add unapproved semantic repair or host checks.

The profile digest and payload digest spans are sufficiently bounded by the current revision. Row/source/config/checkpoint digest values are correctly excluded from computation and validation pending exact domains. No implementation or test execution was performed.
