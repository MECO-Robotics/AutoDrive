# Independent review: proposed V4 payload section-order amendment

- **Verdict:** PASS 1 ACCEPT — proposal only; no ABI option is adopted by this review.
- **Reviewed commit:** `32edc197cc53812cdff1b74190629db1a2acdc72`
- **Reviewed file:** `docs/plans/meco-v4-payload-section-order-owner-amendment-20261026.md`
- **File SHA-256:** `7ef22d7afbcb7897066d32e9a231f516854d53be84380279f13191c761bc0525`
- **Review mode:** static document and Git-object inspection only.

## Arithmetic and alternatives

Both alternatives account for the fixed 123-byte identity prefix, 314,332,992-byte transition stream, and 135,936-byte event table:

- Choice A: identity `[0,123)`, transitions `[123,314,333,115)`, events `[314,333,115,314,469,051)`. Lengths sum to 314,469,051 payload bytes.
- Choice B: identity `[0,123)`, events `[123,136,059)`, transitions `[136,059,314,469,051)`. Lengths also sum to 314,469,051 payload bytes.
- Adding the 96-byte envelope header gives 314,469,147 bytes for either complete bundle.

The options preserve the accepted internal transition/event inventories while making previously unspecified section placement, transition field-major row traversal, inner-axis C order, and event record order explicit. No gaps, overlap, padding, or trailing bytes are proposed.

## Authority boundary

The document repeatedly labels A and B as proposals, requests an explicit Artifact/manifest and shared-ABI owner decision with maintainer ratification, allows rejection or a fully specified alternative, and requires a reviewed successor proposal/reconciliation/profile/schema pin before full codec work. It explicitly withholds full-bundle packing/reading, payload digest verification, and persistence until that chain is complete. It does not alter the accepted ABI or authorize source changes. The exact payload digest span and raw-proposal profile digest are specified for the eventual adopted layout; row-level and identity digest preimages remain out of scope.

This acceptance is solely an independent review of the proposal’s arithmetic and approval gates. It is not owner adoption, maintainer approval, ABI reconciliation, or codec authorization. No code, tests, imports, builds, simulator, dashboard, GPU/device, campaign, lease, checkpoint, or artifact operations were performed.
