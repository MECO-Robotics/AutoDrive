# Independent pre-G0 SOURCE/API review: P1.2 transition-row-block codec

- **Verdict:** SOURCE/API ACCEPT
- **Reviewed commit:** `8bd253042ee88d69afa7653886bf3d744e5c8d5e`
- **Reviewed paths:**
  - `frc_defense/meco_transition_v4.py` — SHA-256 `8c2bc3797b451e22fc394f3a650170b29052ff14f9df0d6974ea1477162492ef`
  - `tests/test_meco_transition_v4.py` — SHA-256 `1dbfcc62061818e55210df08839c4a25aa1ad85d430961d36469d153ee305b76`
- **Accepted plan:** `ff8850d26a7311ccfe11b421299d45af43d23a8f`, plan SHA-256 `5db9b5cc3bd6ac12d94795d7647fa8961f4a373bfc745b904d3d18103a41c25b`
- **Plan review:** `bdcd916c39cd5eb6ea90a8eae22192492e852566`, record SHA-256 `73ae50f7fce64c8622cf304d9b438292d3120ce277e24240a302f6b4c74a0ad7`
- **Review mode:** exact-head, static source/API inspection only.

## Findings

The codec source is unchanged from the previously reviewed source at `9257d6c577980c56fd103af96509d26c7f78bd2a`; its SOURCE/API ACCEPT remains valid. It is based directly on the accepted schema head and the exact changes stay within the plan's two-path allowlist. The ordered sequence/tuple API keeps duplicate field names as separate declaration positions. Pack/unpack operate on CUDA/HIP tensors, validate device, shape, and dtype metadata, and keep tensor contiguity conversion, byte reinterpretation, concatenation, slicing, cloning, reshape, and outputs on the device. Field declaration order implements the accepted group/field order; row-major contiguity implements row order then C-order inner axes. The computed fixed block size is 314,332,992 bytes. No host numeric inspection, hashing, event semantics, bundle layout, payload digest, or persistence API is added.

The updated test fixture adds distinct first-element sentinels to each field and constructs a direct expected whole-block byte sequence from the ordered fields before comparing it to the packed output. These authored assertions are within the accepted test-fixture scope and are not imports or execution performed by this review. Existing checks continue to cover first-field boundaries, raster placement/axis byte order, shape/dtype rejection, output metadata, and raw float/bool byte round-trip. The fixtures do not assert semantic mask/finite-value validity or full V4 compliance.

The updated exact head is therefore accepted for its bounded static SOURCE/API scope. This review does not satisfy G0 or authorize imports, test execution, builds, GPU/device operations, simulation, dashboard, training, integration, merge, promotion, or full-bundle/P1.3/P1.4 work. The accepted plan's gates remain in force.

No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, lease, checkpoint, or artifact operations were performed.
