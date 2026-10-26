# Independent pre-G0 SOURCE/API review: P1.2 transition-row-block codec

- **Verdict:** SOURCE/API ACCEPT
- **Reviewed commit:** `9257d6c577980c56fd103af96509d26c7f78bd2a`
- **Reviewed paths:**
  - `frc_defense/meco_transition_v4.py` — SHA-256 `8c2bc3797b451e22fc394f3a650170b29052ff14f9df0d6974ea1477162492ef`
  - `tests/test_meco_transition_v4.py` — SHA-256 `9eacaa8a4a36476a95c47b98ba2629aaf976ca85aa86b83a8048e4218004a65f`
- **Accepted plan:** `ff8850d26a7311ccfe11b421299d45af43d23a8f`, plan SHA-256 `5db9b5cc3bd6ac12d94795d7647fa8961f4a373bfc745b904d3d18103a41c25b`
- **Plan review:** `bdcd916c39cd5eb6ea90a8eae22192492e852566`, record SHA-256 `73ae50f7fce64c8622cf304d9b438292d3120ce277e24240a302f6b4c74a0ad7`
- **Review mode:** exact-head, static source/API inspection only.

## Findings

The implementation is based directly on accepted schema commit `38fdb1d0423f0e723c4e1fb533e36eb7b88c4715` and changes exactly the two allowlisted paths: the transition schema module and its test fixture. No event registry, native HIP/C++, build configuration, runtime integration, persistence, campaign, or dashboard path is changed.

The API takes an ordered sequence of field tensors and returns an ordered tuple. This preserves distinct positions for duplicate declaration names (including schema version, track code, route cost, target handle, goal coordinates, and fallback) rather than collapsing them into a name-keyed mapping. The implementation walks the flattened `TRANSITION_FIELDS` declaration in its accepted group/field order; for each field, contiguous row-major bytes implement increasing row index followed by C-order inner axes. The little-endian declared widths and absence of padding yield the planned 314,332,992-byte block, with structural length checks on pack and unpack.

Pack and unpack require CUDA/HIP tensors, reject CPU tensors, validate shapes and dtypes, and require all packed fields on the same device. Tensor contiguity conversion, byte reinterpretation, concatenation, slicing, cloning, reshape, and output remain tensor operations on that device. The only host-side checks inspect metadata (device, shape, dtype, byte order, declared sizes); there is no host numeric inspection, hash/digest computation, or CPU fallback/migration path.

The byte reinterpretation and round-trip fixtures preserve raw bool and IEEE-754 patterns without normalization or semantic checks. The implementation does not inspect mask semantics, finite values, event identity/order/clock semantics, or claim complete V4 validity. It adds no bundle/header/identity/event placement, payload digest, reader, or persistence API. Test cases are authored for byte length, field and axis order, dtype/shape rejection, and raw-byte round trip; they were not executed.

The exact diff is within the accepted standalone row-block scope. This SOURCE/API ACCEPT applies only to this exact head and its static API/code review. It does not satisfy G0 or authorize imports, test execution, builds, GPU operations, simulation, dashboard, training, integration, merge, promotion, or full-bundle/P1.3/P1.4 work. Those remain subject to the accepted plan's gates.

No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, lease, checkpoint, or artifact operations were performed.
