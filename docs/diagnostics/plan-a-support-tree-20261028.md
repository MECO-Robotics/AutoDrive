# Plan A support tree evidence

A-track training binds the same ordered tensor bytes and metadata as the
existing HIP support digest using a domain-separated tree SHA-256, labeled
`tree-sha256-v2`. PPO requires that scheme and rehashes support before any
optimizer step. The helper offers a device-byte result for a later combined
readback, and a hex wrapper for reporting and validation. The pinned Plan A
route policy source and production gates are unchanged.

Evaluation and counterfactual generation retain `sha256-v1` for compatibility.
Their unpinned v1 wrapper preserves the v1 stream while flattening scalar root
and fallback tensors before the dtype byte view. The old pinned implementation
attempts a byte view directly on zero-dimensional tensors, which PyTorch rejects.

A synthetic 12,096-node support and graph pair measured median 2,924.56 ms for
the scalar-corrected serial HIP v1 and 2.46 ms for tree v2 across three paired
samples. This includes header/packing and the 32-byte digest readback; it does
not include routes or policy/training. It is not an end-to-end speedup, and the
original v1 does not complete on the scalar input. The measurement and source
hashes are retained in the support-tree evidence directory.
