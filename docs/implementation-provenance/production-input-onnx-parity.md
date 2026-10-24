# Phase 5 production-input ONNX parity tranche

**Scope:** bounded capture of real production scorer request tensors and
checkpoint-backed ONNX export/parity against those exact tensors. This is a
scoped transcript summary, not whole-plan approval or runtime qualification.

The planner approved a narrow Phase 5 tranche to close the production-input
gap in the existing synthetic scorer parity harness. The implementation adds
explicit opt-in capture with a request-count and tensor-size bound, atomic
artifacts, canonical input and request validation, checkpoint/protocol/source
and candidate/legal-pose provenance, and per-tensor plus aggregate content
digests. Export checks capture provenance and the exact tensor content before
writing ONNX and records the capture binding and graph digest in its manifest.
Captured parity requires that manifest and rejects changed capture contents,
request provenance, input shapes/dtypes, or ONNX graph identity. Existing
synthetic export and parity remain clearly distinguished.

The implementation was merged at
`eebd4ae57b9f226dade8bac99a6734596381759e`, retaining feature commits
`77e12b37ea8702221401b66ab0fe8586c81cd00e`,
`e09f31d022c8d59dc0ab41da0e361ca711e1fd7b`, and
`b098d725b7c2c2da90ddd6b228989bfc27d949ac`. The integrated candidate at
`d10764da3ee426c2524f56d85a806725b307e994` received scoped reviewer approval;
the feature-level review approved `b098d725b7c2c2da90ddd6b228989bfc27d949ac`.

No real production capture or captured parity run was performed. No simulator,
tests, Hailo compiler/device, or deployment qualification was run or established.
Capture parity implementation is not a qualification result and makes no claim
about simulator behavior, route correctness, actuation, Hailo support, or robot
hardware.
