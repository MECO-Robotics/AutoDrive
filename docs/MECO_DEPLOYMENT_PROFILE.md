# MECO deployment export and profiling

The Phase 5 harness exports fixed-shape A1, A2, A3, C1, and C2 scorer graphs
from explicit trained checkpoints, checks each ONNX graph, and records its
operator inventory. Each artifact/report records checkpoint path, SHA-256,
track, update, run ID, training sources, and configuration. It refuses missing
or track-mismatched checkpoints. Run from the repository root:

```bash
python scripts/export_meco_onnx.py --checkpoint-root checkpoints/meco-five --output artifacts/meco-onnx
python scripts/verify_meco_onnx_parity.py --checkpoint-root checkpoints/meco-five --onnx-root artifacts/meco-onnx --output evaluations/meco-deployment/onnx-parity.json --seed 20261024 --provider CPUExecutionProvider --atol 1e-5 --rtol 1e-4
python scripts/profile_meco_deployment.py --checkpoint-root checkpoints/meco-five --output evaluations/meco-deployment/profile.json
```

Numerical parity is a separate, fail-closed gate. It always requires A1, A2,
A3, C1, and C2; a partial set cannot pass. For each fixed-shape adapter it
runs identical deterministic seeded inputs through the checkpoint-backed
PyTorch model and ONNX Runtime, checks input/output names, exact graph metadata
dimensions and tensor types against the adapter contract, rejects symbolic,
unknown, or mismatched dimensions, rejects nonfinite values, and compares
every output tensor with the recorded absolute and relative tolerances. The
JSON evidence binds each comparison to the
checkpoint SHA-256, checkpoint source snapshot and its digest, run/update,
protocol ID, ONNX SHA-256, verifier/source identity, input seed and input-byte
digest, requested/active provider, and tolerances. Missing ONNX Runtime or an
unavailable requested provider yields `unavailable`; missing or invalid
artifacts, execution errors, nonfinite values, contract mismatches, or any
out-of-tolerance tensor yield `failed`. The command exits nonzero for either
status. Extreme finite error summaries that cannot be represented safely in the
JSON numeric format are recorded as null with an overflow marker and fail the
track. Only all five complete track comparisons can produce `pass`.

Parity checks tensor numerics for synthetic scorer inputs only. They do not
validate simulator behavior, route construction, action arbitration, physical
actuation, Hailo compilation, or robot hardware. A parity pass is not a
deployment qualification or promotion decision.

Export requires PyTorch's ONNX exporter and the `onnx` package for graph
inspection. Profiling benchmarks CPU and each visible ROCm/CUDA device. It
reports model-only and synthetic raw-scorer-plus-argmax p50/p95/p99 latency,
process peak RSS, and accelerator peak allocated memory. The scorer-plus-argmax
interval uses one model forward pass and applies argmax to that pass's output.
The argmax diagnostic
does not apply candidate feasibility, route/risk cost, or production action
arbitration. Its reported age proxy repeats the same scorer-plus-argmax timing;
it is not an independent measurement and is not real inference age or
sensor-to-actuator age. Inputs are preloaded on device, so transfer time is
excluded.

Inputs use fixed shapes: 64×64 six-channel raster, seven globals, 256 graph
nodes at degree eight, 64 regions, and 16 candidate rows. This is a bounded
export/profile contract, not a claim that all field-sized production scenes
fit these dimensions. Dynamic route construction, AD* search, simulator feature
assembly, valid-candidate filtering, production action execution, and physical
actuation are not measured. These synthetic scorer timings are explicitly
non-qualification and must not be used as deployment evidence.

For C1 and C2, each candidate row has 20 named values (14 safe-action columns
and six route/commitment columns), plus 20 missingness indicators. The shared
feature contract supplies both the trainer model constructor and profiler input
shape, so a checkpoint with a different layer shape fails strict state loading.

Checkpoint provenance is accepted only when the adjacent `run.json` agrees on
schema version, run ID, track configuration, and the complete source snapshot;
the manifest's source and immutable-configuration digests are recomputed. This
also requires an explicitly present `protocol_id` in both checkpoint and
manifest with exact equality (including `null` only when both explicitly use
`null`). For `complete` and `stopped` runs, manifest `completed_updates` must
equal the checkpoint update. Checkpoints from `failed` runs are rejected and
the recorded failure is reported. A `running` run records whether a matching
heartbeat is present for the same run and track with an update at least as new
as the checkpoint. The heartbeat timestamp is retained, but its age is not
measured; reports explicitly set heartbeat age to unknown and make no liveness
or freshness claim. An ahead heartbeat warns that the checkpoint may lag the
active run. These checks establish consistency between trainer artifacts, not
a signature or proof of artifact authenticity. The inspector intentionally
does not compare historical checkpoint source hashes with its current checkout,
because the deployment inspector itself is among the tracked sources and that
would invalidate older checkpoints whenever inspection code changes.
The profile reports Hailo compiler package and `/dev/hailo*` visibility only;
operator support, compilation, memory fit, and latency on Hailo hardware remain
unverified until the actual toolchain and device are exercised.

No model is declared deployable from parameter count or this synthetic profile.
Retain the generated JSON and ONNX files together with the source revision,
device/driver details, simulator scenario, and seed when extending the work to
end-to-end simulator or robot measurements.

## Production inference request boundary (Phase 5A)

`meco-inference-v1` in `frc_defense.meco_tracks` is the versioned request and
response boundary for checkpoint-backed scorers. `meco_track_training`
dispatches A1–A3 through the adapter in `_prepare_plan_a_decision`, where the
real full legal pose/behavior support is bound to the request. C1/C2 decisions
flow through `_prepare_decision` into `score_candidates`, which builds the
request from the real route-feasible candidate table. The request binds the
track, checkpoint SHA-256, protocol ID, observation schema, snapshot ID,
simulation timestamp, monotonic snapshot-capture timestamp, dispatch time,
capture-derived expiry, canonical named input order and dimensions, and stable
candidate IDs and feasibility mask. A1 accepts six-channel raster input and returns dense `U/P`
plus five behavior logits; A2/A3 return node-wise `U/P` plus behavior logits;
C1/C2 return one utility and uncertainty value per candidate. All floating
inputs and required outputs must be finite, graph/candidate dimensions and
masks must align, and C uncertainty/A penalty must be nonnegative.

The adapter rejects mismatched identity, schema, input order/shape/dtype,
expired observations, malformed/nonfinite outputs, and ordinary scorer
exceptions. On failure it uses zero neural utility while retaining the
existing feasibility mask and route/risk costs; a designated valid candidate
is used only when route/risk costs are absent. Plan A also retains its
deterministic route and learned-P path-cost calculation, with a zero P output
on inference failure. No model result can modify feasibility.
`observation_timestamp` is simulation match elapsed time in seconds from
`ObservationSnapshot.timestamp`. `capture_monotonic` is captured by
`ObservationSnapshot` construction using `time.monotonic()` after its
immutable raster/global payload is packed. `dispatch_monotonic` and
`completed_monotonic` use that same host monotonic clock. Expiry is
`capture_monotonic + 0.25 s`; dispatch and completion are rejected when they
exceed that age. Simulation time is never subtracted from host monotonic time.
The monotonic freshness marker is carried explicitly in requests/responses and
is excluded from deterministic content digests.

`load_production_adapter` requires an explicit available `cuda:N` device and
moves the checkpoint model to that device. PyTorch uses the same spelling for
CUDA and ROCm/HIP devices. Missing accelerators and CPU fallback are rejected.
The `ProductionMecoAdapter` constructor also rejects CPU-resident or mixed-
device models, and inference rejects CPU tensors or tensors on another device.
Existing training calls still use their direct model path and do not establish
production qualification.

This is an implemented interface and static validation boundary, not evidence
of ONNX parity, Hailo operator support, accelerator execution, full planner
latency, simulator fallback behavior, or robot actuation. The synthetic export
and profile harness remains diagnostic and its zero tensors are never passed
into this production request adapter.
