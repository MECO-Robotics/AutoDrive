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

The default export and parity commands use synthetic inputs. To capture bounded
real requests at the production adapter boundary, set both
`MECO_REQUEST_CAPTURE_DIR` and `MECO_REQUEST_CAPTURE_MAX` in the process that
runs checkpoint-backed production inference. The maximum must be between 1 and
100 requests per process; capture copies tensors to host and should be enabled
only for a deliberate diagnostic run. Each request is limited to 64 MiB of
tensor payload. Each atomic `.npz` contains canonical
input names/order, dtype/shape and per-tensor plus aggregate content digests,
the validated model-input contract digest, checkpoint/protocol/source snapshot,
snapshot and request/generation identity,
candidate IDs/mask and digest, legal-pose-order digest where applicable, and
capture/dispatch/expiry provenance. `load_capture` rejects malformed metadata,
noncanonical order, digest mismatch, and incomplete request provenance.

With exactly one capture for each requested track, export graph shapes from
the captured tensors and run parity on those exact captured values:

```bash
python scripts/export_meco_onnx.py --checkpoint-root checkpoints/meco-five --output artifacts/meco-onnx-captured --production-capture-dir evaluations/meco-production-captures
python scripts/verify_meco_onnx_parity.py --checkpoint-root checkpoints/meco-five --onnx-root artifacts/meco-onnx-captured --output evaluations/meco-deployment/onnx-captured-parity.json --production-capture-dir evaluations/meco-production-captures --provider CPUExecutionProvider --atol 1e-5 --rtol 1e-4
```

Export rejects a capture whose checkpoint, protocol, or source snapshot does
not match the selected checkpoint. The export manifest pins the capture's
aggregate tensor-content and metadata digests plus request provenance and the
ONNX graph digest. The verifier requires those bindings to match the current
capture byte content and export manifest, so a replaced or edited capture is
rejected. It also checks capture checkpoint, protocol, and source-snapshot
identity against the checkpoint being compared. Its report labels the input
mode as `production_capture`; the existing seeded mode remains
`synthetic_seeded`. Capture parity checks scorer tensor numerics on one
recorded request per track. It does not establish request freshness under
load, simulator behavior, route construction correctness, action arbitration,
physical actuation, Hailo compilation, or robot hardware. Neither parity mode
is a deployment qualification or promotion decision.

Export requires PyTorch's ONNX exporter and the `onnx` package for graph
inspection. With no `--production-capture-dir`, profiling retains the existing
synthetic diagnostic: it benchmarks CPU and each visible ROCm/CUDA device,
uses fixed synthetic inputs, and labels the repeated timing-as-age field as a
synthetic diagnostic. It does not infer capture provenance or deployment
evidence from the checkpoint alone.

For a capture-backed scorer-only measurement, first produce one fresh,
checkpoint-matched production request capture for each of A1, A2, A3, C1, and
C2 by running the actual production adapter with both bounded capture
environment variables enabled. The selected `TRACK/latest.pt`, adjacent
`run.json`, capture checkpoint SHA-256, protocol, source snapshot digest, and
current input contract must agree. Use a clean profiler checkout and the
exclusive device lease must be available for both pinned WX9100 cards. Invoke:

```bash
python scripts/profile_meco_deployment.py \
  --checkpoint-root checkpoints/meco-five \
  --production-capture-dir evaluations/meco-production-captures \
  --device cuda:0 --device cuda:1 \
  --output evaluations/meco-deployment/production-capture-profile.json
```

This mode requires exactly one canonical `<TRACK>-*.npz` for each of the five
tracks, and exactly the explicit device pair `cuda:0`, `cuda:1`. It acquires an
exclusive lease and attests both logical devices against the pinned host GPU
inventory, including BDF, PCI ID, HIP UUID, and the P600 exclusion. Missing or
changed captures/checkpoints, source or protocol mismatches, a dirty profiler
checkout, unavailable tools/devices, or either physical-identity mismatch fail
the run. It never falls back to synthetic inputs, CPU, one GPU, or another
visible accelerator. Every capture is validated before timing and rechecked
along with its checkpoint immediately before and after each device run.

The immutable JSON report uses a distinct
`meco-production-capture-scorer-profile-v1` schema and
`production-capture-backed scorer microbenchmark; not deployment qualification`
evidence class. It records the all-five validation result, capture file and
content/metadata/contract digests, request/generation/snapshot identity,
consumed input names/shapes/dtypes/content digest, checkpoint hash/run/update/
protocol/training-source snapshot, profiler Git commit plus clean-tree
assertion, exact physical device identity and inventory hash, runtime versions,
and warmup/iteration counts. Scorer-forward and scorer-plus-argmax latency
distributions and peak allocated memory are reported separately for each card.
Captured tensors are copied to each card before warmup; host-to-device transfer
is excluded and labelled. The profile does not report inference age or
end-to-end latency. Existing reports are never overwritten; the command prints
the successful report SHA-256 for the evidence ledger.

Both profile modes measure scorer work only. The argmax diagnostic does not
apply candidate feasibility, route/risk cost, or production action arbitration.
Neither mode measures request freshness, host-to-device transfer, route
construction, AD* search, simulator feature assembly, action execution, or
physical actuation. Neither mode establishes Hailo compilation, target-device
memory fit, deployment qualification, or promotion readiness. Those gates
remain false until independently measured on the actual target and through the
required end-to-end evaluation.

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

`meco-inference-v2` in `frc_defense.meco_tracks` is the versioned request and
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

The v2 request also binds `model_id`, a per-adapter monotonic request sequence,
and an adapter generation ID. C1/C2 bind the ordered candidate IDs and mask
digest. A1/A2/A3 bind the ordered legal node/pose/behavior support digest,
including descriptor order. The response echoes these identities and the
expiry time. Consumers reject any identity, digest, sequence, or timestamp
mismatch, including a result superseded by a newer submitted request.

Production inference runs through a per-adapter asynchronous broker with one
worker and one replaceable pending slot. Inputs are cloned before enqueueing;
an older pending request is discarded when a newer request arrives. Expired
requests are rejected before execution and responses are published only after
all outputs pass shape, device, finiteness, and freshness checks. The action
boundary polls once without waiting before committing the action; if no
complete matching result is already available, the deterministic
route/risk/feasibility fallback is used for that snapshot. It never carries a
late result into a later snapshot. Each decision carries
request/checkpoint/protocol/snapshot provenance and the broker's dropped, late,
and fault counters. The worker runs the model forward under
`torch.inference_mode()`. Plan A's P path-cost semantics
remain active, with zero P on inference failure.

`evaluate_meco` owns every adapter it constructs through a scoped lifecycle.
Its `finally` cleanup closes adapters on success, evaluation errors, and
partial checkpoint loading. Closing is idempotent, rejects later submissions,
discards queued work, sends a worker stop sentinel, and joins the worker before
the evaluator releases checkpoint/model references.

`load_production_adapter` requires an explicit available `cuda:N` device and
moves the checkpoint model to that device. PyTorch uses the same spelling for
CUDA and ROCm/HIP devices. Missing accelerators and CPU fallback are rejected.
The `ProductionMecoAdapter` constructor also rejects CPU-resident or mixed-
device models, and inference rejects CPU tensors or tensors on another device.
Existing training calls still use their direct model path and do not establish
production qualification.

Focused deterministic CPU tests now exercise the broker state machine with a
fake adapter: latest-pending replacement, identity rejection, one-time
publication, poll-before-completion non-reuse, expired/empty/raised inference
faults, and idempotent close/join/reject-after-close. A raised adapter
exception is counted as a per-request fault and does not terminate the worker.
These tests do not construct the production CUDA-only adapter or demonstrate
GPU-load freshness, evaluator lifecycle under real checkpoint loading,
simulator fallback behavior, or replay correctness. ONNX parity, Hailo operator
support, accelerator execution, full planner latency, and robot actuation also
remain unvalidated. The synthetic export/profile harness remains diagnostic;
its zero tensors are never passed into this production request adapter.
