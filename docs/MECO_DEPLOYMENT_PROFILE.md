# MECO deployment export and profiling

The Phase 5 harness exports fixed-shape A1, A2, A3, C1, and C2 scorer graphs
from explicit trained checkpoints, checks each ONNX graph, and records its
operator inventory. Each artifact/report records checkpoint path, SHA-256,
track, update, run ID, training sources, and configuration. It refuses missing
or track-mismatched checkpoints. Run from the repository root:

```bash
python scripts/export_meco_onnx.py --checkpoint-root checkpoints/meco-five --output artifacts/meco-onnx
python scripts/profile_meco_deployment.py --checkpoint-root checkpoints/meco-five --output evaluations/meco-deployment/profile.json
```

Export requires PyTorch's ONNX exporter and the `onnx` package for graph
inspection. Profiling benchmarks CPU and each visible ROCm/CUDA device. It
reports model-only and synthetic raw-scorer-plus-argmax p50/p95/p99 latency,
process peak RSS, and accelerator peak allocated memory. The argmax diagnostic
does not apply candidate feasibility, route/risk cost, or production action
arbitration. Its reported age proxy repeats scorer-plus-argmax timing; it is
not independent and is not sensor-to-actuator age. Inputs are preloaded on
device, so transfer time is excluded.

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
is a consistency check between trainer artifacts, not a signature or proof of
artifact authenticity. The inspector intentionally does not compare historical
checkpoint source hashes with its current checkout, because the deployment
inspector itself is among the tracked sources and that would invalidate older
checkpoints whenever inspection code changes.
The profile reports Hailo compiler package and `/dev/hailo*` visibility only;
operator support, compilation, memory fit, and latency on Hailo hardware remain
unverified until the actual toolchain and device are exercised.

No model is declared deployable from parameter count or this synthetic profile.
Retain the generated JSON and ONNX files together with the source revision,
device/driver details, simulator scenario, and seed when extending the work to
end-to-end simulator or robot measurements.
