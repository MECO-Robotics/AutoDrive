"""Fixed-shape export and deployment profiling helpers for MECO models.

The harness measures model execution and the tensor-side decision pipeline. It
does not claim simulator, Hailo compiler, or robot-hardware qualification.
"""
from __future__ import annotations

import importlib.util
import hashlib
import io
import platform
import resource
import time
from pathlib import Path

import torch
from torch import nn

from .meco_tracks import (MECO_TRACKS, MECO_CANDIDATE_RAW_FEATURE_COUNT,
    MECO_CANDIDATE_MODEL_INPUT_WIDTH,
                          make_meco_track)

BATCH = 1
NODES = 256
DEGREE = 8
REGIONS = 64
CANDIDATES = 16
RASTER_H = RASTER_W = 64
GLOBAL_FEATURES = 7
CANDIDATE_FEATURES = MECO_CANDIDATE_MODEL_INPUT_WIDTH


class ExportAdapter(nn.Module):
    """Tensor-only wrapper with stable named outputs and explicit input order."""
    def __init__(self, track: str, model: nn.Module):
        super().__init__()
        self.track = track
        self.model = model

    def forward(self, *inputs):
        result = self.model(**dict(zip(input_names(self.track), inputs)))
        if "utility" in result:
            return result["utility"], result["penalty"], result["behavior_logits"], result["hidden"]
        return result["candidate_utility"], result["uncertainty"], result["hidden"]


def input_names(track: str) -> tuple[str, ...]:
    if track == "A1": return ("raster", "globals_", "hidden")
    if track in ("A2", "A3"):
        base = ("node_features", "edges", "edge_mask", "region_ids", "globals_")
        return base + (("coarse_edges", "coarse_edge_mask") if track == "A2" else ()) + ("hidden",)
    if track == "C1": return ("globals_", "candidate_features", "hidden")
    if track == "C2": return ("raster", "globals_", "candidate_features", "hidden")
    raise ValueError(f"unsupported MECO track: {track}")


def dummy_inputs(track: str, device="cpu") -> tuple[torch.Tensor, ...]:
    f = lambda *shape: torch.zeros(shape, dtype=torch.float32, device=device)
    li = lambda *shape: torch.zeros(shape, dtype=torch.long, device=device)
    bi = lambda *shape: torch.ones(shape, dtype=torch.bool, device=device)
    raster, globals_, hidden = f(BATCH, 6, RASTER_H, RASTER_W), f(BATCH, GLOBAL_FEATURES), f(BATCH, 64)
    candidate = f(BATCH, CANDIDATES, CANDIDATE_FEATURES)
    if track == "A1": return raster, globals_, hidden
    if track in ("A2", "A3"):
        nodes = f(BATCH, NODES, 9)
        # Normalized x/y and heading bin are valid for the production encoders.
        nodes[:, :, 6:8] = .5
        edges = li(BATCH, NODES, DEGREE)
        edge_mask = bi(BATCH, NODES, DEGREE)
        regions = li(BATCH, NODES).remainder(REGIONS)
        if track == "A2":
            coarse_edges = li(BATCH, REGIONS, DEGREE)
            coarse_mask = bi(BATCH, REGIONS, DEGREE)
            return nodes, edges, edge_mask, regions, globals_, coarse_edges, coarse_mask, hidden
        return nodes, edges, edge_mask, regions, globals_, hidden
    if track == "C1": return globals_, candidate, hidden
    if track == "C2": return raster, globals_, candidate, hidden
    raise ValueError(f"unsupported MECO track: {track}")


def load_track_model(track: str, checkpoint: Path) -> tuple[nn.Module, dict]:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"trained checkpoint required for {track}: {checkpoint}")
    checkpoint_bytes = checkpoint.read_bytes()
    if not checkpoint_bytes:
        raise ValueError(f"checkpoint is empty: {checkpoint}")
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    sources = payload.get("sources")
    source_files = ({key: value for key, value in sources.items()
                     if isinstance(key, str) and not key.startswith("_")}
                    if isinstance(sources, dict) else {})
    source_hashes_valid = bool(source_files) and all(
        isinstance(value, str) and len(value) == 64 and
        all(char in "0123456789abcdef" for char in value.lower())
        for value in source_files.values())
    run_id, update = payload.get("run_id"), payload.get("update")
    if (payload.get("schema_version") != 1 or not isinstance(config, dict)
            or config.get("track") != track or not isinstance(payload.get("model"), dict)
            or not payload["model"] or not source_hashes_valid
            or not isinstance(run_id, str) or not run_id.strip()
            or type(update) is not int or update < 1):
        raise ValueError(f"checkpoint does not contain a {track} MECO model: {checkpoint}")
    # Bind checkpoint metadata to the trainer's adjacent run record. This is
    # consistency checking, not a signature; current sources are intentionally
    # not compared because the inspector itself is among the tracked sources.
    manifest_path = checkpoint.parent / "run.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint run manifest is missing or unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"checkpoint run manifest is invalid: {manifest_path}")
    immutable_config = {key: value for key, value in config.items() if key != "updates"}
    manifest_config = manifest.get("config")
    try:
        source_digest = hashlib.sha256(json.dumps(
            sources, sort_keys=True, allow_nan=False).encode()).hexdigest()
        config_digest = hashlib.sha256(json.dumps(
            immutable_config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint provenance is not canonical JSON: {checkpoint}") from exc
    completed_updates = manifest.get("completed_updates")
    if (manifest.get("schema_version") != 1 or manifest.get("run_id") != run_id
            or manifest.get("sources") != sources or manifest_config != config
            or manifest.get("source_identity_sha256") != source_digest
            or manifest.get("configuration_sha256") != config_digest
            or manifest.get("status") not in ("running", "complete", "stopped", "failed")
            or (completed_updates is not None and
                (type(completed_updates) is not int or completed_updates < update))):
        raise ValueError(f"checkpoint provenance does not match its run manifest: {checkpoint}")
    model = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT)
    model.load_state_dict(payload["model"], strict=True)
    provenance = {"checkpoint": str(checkpoint), "sha256": digest,
                  "track": track, "update": update,
                  "run_id": run_id, "sources": sources,
                  "config": config, "run_manifest": str(manifest_path),
                  "source_trust": "checkpoint source snapshot matches adjacent trainer run manifest; current checkout is not compared"}
    return ExportAdapter(track, model).eval(), provenance


def operator_inventory(onnx_path: Path) -> dict:
    try:
        import onnx
    except ImportError:
        return {"onnx_package": "unavailable", "inspected": False}
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    ops = {}
    for node in model.graph.node:
        key = f"{node.domain or 'ai.onnx'}::{node.op_type}"
        ops[key] = ops.get(key, 0) + 1
    return {"onnx_package": onnx.__version__, "inspected": True,
            "opset_imports": [{"domain": x.domain or "ai.onnx", "version": x.version}
                               for x in model.opset_import],
            "operator_counts": dict(sorted(ops.items())),
            "node_count": len(model.graph.node)}


def export_track(track: str, checkpoint: Path, output: Path) -> dict:
    model, provenance = load_track_model(track, checkpoint)
    tensors = dummy_inputs(track)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, tensors, str(output), input_names=input_names(track),
        output_names=("utility", "penalty", "behavior_logits", "hidden") if track.startswith("A")
                      else ("candidate_utility", "uncertainty", "hidden"),
        opset_version=17, do_constant_folding=True, dynamo=False)
    return {"track": track, "checkpoint_provenance": provenance,
            "path": str(output), "size_bytes": output.stat().st_size,
            "inputs": [{"name": name, "shape": list(t.shape), "dtype": str(t.dtype)}
                       for name, t in zip(input_names(track), tensors)],
            "graph": operator_inventory(output)}


def _sync(device):
    if device.type == "cuda": torch.cuda.synchronize(device)


def _pipeline(model, track, inputs, device):
    """Synthetic diagnostic: scorer output plus raw argmax, not policy action."""
    outputs = model(*inputs)
    utility = outputs[0]
    return torch.argmax(utility.flatten(start_dim=1), dim=-1)


def benchmark(track: str, checkpoint: Path, device_name: str, iterations=200, warmup=30) -> dict:
    device = torch.device(device_name)
    model, provenance = load_track_model(track, checkpoint)
    model = model.to(device)
    host_inputs = dummy_inputs(track, "cpu")
    inputs = tuple(x.to(device) for x in host_inputs)
    with torch.inference_mode():
        for _ in range(warmup): model(*inputs)
        _sync(device)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        model_ms, pipeline_ms, age_ms = [], [], []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            out = model(*inputs)
            _sync(device)
            middle = time.perf_counter_ns()
            _pipeline(model, track, inputs, device)
            _sync(device)
            end = time.perf_counter_ns()
            model_ms.append((middle-start)/1e6)
            pipeline_ms.append((end-start)/1e6)
            # Host timestamp through synchronized result is the measured age
            # proxy; it excludes simulator sensing and physical actuation.
            age_ms.append((end-start)/1e6)
        peak_device = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    def pct(values, p):
        values = sorted(values)
        return values[min(len(values)-1, int((len(values)-1)*p))]
    return {"device": device_name, "iterations": iterations, "warmup": warmup,
        "checkpoint_provenance": provenance,
        "model_only_ms": {f"p{p}": pct(model_ms, p/100) for p in (50,95,99)},
        "synthetic_raw_scorer_argmax_ms": {f"p{p}": pct(pipeline_ms, p/100) for p in (50,95,99)},
        "same_timing_relabelled_as_age_proxy_ms": {f"p{p}": pct(age_ms, p/100) for p in (50,95,99)},
        "peak_device_memory_bytes": peak_device,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
            (1024 if platform.system() != "Darwin" else 1),
        "host_to_device_transfer": "excluded: inputs preloaded on device; measure separately on hardware",
        "deployment_evidence": False,
        "synthetic_diagnostic_only": True,
        "feature_route_feasibility_action_pipeline_measured": False,
        "age_proxy_independent_measurement": False}


def profile_report(tracks, checkpoint_root: Path, iterations=200, warmup=30):
    available = ["cpu"]
    if torch.cuda.is_available():
        available += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    hailo = {"compiler_available": importlib.util.find_spec("hailo_sdk_client") is not None,
             "device_visible": any(Path(p).exists() for p in ("/dev/hailo0", "/dev/hailo1")),
             "validated": False}
    results = []
    for track in tracks:
        checkpoint = checkpoint_root / track / "latest.pt"
        for device in available:
            results.append({"track": track, **benchmark(track, checkpoint, device, iterations, warmup)})
    return {"schema": "meco-synthetic-scorer-profile-v1", "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__, "platform": platform.platform(),
        "evidence_class": "synthetic trained-scorer microbenchmark; not deployment qualification",
        "devices_tested": available, "hailo": hailo, "results": results,
        "limitations": ["Synthetic fixed-shape inputs; no simulator feature construction or AD* route search.",
            "Argmax diagnostic ranks raw scorer outputs; it uses no valid-candidate mask, route/risk cost, or production action arbitration.",
            "Inference-age field repeats host scorer-plus-argmax timing; it is not an independent age measurement or sensor-to-actuator latency.",
            "Host-to-device transfer is excluded and no physical actuation is measured.",
            "ONNX export/operator inventory do not establish Hailo compiler support or robot deployability."]}
