"""Fixed-shape export and deployment profiling helpers for MECO models.

The harness measures model execution and the tensor-side decision pipeline. It
does not claim simulator, Hailo compiler, or robot-hardware qualification.
"""
from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import platform
import resource
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from .meco_candidates import CANDIDATE_BASE_FEATURE_COUNT, CANDIDATE_MODEL_FEATURE_COUNT
from .meco_tracks import (HIDDEN, MECO_CANDIDATE_RAW_FEATURE_COUNT,
    MECO_TRACKS, ProductionMecoAdapter, make_meco_track)
from .meco_input_provenance import input_contract_digest

BATCH = 1
NODES = 256
DEGREE = 8
REGIONS = 64
CANDIDATES = 16
RASTER_H = RASTER_W = 64
GLOBAL_FEATURES = 7
CANDIDATE_FEATURES = CANDIDATE_MODEL_FEATURE_COUNT


def _parameter_dimensions(track: str) -> dict:
    if track == "A1":
        return {"raster_channels": 6, "raster_height": RASTER_H,
            "raster_width": RASTER_W, "global_features": GLOBAL_FEATURES,
            "hidden_features": HIDDEN, "heading_bins": 8,
            "behavior_outputs": 5}
    if track in ("A2", "A3"):
        return {"nodes": NODES, "node_features": 9, "edge_degree": DEGREE,
            "regions": REGIONS, "global_features": GLOBAL_FEATURES,
            "hidden_features": HIDDEN, "heading_bins": 8,
            "behavior_outputs": 5}
    if track in ("C1", "C2"):
        return {"raster_channels": 6 if track == "C2" else None,
            "raster_height": RASTER_H if track == "C2" else None,
            "raster_width": RASTER_W if track == "C2" else None,
            "global_features": GLOBAL_FEATURES, "candidates": CANDIDATES,
            "candidate_raw_features": MECO_CANDIDATE_RAW_FEATURE_COUNT,
            "candidate_model_features": CANDIDATE_FEATURES,
            "hidden_features": HIDDEN}
    raise ValueError(f"unsupported MECO track: {track}")


def _output_contract(track: str) -> list[dict]:
    if track == "A1":
        outputs = (("utility", (BATCH, 8, RASTER_H, RASTER_W)),
            ("penalty", (BATCH, 8, RASTER_H, RASTER_W)),
            ("behavior_logits", (BATCH, 5)), ("hidden", (BATCH, HIDDEN)))
    elif track in ("A2", "A3"):
        outputs = (("utility", (BATCH, NODES)),
            ("penalty", (BATCH, NODES)),
            ("behavior_logits", (BATCH, 5)), ("hidden", (BATCH, HIDDEN)))
    else:
        outputs = (("candidate_utility", (BATCH, CANDIDATES)),
            ("uncertainty", (BATCH, CANDIDATES)),
            ("hidden", (BATCH, HIDDEN)))
    return [{"name": name, "shape": list(shape), "dtype": "torch.float32"}
            for name, shape in outputs]


def parameter_inventory() -> dict:
    """Inventory all five model constructors on CPU without loading checkpoints."""
    root = Path(__file__).resolve().parents[1]
    source_paths = ("frc_defense/meco_deployment.py",
        "frc_defense/meco_tracks.py", "frc_defense/meco_candidates.py")
    source_files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in source_paths}
    source_digest = hashlib.sha256(json.dumps(source_files, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    models = {}
    with torch.random.fork_rng(devices=[]):
        for track in MECO_TRACKS:
            constructor_defaults = {
                "A1": {}, "A2": {"region_tokens": REGIONS, "heads": 4},
                "A3": {"region_tokens": REGIONS, "heads": 4},
                "C1": {"width": HIDDEN},
                "C2": {"tokens": REGIONS, "heads": 4},
            }[track]
            dimensions = _parameter_dimensions(track)
            input_tensors = dummy_inputs(track, device="cpu")
            input_contract = [{"name": name, "shape": list(tensor.shape),
                "dtype": str(tensor.dtype)} for name, tensor in
                zip(input_names(track), input_tensors)]
            output_contract = _output_contract(track)
            config = {"factory": "frc_defense.meco_tracks.make_meco_track",
                "track": track,
                "candidate_features_argument": CANDIDATE_BASE_FEATURE_COUNT,
                "resolved_constructor_defaults": constructor_defaults,
                "dimensions": dimensions, "input_contract": input_contract,
                "output_contract": output_contract, "device": "cpu"}
            model = make_meco_track(track,
                candidate_features=CANDIDATE_BASE_FEATURE_COUNT).cpu()
            parameters = list(model.named_parameters())
            total = sum(parameter.numel() for _, parameter in parameters)
            trainable = sum(parameter.numel() for _, parameter in parameters
                            if parameter.requires_grad)
            by_dtype = defaultdict(lambda: {"tensor_count": 0, "parameters": 0,
                                            "trainable_parameters": 0})
            by_shape = defaultdict(lambda: {"tensor_count": 0, "parameters": 0,
                                            "trainable_parameters": 0})
            for _, parameter in parameters:
                dtype = str(parameter.dtype)
                shape = tuple(parameter.shape)
                by_dtype[dtype]["tensor_count"] += 1
                by_dtype[dtype]["parameters"] += parameter.numel()
                by_dtype[dtype]["trainable_parameters"] += (
                    parameter.numel() if parameter.requires_grad else 0)
                shape_key = (dtype, shape)
                by_shape[shape_key]["tensor_count"] += 1
                by_shape[shape_key]["parameters"] += parameter.numel()
                by_shape[shape_key]["trainable_parameters"] += (
                    parameter.numel() if parameter.requires_grad else 0)
            module_counts = []
            for module_name, module in model.named_modules():
                direct = list(module.parameters(recurse=False))
                if not direct:
                    continue
                module_counts.append({"module": module_name or "<root>",
                    "module_type": type(module).__name__,
                    "parameters": sum(parameter.numel() for parameter in direct),
                    "trainable_parameters": sum(parameter.numel() for parameter in direct
                                                  if parameter.requires_grad)})
            config_digest = hashlib.sha256(json.dumps(config, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
            models[track] = {"model_class": type(model).__name__,
                "parameter_count": total, "trainable_parameter_count": trainable,
                "dtype_totals": [{"dtype": dtype, **counts}
                    for dtype, counts in sorted(by_dtype.items())],
                "dtype_shape_totals": [{"dtype": dtype, "shape": list(shape), **counts}
                    for (dtype, shape), counts in sorted(by_shape.items())],
                "per_module_counts": module_counts,
                "configuration": config,
                "configuration_sha256": config_digest,
                "dimensions": dimensions,
                "input_contract": input_contract,
                "output_contract": output_contract}
            del model
    return {"schema": "meco-parameter-inventory-v1",
        "basis": "fresh CPU model constructors; no checkpoint weights loaded",
        "tracks": list(MECO_TRACKS), "source_files": source_files,
        "source_files_sha256": source_digest, "models": models,
        "limitations": ["Parameter counts and tensor interface metadata do not prove checkpoint/export parity, runtime latency, Hailo compiler support, or robot deployment."]}


def export_manifest(reports: list[dict]) -> dict:
    return {"schema": "meco-onnx-export-v1",
        "input_contract_sha256": input_contract_digest(), "exports": reports,
        "parameter_inventory": parameter_inventory()}


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
    from .meco_input_provenance import validate_registry
    data = validate_registry()
    names = tuple(data["serialized_inputs_by_track"].get(track, ()))
    if not names:
        raise ValueError(f"unsupported MECO track: {track}")
    return names


def dummy_inputs(track: str, device="cpu") -> tuple[torch.Tensor, ...]:
    f = lambda *shape: torch.zeros(shape, dtype=torch.float32, device=device)
    li = lambda *shape: torch.zeros(shape, dtype=torch.long, device=device)
    bi = lambda *shape: torch.ones(shape, dtype=torch.bool, device=device)
    raster, globals_, hidden = f(BATCH, 6, RASTER_H, RASTER_W), f(BATCH, GLOBAL_FEATURES), f(BATCH, 64)
    candidate = f(BATCH, CANDIDATES, CANDIDATE_FEATURES)
    by_name = {"raster": raster, "globals_": globals_, "hidden": hidden,
        "candidate_features": candidate}
    if track in ("A2", "A3"):
        nodes = f(BATCH, NODES, 9)
        # Normalized x/y and heading bin are valid for the production encoders.
        nodes[:, :, 6:8] = .5
        edges = li(BATCH, NODES, DEGREE)
        edge_mask = bi(BATCH, NODES, DEGREE)
        regions = li(BATCH, NODES).remainder(REGIONS)
        by_name.update(node_features=nodes, edges=edges, edge_mask=edge_mask,
            region_ids=regions)
        if track == "A2":
            coarse_edges = li(BATCH, REGIONS, DEGREE)
            coarse_mask = bi(BATCH, REGIONS, DEGREE)
            by_name.update(coarse_edges=coarse_edges, coarse_edge_mask=coarse_mask)
    names = input_names(track)
    return tuple(by_name[name] for name in names)


def validate_export_input_order(track: str, names, tensors) -> tuple[str, ...]:
    """Check export names and tensors against the canonical validated registry."""
    from .meco_input_provenance import validate_serialized_inputs
    canonical = input_names(track)
    validate_serialized_inputs(track, dict.fromkeys(names))
    if tuple(names) != canonical or len(tensors) != len(canonical):
        raise ValueError("ONNX export inputs differ from canonical registry order")
    return canonical


def _running_heartbeat_status(checkpoint: Path, run_id: str, track: str,
                              update: int) -> dict:
    """Report whether a running checkpoint has a matching progress heartbeat."""
    heartbeat_path = checkpoint.parent / "heartbeat.json"
    try:
        heartbeat = json.loads(heartbeat_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unknown", "reason": f"heartbeat unavailable: {exc}",
                "path": str(heartbeat_path)}
    if (not isinstance(heartbeat, dict) or heartbeat.get("schema_version") != 1
            or heartbeat.get("run_id") != run_id or heartbeat.get("track") != track
            or type(heartbeat.get("completed_updates")) is not int
            or heartbeat["completed_updates"] < update
            or not isinstance(heartbeat.get("heartbeat_utc"), str)
            or not heartbeat["heartbeat_utc"].strip()):
        return {"status": "unknown",
                "reason": "heartbeat does not match checkpoint run, track, or update",
                "path": str(heartbeat_path),
                "heartbeat_age_seconds": None,
                "heartbeat_age_status": "unknown_not_measured"}
    return {"status": "heartbeat_present_and_matching", "path": str(heartbeat_path),
            "heartbeat_utc": heartbeat["heartbeat_utc"],
            "heartbeat_age_seconds": None,
            "heartbeat_age_status": "unknown_not_measured",
            "heartbeat_completed_updates": heartbeat["completed_updates"],
            "checkpoint_update": update,
            "checkpoint_matches_heartbeat_update": heartbeat["completed_updates"] == update,
            "checkpoint_may_be_behind_running_update": heartbeat["completed_updates"] > update}


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
    status = manifest.get("status")
    if status == "failed":
        failure = manifest.get("error")
        detail = f": {failure}" if isinstance(failure, str) and failure else " without an error detail"
        raise ValueError(f"refusing checkpoint from failed training run {run_id}{detail}")
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
    checkpoint_protocol = payload.get("protocol_id")
    manifest_protocol = manifest.get("protocol_id")
    if (manifest.get("schema_version") != 1 or manifest.get("run_id") != run_id
            or manifest.get("sources") != sources or manifest_config != config
            or "protocol_id" not in payload or "protocol_id" not in manifest
            or checkpoint_protocol != manifest_protocol
            or manifest.get("source_identity_sha256") != source_digest
            or manifest.get("configuration_sha256") != config_digest
            or status not in ("running", "complete", "stopped")
            or (completed_updates is not None and
                (type(completed_updates) is not int or completed_updates < update))
            or (status in ("complete", "stopped") and
                (type(completed_updates) is not int or completed_updates != update))):
        raise ValueError(f"checkpoint provenance does not match its run manifest: {checkpoint}")
    heartbeat_status = (_running_heartbeat_status(checkpoint, run_id, track, update)
                        if status == "running" else None)
    model = make_meco_track(track, candidate_features=CANDIDATE_BASE_FEATURE_COUNT)
    model.load_state_dict(payload["model"], strict=True)
    provenance = {"checkpoint": str(checkpoint), "sha256": digest,
                  "input_contract_sha256": payload.get("input_contract_sha256"),
                  "track": track, "update": update,
                  "run_id": run_id, "sources": sources,
                  "config": config,
                  "protocol_id": checkpoint_protocol,
                  "training_status": status,
                  "manifest_completed_updates": completed_updates,
                  "running_checkpoint_heartbeat": heartbeat_status,
                  "run_manifest": str(manifest_path),
                  "source_trust": "checkpoint source snapshot matches adjacent trainer run manifest; current checkout is not compared"}
    return ExportAdapter(track, model).eval(), provenance


def load_production_adapter(track: str, checkpoint: Path, *, device: str) -> tuple[ProductionMecoAdapter, dict]:
    """Load a checkpoint on an explicitly available CUDA/HIP device.

    Production loading never silently falls back to CPU. PyTorch exposes both
    CUDA and ROCm devices through its ``cuda:N`` device spelling.
    """
    if (not isinstance(device, str) or not device.startswith("cuda:")
            or not torch.cuda.is_available()):
        raise RuntimeError("production MECO inference requires an explicit available CUDA/HIP device")
    try:
        selected = torch.device(device)
        index = selected.index
        count = torch.cuda.device_count()
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"invalid production CUDA/HIP device {device!r}") from exc
    if selected.type != "cuda" or index is None or index < 0 or index >= count:
        raise RuntimeError(f"production CUDA/HIP device is unavailable: {device}")
    export_adapter, provenance = load_track_model(track, checkpoint)
    model = export_adapter.model.to(selected).eval()
    if any(parameter.device != selected for parameter in model.parameters()):
        raise RuntimeError("production MECO checkpoint did not load on the selected accelerator")
    return ProductionMecoAdapter(track, export_adapter.model,
        checkpoint_sha256=provenance["sha256"],
        protocol_id=provenance["protocol_id"],
        source_snapshot_sha256=hashlib.sha256(json.dumps(
            provenance["sources"], sort_keys=True, allow_nan=False,
            separators=(",", ":")).encode()).hexdigest()), provenance


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


def export_track(track: str, checkpoint: Path, output: Path,
                 input_tensors: tuple[torch.Tensor, ...] | None = None,
                 capture_provenance: dict | None = None) -> dict:
    model, provenance = load_track_model(track, checkpoint)
    names = input_names(track)
    if capture_provenance is not None and input_tensors is None:
        raise ValueError("production capture provenance requires its captured input tensors")
    capture_binding = None
    if capture_provenance is not None:
        from .meco_request_capture import validate_capture_inputs
        validate_capture_inputs(capture_provenance, input_tensors,
                               expected_track=track)
        source_digest = hashlib.sha256(json.dumps(
            provenance["sources"], sort_keys=True, allow_nan=False,
            separators=(",", ":")).encode()).hexdigest()
        if (capture_provenance.get("track") != track
                or capture_provenance.get("checkpoint_sha256") != provenance["sha256"]
                or capture_provenance.get("protocol_id") != provenance["protocol_id"]
                or capture_provenance.get("source_snapshot_sha256") != source_digest):
            raise ValueError("production capture checkpoint/protocol/source provenance mismatch")
        capture_binding = {key: capture_provenance.get(key) for key in (
            "track", "checkpoint_sha256", "protocol_id", "source_snapshot_sha256",
            "input_contract_sha256",
            "content_sha256", "metadata_sha256", "request_sequence",
            "generation_id", "snapshot_id", "candidate_mask_digest",
            "legal_pose_order_digest")}
    tensors = dummy_inputs(track) if input_tensors is None else tuple(input_tensors)
    validate_export_input_order(track, names, tensors)
    if any(not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
           or not tensor.is_contiguous() or not torch.isfinite(tensor).all()
           for tensor in tensors):
        raise ValueError("ONNX export inputs must be finite, contiguous CPU tensors")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, tensors, str(output), input_names=names,
        output_names=("utility", "penalty", "behavior_logits", "hidden") if track.startswith("A")
                      else ("candidate_utility", "uncertainty", "hidden"),
        opset_version=17, do_constant_folding=True, dynamo=False)
    return {"track": track, "checkpoint_provenance": provenance,
            "path": str(output), "size_bytes": output.stat().st_size,
            "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "input_mode": "production_capture" if capture_provenance else "synthetic",
            "capture_binding": capture_binding,
            "capture_provenance": capture_provenance,
            "inputs": [{"name": name, "shape": list(t.shape), "dtype": str(t.dtype)}
                       for name, t in zip(names, tensors)],
            "graph": operator_inventory(output)}


def _sync(device):
    if device.type == "cuda": torch.cuda.synchronize(device)


def _pipeline(outputs, track):
    """Synthetic diagnostic over an existing scorer result, not a policy action."""
    if isinstance(outputs, dict):
        output_name = "utility" if track.startswith("A") else "candidate_utility"
        utility = outputs[output_name]
    else:
        utility = outputs[0]
    return torch.argmax(utility.flatten(start_dim=1), dim=-1)


def benchmark(track: str, checkpoint: Path, device_name: str, iterations=200, warmup=30) -> dict:
    device = torch.device(device_name)
    model, provenance = load_track_model(track, checkpoint)
    model = model.to(device)
    host_inputs = dummy_inputs(track, "cpu")
    inputs = tuple(x.to(device) for x in host_inputs)
    with torch.inference_mode():
        for _ in range(warmup):
            warm_outputs = model(*inputs)
            _pipeline(warm_outputs, track)
        _sync(device)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        model_ms, pipeline_ms, age_ms = [], [], []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            out = model(*inputs)
            _sync(device)
            middle = time.perf_counter_ns()
            _pipeline(out, track)
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
        "parameter_inventory": parameter_inventory(),
        "evidence_class": "synthetic trained-scorer microbenchmark; not deployment qualification",
        "devices_tested": available, "hailo": hailo, "results": results,
        "limitations": ["Synthetic fixed-shape inputs; no simulator feature construction or AD* route search.",
            "Argmax diagnostic ranks raw scorer outputs; it uses no valid-candidate mask, route/risk cost, or production action arbitration.",
            "Inference-age field repeats host scorer-plus-argmax timing; it is not an independent age measurement or sensor-to-actuator latency.",
            "Host-to-device transfer is excluded and no physical actuation is measured.",
            "ONNX export/operator inventory do not establish Hailo compiler support or robot deployability."]}
