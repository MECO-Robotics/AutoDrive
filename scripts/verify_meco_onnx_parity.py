#!/usr/bin/env python3
"""Fail-closed numerical parity gate for all five MECO ONNX scorer adapters."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

from frc_defense import curriculum_training
from frc_defense.meco_deployment import dummy_inputs, input_names, load_track_model
from frc_defense.meco_tracks import MECO_TRACKS


OUTPUT_NAMES = {
    "A1": ("utility", "penalty", "behavior_logits", "hidden"),
    "A2": ("utility", "penalty", "behavior_logits", "hidden"),
    "A3": ("utility", "penalty", "behavior_logits", "hidden"),
    "C1": ("candidate_utility", "uncertainty", "hidden"),
    "C2": ("candidate_utility", "uncertainty", "hidden"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seeded_inputs(track: str, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + MECO_TRACKS.index(track))
    tensors = []
    for name, base in zip(input_names(track), dummy_inputs(track)):
        value = base.clone()
        if value.is_floating_point():
            value.normal_(mean=0.0, std=0.1, generator=generator)
            # The graph encoder expects normalized node coordinates.
            if name == "node_features":
                value[:, :, 6:8] = torch.rand(
                    value[:, :, 6:8].shape, generator=generator, dtype=value.dtype)
        tensors.append(value.contiguous())
    return tuple(tensors)


def _input_digest(names, tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in zip(names, tensors):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _compare_outputs(track, torch_outputs, onnx_outputs, atol, rtol):
    if len(torch_outputs) != len(OUTPUT_NAMES[track]) or len(onnx_outputs) != len(OUTPUT_NAMES[track]):
        raise ValueError("output count does not match the fixed adapter contract")
    reports = []
    for name, expected_tensor, actual_value in zip(
            OUTPUT_NAMES[track], torch_outputs, onnx_outputs):
        expected = expected_tensor.detach().cpu().numpy()
        actual = np.asarray(actual_value)
        item = {"name": name, "torch_shape": list(expected.shape),
                "onnx_shape": list(actual.shape),
                "torch_finite": bool(np.isfinite(expected).all()),
                "onnx_finite": bool(np.isfinite(actual).all()),
                "max_abs_error": None, "max_rel_error": None, "allclose": False}
        if expected.shape != actual.shape:
            item["error"] = "shape_mismatch"
        elif item["torch_finite"] and item["onnx_finite"]:
            delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
            denominator = np.maximum(np.abs(expected.astype(np.float64)), np.finfo(np.float64).tiny)
            item["max_abs_error"] = float(delta.max(initial=0.0))
            item["max_rel_error"] = float((delta / denominator).max(initial=0.0))
            item["allclose"] = bool(np.allclose(expected, actual, atol=atol, rtol=rtol,
                                                 equal_nan=False))
        else:
            item["error"] = "nonfinite_output"
        reports.append(item)
    return reports


def verify(checkpoint_root: Path, onnx_root: Path, seed: int,
           provider: str, atol: float, rtol: float) -> dict:
    current_sources = curriculum_training.source_identity()
    report = {
        "schema": "meco-onnx-numerical-parity-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "scope": {"required_tracks": list(MECO_TRACKS), "tracks_requested": list(MECO_TRACKS)},
        "checkpoint_root": str(checkpoint_root.resolve()),
        "onnx_root": str(onnx_root.resolve()),
        "input_seed": seed,
        "requested_provider": provider,
        "provider": None,
        "tolerance": {"atol": atol, "rtol": rtol},
        "verifier": {"path": str(Path(__file__).resolve()),
                     "sha256": _sha256(Path(__file__).resolve()),
                     "current_source_identity_sha256": _canonical_sha256(current_sources)},
        "tracks": [],
    }
    try:
        import onnxruntime as ort
    except Exception as exc:
        report["status"] = "unavailable"
        report["unavailable_reason"] = f"ONNX Runtime unavailable: {type(exc).__name__}: {exc}"
        return report

    report["onnxruntime_version"] = getattr(ort, "__version__", "unknown")
    available = list(ort.get_available_providers())
    report["available_providers"] = available
    if provider not in available:
        report["status"] = "unavailable"
        report["unavailable_reason"] = f"requested provider {provider!r} is not available"
        return report

    for track in MECO_TRACKS:
        checkpoint = checkpoint_root / track / "latest.pt"
        graph = onnx_root / f"{track}.onnx"
        item = {"track": track, "checkpoint": str(checkpoint.resolve()),
                "onnx": str(graph.resolve()), "status": "failed", "outputs": []}
        report["tracks"].append(item)
        try:
            model, provenance = load_track_model(track, checkpoint)
            item["checkpoint_provenance"] = provenance
            item["checkpoint_sha256"] = provenance["sha256"]
            item["run_id"] = provenance["run_id"]
            item["update"] = provenance["update"]
            item["protocol_id"] = provenance["protocol_id"]
            item["checkpoint_source_snapshot"] = provenance["sources"]
            item["checkpoint_source_identity_sha256"] = _canonical_sha256(provenance["sources"])
            if not graph.is_file() or graph.stat().st_size == 0:
                raise FileNotFoundError(f"ONNX graph missing or empty: {graph}")
            item["onnx_sha256"] = _sha256(graph)
            inputs = _seeded_inputs(track, seed)
            names = input_names(track)
            if not all(bool(torch.isfinite(tensor).all()) for tensor in inputs):
                raise ValueError("seeded verifier input contains nonfinite values")
            item["input_sha256"] = _input_digest(names, inputs)
            item["inputs"] = [{"name": name, "shape": list(tensor.shape),
                                "dtype": str(tensor.dtype)}
                               for name, tensor in zip(names, inputs)]
            session = ort.InferenceSession(str(graph), providers=[provider])
            actual_providers = list(session.get_providers())
            item["active_providers"] = actual_providers
            report["provider"] = actual_providers
            if provider not in actual_providers:
                raise RuntimeError(
                    f"requested provider {provider!r} is not active in the session")
            graph_inputs = [entry.name for entry in session.get_inputs()]
            graph_outputs = [entry.name for entry in session.get_outputs()]
            if graph_inputs != list(names):
                raise ValueError(f"ONNX input names/order mismatch: {graph_inputs!r}")
            if graph_outputs != list(OUTPUT_NAMES[track]):
                raise ValueError(f"ONNX output names/order mismatch: {graph_outputs!r}")
            feeds = {name: tensor.numpy() for name, tensor in zip(names, inputs)}
            with torch.inference_mode():
                torch_outputs = model(*inputs)
            onnx_outputs = session.run(None, feeds)
            item["outputs"] = _compare_outputs(track, torch_outputs, onnx_outputs, atol, rtol)
            item["status"] = ("pass" if all(output["allclose"] for output in item["outputs"])
                              else "failed")
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
            item["traceback"] = traceback.format_exc(limit=8)
    report["status"] = ("pass" if len(report["tracks"]) == len(MECO_TRACKS)
                        and all(track["status"] == "pass" for track in report["tracks"])
                        else "failed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--onnx-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.seed < 0 or not np.isfinite(args.atol) or not np.isfinite(args.rtol) \
            or args.atol < 0 or args.rtol < 0:
        parser.error("seed and tolerances must be finite and nonnegative")
    report = verify(args.checkpoint_root, args.onnx_root, args.seed,
                    args.provider, args.atol, args.rtol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "evidence": str(args.output)}, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
