"""Canonical execution-device provenance shared by MECO evaluation paths."""
from __future__ import annotations


def require_indexed_cuda_device(device, *, purpose="MECO compute"):
    """Require an explicitly indexed, currently available CUDA/ROCm device."""
    import torch

    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise ValueError(f"{purpose} requires an explicit indexed GPU such as cuda:0")
    try:
        selected = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"{purpose} requires an explicit indexed GPU such as cuda:0") from exc
    if (selected.type != "cuda" or selected.index is None or selected.index < 0
            or not torch.cuda.is_available()
            or selected.index >= torch.cuda.device_count()):
        raise ValueError(f"{purpose} device is unavailable: {device}")
    return selected


def normalized_device_info(device) -> dict[str, str]:
    """Resolve aliases such as ``cuda`` to a stable device/backend identity."""
    import torch

    selected = torch.device(device)
    if selected.type == "cuda":
        index = selected.index
        if index is None:
            index = torch.cuda.current_device()
        selected = torch.device("cuda", index)
        backend = "ROCm" if torch.version.hip else "CUDA"
    elif selected.type == "cpu":
        selected = torch.device("cpu")
        backend = "CPU"
    else:
        backend = selected.type.upper()
    return {"device": str(selected), "accelerator_backend": backend}
