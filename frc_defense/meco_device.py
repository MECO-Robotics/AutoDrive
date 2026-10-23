"""Canonical execution-device provenance shared by MECO evaluation paths."""
from __future__ import annotations


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
