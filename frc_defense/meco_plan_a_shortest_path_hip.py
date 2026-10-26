"""On-demand loader for the Plan A exact-heap HIP prototype.

This loader is intentionally not imported by runtime support construction.
Calling ``extension()`` explicitly is the only way to compile/load it.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_EXTENSION = None


def extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("Plan A shortest-path prototype requires a HIP runtime")
    from torch.utils import cpp_extension

    bundled = (Path(torch.__file__).parent / ".." / "_rocm_sdk_core").resolve()
    rocm = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or
        (bundled if bundled.exists() else cpp_extension._find_rocm_home())).resolve()
    rocm_alias = rocm
    if " " in str(rocm):
        rocm_alias = Path("/tmp/autodrive-rocm-sdk")
        if rocm_alias.is_symlink() and rocm_alias.resolve() != rocm:
            rocm_alias.unlink()
        if not rocm_alias.exists():
            rocm_alias.symlink_to(rocm, target_is_directory=True)
        os.environ.setdefault("ROCM_HOME", str(rocm_alias))
        os.environ.setdefault("HIP_CLANG_PATH", str(rocm_alias / "lib/llvm/bin"))
    torch_lib = Path(torch.__file__).parent / "lib"
    if " " in str(torch_lib.resolve()):
        alias = Path("/tmp/autodrive-torch-lib")
        if alias.is_symlink() and alias.resolve() != torch_lib.resolve():
            alias.unlink()
        if not alias.exists():
            alias.symlink_to(torch_lib.resolve(), target_is_directory=True)
        torch_lib = alias
    old = (cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME)
    cpp_extension.TORCH_LIB_PATH = str(torch_lib)
    cpp_extension.ROCM_HOME = str(rocm_alias)
    cpp_extension.HIP_HOME = str(rocm_alias / "hip")
    source = Path(__file__).parent / "csrc"
    flags = ["-O3", "-ffp-contract=off"]
    bitcode = next((path for path in (rocm_alias / "lib/llvm/amdgcn/bitcode",
        rocm_alias / "amdgcn/bitcode") if path.is_dir()), None)
    if bitcode is not None:
        flags.append(f"--rocm-device-lib-path={bitcode}")
    try:
        _EXTENSION = cpp_extension.load(name="autodrive_meco_plan_a_heap_hip_prototype",
            sources=[str(source / "meco_plan_a_shortest_path_hip.cpp"),
                     str(source / "meco_plan_a_shortest_path_hip_kernel.cu")],
            with_cuda=True, extra_cflags=["-O3", "-ffp-contract=off"],
            extra_cuda_cflags=flags, verbose=False)
    finally:
        cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME = old
    return _EXTENSION


def dijkstra_graph(xy, heading, edge_dst, edge_allowed, node_valid, root,
                   root_start_cost, root_connector_m):
    """Run the tensor-root graph kernel under the closed-by-default gate.

    Every input remains on the selected device. This wrapper validates only
    static shape, dtype and placement metadata; kernel status stays a device
    tensor and no scalar is read back to Python.
    """
    import torch

    from .meco_plan_a_shortest_path import (PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED,
        _require_wx9100_device)
    if not PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED:
        raise RuntimeError("Plan A HIP dispatch is disabled pending WX9100 parity and profiling")
    if (not torch.is_tensor(xy) or xy.ndim != 2 or xy.shape[1] != 2 or
            xy.dtype not in (torch.float32, torch.float64) or not xy.is_cuda):
        raise ValueError("xy must be HIP float32/float64 [N,2]")
    device = xy.device
    n = xy.shape[0]
    if (not torch.is_tensor(heading) or heading.shape != (n,) or
            heading.dtype not in (torch.float32, torch.float64) or
            not torch.is_tensor(edge_dst) or edge_dst.ndim != 2 or
            edge_dst.shape[0] != n or edge_dst.dtype != torch.int64 or
            not torch.is_tensor(edge_allowed) or edge_allowed.shape != edge_dst.shape or
            edge_allowed.dtype != torch.bool or
            not torch.is_tensor(node_valid) or node_valid.shape != (n,) or
            node_valid.dtype != torch.bool or
            not torch.is_tensor(root) or root.shape != () or root.dtype != torch.int64 or
            not torch.is_tensor(root_start_cost) or root_start_cost.shape != () or
            root_start_cost.dtype != torch.float64 or
            not torch.is_tensor(root_connector_m) or root_connector_m.shape != () or
            root_connector_m.dtype != torch.float64):
        raise ValueError("Plan A HIP graph tensors have an invalid shape or dtype")
    if n < 1 or edge_dst.shape[1] < 1:
        raise ValueError("Plan A HIP graph capacity and degree must be positive")
    tensors = (heading, edge_dst, edge_allowed, node_valid, root,
               root_start_cost, root_connector_m)
    if any(value.device != device for value in tensors):
        raise ValueError("Plan A HIP graph tensors must share the XY device")
    _require_wx9100_device(device)
    return extension().dijkstra_graph(xy.contiguous(), heading.contiguous(),
        edge_dst.contiguous(), edge_allowed.contiguous(), node_valid.contiguous(),
        root.contiguous(), root_start_cost.contiguous(), root_connector_m.contiguous())
