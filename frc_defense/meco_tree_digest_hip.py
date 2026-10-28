"""On-demand loader for the Plan A parallel tree digest prototype.

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
        _EXTENSION = cpp_extension.load(name="autodrive_meco_tree_digest_hip",
            sources=[str(source / "meco_tree_digest_hip.cpp"),
                     str(source / "meco_tree_digest_hip_kernel.cu")],
            with_cuda=True, extra_cflags=["-O3", "-ffp-contract=off"],
            extra_cuda_cflags=flags, verbose=False)
    finally:
        cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME = old
    return _EXTENSION
