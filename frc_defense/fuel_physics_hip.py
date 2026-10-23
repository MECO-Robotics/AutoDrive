"""Device-resident cached linked-list broadphase and fused fuel contacts."""
from __future__ import annotations
import math
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import warnings
import fcntl
import threading
from pathlib import Path
import torch
_EXT = None
_ATTEMPTED = False
_EXT_IDENTITY = None
_BUILD_FINGERPRINT = None
_MANIFEST_STAT_IDENTITY = None
_EXTENSION_LOCK = threading.RLock()

_EXTENSION_NAME = "autodrive_fuel_physics_hip"
_BUILD_FLAGS = {"extra_cflags": ["-O3"],
    "extra_cuda_cflags": ["-O3", "-ffp-contract=off"],
    "with_cuda": True}

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _extension_path():
    explicit = os.environ.get("AUTODRIVE_FUEL_HIP_EXTENSION_PATH")
    path = (Path(explicit).expanduser() if explicit else
        Path(__file__).resolve().parent.parent /
        "outputs/fuel_physics/extensions/native" / f"{_EXTENSION_NAME}.so")
    path = path.resolve()
    if path.name != f"{_EXTENSION_NAME}.so":
        raise ValueError("AUTODRIVE_FUEL_HIP_EXTENSION_PATH must name the native HIP .so")
    return path

def _path_stat_identity(path):
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return {"path": str(Path(path).resolve()), "device": stat.st_dev,
        "inode": stat.st_ino, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

def _same_stat_identity(left, right):
    return bool(left and right and all(left.get(key) == right.get(key)
        for key in ("path", "device", "inode", "size", "mtime_ns")))

def _build_inputs():
    from torch.utils import cpp_extension
    from torch.utils.hipify import hipify_python
    src = Path(__file__).parent / "csrc"
    rocm_bundled = (Path(torch.__file__).parent / ".." / "_rocm_sdk_core").resolve()
    rocm_value = (os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or
        (rocm_bundled if rocm_bundled.exists() else
         (cpp_extension._find_rocm_home() if torch.version.hip is not None else None)))
    rocm = Path(rocm_value).resolve() if rocm_value else None
    hip_clang_value = os.environ.get("HIP_CLANG_PATH")
    hip_clang_path = (Path(hip_clang_value).expanduser().resolve() if hip_clang_value
        else (rocm / "lib/llvm/bin").resolve() if rocm else None)
    device_lib = next((p for p in (rocm / "lib/llvm/amdgcn/bitcode",
                                    rocm / "amdgcn/bitcode") if p.is_dir()), None) if rocm else None
    base_flags = list(_BUILD_FLAGS["extra_cuda_cflags"])
    if device_lib is not None:
        base_flags.append(f"--rocm-device-lib-path={device_lib}")
    arch_flags = (sorted(set(cpp_extension._get_rocm_arch_flags(base_flags)))
        if torch.version.hip is not None else [])
    compiler = (rocm / "bin/hipcc") if rocm else None
    if compiler and not compiler.exists():
        compiler = rocm / "lib/llvm/bin/clang++"
    relevant_env = {name: os.environ.get(name) for name in (
        "PYTORCH_ROCM_ARCH",
        "TORCH_CUDA_ARCH_LIST", "CXX", "CC", "CFLAGS", "CXXFLAGS",
        "CPPFLAGS", "LDFLAGS", "HIPCC_COMPILE_FLAGS_APPEND",
        "HIPCC_LINK_FLAGS_APPEND")}
    cxx = shutil.which(os.environ.get("CXX", "c++"))
    cc = shutil.which(os.environ.get("CC", "cc"))
    return {"schema": 1, "name": _EXTENSION_NAME,
        "sources": {name: _sha256(src / name) for name in
            ("fuel_physics_hip.cpp", "fuel_physics_hip_kernel.cu")},
        "torch": {"version": str(torch.__version__), "hip": torch.version.hip,
            "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            "package": str(Path(torch.__file__).resolve()),
            "cpp_extension": _path_stat_identity(cpp_extension.__file__),
            "hipify": _path_stat_identity(hipify_python.__file__),
            "torch_c": _path_stat_identity(torch._C.__file__)},
        "python": {"version": platform.python_version(),
            "abi_tag": getattr(sys.implementation, "cache_tag", None),
            "abiflags": getattr(sys, "abiflags", "")},
        "toolchain_environment": {"effective_rocm_home": str(rocm) if rocm else None,
            "effective_hip_clang_path": str(hip_clang_path) if hip_clang_path else None},
        "rocm": str(rocm) if rocm else None,
        "hipcc": _path_stat_identity(compiler) if compiler else None,
        "host_compilers": {"cxx": _path_stat_identity(cxx) if cxx else None,
            "cc": _path_stat_identity(cc) if cc else None},
        "device_lib": ({"directory": _path_stat_identity(device_lib),
            "files": [_path_stat_identity(path) for path in sorted(device_lib.rglob("*"))
                if path.is_file()]} if device_lib else None),
        "flags": {**_BUILD_FLAGS, "device_lib_flag": (
            f"--rocm-device-lib-path={device_lib}" if device_lib else None),
            "arch_flags": arch_flags},
        "environment": relevant_env}

def _fingerprint(build_inputs):
    payload = json.dumps(build_inputs, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()

def _manifest_path(binary):
    return Path(str(binary) + ".manifest.json")

def _manifest_identity(binary, fingerprint, build_inputs):
    try:
        manifest = json.loads(_manifest_path(binary).read_text())
        identity = _binary_identity(binary)
    except (OSError, ValueError, RuntimeError):
        return None
    if (manifest.get("schema") != 1 or manifest.get("binary_path") != str(binary) or
            manifest.get("build_fingerprint") != fingerprint or
            manifest.get("build_inputs") != build_inputs or
            manifest.get("binary_sha256") != identity["sha256"]):
        return None
    return {**identity, "build_fingerprint": fingerprint}

def _write_manifest(binary, fingerprint, identity, build_inputs):
    payload = {"schema": 1, "binary_path": str(binary),
        "build_fingerprint": fingerprint, "build_inputs": build_inputs,
        "binary_sha256": identity["sha256"]}
    fd, temp_name = tempfile.mkstemp(prefix=f".{binary.name}.", suffix=".tmp",
                                     dir=binary.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, _manifest_path(binary))
    finally:
        Path(temp_name).unlink(missing_ok=True)

def _load_existing(binary):
    existing = sys.modules.get(_EXTENSION_NAME)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != binary:
            raise RuntimeError("native HIP extension name is already loaded from another path")
        return existing
    spec = importlib.util.spec_from_file_location(_EXTENSION_NAME, binary)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create loader for prebuilt native HIP extension: {binary}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_EXTENSION_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_EXTENSION_NAME, None)
        raise
    return module

def _build_lock(binary):
    lock_path = Path(str(binary) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    return stream

def _mapped_extension_identity(path):
    """Find the live dlopen mapping by path, device, and inode."""
    path = Path(path).resolve()
    expected_path = str(path)
    maps_path = Path("/proc/self/maps")
    try:
        lines = maps_path.read_text().splitlines()
    except OSError as exc:
        raise RuntimeError("cannot attest the loaded native HIP mapping via /proc/self/maps") from exc
    matching = []
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 5 or len(fields) < 6:
            continue
        try:
            major_hex, minor_hex = fields[3].split(":", 1)
            mapped_device = (int(major_hex, 16), int(minor_hex, 16))
            mapped_inode = int(fields[4])
        except (ValueError, IndexError):
            continue
        mapped_path = re.sub(r"\\([0-7]{3})",
            lambda match: chr(int(match.group(1), 8)), fields[5])
        if mapped_path.endswith(" (deleted)"):
            mapped_path = mapped_path[:-10]
        try:
            same_path = str(Path(mapped_path).resolve()) == expected_path
        except OSError:
            same_path = mapped_path == expected_path
        if same_path:
            matching.append((mapped_device, mapped_inode))
    stat = path.stat()
    expected_device = (os.major(stat.st_dev), os.minor(stat.st_dev))
    if not any(device == expected_device and inode == stat.st_ino
               for device, inode in matching):
        raise RuntimeError("loaded native HIP mapping does not match its current path inode")


def _binary_identity(path, *, require_loaded_mapping=False):
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_id != after_id:
        raise RuntimeError("native HIP extension changed while its identity was read")
    if require_loaded_mapping:
        _mapped_extension_identity(path)
    return {"module": "autodrive_fuel_physics_hip", "path": str(path),
        "sha256": digest, "device": after.st_dev, "inode": after.st_ino,
        "size": after.st_size, "mtime_ns": after.st_mtime_ns,
        "status": "present"}

def _verify_loaded_identity(*, strong=False):
    if _EXT is None or _EXT_IDENTITY is None:
        return
    binary = _extension_path()
    if Path(_EXT.__file__).resolve() != binary:
        raise RuntimeError("loaded native HIP module path differs from the pinned extension path")
    current_stat = _path_stat_identity(binary)
    manifest_stat = _path_stat_identity(_manifest_path(binary))
    if (not _same_stat_identity(current_stat, _EXT_IDENTITY) or
            not _same_stat_identity(manifest_stat, _MANIFEST_STAT_IDENTITY)):
        raise RuntimeError("loaded native HIP extension path or manifest changed after load")
    _mapped_extension_identity(binary)
    if not strong:
        return
    build_inputs = _build_inputs()
    fingerprint = _fingerprint(build_inputs)
    current = _binary_identity(binary, require_loaded_mapping=True)
    manifest_identity = _manifest_identity(binary, fingerprint, build_inputs)
    pinned = {**current, "build_fingerprint": fingerprint}
    if manifest_identity != pinned or pinned != _EXT_IDENTITY:
        raise RuntimeError("loaded native HIP extension path or binary changed after load")

def native_extension_identity():
    """Describe the loaded or expected HIP binary without importing/building it."""
    with _EXTENSION_LOCK:
        if _EXT is not None:
            _verify_loaded_identity()
            return dict(_EXT_IDENTITY)
        path = _extension_path()
        # CPU-only provenance must not walk ROCm's device libraries or hash
        # an unused binary. GPU entry points call prepare_native_fuel_extension
        # before capturing provenance, which performs the full attestation.
        candidate = _path_stat_identity(path)
        result = {"module": _EXTENSION_NAME, "path": str(path),
            "status": "unprepared"}
        if candidate is not None:
            result.update(candidate_device=candidate["device"],
                candidate_inode=candidate["inode"], candidate_size=candidate["size"],
                candidate_mtime_ns=candidate["mtime_ns"])
        return result

def prepare_native_fuel_extension(device):
    """Explicitly build/load and pin HIP before capturing GPU run provenance."""
    requested = torch.device(device)
    if requested.type != "cuda":
        return None
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("CUDA-device MECO work requires an available HIP runtime")
    if requested.index is not None and requested.index >= torch.cuda.device_count():
        raise RuntimeError(f"requested HIP device is unavailable: {requested}")
    extension = _extension()
    with _EXTENSION_LOCK:
        if extension is None or _EXT_IDENTITY is None:
            raise RuntimeError("native HIP fuel extension could not be prepared; refusing GPU fallback")
        _verify_loaded_identity(strong=True)
        return dict(_EXT_IDENTITY)

def _extension():
    with _EXTENSION_LOCK:
        return _extension_locked()

def _extension_locked():
    global _EXT, _ATTEMPTED, _EXT_IDENTITY, _BUILD_FINGERPRINT, _MANIFEST_STAT_IDENTITY
    if _ATTEMPTED:
        _verify_loaded_identity()
        return _EXT
    if not (torch.cuda.is_available() and torch.version.hip):
        return None
    _ATTEMPTED = True
    try:
        from torch.utils import cpp_extension
        binary = _extension_path()
        build_dir = binary.parent
        build_dir.mkdir(parents=True, exist_ok=True)
        lock_stream = _build_lock(binary)
        try:
            build_inputs = _build_inputs()
            fingerprint = _fingerprint(build_inputs)
            manifest_identity = _manifest_identity(binary, fingerprint, build_inputs)
            if manifest_identity is not None:
                _EXT = _load_existing(binary)
                current = _binary_identity(binary, require_loaded_mapping=True)
                pinned = {**current, "build_fingerprint": fingerprint}
                if pinned != manifest_identity:
                    raise RuntimeError("prebuilt native HIP binary changed while loading")
                _EXT_IDENTITY = pinned
                _BUILD_FINGERPRINT = fingerprint
                _MANIFEST_STAT_IDENTITY = _path_stat_identity(_manifest_path(binary))
                _verify_loaded_identity(strong=True)
                return _EXT

            if _EXTENSION_NAME in sys.modules:
                raise RuntimeError(
                    "native HIP extension is already loaded without a valid build manifest; "
                    "restart the process before rebuilding it")
            _manifest_path(binary).unlink(missing_ok=True)
            binary.unlink(missing_ok=True)
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
            src = Path(__file__).parent / "csrc"
            device_lib = next((p for p in (rocm_alias / "lib/llvm/amdgcn/bitcode",
                                            rocm_alias / "amdgcn/bitcode") if p.is_dir()), None)
            flags = list(_BUILD_FLAGS["extra_cuda_cflags"])
            if device_lib is not None:
                flags.append(f"--rocm-device-lib-path={device_lib}")
            old_env = {name: os.environ.get(name) for name in ("ROCM_HOME", "HIP_CLANG_PATH")}
            if " " in str(rocm):
                os.environ["ROCM_HOME"] = str(rocm_alias)
                os.environ["HIP_CLANG_PATH"] = str(rocm_alias / "lib/llvm/bin")
            try:
                _EXT = cpp_extension.load(name=_EXTENSION_NAME,
                    sources=[str(src / "fuel_physics_hip.cpp"),
                             str(src / "fuel_physics_hip_kernel.cu")],
                    with_cuda=True, build_directory=str(build_dir),
                    extra_cflags=list(_BUILD_FLAGS["extra_cflags"]),
                    extra_cuda_cflags=flags, verbose=False)
            finally:
                for name, value in old_env.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
                cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME = old
            if Path(_EXT.__file__).resolve() != binary:
                raise RuntimeError("Torch built the HIP extension outside its pinned path")
            current = _binary_identity(binary, require_loaded_mapping=True)
            after_build_inputs = _build_inputs()
            if _fingerprint(after_build_inputs) != fingerprint:
                raise RuntimeError("native HIP build inputs changed while extension was compiling")
            _EXT_IDENTITY = {**current, "build_fingerprint": fingerprint}
            _BUILD_FINGERPRINT = fingerprint
            _write_manifest(binary, fingerprint, current, build_inputs)
            _MANIFEST_STAT_IDENTITY = _path_stat_identity(_manifest_path(binary))
            _verify_loaded_identity(strong=True)
        finally:
            lock_stream.close()
    except Exception as exc:
        _EXT = None
        _EXT_IDENTITY = None
        warnings.warn(f"Coupled HIP fuel physics unavailable: {exc}",
                      RuntimeWarning, stacklevel=2)
    return _EXT



class FuelHIP:
    def __init__(self, physics):
        self.physics = physics
        self.ext = _extension()
        self.available = self.ext is not None
        if not self.available:
            return
        p,c,s = physics,physics.config,physics.sim
        if p.pos.dtype != torch.float32:
            raise ValueError("HIP fuel state requires float32")
        self.cell = 2*c.radius+c.skin
        self.xyz = (math.ceil(s.field_length/self.cell)+2,
                    math.ceil(s.field_width/self.cell)+2,math.ceil(8/self.cell)+2)
        kw = dict(device=p.device,dtype=torch.int32)
        self.heads = torch.full((p.n,math.prod(self.xyz)),-1,**kw)
        self.next = torch.zeros((p.n,p.p),**kw)
        self.list = torch.zeros((p.n,p.p,c.max_neighbors),**kw)
        self.counts = torch.zeros((p.n,p.p),**kw)
        self.overflow = torch.zeros(p.n,**kw)
        self.rebuild = torch.ones(p.n,**kw)
        self.metrics = torch.zeros(2,**kw)
        self.reference = p.pos.clone()
        self.oldfree = torch.zeros_like(p.sleeping)
        self.dv = torch.zeros_like(p.vel)
        self.dw = torch.zeros_like(p.angular)
        self.rdv = torch.zeros_like(s.velocity)
        # Grid compaction bounds storage by neighbor capacity; overflowing
        # contacts use the fused path, preserving every contact impulse.
        capacity = (p.n*p.p*(p.p-1)//2 if c.broadphase == "all_pairs" or c.compact_capacity == "full" else min(p.n*p.p*c.max_neighbors,p.n*p.p*(p.p-1)//2)) if c.contact_mode == "compact" else 1
        self.contacts = torch.zeros((max(1,capacity),2),**kw)
        self.contact_count = torch.zeros(1,**kw)
        self.robot_candidates = torch.zeros((p.n,s.pose.shape[1],p.p),device=p.device,dtype=torch.bool)
        self.bumps = getattr(s,"bump_regions",torch.empty((0,4),device=p.device)).contiguous()
        self.support_height = torch.zeros((p.n,p.p),device=p.device,dtype=p.pos.dtype)
        p._support_height = self.support_height
        self.spawn_heights = torch.empty((p.n*p.p,),device=p.device,dtype=p.pos.dtype)
        self.boxes = getattr(s,"field_colliders",torch.empty((0,4),device=p.device)).contiguous()

    def spawn_height(self, xy):
        from .field import BUMP_PANEL_THICKNESS, BUMP_RAMP_RISE
        count=xy.numel()//2
        if count>self.spawn_heights.numel():
            raise ValueError("spawn positions exceed allocated fuel slots")
        out=self.spawn_heights[:count].view(xy.shape[:-1])
        self.ext.spawn_height(xy.contiguous(),self.bumps,out,self.physics.config.radius,
                              BUMP_PANEL_THICKNESS,BUMP_RAMP_RISE)
        return out

    def invalidate(self):
        self.rebuild.fill_(1)

    def step(self, free, changed, dt, substeps):
        p,c,s = self.physics,self.physics.config,self.physics.sim
        # Spawns/teleports must invalidate even when displacement ends inside
        # the previous margin. No device-to-host check is needed.
        torch.maximum(self.rebuild, changed.any(-1).to(torch.int32), out=self.rebuild)
        tensors = [p.pos,p.vel,p.angular,p.sleeping,p.sleep_clock,free.contiguous(),
                   s.pose,s.velocity,s.length,s.width,s.mass,s.yaw_inertia_multiplier,
                   self.boxes,self.reference,self.oldfree,self.heads,self.next,self.list,
                   self.counts,self.overflow,self.rebuild,self.metrics,self.dv,self.dw,
                   self.rdv,self.contacts,self.contact_count,self.robot_candidates,self.bumps,self.support_height]
        from .field import BUMP_PANEL_THICKNESS, BUMP_RAMP_RISE
        params = [c.radius,c.mass,c.gravity,c.stiffness,c.damping,c.friction,
                  c.rolling_friction,c.robot_height,c.skin,dt,s.field_length,s.field_width,
                  c.sleep_time,c.sleep_speed,c.sleep_angular_speed,c.max_compression_fraction,BUMP_PANEL_THICKNESS,BUMP_RAMP_RISE]
        dims = [p.n,p.p,s.pose.shape[1],self.boxes.shape[0],c.max_neighbors,*self.xyz,
                int(c.cache),(2 if c.sleep and c.sleep_islands else int(c.sleep)),int(c.broadphase=="all_pairs"),int(c.contact_mode=="compact"),int(c.solver=="colored"),int(c.robot_broadphase=="grid"),self.bumps.shape[0],self.contacts.shape[0]]
        for _ in range(substeps):
            self.ext.substep(tensors,params,dims)
            if c.sleep and c.sleep_islands:
                from .fuel_physics_sleep import update_island_sleep
                update_island_sleep(p,free,dt)

    def refresh_stats(self):
        p=self.physics
        values=self.metrics.cpu().tolist()
        p.stats["rebuilds"],p.stats["overflows"]=values
        p.stats["candidate_pairs"]=int(self.counts.sum().item())
        if p.config.broadphase=="all_pairs":
            p.stats["candidate_pairs"]=p.n*p.p*(p.p-1)//2
        return p.stats
