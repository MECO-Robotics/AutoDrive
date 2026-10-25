"""Fail-fast exclusive leases for indexed MECO accelerator devices."""
from __future__ import annotations

from contextlib import AbstractContextManager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
import uuid


def _index(device: str) -> int:
    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise ValueError("device lease requires an indexed CUDA/HIP device")
    raw = device[5:]
    if not raw.isdecimal() or (len(raw) > 1 and raw.startswith("0")):
        raise ValueError("device lease requires a canonical indexed device")
    return int(raw)


def _stable_devices(devices):
    devices = tuple(devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("device lease set must be nonempty and duplicate-free")
    for device in devices:
        _index(device)
    return tuple(sorted(devices, key=_index))


class DeviceLease(AbstractContextManager):
    """Exclusive flock leases; paths are stable and lockfiles are never unlinked."""
    def __init__(self, devices, *, owner="meco", lock_root=None):
        self.devices = _stable_devices(devices)
        self.owner = str(owner)
        self.lock_root = Path(lock_root or
            (Path(tempfile.gettempdir()) / "autodrive-meco-device-leases"))
        self._handles = []
        self.lease_id = None

    @property
    def acquired(self):
        return len(self._handles) == len(self.devices) and bool(self._handles)

    def acquire(self):
        if self._handles:
            raise RuntimeError("device lease is already acquired")
        self.lease_id = uuid.uuid4().hex
        self.lock_root.mkdir(parents=True, exist_ok=True)
        try:
            for device in self.devices:
                token = hashlib.sha256(device.encode()).hexdigest()[:16]
                path = self.lock_root / f"{token}.lock"
                handle = path.open("a+")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    handle.close()
                    raise RuntimeError(f"indexed device {device} is already leased") from exc
                self._handles.append(handle)
                handle.seek(0)
                handle.truncate()
                json.dump({"device": device, "owner": self.owner, "pid": os.getpid(),
                    "lease_id": self.lease_id,
                    "acquired_at": datetime.now(timezone.utc).isoformat()}, handle,
                    sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            return self
        except BaseException:
            self.release()
            raise

    def release(self):
        while self._handles:
            handle = self._handles.pop()
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


def lease_devices(devices, *, owner="meco", lock_root=None):
    return DeviceLease(devices, owner=owner, lock_root=lock_root)
