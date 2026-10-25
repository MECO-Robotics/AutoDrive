"""Read-only, fail-closed r4/WX9100 release observation for paired collection.

This module deliberately uses only the standard library. It is safe to import
before the evaluator, PyTorch, HIP, or any native simulator extension.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import copy

R4_CAMPAIGN_ID = "meco-five-continuous-20261025-r4"
R4_CAMPAIGN_ROOT = Path(
    "/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDriveCampaigns"
) / R4_CAMPAIGN_ID
R4_MANIFEST_PATH = R4_CAMPAIGN_ROOT.parent / "manifests" / f"{R4_CAMPAIGN_ID}.json"
WX9100_BY_INDEX = {
    "cuda:0": {"bdf": "0000:19:00.0", "uuid": "30323133-6632-6261-3734-313632316134"},
    "cuda:1": {"bdf": "0000:67:00.0", "uuid": "30323133-6661-3836-3737-346334386134"},
}
_CAPABILITY_TOKEN = object()


class _LiveReleaseCapability:
    """Process-local proof that this module just completed both live scans."""
    __slots__ = ("_token", "_receipt")

    def __init__(self, token, receipt):
        if token is not _CAPABILITY_TOKEN:
            raise TypeError("live release capabilities can only come from the observer")
        self._token = token
        self._receipt = copy.deepcopy(receipt)

    def record(self):
        if self._token is not _CAPABILITY_TOKEN:
            raise ValueError("live release capability is invalid")
        return copy.deepcopy(self._receipt)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _cmdline(path: Path) -> tuple[str, ...]:
    raw = path.read_bytes()
    if not raw:
        return ()
    return tuple(token.decode("utf-8", "strict") for token in raw.rstrip(b"\0").split(b"\0"))


def _argv_value(argv: tuple[str, ...], name: str):
    try:
        index = argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        raise RuntimeError(f"process command line has an incomplete {name} option")
    return argv[index + 1]


def _r4_process_role(argv: tuple[str, ...]) -> str | None:
    """Recognize the canonical supervisor or its spawned evaluator/trainer."""
    module_index = None
    for index in range(len(argv) - 1):
        if argv[index] in {"-m", "--module"}:
            module_index = index + 1
            break
    if module_index is None or module_index >= len(argv):
        return None
    module = argv[module_index]
    if module == "frc_defense.meco_supervisor":
        path = _argv_value(argv, "--campaign-manifest")
        if path is None:
            return None
        try:
            if Path(path).expanduser().resolve(strict=False) == R4_MANIFEST_PATH:
                return "supervisor"
        except (OSError, RuntimeError):
            raise RuntimeError("could not resolve a MECO supervisor manifest argument")
        return None
    if module != "frc_defense.meco_track_training":
        return None
    if len(argv) <= module_index + 1 or argv[module_index + 1] not in {"train", "evaluate"}:
        return None
    output = _argv_value(argv, "--output")
    if output is None:
        return None
    try:
        resolved = Path(output).expanduser().resolve(strict=False)
        root = R4_CAMPAIGN_ROOT.resolve(strict=False)
    except (OSError, RuntimeError):
        raise RuntimeError("could not resolve a MECO worker output argument")
    if resolved == root or root in resolved.parents:
        return "trainer" if argv[module_index + 1] == "train" else "evaluator"
    return None


def _process_rows(proc_root: Path) -> list[dict]:
    rows = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError("cannot enumerate process table for release verification") from exc
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            argv = _cmdline(entry / "cmdline")
        except FileNotFoundError:
            # A process may exit during the scan. It cannot continue to own a
            # device once /proc has removed its entry.
            continue
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"cannot inspect process {pid} command line") from exc
        role = _r4_process_role(argv)
        if role:
            rows.append({"pid": pid, "role": role, "argv_sha256": _sha256(
                b"\0".join(part.encode("utf-8") for part in argv))})
    return sorted(rows, key=lambda row: (row["pid"], row["role"]))


def _render_node_bdf(link: str, sys_root: Path) -> str | None:
    match = re.fullmatch(r"/dev/dri/renderD[0-9]+", link)
    if match is None:
        return None
    node = Path(link).name
    device = sys_root / "class" / "drm" / node / "device"
    try:
        target = device.resolve(strict=True)
    except FileNotFoundError:
        raise RuntimeError(f"cannot map open DRM node {node} to a physical device")
    bdf_match = re.search(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$", str(target))
    if bdf_match is None:
        raise RuntimeError(f"DRM node {node} has no canonical PCI BDF")
    return bdf_match.group(0).lower()


def _device_owners(proc_root: Path, sys_root: Path) -> dict[str, list[int]]:
    by_bdf = {item["bdf"]: device for device, item in WX9100_BY_INDEX.items()}
    owners = {device: set() for device in WX9100_BY_INDEX}
    try:
        processes = list(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError("cannot enumerate process table for GPU ownership check") from exc
    for process in processes:
        if not process.name.isdecimal():
            continue
        pid = int(process.name)
        fd_root = process / "fd"
        try:
            fds = list(fd_root.iterdir())
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"cannot inspect file descriptors for process {pid}") from exc
        for fd in fds:
            try:
                link = os.readlink(fd)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RuntimeError(f"cannot inspect process {pid} device ownership") from exc
            bdf = _render_node_bdf(link, sys_root)
            device = by_bdf.get(bdf)
            if device is not None:
                owners[device].add(pid)
    return {device: sorted(pids) for device, pids in owners.items()}


def _validate_worker_index_uuid(campaign):
    workers = campaign.get("worker_assignments") if isinstance(campaign, dict) else None
    if not isinstance(workers, list) or len(workers) != len(WX9100_BY_INDEX):
        raise ValueError("release check requires both explicitly assigned WX9100 workers")
    observed = {}
    for worker in workers:
        if not isinstance(worker, dict):
            raise ValueError("paired calibration worker assignment is malformed")
        device = worker.get("device")
        expected = WX9100_BY_INDEX.get(device)
        if (expected is None or worker.get("device_uuid", "").lower() != expected["uuid"]
                or worker.get("device_model") != "AMD Radeon Pro WX 9100"
                or worker.get("backend") != "hip"):
            raise ValueError("worker assignment does not bind the indexed WX9100 UUID")
        if device in observed:
            raise ValueError("worker assignments duplicate an indexed WX9100")
        observed[device] = worker["device_uuid"].lower()
    if set(observed) != set(WX9100_BY_INDEX):
        raise ValueError("worker assignments do not cover the two indexed WX9100s")


def verify_external_release_before_paired_run(campaign, *, proc_root="/proc",
        sys_root="/sys", settle_seconds=1.0, clock=time.monotonic,
        sleep=time.sleep) -> dict:
    """Observe r4 process absence and exact WX9100 release before collection.

    The scan is read-only and uses live process command lines plus open DRM
    render-node ownership mapped to pinned physical BDF/UUID identities. It
    rejects partial visibility and repeats the observations after a short
    settle interval to reduce process-exit/start races. This is not a durable
    operator approval; it proves only that the named r4 processes and device
    owners were absent at the observation time.
    """
    _validate_worker_index_uuid(campaign)
    proc_root, sys_root = Path(proc_root), Path(sys_root)
    if not isinstance(settle_seconds, (int, float)) or settle_seconds < 0:
        raise ValueError("release observation settle interval is invalid")
    start = clock()
    observations = []
    for iteration in range(2):
        processes = _process_rows(proc_root)
        owners = _device_owners(proc_root, sys_root)
        if processes:
            raise RuntimeError("canonical r4 supervisor/trainer/evaluator process is still active")
        if any(owners.values()):
            raise RuntimeError("an assigned WX9100 still has active DRM process ownership")
        observations.append({"r4_processes": processes, "device_owners": owners})
        if iteration == 0 and settle_seconds:
            sleep(settle_seconds)
    elapsed = clock() - start
    receipt = {"schema": "meco-external-release-observation-v1",
        "campaign_id": R4_CAMPAIGN_ID,
        "manifest_path": str(R4_MANIFEST_PATH),
        "manifest_id": "acaf745eb84eb66ea8447a884ad6eb036166abe2f596b168ee392fb6d423bc03",
        "devices": {device: {**identity, "active_owners": []}
                    for device, identity in sorted(WX9100_BY_INDEX.items())},
        "observations": observations, "settle_seconds": elapsed,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "method": "proc-cmdline-and-proc-fd-render-node-bdf-v1"}
    receipt["observation_sha256"] = _sha256(json.dumps(receipt,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return _LiveReleaseCapability(_CAPABILITY_TOKEN, receipt)


def require_live_release_capability(capability, campaign):
    """Accept only the process-local capability returned by this live observer."""
    if (not isinstance(capability, _LiveReleaseCapability)
            or capability._token is not _CAPABILITY_TOKEN):
        raise ValueError("finalization requires a fresh live release observer capability")
    receipt = capability.record()
    digest = validate_persisted_release_observation_record(receipt, campaign)
    return receipt, digest


def validate_persisted_release_observation_record(receipt, campaign):
    """Validate stored receipt integrity only; this is never run authorization."""
    if not isinstance(receipt, dict) or receipt.get("schema") != "meco-external-release-observation-v1":
        raise ValueError("paired calibration release observation is missing or malformed")
    body = {key: value for key, value in receipt.items() if key != "observation_sha256"}
    if (receipt.get("observation_sha256") != _sha256(json.dumps(body,
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            or receipt.get("campaign_id") != R4_CAMPAIGN_ID
            or receipt.get("manifest_path") != str(R4_MANIFEST_PATH)
            or receipt.get("manifest_id") != "acaf745eb84eb66ea8447a884ad6eb036166abe2f596b168ee392fb6d423bc03"
            or receipt.get("method") != "proc-cmdline-and-proc-fd-render-node-bdf-v1"):
        raise ValueError("paired calibration release observation identity/digest changed")
    _validate_worker_index_uuid(campaign)
    expected_devices = {device: {**identity, "active_owners": []}
        for device, identity in sorted(WX9100_BY_INDEX.items())}
    observations = receipt.get("observations")
    if (receipt.get("devices") != expected_devices
            or not isinstance(observations, list) or len(observations) != 2
            or any(row != {"r4_processes": [], "device_owners":
                {device: [] for device in WX9100_BY_INDEX}} for row in observations)):
        raise ValueError("paired calibration release observation does not prove empty r4/device ownership")
    return receipt["observation_sha256"]
