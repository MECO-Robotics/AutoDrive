"""Fail-closed physical WX9100 identity checks for leased MECO devices."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import re
import subprocess
import uuid


EXPECTED_IDENTITIES = {
    "cuda:0": {
        "name": "AMD Radeon Pro WX 9100",
        "pci_address": "0000:19:00.0",
        "pci_vendor_device": "1002:6861",
        "hip_uuid": "30323133-6632-6261-3734-313632316134",
    },
    "cuda:1": {
        "name": "AMD Radeon Pro WX 9100",
        "pci_address": "0000:67:00.0",
        "pci_vendor_device": "1002:6861",
        "hip_uuid": "30323133-6661-3836-3737-346334386134",
    },
}
P600_IDENTITY = ("0000:b3:00.0", "10de:1cb2")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _model_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", value.casefold())
    return key[3:] if key.startswith("amd") else key


def _normalize_uuid(value) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            if len(value) != 16:
                raise RuntimeError("HIP device UUID bytes are malformed")
            return str(uuid.UUID(bytes=value))
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("HIP device UUID is unavailable")
    text = value.strip()
    if text.casefold().startswith("gpu-"):
        text = text[4:]
    try:
        return str(uuid.UUID(text))
    except (ValueError, AttributeError) as exc:
        raise RuntimeError("HIP device UUID is malformed") from exc


def _normalize_bdf(value) -> str:
    if not isinstance(value, str):
        raise RuntimeError("HIP runtime PCI BDF is unavailable")
    match = re.fullmatch(
        r"(?:(?P<domain>[0-9a-fA-F]{4,8}):)?(?P<bus>[0-9a-fA-F]{2}):"
        r"(?P<slot>[0-9a-fA-F]{2})\.(?P<function>[0-7])", value.strip())
    if match is None:
        raise RuntimeError("HIP runtime PCI BDF is malformed")
    return (f"{match.group('domain') or '0000'}:{match.group('bus')}:"
            f"{match.group('slot')}.{match.group('function')}").lower()


def _runtime_bdf(properties, pci_devices: dict[str, str], expected: dict) -> str:
    """Resolve the runtime PCI identity across CUDA and ROCm property shapes.

    Some HIP builds expose numeric domain/bus/device fields instead of a BDF
    string. In that case, use the numeric tuple only to select the lspci
    function and require a unique function with the pinned vendor/device ID.
    """
    for name in ("pci_bus_id_string", "pciBusID", "pci_bus_id"):
        value = getattr(properties, name, None)
        if isinstance(value, str):
            return _normalize_bdf(value)

    fields = (getattr(properties, "pci_domain_id", None),
              getattr(properties, "pci_bus_id", None),
              getattr(properties, "pci_device_id", None))
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in fields):
        raise RuntimeError("HIP runtime PCI BDF fields are unavailable")
    domain, bus, slot = fields
    if not (0 <= domain <= 0xffff and 0 <= bus <= 0xff and 0 <= slot <= 0x1f):
        raise RuntimeError("HIP runtime PCI BDF fields are malformed")
    prefix = f"{domain:04x}:{bus:02x}:{slot:02x}."
    matching = [bdf for bdf, vendor_device in pci_devices.items()
                if bdf.startswith(prefix) and
                vendor_device == expected["pci_vendor_device"]]
    if len(matching) != 1 or matching[0] != expected["pci_address"]:
        raise RuntimeError("HIP runtime PCI fields do not resolve uniquely to the pinned device")
    return matching[0]


def _run_json_command(argv: list[str], label: str):
    try:
        result = subprocess.run(argv, check=True, capture_output=True, text=True,
                                timeout=15)
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} identity query failed") from exc


def _lspci_devices() -> dict[str, str]:
    try:
        result = subprocess.run(["lspci", "-Dnn"], check=True,
            capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("lspci physical-device query failed") from exc
    devices = {}
    for line in result.stdout.splitlines():
        match = re.match(r"^(\S+)\s+", line)
        if match is None:
            continue
        address = _normalize_bdf(match.group(1))
        pairs = re.findall(r"\[([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\]", line)
        if pairs:
            devices[address] = f"{pairs[-1][0]}:{pairs[-1][1]}".lower()
    if not devices:
        raise RuntimeError("lspci returned no parseable PCI device identities")
    return devices


def _rocm_product_names(payload) -> list[str]:
    names = []

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if (isinstance(child, str) and
                        re.search(r"product|model|series|name", str(key), re.I)):
                    names.append(child.strip())
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    if not names:
        raise RuntimeError("rocm-smi returned no product-name fields")
    return names


def _runtime_property(properties, *names):
    for name in names:
        value = getattr(properties, name, None)
        if value is not None and value != "":
            return value
    raise RuntimeError(f"HIP device properties omit required field {names[0]}")


def verify_post_lease_wx9100(lease, devices) -> dict:
    """Attest requested logical indices against the pinned physical inventory.

    This performs identity queries only. It does not allocate tensors or launch
    kernels. Any absent property/tool, inventory drift, or identity mismatch
    prevents the caller from reserving campaign output roots.
    """
    from .meco_training_manifest import (EXPECTED_WX9100_BDFS,
        HOST_GPU_INVENTORY, HOST_GPU_INVENTORY_SHA256)

    raw_devices = tuple(devices)
    if not raw_devices or any(device not in EXPECTED_IDENTITIES
                              for device in raw_devices):
        raise RuntimeError("physical identity check requires indexed WX9100 lease devices")
    requested = tuple(sorted(set(raw_devices),
        key=lambda item: int(item.split(":", 1)[1])))
    if len(requested) != len(raw_devices):
        raise RuntimeError("physical identity check requires unique indexed WX9100 lease devices")
    if (not getattr(lease, "acquired", False) or
            tuple(getattr(lease, "devices", ())) != requested):
        raise RuntimeError("physical identity check requires the matching acquired logical lease")
    lease_id = getattr(lease, "lease_id", None)
    if not isinstance(lease_id, str) or re.fullmatch(r"[0-9a-f]{32}", lease_id) is None:
        raise RuntimeError("acquired logical device lease ID is unavailable")
    owner = getattr(lease, "owner", None)
    if not isinstance(owner, str) or not owner:
        raise RuntimeError("acquired logical device lease owner is unavailable")

    try:
        inventory_bytes = HOST_GPU_INVENTORY.read_bytes()
    except OSError as exc:
        raise RuntimeError("pinned host GPU inventory is unavailable") from exc
    inventory_sha256 = _sha256(inventory_bytes)
    if inventory_sha256 != HOST_GPU_INVENTORY_SHA256:
        raise RuntimeError("pinned host GPU inventory SHA-256 mismatch")
    try:
        inventory = json.loads(inventory_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError("pinned host GPU inventory is malformed") from exc
    if (not isinstance(inventory, dict) or
            inventory.get("schema") != "meco-host-gpu-inventory-v1"):
        raise RuntimeError("pinned host GPU inventory schema mismatch")
    expected_rows = {row.get("device"): row for row in inventory.get("devices", [])
                     if isinstance(row, dict)}
    if set(expected_rows) != set(EXPECTED_IDENTITIES):
        raise RuntimeError("pinned inventory does not contain the exact WX9100 set")
    for device, expected in EXPECTED_IDENTITIES.items():
        row = expected_rows[device]
        if (row.get("name") != expected["name"] or
                row.get("pci_address") != EXPECTED_WX9100_BDFS.get(device) or
                row.get("pci_address") != expected["pci_address"] or
                row.get("pci_vendor_device") != expected["pci_vendor_device"] or
                _normalize_uuid(row.get("hip_uuid")) != expected["hip_uuid"]):
            raise RuntimeError(f"pinned inventory identity mismatch for {device}")
    excluded = inventory.get("excluded_devices", [])
    if not any(isinstance(row, dict) and row.get("pci_address") == P600_IDENTITY[0]
            and row.get("pci_vendor_device", "").lower() == P600_IDENTITY[1]
            for row in excluded):
        raise RuntimeError("pinned inventory does not explicitly exclude the P600")

    import torch
    if (torch.version.hip is None or not torch.cuda.is_available() or
            torch.cuda.device_count() < 1 + max(int(device.split(":", 1)[1])
                                                for device in requested)):
        raise RuntimeError("live HIP runtime does not expose every leased device")
    pci_devices = _lspci_devices()
    product_payload = _run_json_command(
        ["rocm-smi", "--showproductname", "--json"], "rocm-smi product-name")
    rocm_products = _rocm_product_names(product_payload)
    expected_product_keys = {_model_key(row["name"]) for row in
                             (expected_rows[device] for device in requested)}
    observed_product_keys = {_model_key(name) for name in rocm_products}
    if not expected_product_keys <= observed_product_keys:
        # Some ROCm builds expose only the PCI device ID in Card Model and
        # report Card Series as N/A. Accept that limited response only when
        # lspci independently confirms every requested pinned PCI identity;
        # runtime model and UUID are still checked below.
        unavailable_product_identity = all(
            key in {"na", f"0x{expected_rows[device]['pci_vendor_device'].split(':')[1]}"}
            for key in observed_product_keys
            for device in requested)
        if not unavailable_product_identity or any(
                pci_devices.get(expected_rows[device]["pci_address"]) !=
                expected_rows[device]["pci_vendor_device"]
                for device in requested):
            raise RuntimeError(
                "rocm-smi product name does not match the leased WX9100 inventory")

    observed = []
    for device in requested:
        index = int(device.split(":", 1)[1])
        expected = expected_rows[device]
        try:
            properties = torch.cuda.get_device_properties(index)
        except Exception as exc:
            raise RuntimeError(f"HIP properties unavailable for {device}") from exc
        runtime_name = _runtime_property(properties, "name")
        runtime_uuid = _normalize_uuid(_runtime_property(properties, "uuid"))
        runtime_bdf = _runtime_bdf(properties, pci_devices, expected)
        vendor_device = pci_devices.get(runtime_bdf)
        if (runtime_bdf, vendor_device) == P600_IDENTITY:
            raise RuntimeError(f"refusing NVIDIA Quadro P600 at logical index {device}")
        if (not isinstance(runtime_name, str) or
                _model_key(runtime_name) != _model_key(expected["name"]) or
                runtime_uuid != expected["hip_uuid"] or
                runtime_bdf != expected["pci_address"] or
                vendor_device != expected["pci_vendor_device"]):
            raise RuntimeError(f"live physical identity mismatch for {device}")
        observed.append({"device": device, "model": runtime_name,
            "pci_bdf": runtime_bdf, "pci_vendor_device": vendor_device,
            "hip_uuid": runtime_uuid})

    attestation = {"schema": "meco-post-lease-wx9100-attestation-v1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_path": str(HOST_GPU_INVENTORY),
        "inventory_sha256": inventory_sha256,
        "logical_lease": {"lease_id": lease_id, "owner": owner,
            "pid": os.getpid(), "devices": list(requested)},
        "devices": observed,
        "rocm_smi_product_names": sorted(set(rocm_products))}
    attestation["attestation_sha256"] = _sha256(json.dumps(attestation,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return attestation
