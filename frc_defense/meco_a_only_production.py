"""Source-bound production launcher for the Plan A training tracks.

This campaign trains fresh A1/A2/A3 policies to a fixed update frontier. It
does not claim Phase 2 readiness, model qualification, or held-out performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from .meco_training_manifest import (_file_sha,
    _valid_native_identity, _protocol, _canonical_bytes,
    source_identity_projection, ROOT, EXPECTED_WX9100_BDFS,
    HOST_GPU_INVENTORY, HOST_GPU_INVENTORY_SHA256)


SCHEMA = "meco-a-only-production-campaign-v1"
TRACKS = ("A1", "A2", "A3")
TARGET_UPDATES = 1000
CHUNK_UPDATES = 100
GENERATED_HIPIFY = frozenset({
    "frc_defense/csrc/fuel_physics_hip_kernel.hip",
    "frc_defense/csrc/fuel_sleep_hip_kernel.hip",
    "frc_defense/csrc/tensor_collision_multi_hip_kernel.hip",
    "frc_defense/csrc/meco_tree_digest_hip_kernel.hip",
    "frc_defense/csrc/meco_tree_digest_hip_hip.cpp",
    "frc_defense/csrc/tensor_adstar_shortcut_hip_hip.cpp",
})
PARITY_REPORTS = {
    "cuda:0": ("hip-parity-20261029/plan-a/cuda0/plan-a-hip-evidence-20261029T213604.450586Z-af6cb61ce4b6.json",
        "860489b4a6f89f961fbd6bbb586cd6c97267d0ebc2a494c264f8add246f54e77"),
    "cuda:1": ("hip-parity-20261029/plan-a/cuda1/plan-a-hip-evidence-20261029T213704.444503Z-268a67037b9f.json",
        "b85102fb3bc678b3f51a15da4eedc52b83948280ab9ea6c16f4afc6a1e972e21"),
}
PARITY_GATE_OPENING_COMMIT = "a1dbf21152f9c920c762c984352445ab71ab403a"


def _sha(value) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = _canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2,
        ensure_ascii=False, allow_nan=False).encode() + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
        dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _clean_source(source_root: Path) -> dict:
    commit = subprocess.check_output(["git", "-C", str(source_root),
        "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(source_root),
        "status", "--porcelain=v1", "--untracked-files=all"], text=True)
    unexpected = [line[3:].strip().strip('"') for line in status.splitlines()
        if line[3:].strip().strip('"') not in GENERATED_HIPIFY]
    if unexpected:
        raise ValueError("A-only production source has uncommitted files: " +
            ", ".join(unexpected[:8]))
    return {"commit": commit, "clean": True,
        "generated_hipify_status_sha256": hashlib.sha256(status.encode()).hexdigest()}


def _parity_evidence(campaigns_root: Path, device: str, source_root: Path) -> dict:
    from .meco_plan_a_shortest_path import PLAN_A_HIP_GRAPH_PARITY_PROFILE_GATE_PASSED
    if PLAN_A_HIP_GRAPH_PARITY_PROFILE_GATE_PASSED is not True:
        raise ValueError("the Plan A HIP graph parity gate is closed in this source")
    relative, expected_sha = PARITY_REPORTS[device]
    path = campaigns_root / relative
    if not path.is_file() or _file_sha(path) != expected_sha:
        raise ValueError(f"pinned HIP parity evidence is missing or changed for {device}")
    report = json.loads(path.read_text())
    if (report.get("passed") is not True or report.get("git_dirty") is not False or
            report.get("requested_device") != device or
            report.get("device", {}).get("model") != "AMD Radeon Pro WX 9100" or
            report.get("training_configuration_modified") is not False or
            report.get("source_identity_verified_at_start_and_end") is not True or
            report.get("fixture_cases_expected") != report.get("fixture_cases_reported")):
        raise ValueError(f"pinned HIP parity evidence does not qualify {device}")
    # The reports were produced with the graph gate closed, then reviewed and
    # source-bound by the main commit that opened this exact graph path.
    if report.get("production_gates_open", {}).get("graph") is not False:
        raise ValueError(f"unexpected HIP parity report gate provenance for {device}")
    if subprocess.run(["git", "-C", str(source_root), "merge-base", "--is-ancestor",
            str(report.get("git_commit")), "HEAD"], check=False).returncode != 0:
        raise ValueError(f"HIP parity source is not an ancestor of this checkout for {device}")
    if subprocess.run(["git", "-C", str(source_root), "merge-base", "--is-ancestor",
            PARITY_GATE_OPENING_COMMIT, "HEAD"], check=False).returncode != 0:
        raise ValueError("the reviewed Plan A HIP graph gate-opening commit is absent")
    for relative_path, expected_sha in report.get("source_sha256", {}).items():
        source_bytes = (source_root / relative_path).read_bytes()
        if relative_path == "frc_defense/meco_plan_a_shortest_path.py":
            new = b"PLAN_A_HIP_GRAPH_PARITY_PROFILE_GATE_PASSED = True"
            old = b"PLAN_A_HIP_GRAPH_PARITY_PROFILE_GATE_PASSED = False"
            if source_bytes.count(new) != 1:
                raise ValueError("current graph parity gate source differs from the reviewed opening change")
            source_bytes = source_bytes.replace(new, old, 1)
        if _sha(source_bytes) != expected_sha:
            raise ValueError(f"HIP parity evidence does not bind current source {relative_path}")
    return {"path": str(path), "sha256": expected_sha,
        "source_commit": report.get("git_commit"),
        "fixture_manifest_sha256": report.get("fixture_manifest_sha256"),
        "extension_binary_sha256": report.get("extension_binary_sha256"),
        "torch_version": report.get("torch_version"),
        "hip_runtime_version": report.get("hip_runtime_version"),
        "hip_architecture": report.get("hip_architecture")}


def describe_campaign(*, protocol_path: Path, campaign_id: str,
        campaign_root: Path, manifest_path: Path,
        source_root: Path = ROOT,
        campaigns_root: Path | None = None,
        devices: tuple[str, ...] = ("cuda:0", "cuda:1"),
        scenario_id: str = "pickup_simple", category: str = "single") -> dict:
    """Freeze a new A-only campaign without acquiring a device lease."""
    source_root = Path(source_root).expanduser().resolve(strict=True)
    git = _clean_source(source_root)
    campaigns_root = Path(campaigns_root or Path(campaign_root).expanduser().absolute().parent).resolve()
    protocol_path, protocol, protocol_sha, _ = _protocol(protocol_path)
    protocol_id = protocol["protocol_id"]
    if protocol.get("master_seed") is None:
        raise ValueError("qualification protocol has no master seed")
    cases = [row for row in protocol.get("cases", []) if
        row.get("scenario_id") == scenario_id and row.get("category") == category and
        row.get("split") == "train"]
    if (len(cases) != 1 or not isinstance(cases[0].get("seeds"), list) or
            len(cases[0]["seeds"]) < len(TRACKS)):
        raise ValueError("campaign requires one frozen train stratum with three or more seeds")
    if (not isinstance(devices, tuple) or not devices or
            any(device not in EXPECTED_WX9100_BDFS for device in devices) or
            len(set(devices)) != len(devices)):
        raise ValueError("devices must be a duplicate-free tuple of pinned WX9100 indices")
    if set(devices) != set(EXPECTED_WX9100_BDFS):
        raise ValueError("production campaign requires both independently pinned WX9100s")
    campaign_root = Path(campaign_root).expanduser().absolute()
    manifest_path = Path(manifest_path).expanduser().absolute()
    if campaign_root.name != campaign_id or campaign_root.exists() or campaign_root.is_symlink():
        raise ValueError("campaign root must be a fresh directory named by campaign_id")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", campaign_id) is None:
        raise ValueError("campaign_id must be a filesystem-safe name")
    if (campaign_root == source_root or source_root in campaign_root.parents or
            campaign_root in source_root.parents):
        raise ValueError("campaign output must be isolated from source checkout")
    if campaign_root == manifest_path or campaign_root in manifest_path.parents:
        raise ValueError("frozen manifest must be stored outside campaign output")
    if (manifest_path.exists() or manifest_path.is_symlink() or
            manifest_path == source_root or source_root in manifest_path.parents or
            manifest_path in source_root.parents):
        raise ValueError("manifest output must be fresh and outside the source checkout")
    inventory_sha = _file_sha(HOST_GPU_INVENTORY)
    if inventory_sha != HOST_GPU_INVENTORY_SHA256:
        raise ValueError("pinned WX9100 host inventory digest mismatch")
    parity = {device: _parity_evidence(campaigns_root, device, source_root)
        for device in devices}
    seeds = cases[0]["seeds"][:len(TRACKS)]
    assignment = {track: {"device": devices[index % len(devices)],
        "seed": seeds[index], "output": str(campaign_root / track)}
        for index, track in enumerate(TRACKS)}
    source = source_identity_projection(__import__(
        "frc_defense.curriculum_training", fromlist=["source_identity"]).source_identity())
    manifest = {"schema": SCHEMA, "campaign_id": campaign_id,
        "campaign_root": str(campaign_root), "manifest_path": str(manifest_path),
        "source": {**git, "source_identity": source,
            "source_identity_sha256": _sha(source)},
        "qualification_protocol": {"path": str(protocol_path),
            "id": protocol_id, "sha256": protocol_sha,
            "split": "train", "scenario_id": scenario_id,
            "category": category, "master_seed": protocol["master_seed"],
            "available_train_seeds": cases[0]["seeds"]},
        "tracks": list(TRACKS), "track_assignments": {
            track: {**assignment[track], "seed_schedule": list(cases[0]["seeds"])}
            for track in TRACKS},
        "training": {"initialization": "fresh", "target_updates_per_track": TARGET_UPDATES,
            "rollout_steps": 32, "epochs": 4, "action_ticks": 12,
            "learning_rate": 0.0003, "gamma": 0.993,
            "gae_lambda": 0.95, "clip_coef": 0.2,
            "entropy_coef": 0.01, "value_coef": 0.5,
            "max_grad_norm": 0.5, "candidate_k": 16,
            "p_mode": "learned", "p_cost_scale": 1.0,
            "p_points_per_grid_edge": 0.05,
            "prediction_mode": "cv"},
        "devices": list(devices), "device_inventory_sha256": inventory_sha,
        "hip_graph_parity_evidence": parity,
        "phase2_ready": False, "phase2_complete": False,
        "qualification_eligible": False, "launch_ready": True,
        "launch_readiness_basis": ["clean source identity", "frozen train protocol",
            "pinned WX9100 inventory", "source-bound Plan A HIP graph parity evidence"],
        "created_at_utc": datetime.now(timezone.utc).isoformat()}
    manifest["canonical_sha256"] = _sha(manifest)
    return manifest


def validate_manifest(manifest: dict, *, source_root: Path = ROOT,
        campaigns_root: Path | None = None) -> dict:
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported A-only production campaign manifest")
    body = {key: value for key, value in manifest.items() if key != "canonical_sha256"}
    if manifest.get("canonical_sha256") != _sha(body):
        raise ValueError("A-only production manifest digest mismatch")
    source_root = Path(source_root).expanduser().resolve(strict=True)
    git = _clean_source(source_root)
    current = source_identity_projection(__import__(
        "frc_defense.curriculum_training", fromlist=["source_identity"]).source_identity())
    if (manifest["source"].get("commit") != git["commit"] or
            manifest["source"].get("source_identity") != current or
            manifest["source"].get("source_identity_sha256") != _sha(current)):
        raise ValueError("A-only production manifest source binding is stale")
    path, protocol, protocol_sha, _ = _protocol(
        Path(manifest["qualification_protocol"]["path"]))
    protocol_id = protocol["protocol_id"]
    expected_protocol = manifest["qualification_protocol"]
    if (protocol_id != expected_protocol.get("id") or
            protocol_sha != expected_protocol.get("sha256") or
            str(path) != expected_protocol.get("path")):
        raise ValueError("A-only production protocol binding is stale")
    if (manifest.get("tracks") != list(TRACKS) or
            manifest.get("training", {}).get("target_updates_per_track") != TARGET_UPDATES or
            manifest.get("launch_ready") is not True or
            manifest.get("qualification_eligible") is not False or
            manifest.get("phase2_ready") is not False or
            manifest.get("phase2_complete") is not False):
        raise ValueError("A-only production scope or readiness claims are invalid")
    if (set(manifest.get("devices", [])) != set(EXPECTED_WX9100_BDFS) or
            len(manifest["devices"]) != len(EXPECTED_WX9100_BDFS)):
        raise ValueError("A-only production requires the exact two pinned WX9100 devices")
    root = Path(manifest["campaign_root"]).expanduser().absolute()
    if root.name != manifest.get("campaign_id"):
        raise ValueError("A-only campaign root/ID mismatch")
    assignments = manifest.get("track_assignments")
    if not isinstance(assignments, dict) or set(assignments) != set(TRACKS):
        raise ValueError("A-only manifest must contain exact A1/A2/A3 assignments")
    allowed_seeds = set(expected_protocol.get("available_train_seeds", []))
    for track in TRACKS:
        item = assignments[track]
        if (item.get("device") not in manifest.get("devices", []) or
                item.get("output") != str(root / track) or
                not item.get("seed_schedule") or
                any(type(seed) is not int or seed not in allowed_seeds
                    for seed in item["seed_schedule"])):
            raise ValueError(f"{track} assignment is outside the frozen train contract")
    inv_sha = _file_sha(HOST_GPU_INVENTORY)
    if inv_sha != HOST_GPU_INVENTORY_SHA256 or inv_sha != manifest.get("device_inventory_sha256"):
        raise ValueError("pinned WX9100 inventory changed")
    campaigns_root = Path(campaigns_root or root.parent).expanduser().resolve()
    for device in manifest["devices"]:
        current_evidence = _parity_evidence(campaigns_root, device, source_root)
        if current_evidence != manifest["hip_graph_parity_evidence"].get(device):
            raise ValueError(f"Plan A HIP parity evidence changed for {device}")
    return {"valid": True, "launch_ready": True,
        "phase2_ready": False, "qualification_eligible": False,
        "canonical_sha256": manifest["canonical_sha256"]}


def _run_track(manifest: dict, track: str, lease_attestation: dict,
        native_attestation: dict, source_sha256: str, *, resume: bool) -> dict:
    import torch
    assignment = manifest["track_assignments"][track]
    output = Path(assignment["output"])
    checkpoint = output / "latest.pt"
    prior = torch.load(checkpoint, map_location="cpu", weights_only=True) if resume else None
    if resume and (not isinstance(prior, dict) or prior.get("config", {}).get("track") != track or
            prior.get("sources") is None or prior.get("update", 0) > TARGET_UPDATES):
        raise ValueError(f"{track} resume checkpoint does not match the production campaign")
    if resume and prior["sources"].get("_native_fuel_extension") != native_attestation.get("identity"):
        raise ValueError(f"{track} resume checkpoint native HIP identity changed")
    config = manifest["training"]
    output.mkdir(parents=True, exist_ok=True)
    log_path = Path(manifest["campaign_root"]) / f"{track}.log"
    start_update = int(prior["update"]) if resume else 0
    seed_schedule = assignment["seed_schedule"]
    if not seed_schedule or any(seed not in manifest["qualification_protocol"]["available_train_seeds"]
            for seed in seed_schedule):
        raise ValueError(f"{track} seed schedule is not confined to the frozen train split")
    update_start = start_update
    while update_start < TARGET_UPDATES:
        chunk_index = update_start // CHUNK_UPDATES
        update_end = min((chunk_index + 1) * CHUNK_UPDATES, TARGET_UPDATES)
        updates = update_end - update_start
        seed = seed_schedule[chunk_index % len(seed_schedule)]
        task = {"scenario_id": manifest["qualification_protocol"]["scenario_id"],
            "category": manifest["qualification_protocol"]["category"],
            "seed": seed, "cycle": chunk_index + 1,
            "sampling_role": "a_only_production",
            "target_update": update_end,
            "assignment_id": _sha({"campaign": manifest["canonical_sha256"],
                "track": track, "cycle": chunk_index + 1, "seed": seed})}
        command = [sys.executable, "-m", "frc_defense.meco_track_training", "train",
            "--track", track, "--scenario", task["scenario_id"],
            "--category", task["category"], "--updates", str(updates),
            "--rollout-steps", str(config["rollout_steps"]), "--epochs", str(config["epochs"]),
            "--action-ticks", str(config["action_ticks"]),
            "--seed", str(manifest["qualification_protocol"]["master_seed"] + TRACKS.index(track)),
            "--device", assignment["device"], "--learning-rate", str(config["learning_rate"]),
            "--gamma", str(config["gamma"]), "--gae-lambda", str(config["gae_lambda"]),
            "--clip-coef", str(config["clip_coef"]), "--entropy-coef", str(config["entropy_coef"]),
            "--value-coef", str(config["value_coef"]), "--max-grad-norm", str(config["max_grad_norm"]),
            "--candidate-k", str(config["candidate_k"]), "--p-mode", config["p_mode"],
            "--p-cost-scale", str(config["p_cost_scale"]),
            "--p-points-per-grid-edge", str(config["p_points_per_grid_edge"]),
            "--prediction-mode", config["prediction_mode"], "--output", str(output),
            "--qualification-protocol", manifest["qualification_protocol"]["path"],
            "--protocol-id", manifest["canonical_sha256"],
            "--task-assignment-json", json.dumps(task, sort_keys=True),
            "--expected-source-sha256", source_sha256,
            "--native-extension-attestation-json", json.dumps(native_attestation,
                sort_keys=True)]
        if checkpoint.exists():
            command.extend(("--resume", str(checkpoint)))
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n### update {update_start} to {update_start + updates}; seed {seed}\n")
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"{track} trainer exited with status {result.returncode}; see {log_path}")
        prior = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (not isinstance(prior, dict) or prior.get("config", {}).get("track") != track or
                prior.get("update") != update_start + updates or prior.get("sources") is None):
            raise RuntimeError(f"{track} checkpoint failed to reach the expected chunk frontier")
        update_start = update_end
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("config", {}).get("track") != track or
            payload.get("update") != TARGET_UPDATES or
            payload.get("sources", {}).get("_native_fuel_extension") !=
                native_attestation.get("identity")):
        raise RuntimeError(f"{track} checkpoint did not reach the frozen production frontier")
    return {"track": track, "device": assignment["device"],
        "seed_schedule_sha256": _sha(seed_schedule), "completed_updates": payload["update"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": _file_sha(checkpoint),
        "log": str(log_path), "status": "complete"}


def _worker_source_sha(sources: dict) -> str:
    # Match meco_track_training._validate_supervisor_training_binding exactly.
    return hashlib.sha256(json.dumps(sources, sort_keys=True,
        allow_nan=False).encode()).hexdigest()


def _worker_identity_sha(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True,
        allow_nan=False).encode()).hexdigest()


def launch_campaign(manifest: dict, *, source_root: Path = ROOT,
        campaigns_root: Path | None = None, resume: bool = False) -> dict:
    """Launch the frozen A-only campaign, holding both leases throughout."""
    source_root = Path(source_root).expanduser().resolve(strict=True)
    campaigns_root = Path(campaigns_root or Path(manifest["campaign_root"]).parent).expanduser().resolve()
    validate_manifest(manifest, source_root=source_root,
        campaigns_root=campaigns_root)
    from .meco_device_lease import lease_devices
    from .meco_gpu_attestation import verify_post_lease_wx9100
    from .fuel_physics_hip import prepare_native_fuel_extension, native_extension_identity
    root = Path(manifest["campaign_root"])
    if root.exists() != resume:
        raise ValueError("fresh launch requires an absent campaign root; resume requires an existing root")
    frozen = root / "campaign-manifest.json"
    if resume:
        if not frozen.is_file() or json.loads(frozen.read_text()) != manifest:
            raise ValueError("resume campaign does not contain the exact frozen manifest")
    status_path = root / "status.json"
    status = {"schema": SCHEMA, "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["canonical_sha256"], "status": "running",
        "launch_ready": True, "phase2_ready": False, "qualification_eligible": False,
        "tracks": {}, "started_at_utc": datetime.now(timezone.utc).isoformat()}
    if resume and status_path.is_file():
        prior_status = json.loads(status_path.read_text())
        if prior_status.get("manifest_sha256") != manifest["canonical_sha256"]:
            raise ValueError("resume status does not bind the frozen campaign manifest")
        status["tracks"] = prior_status.get("tracks", {})
    try:
        with lease_devices(manifest["devices"],
                owner=f"a-only-production-{manifest['campaign_id']}") as lease:
            validate_manifest(manifest, source_root=source_root,
                campaigns_root=campaigns_root)
            device_attestation = verify_post_lease_wx9100(lease, manifest["devices"])
            native_by_device = {}
            worker_source_sha = {}
            for device in manifest["devices"]:
                import torch
                torch.cuda.set_device(torch.device(device))
                prepare_native_fuel_extension(device)
                identity = native_extension_identity()
                if not _valid_native_identity(identity):
                    raise RuntimeError(f"native HIP extension identity is invalid for {device}")
                native_by_device[device] = identity
                from .curriculum_training import source_identity
                worker_source_sha[device] = _worker_source_sha(source_identity())
            if len({_sha(identity) for identity in native_by_device.values()}) != 1:
                raise RuntimeError("WX9100s loaded different native fuel HIP extension identities")
            from .meco_plan_a_shortest_path_hip import extension as plan_a_hip_extension
            route_extension = plan_a_hip_extension()
            route_binary = Path(route_extension.__file__).resolve(strict=True)
            route_binary_sha = _file_sha(route_binary)
            if not resume:
                root.mkdir(parents=True, mode=0o700)
                _atomic_json(frozen, manifest)
            _atomic_json(root / "runtime-attestation.json", {
                "device_identity_attestation": device_attestation,
                "native_extension_identity_by_device": native_by_device,
                "worker_source_sha256_by_device": worker_source_sha,
                "plan_a_graph_extension": {"path": str(route_binary),
                    "sha256": route_binary_sha,
                    "source_files": {name: _file_sha(Path(source_root) / name)
                        for name in ("frc_defense/csrc/meco_plan_a_shortest_path_hip.cpp",
                            "frc_defense/csrc/meco_plan_a_shortest_path_hip_kernel.cu")}},
                "hip_graph_parity_evidence": manifest["hip_graph_parity_evidence"],
                "attested_at_utc": datetime.now(timezone.utc).isoformat()})
            for track in TRACKS:
                state = status["tracks"].get(track, {})
                if state.get("status") == "complete":
                    checkpoint = Path(manifest["track_assignments"][track]["output"]) / "latest.pt"
                    if (not checkpoint.is_file() or _file_sha(checkpoint) !=
                            state.get("checkpoint_sha256")):
                        raise ValueError(f"completed {track} checkpoint changed after campaign status")
                    import torch
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    expected_sources = manifest["source"]["source_identity"]
                    current_sources = __import__("frc_defense.curriculum_training",
                        fromlist=["source_identity"]).source_identity()
                    if (payload.get("config", {}).get("track") != track or
                            payload.get("update") != TARGET_UPDATES or
                            source_identity_projection(payload.get("sources", {})) != expected_sources or
                            payload["sources"].get("_native_fuel_extension") !=
                                native_by_device[manifest["track_assignments"][track]["device"]] or
                            payload["sources"] != current_sources):
                        raise ValueError(f"completed {track} checkpoint provenance changed")
                    continue
                device = manifest["track_assignments"][track]["device"]
                result = _run_track(manifest, track, device_attestation,
                    {"schema": 1, "identity": native_by_device[device],
                        "identity_sha256": _worker_identity_sha(native_by_device[device])},
                    worker_source_sha[device], resume=resume)
                if _file_sha(route_binary) != route_binary_sha:
                    raise RuntimeError("Plan A HIP graph binary changed during the production campaign")
                status["tracks"][track] = result
                _atomic_json(status_path, status)
        if set(status["tracks"]) != set(TRACKS) or any(
                status["tracks"][track].get("completed_updates") != TARGET_UPDATES
                for track in TRACKS):
            raise RuntimeError("campaign ended without all A tracks at the 1000-update frontier")
        status.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
        _atomic_json(status_path, status)
        return status
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}",
            failed_at_utc=datetime.now(timezone.utc).isoformat())
        if root.exists():
            _atomic_json(status_path, status)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    desc = subs.add_parser("describe")
    desc.add_argument("--protocol", required=True, type=Path)
    desc.add_argument("--campaign-id", required=True)
    desc.add_argument("--campaign-root", required=True, type=Path)
    desc.add_argument("--manifest", required=True, type=Path)
    desc.add_argument("--source-root", type=Path, default=ROOT)
    desc.add_argument("--campaigns-root", type=Path)
    run = subs.add_parser("launch")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--source-root", type=Path, default=ROOT)
    run.add_argument("--campaigns-root", type=Path)
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "describe":
        value = describe_campaign(protocol_path=args.protocol,
            campaign_id=args.campaign_id, campaign_root=args.campaign_root,
            manifest_path=args.manifest, source_root=args.source_root,
            campaigns_root=args.campaigns_root)
        _atomic_json(args.manifest, value)
        print(json.dumps({"valid": True, "launch_ready": value["launch_ready"],
            "manifest_sha256": value["canonical_sha256"],
            "tracks": value["tracks"], "target_updates_per_track": TARGET_UPDATES}, indent=2))
        return 0
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, source_root=args.source_root,
        campaigns_root=args.campaigns_root)
    status = launch_campaign(manifest, source_root=args.source_root,
        campaigns_root=args.campaigns_root, resume=args.resume)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
