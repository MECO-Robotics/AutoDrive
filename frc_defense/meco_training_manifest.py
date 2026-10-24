"""Offline, source-bound manifest for the five-track MECO campaign.

This module only inspects repository/configuration/artifact metadata.  It never
starts training, imports the simulator, or reserves a device.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

from .curriculum_training import source_identity
from .meco_tracks import (MECO_TRACKS, MECO_CANDIDATE_RAW_FEATURE_COUNT,
                          PREDICTION_MODES, make_meco_track)
from .meco_track_training import CHUNK_RESET_POLICY, MecoTrainConfig
from .training_curriculum import get_scenario

SCHEMA = "meco-five-track-campaign-manifest-v2"
ROOT = Path(__file__).resolve().parents[1]
LEASE_CONTRACT = {"owner_limit_per_index": 1,
    "training_policy": "sequential_round_robin_tracks",
    "evaluation_policy": "wait_for_device_lease"}
PINNED_LIVE_ROOTS = ((ROOT / "checkpoints" / "meco-five").resolve(),
                     (ROOT / "evaluations" / "meco-five").resolve())
EXPECTED_WX9100_BDFS = {"cuda:0": "0000:19:00.0", "cuda:1": "0000:67:00.0"}
HOST_GPU_INVENTORY = ROOT / "qualifications" / "meco" / "host-gpu-inventory-20261024.json"
HOST_GPU_INVENTORY_SHA256 = "5f7518b9c0a25687947a5375eb01a10e67a7a9089f7264eb44625bab52b543af"
NATIVE_SOURCE_FIELD = "_native_fuel_extension"
SOURCE_IDENTITY_POLICY = {"projection": "all source_identity fields except _native_fuel_extension",
    "native_runtime_attestation": "required after HIP preparation and before worker launch"}


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def source_identity_projection(identity):
    """Return the stable code/config identity while preserving all other inputs."""
    if not isinstance(identity, dict) or not identity:
        raise ValueError("source identity must be a nonempty object")
    return {key: value for key, value in identity.items() if key != NATIVE_SOURCE_FIELD}


def _source_artifact_hash(identity):
    # Matches meco_supervised._canonical_hash for artifact metadata.
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _valid_native_identity(identity):
    if not isinstance(identity, dict) or identity.get("status") != "present":
        return False
    if (not isinstance(identity.get("module"), str) or not identity["module"] or
            not isinstance(identity.get("path"), str) or not Path(identity["path"]).is_absolute()):
        return False
    for key in ("sha256", "build_fingerprint"):
        value = identity.get(key)
        if (not isinstance(value, str) or len(value) != 64 or
                any(char not in "0123456789abcdef" for char in value)):
            return False
    return all(type(identity.get(key)) is int and identity[key] >= 0
               for key in ("device", "inode", "size", "mtime_ns")) and identity["size"] > 0


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_binding(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {"commit": commit, "clean": not bool(status),
            "status_sha256": hashlib.sha256(status.encode()).hexdigest()}


def _protocol(path):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("qualification protocol must be an object")
    pid = value.get("protocol_id")
    body = {k: v for k, v in value.items() if k != "protocol_id"}
    version = value.get("protocol_version")
    if pid != _sha(body) or version not in ("meco-qualification-v1", "meco-qualification-v2"):
        raise ValueError("qualification protocol ID/version is invalid")
    if version == "meco-qualification-v2":
        from .meco_qualification import _validate_v2_protocol
        _validate_v2_protocol(value, value.get("promotion_target_scenario"))
    split_seeds = {name: set() for name in ("train", "validation", "test")}
    for case in value.get("cases", []):
        split = case.get("split")
        seeds = case.get("seeds")
        key = (case.get("scenario_id"), case.get("category"), split)
        if (split not in split_seeds or not all(isinstance(x, str) and x for x in key[:2])
                or not isinstance(seeds, list) or not seeds
                or any(type(s) is not int or s < 0 for s in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError("qualification protocol contains malformed cases")
        split_seeds[split].update(seeds)
    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        if split_seeds[left] & split_seeds[right]:
            raise ValueError("qualification protocol seed partitions overlap")
    return path, value, hashlib.sha256(raw).hexdigest(), {
        k: sorted(v) for k, v in split_seeds.items()}


def _model_schema(track):
    import torch
    # Schema inspection must not perturb a caller's training RNG state.
    with torch.random.fork_rng(devices=[]):
        model = make_meco_track(track, candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT)
    schema = [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
              for name, tensor in model.state_dict().items()]
    return {"sha256": _sha(schema), "parameters": schema}


def _device_index(device):
    if not isinstance(device, str) or ":" not in device:
        raise ValueError("every training/evaluation device must be explicitly indexed CUDA")
    kind, index = device.split(":", 1)
    if kind != "cuda" or not index.isdecimal():
        raise ValueError("every training/evaluation device must be explicitly indexed CUDA")
    return int(index)


def _validate_config_metadata(cfg):
    """Validate config shape without querying CUDA availability or building envs."""
    if cfg.track not in MECO_TRACKS:
        raise ValueError("invalid MECO track")
    definition = get_scenario(cfg.scenario_id)
    if cfg.category not in definition.required_categories:
        raise ValueError("category is not registered for scenario")
    if cfg.prediction_mode not in PREDICTION_MODES or cfg.p_mode not in ("learned", "p0"):
        raise ValueError("unsupported prediction/P configuration")
    for name in ("updates", "rollout_steps", "epochs", "action_ticks", "seed", "candidate_k"):
        value = getattr(cfg, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid integer config field {name}")
    if min(cfg.updates, cfg.rollout_steps, cfg.epochs, cfg.action_ticks) < 1:
        raise ValueError("training iteration counts must be positive")
    if cfg.candidate_k not in (8, 16, 32):
        raise ValueError("candidate_k must be 8, 16, or 32")
    _device_index(cfg.device)
    for name in ("learning_rate", "gamma", "gae_lambda", "clip_coef", "entropy_coef",
                 "value_coef", "max_grad_norm", "route_cost_points_per_second",
                 "p_cost_scale", "p_points_per_grid_edge"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if (cfg.learning_rate <= 0 or cfg.gamma > 1 or cfg.gae_lambda > 1 or
            cfg.max_grad_norm <= 0):
        raise ValueError("invalid optimizer or discount configuration")
    if not isinstance(cfg.output, str) or not cfg.output:
        raise ValueError("every track requires an explicit campaign output path")


def _path_is_within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _paths_overlap(first, second):
    first, second = Path(first).resolve(), Path(second).resolve()
    return first == second or _path_is_within(first, second) or _path_is_within(second, first)


def _validate_campaign_paths(campaign_id, campaign_root, tracks, evaluation_output, output=None,
                             check_existing_output=True, check_namespace_available=True):
    if (not isinstance(campaign_id, str) or
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", campaign_id) is None):
        raise ValueError("campaign_id must be a filesystem-safe namespace")
    campaign_root = Path(campaign_root).expanduser().resolve()
    if campaign_root.name != campaign_id:
        raise ValueError("campaign_root must be namespaced by the exact campaign_id")
    if check_namespace_available and campaign_root.exists():
        raise ValueError("campaign namespace already exists; use a fresh campaign_id/root")
    for pinned in PINNED_LIVE_ROOTS:
        if _path_is_within(campaign_root, pinned) or _path_is_within(pinned, campaign_root):
            raise ValueError("campaign_root overlaps a pinned live checkpoint/evaluation root")
    resolved = []
    for track in MECO_TRACKS:
        path = Path(tracks[track]["output"]).expanduser().resolve()
        if not _path_is_within(path, campaign_root) or path == campaign_root:
            raise ValueError(f"{track} output must be a child inside the campaign namespace")
        if any(_paths_overlap(path, prior) for prior in resolved):
            raise ValueError("track output paths must be unique and non-overlapping")
        resolved.append(path)
    if not isinstance(evaluation_output, str) or not evaluation_output:
        raise ValueError("evaluation_output must be explicit inside campaign_root")
    eval_output = Path(evaluation_output).expanduser().resolve()
    if (not _path_is_within(eval_output, campaign_root) or
            any(_paths_overlap(eval_output, prior) for prior in resolved)):
        raise ValueError("evaluation output must be unique and inside campaign_root")
    resolved.append(eval_output)
    if check_namespace_available and any(path.exists() for path in resolved):
        raise ValueError("campaign checkpoint/evaluation output directory already exists")
    if output is not None:
        output = Path(output).expanduser().resolve()
        if (not _path_is_within(output, campaign_root) or
                any(_paths_overlap(output, prior) for prior in resolved)):
            raise ValueError("manifest output must be isolated inside campaign_root")
        if check_existing_output and output.exists():
            raise ValueError("refusing to replace an existing frozen manifest")
        for pinned in PINNED_LIVE_ROOTS:
            if _path_is_within(output, pinned) or _path_is_within(pinned, output):
                raise ValueError("manifest output overlaps a pinned live run root")


def _device_inventory(device_pool):
    if not HOST_GPU_INVENTORY.is_file():
        return {device: {"hardware_verified": False, "reason": "host inventory artifact missing"}
            for device in device_pool}, {"path": str(HOST_GPU_INVENTORY), "sha256": None,
            "verified": False}
    digest = _file_sha(HOST_GPU_INVENTORY)
    if digest != HOST_GPU_INVENTORY_SHA256:
        raise ValueError("host GPU inventory bytes differ from the reviewed pinned SHA-256")
    payload = json.loads(HOST_GPU_INVENTORY.read_text())
    if payload.get("schema") != "meco-host-gpu-inventory-v1":
        raise ValueError("unsupported host GPU inventory schema")
    actual = {row.get("device"): row for row in payload.get("devices", [])}
    excluded = payload.get("excluded_devices", [])
    p600_excluded = any(row.get("name") == "NVIDIA Quadro P600" and
        row.get("pci_address") == "0000:b3:00.0" for row in excluded)
    if set(actual) != set(EXPECTED_WX9100_BDFS) or not p600_excluded:
        raise ValueError("host inventory must identify two WX9100s and separately exclude the P600")
    rows = {}
    for device in device_pool:
        observed = actual[device]
        if (observed.get("pci_address") != EXPECTED_WX9100_BDFS[device] or
                re.search(r"WX\s*9100", observed.get("name", ""), re.IGNORECASE) is None or
                observed.get("pci_vendor_device") != "1002:6861"):
            raise ValueError("host inventory index/BDF/model does not match reviewed WX9100 mapping")
        rows[device] = {"hardware_verified": True, "observed_hardware": observed["name"],
            "observed_pci_bdf": observed["pci_address"],
            "evidence_sha256": digest}
    binding = {"path": str(HOST_GPU_INVENTORY), "sha256": digest,
        "schema": payload["schema"], "captured_at_utc": payload.get("captured_at_utc"),
        "excluded_devices": excluded, "verified": True}
    return rows, binding


def _gate_calibration(protocol):
    return {"protocol_version": protocol.get("protocol_version"),
        "status": protocol.get("gate_calibration_status"),
        "trusted_calibration_report_binding": False,
        "calibrated_and_reviewed": False,
        "reason": "manifest API has no trusted calibration-report validator binding"}


def _atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError("refusing to replace an existing frozen manifest") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _warm_start(path, track, expected_sources, protocol_path, protocol_id, protocol_sha):
    required_targets = (("U", "P", "behavior") if track.startswith("A") else
                        ("gross_reward", "pairwise_preference"))
    unverified_targets = {name: {"valid": False,
        "reason": "target dataset manifest bytes/schema/coverage are not validated by this API"}
        for name in required_targets}
    if path is None:
        return {"present": False, "path": None, "sha256": None,
                "native_extension_identity": None, "native_extension_identity_sha256": None,
                "native_runtime_attested": False,
                "binding_valid": False, "targets": unverified_targets, "phase2_covered": False,
                "reason": "artifact not supplied"}
    path = Path(path).expanduser().resolve()
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return {"present": path.is_file(), "path": str(path),
                "sha256": _file_sha(path) if path.is_file() else None,
                "native_extension_identity": None, "native_extension_identity_sha256": None,
                "native_runtime_attested": False,
                "binding_valid": False, "targets": unverified_targets, "phase2_covered": False,
                "reason": f"artifact unreadable: {type(exc).__name__}"}
    if not isinstance(payload, dict):
        payload = {}
    payload_sources = payload.get("sources")
    native_identity = (payload_sources.get(NATIVE_SOURCE_FIELD)
                       if isinstance(payload_sources, dict) else None)
    native_valid = _valid_native_identity(native_identity)
    source_binding = (isinstance(payload_sources, dict) and
        source_identity_projection(payload_sources) == expected_sources and native_valid and
        payload.get("source_identity_sha256") == _source_artifact_hash(payload_sources))
    protocol_binding = (payload.get("protocol_id") == protocol_id and
        payload.get("protocol_sha256", payload.get("qualification_protocol_sha256")) == protocol_sha)
    track_binding = payload.get("track") == track
    # This offline inspection binds artifact metadata only. The supervisor
    # revalidates the complete artifact against the loaded binary before launch.
    targets = unverified_targets
    return {"present": True, "path": str(path), "sha256": _file_sha(path),
        "artifact_schema": payload.get("schema"), "track": payload.get("track"),
        "source_identity_sha256": payload.get("source_identity_sha256"),
        "native_extension_identity": native_identity,
        "native_extension_identity_sha256": (_sha(native_identity) if native_valid else None),
        "native_runtime_attested": False,
        "protocol_id": payload.get("protocol_id"),
        "protocol_sha256": payload.get("protocol_sha256",
            payload.get("qualification_protocol_sha256")),
        "binding_valid": bool(source_binding and protocol_binding and track_binding),
        "targets": targets,
        "phase2_covered": False,
        "reason": ("loaded HIP parity is required at runtime" if
            source_binding and protocol_binding and track_binding else
            "source projection, native identity, protocol, or track binding mismatch")}


def describe_campaign(config, *, source_root=ROOT, output=None):
    """Describe a campaign from explicit config; write only when output is passed."""
    source_root = Path(source_root).resolve()
    if not isinstance(config, dict) or set(config.get("tracks", {})) != set(MECO_TRACKS):
        raise ValueError(f"config must define exactly these tracks: {list(MECO_TRACKS)}")
    campaign_id = config.get("campaign_id")
    campaign_root = config.get("campaign_root")
    evaluation_output = config.get("evaluation_output")
    _validate_campaign_paths(campaign_id, campaign_root, config["tracks"], evaluation_output, output)
    protocol_path, protocol, protocol_sha, seed_partitions = _protocol(config["qualification_protocol"])
    base_seed = config.get("seed")
    if type(base_seed) is not int or base_seed != protocol.get("master_seed"):
        raise ValueError("campaign seed must equal the frozen protocol master seed")
    entries = {}
    for track in MECO_TRACKS:
        raw = config["tracks"][track]
        if not isinstance(raw, dict) or raw.get("track") != track:
            raise ValueError(f"{track} config must explicitly bind its track name")
        cfg = MecoTrainConfig(**raw)
        _validate_config_metadata(cfg)
        data = asdict(cfg)
        entries[track] = {"config": data, "config_sha256": _sha(data),
                          "model_schema": _model_schema(track)}
    shared_fields = set(entries[MECO_TRACKS[0]]["config"])
    shared_fields -= {"track", "device", "output"}
    shared = {name: entries[MECO_TRACKS[0]]["config"][name]
              for name in sorted(shared_fields)}
    for track, entry in entries.items():
        for name, expected in shared.items():
            if entry["config"][name] != expected:
                raise ValueError(f"shared config field {name} differs for {track}")
    devices = {track: entries[track]["config"]["device"] for track in MECO_TRACKS}
    device_pool = config.get("training_devices")
    if not isinstance(device_pool, list):
        raise ValueError("training_devices must be an indexed CUDA device list")
    indices = [_device_index(d) for d in device_pool]
    if len(device_pool) != 2:
        raise ValueError("current campaign requires exactly two indexed training GPUs")
    if len(indices) != len(set(indices)):
        raise ValueError("training_devices lease pool must not contain duplicate devices")
    if set(device_pool) != set(EXPECTED_WX9100_BDFS):
        raise ValueError("current campaign requires the two indexed WX9100 devices cuda:0 and cuda:1")
    inventory, host_inventory = _device_inventory(device_pool)
    hardware_verified = host_inventory["verified"] and all(inventory[d].get("hardware_verified")
        for d in device_pool)
    if any(device not in device_pool for device in devices.values()):
        raise ValueError("every track assignment must use a device in the indexed lease pool")
    eval_device = config.get("evaluation_device")
    eval_index = _device_index(eval_device)
    if eval_device not in device_pool:
        raise ValueError("evaluation device must use a declared indexed lease device")
    lease_contract = config.get("lease_contract")
    if lease_contract != LEASE_CONTRACT:
        raise ValueError("device lease contract must enforce sequential exclusive ownership")
    leases = {"pool": device_pool, "training_by_track": devices,
        "evaluation": eval_device, "contract": LEASE_CONTRACT}
    source = source_identity_projection(source_identity())
    source_hash = _sha(source)
    git = _git_binding(source_root)
    warm = {track: _warm_start(config.get("warm_start_artifacts", {}).get(track), track,
            source, protocol_path, protocol["protocol_id"], protocol_sha) for track in MECO_TRACKS}
    schedule = config.get("scheduler")
    if not isinstance(schedule, dict) or schedule.get("master_seed") != base_seed:
        raise ValueError("scheduler must explicitly bind the protocol master seed")
    if schedule.get("seed_partitions") != seed_partitions:
        raise ValueError("scheduler seed partitions must exactly match protocol partitions")
    assignment_contract = ("sequential-cycle deterministic curriculum assignment; all five tracks share "
        "scenario/category/seed; train seed indexed by cycle modulo frozen protocol pool")
    if (schedule.get("algorithm") != "MecoTaskScheduler" or schedule.get("version") != 1 or
            schedule.get("task_assignment_contract") != assignment_contract):
        raise ValueError("scheduler algorithm/version/task assignment contract is not explicit")
    evaluation_path, evaluation_protocol, evaluation_sha, evaluation_partitions = _protocol(
        config["evaluation_protocol"])
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation policy must be explicit and separate from training protocol")
    eval_split = evaluation.get("split")
    scenario_id, category = evaluation.get("scenario_id"), evaluation.get("category")
    episode_seeds = evaluation.get("episode_seeds")
    if (eval_split not in ("diagnostic", "validation", "test") or
            not isinstance(scenario_id, str) or not isinstance(category, str) or
            not isinstance(episode_seeds, list) or not episode_seeds or
            any(type(s) is not int or s < 0 for s in episode_seeds) or
            len(set(episode_seeds)) != len(episode_seeds)):
        raise ValueError("evaluation split/scenario/category/episode seeds are malformed")
    eval_case_seeds = {seed for case in evaluation_protocol.get("cases", [])
        if case.get("scenario_id") == scenario_id and case.get("category") == category
        and case.get("split") == eval_split for seed in case.get("seeds", [])}
    if eval_split == "diagnostic":
        if set(episode_seeds) & set(sum(evaluation_partitions.values(), [])):
            raise ValueError("diagnostic evaluation seeds overlap frozen protocol partitions")
    elif not set(episode_seeds) <= eval_case_seeds:
        raise ValueError("evaluation episode seeds are outside the bound protocol split/case")
    if evaluation.get("protocol_id") not in (None, evaluation_protocol["protocol_id"]):
        raise ValueError("evaluation protocol ID does not match protocol file")
    manifest = {"schema": SCHEMA, "source_identity_policy": SOURCE_IDENTITY_POLICY,
        "source": {"commit": git["commit"],
            "clean": git["clean"], "status_sha256": git["status_sha256"],
            "identity_sha256": source_hash, "identity": source},
        "qualification_protocol": {"path": str(protocol_path),
            "protocol_id": protocol["protocol_id"], "sha256": protocol_sha,
            "split_seed_partitions": seed_partitions},
        "qualification_gate_calibration": _gate_calibration(protocol),
        "evaluation_protocol": {"path": str(evaluation_path),
            "protocol_id": evaluation_protocol["protocol_id"], "sha256": evaluation_sha,
            "split_seed_partitions": evaluation_partitions, "split": eval_split,
            "scenario_id": scenario_id, "category": category,
            "episode_seeds": episode_seeds,
            "gate_calibration": _gate_calibration(evaluation_protocol),
            "qualification_eligible": False},
        "campaign": {"id": campaign_id, "root": str(Path(campaign_root).expanduser().resolve()),
            "evaluation_output": str(Path(evaluation_output).expanduser().resolve()),
            "manifest_output": (str(Path(output).expanduser().resolve()) if output is not None else None)},
        "device_inventory": inventory, "host_gpu_inventory": host_inventory,
        "seed": base_seed, "shared_training_fields": shared,
        "tracks": entries, "scheduler": schedule,
        "devices": {"training_pool": device_pool, "training_by_track": devices,
            "evaluation": eval_device,
            "indexed_device_leases": leases},
        "chunk_reset_policy": CHUNK_RESET_POLICY,
        "warm_starts": warm}
    implementation_ready = bool(git["clean"] and hardware_verified and len(entries) == 5 and
        len(indices) == 2 and
        all(entry["config_sha256"] for entry in entries.values()))
    phase2_ready = bool(implementation_ready and all(x["phase2_covered"] for x in warm.values()))
    validation_ready = bool(implementation_ready and eval_split == "validation")
    qualification_ready = bool(implementation_ready and phase2_ready and
        manifest["evaluation_protocol"]["qualification_eligible"])
    manifest["readiness"] = {"hardware_ready": hardware_verified,
        "implementation_ready": implementation_ready,
        "validation_ready": validation_ready,
        "qualification_ready": qualification_ready,
        "launch_ready": False,
        "launch_blocker": "supervisor launch requires explicit --diagnostic-launch, starts fresh outputs, and remains permanently non-qualifying without Phase 2 evidence",
        "phase2_ready": phase2_ready, "phase2_complete": False,
        "phase2_blockers": [f"{t}: valid U/P/behavior or gross/pairwise target coverage missing"
            for t, item in warm.items() if not item["phase2_covered"]]}
    manifest["canonical_sha256"] = _sha(manifest)
    if output is not None:
        _atomic_write_json(Path(output).expanduser().resolve(), manifest)
    return manifest


def validate_campaign_manifest(manifest, *, source_root=ROOT):
    """Validate canonical digest, clean source binding, protocol bytes and track map."""
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported campaign manifest")
    data = dict(manifest)
    digest = data.pop("canonical_sha256", None)
    if digest != _sha(data):
        raise ValueError("campaign manifest canonical digest mismatch")
    expected = _git_binding(Path(source_root).resolve())
    src = manifest.get("source", {})
    current_identity = source_identity_projection(source_identity())
    if (not expected["clean"] or not src.get("clean") or
            src.get("commit") != expected["commit"] or
            src.get("status_sha256") != expected["status_sha256"] or
            manifest.get("source_identity_policy") != SOURCE_IDENTITY_POLICY or
            not isinstance(src.get("identity"), dict) or
            NATIVE_SOURCE_FIELD in src["identity"] or
            src.get("identity") != current_identity or
            src.get("identity_sha256") != _sha(src.get("identity")) or
            src.get("identity_sha256") != _sha(current_identity)):
        raise ValueError("campaign manifest source binding is stale or dirty")
    protocol = manifest.get("qualification_protocol", {})
    path, parsed, sha, seeds = _protocol(protocol.get("path", ""))
    if (sha != protocol.get("sha256") or parsed["protocol_id"] != protocol.get("protocol_id")
            or seeds != protocol.get("split_seed_partitions") or
            parsed.get("master_seed") != manifest.get("seed")):
        raise ValueError("campaign manifest protocol binding is stale")
    if manifest.get("qualification_gate_calibration") != _gate_calibration(parsed):
        raise ValueError("qualification gate calibration binding is stale")
    if set(manifest.get("tracks", {})) != set(MECO_TRACKS):
        raise ValueError("campaign manifest must contain the exact five-track map")
    entries = manifest["tracks"]
    for track in MECO_TRACKS:
        entry = entries[track]
        cfg = entry.get("config")
        if not isinstance(cfg, dict) or cfg.get("track") != track or _sha(cfg) != entry.get("config_sha256"):
            raise ValueError(f"{track} configuration hash/binding mismatch")
        if _sha(_model_schema(track)["parameters"]) != entry.get("model_schema", {}).get("sha256"):
            raise ValueError(f"{track} model schema binding mismatch")
        _validate_config_metadata(MecoTrainConfig(**cfg))
    reference = entries[MECO_TRACKS[0]]["config"]
    expected_shared_fields = set(reference) - {"track", "device", "output"}
    shared = manifest.get("shared_training_fields", {})
    if set(shared) != expected_shared_fields or any(
            entries[t]["config"][name] != reference[name]
            for t in MECO_TRACKS for name in expected_shared_fields):
        raise ValueError("shared training fields are incomplete or differ across tracks")
    devices = manifest.get("devices", {})
    training = devices.get("training_by_track", {})
    if set(training) != set(MECO_TRACKS):
        raise ValueError("device map must bind every exact MECO track")
    pool = devices.get("training_pool")
    if not isinstance(pool, list):
        raise ValueError("device lease pool must be an indexed CUDA list")
    indices = [_device_index(value) for value in pool]
    if len(pool) != 2:
        raise ValueError("device lease pool must contain the two indexed WX9100 devices")
    if len(set(indices)) != len(indices):
        raise ValueError("device lease map contains duplicate indexed GPUs")
    if set(pool) != set(EXPECTED_WX9100_BDFS) or devices.get("evaluation") not in pool:
        raise ValueError("device lease map contains undeclared indexed WX9100 GPUs")
    _device_index(devices.get("evaluation"))
    if devices.get("indexed_device_leases") != {
            "pool": pool, "training_by_track": training,
            "evaluation": devices.get("evaluation"), "contract": LEASE_CONTRACT}:
        raise ValueError("serialized device leases do not match the exclusive lease contract")
    if any(training[t] not in pool for t in MECO_TRACKS):
        raise ValueError("a track assignment lies outside the declared lease pool")
    if training != {t: entries[t]["config"].get("device") for t in MECO_TRACKS}:
        raise ValueError("device assignments disagree with per-track configs")
    inventory, host_inventory = _device_inventory(pool)
    if inventory != manifest.get("device_inventory"):
        raise ValueError("device inventory binding is malformed")
    if host_inventory != manifest.get("host_gpu_inventory"):
        raise ValueError("host GPU inventory path/hash/contents changed")
    scheduler = manifest.get("scheduler", {})
    if scheduler.get("seed_partitions") != seeds or scheduler.get("master_seed") != manifest.get("seed"):
        raise ValueError("scheduler seed partitions/master seed differ from frozen protocol")
    assignment_contract = ("sequential-cycle deterministic curriculum assignment; all five tracks share "
        "scenario/category/seed; train seed indexed by cycle modulo frozen protocol pool")
    if (scheduler.get("algorithm") != "MecoTaskScheduler" or scheduler.get("version") != 1 or
            scheduler.get("task_assignment_contract") != assignment_contract):
        raise ValueError("scheduler algorithm/task assignment contract is invalid")
    evaluation = manifest.get("evaluation_protocol", {})
    _, eval_protocol, eval_sha, eval_partitions = _protocol(evaluation.get("path", ""))
    if (eval_sha != evaluation.get("sha256") or
            eval_protocol.get("protocol_id") != evaluation.get("protocol_id") or
            eval_partitions != evaluation.get("split_seed_partitions")):
        raise ValueError("evaluation protocol bytes/ID/splits are stale")
    if evaluation.get("gate_calibration") != _gate_calibration(eval_protocol):
        raise ValueError("evaluation gate calibration binding is stale")
    eval_split = evaluation.get("split")
    if (eval_split not in ("diagnostic", "validation", "test") or
            not isinstance(evaluation.get("scenario_id"), str) or
            not isinstance(evaluation.get("category"), str) or
            not isinstance(evaluation.get("episode_seeds"), list) or
            not evaluation["episode_seeds"] or
            any(type(seed) is not int or seed < 0 for seed in evaluation["episode_seeds"])):
        raise ValueError("evaluation split/case/seeds are malformed")
    qualification_eligible = False
    if evaluation.get("qualification_eligible") is not qualification_eligible:
        raise ValueError("qualification eligibility disagrees with split and calibrated gates")
    eval_case_seeds = {seed for case in eval_protocol.get("cases", [])
        if case.get("scenario_id") == evaluation.get("scenario_id")
        and case.get("category") == evaluation.get("category")
        and case.get("split") == eval_split for seed in case.get("seeds", [])}
    episode_seeds = evaluation.get("episode_seeds", [])
    if ((eval_split == "diagnostic" and set(episode_seeds) & set(sum(eval_partitions.values(), []))) or
            (eval_split != "diagnostic" and not set(episode_seeds) <= eval_case_seeds)):
        raise ValueError("evaluation seed case no longer matches the bound protocol split")
    warm = manifest.get("warm_starts", {})
    if set(warm) != set(MECO_TRACKS):
        raise ValueError("warm-start map must contain all five tracks, including absent entries")
    for track, item in warm.items():
        current = _warm_start(item.get("path"), track, src.get("identity", {}),
            protocol.get("path"), protocol["protocol_id"], protocol["sha256"])
        if current != item:
            raise ValueError(f"{track} warm-start bytes, binding, or target coverage changed")
    all_covered = all(warm[t].get("phase2_covered") is True for t in MECO_TRACKS)
    readiness = manifest.get("readiness", {})
    hardware_ready = bool(host_inventory["verified"] and
        all(inventory[d].get("hardware_verified") for d in pool))
    implementation_ready = bool(expected["clean"] and hardware_ready and all(
        entries[t].get("config_sha256") == _sha(entries[t]["config"]) for t in MECO_TRACKS) and
        len(pool) == 2)
    if readiness.get("implementation_ready") is not implementation_ready:
        raise ValueError("implementation readiness does not match clean source/config bindings")
    if readiness.get("hardware_ready") is not hardware_ready:
        raise ValueError("hardware readiness does not match byte-verified device inventory")
    campaign = manifest.get("campaign", {})
    _validate_campaign_paths(campaign.get("id"), campaign.get("root"),
        {t: entries[t]["config"] for t in MECO_TRACKS},
        campaign.get("evaluation_output"), campaign.get("manifest_output"),
        check_existing_output=False, check_namespace_available=False)
    if campaign.get("manifest_output") is not None:
        try:
            persisted_manifest = json.loads(Path(campaign["manifest_output"]).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("persisted campaign manifest output is missing or malformed") from exc
        if persisted_manifest != manifest:
            raise ValueError("persisted campaign manifest differs from validated manifest bytes")
    validation_ready = bool(implementation_ready and eval_split == "validation")
    qualification_ready = bool(implementation_ready and all_covered and qualification_eligible)
    if readiness.get("validation_ready") is not validation_ready:
        raise ValueError("validation readiness does not match the frozen validation protocol")
    if readiness.get("qualification_ready") is not qualification_ready:
        raise ValueError("qualification readiness does not match calibrated gates and Phase 2 coverage")
    if readiness.get("launch_ready") is not False or not isinstance(
            readiness.get("launch_blocker"), str) or not readiness["launch_blocker"]:
        raise ValueError("manifest cannot claim launch readiness before supervisor integration")
    if readiness.get("phase2_ready") is not bool(implementation_ready and all_covered):
        raise ValueError("Phase 2 readiness does not match bound per-track target coverage")
    if readiness.get("phase2_complete") is not False:
        raise ValueError("manifest cannot claim Phase 2 completed without campaign evidence")
    return {"valid": True, "canonical_sha256": digest,
            "implementation_ready": manifest["readiness"]["implementation_ready"],
            "launch_ready": manifest["readiness"]["launch_ready"],
            "validation_ready": manifest["readiness"]["validation_ready"],
            "qualification_ready": manifest["readiness"]["qualification_ready"],
            "phase2_ready": manifest["readiness"]["phase2_ready"],
            "phase2_complete": manifest["readiness"]["phase2_complete"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Describe/validate an offline five-track MECO campaign")
    subs = parser.add_subparsers(dest="command", required=True)
    desc = subs.add_parser("describe")
    desc.add_argument("--config", required=True)
    desc.add_argument("--output", help="optional explicit manifest output path")
    desc.add_argument("--source-root", default=str(ROOT))
    val = subs.add_parser("validate")
    val.add_argument("manifest")
    val.add_argument("--source-root", default=str(ROOT))
    args = parser.parse_args(argv)
    if args.command == "describe":
        result = describe_campaign(json.loads(Path(args.config).read_text()),
            source_root=args.source_root, output=args.output)
    else:
        result = validate_campaign_manifest(json.loads(Path(args.manifest).read_text()),
            source_root=args.source_root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
