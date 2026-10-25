"""Coordinate a validated five-track Phase 2 warm-start artifact set.

Descriptor and report operations are metadata-only. Explicit execution invokes
the existing C1/C2 simulator campaign and Plan A U+B+P collector/fitter.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import uuid
import weakref
from datetime import datetime, timezone

from .meco_phase2_campaign import (_protocol, _source_snapshot, _sha,
    _canonical, _safe_device, describe_campaign, FIXED as C_FIXED,
    SCHEMA as C_SCHEMA)
from .curriculum_training import source_identity
from .meco_tracks import MECO_TRACKS

SCHEMA = "meco-phase2-five-track-complete-set-v1"
STAGE_C = "phase2-c"
STAGE_A = "plan-a-ubp"
MANIFEST_NAME = "complete-set-manifest.json"
STATE_NAME = "coordinator-state.json"


def _digest(value) -> str:
    return _sha(_canonical(value).encode())


def _sha_file(path: Path) -> str:
    return _sha(Path(path).read_bytes())


def _regular_nonsymlink(path: Path) -> Path:
    path = Path(path).absolute()
    cursor = path
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValueError(f"artifact path contains symlink: {cursor}")
        cursor = cursor.parent
    if not path.is_file():
        raise ValueError(f"artifact is not a regular file: {path}")
    return path


def _reject_symlink_path(path: Path, *, allow_missing_leaf=True) -> None:
    path = Path(path).absolute()
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"output path contains symlink: {cursor}")
        if not cursor.exists() and (not allow_missing_leaf or cursor != path):
            raise ValueError(f"output path ancestor is missing: {cursor}")


def _open_pinned_directory(path: Path) -> int:
    """Open every absolute path component with O_NOFOLLOW and return pinned fd."""
    path = Path(path).absolute()
    if not path.is_absolute():
        raise ValueError("output directory must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _verify_output_namespace(path: Path) -> None:
    """Fail closed on writable non-sticky ancestors; trust the coordinator UID."""
    path = Path(path).absolute()
    cursor = Path(path.anchor)
    euid = os.geteuid()
    for part in path.parts[1:]:
        cursor /= part
        stat = cursor.stat(follow_symlinks=False)
        if not cursor.is_dir() or cursor.is_symlink():
            raise ValueError(f"output namespace contains a non-directory or symlink: {cursor}")
        mode = stat.st_mode
        writable_by_others = bool(mode & 0o022)
        sticky = bool(mode & 0o1000)
        if writable_by_others and not sticky:
            raise ValueError(f"output namespace ancestor is writable by other users: {cursor}")
        if stat.st_uid not in (0, euid) and writable_by_others:
            raise ValueError(f"untrusted output namespace owner: {cursor}")


def _nonoverlap(path: Path, protected: Path) -> bool:
    path, protected = Path(path).absolute(), Path(protected).resolve()
    return not (path == protected or path in protected.parents or protected in path.parents)


def _expected_outputs(root: Path) -> dict:
    c_root, a_root = root / STAGE_C, root / STAGE_A
    artifacts = {
        "A1": a_root / "a1-plan-a-joint-ubp.pt",
        "A2": a_root / "a2-plan-a-joint-ubp.pt",
        "A3": a_root / "a3-plan-a-joint-ubp.pt",
        "C1": c_root / "C1.pt",
        "C2": c_root / "C2.pt",
    }
    return {"root": str(root), "manifest": str(root / MANIFEST_NAME),
        "state": str(root / STATE_NAME), "stage_journal": str(root / "stage-receipts.jsonl"),
        "source_dataset": str(c_root / "matched-prefix.jsonl"),
        "source_manifest": str(c_root / "matched-prefix.jsonl.manifest.json"),
        "c_campaign_report": str(c_root / "campaign-report.json"),
        "a_bundle": str(a_root / "bundle.json"),
        "artifacts": {track: str(artifacts[track]) for track in MECO_TRACKS}}


def describe_complete_set(*, protocol_path: Path, scenario_id: str, category: str,
        seeds: list[int], device: str, output_root: Path,
        repo_root: Path | None = None) -> dict:
    """Create a no-write immutable binding for one train dataset and five outputs."""
    repo = Path(repo_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    if repo != Path(__file__).resolve().parents[1]:
        raise ValueError("coordinator source root must be the current clean checkout")
    commit, source_sha = _source_snapshot(repo)
    protocol_file, _protocol_doc, protocol_id, protocol_sha = _protocol(protocol_path)
    device = _safe_device(device)
    root = Path(output_root).expanduser().absolute()
    if root.exists() or root.is_symlink() or not root.parent.is_dir():
        raise ValueError("complete-set output root must be absent under an existing directory")
    if root.parent.resolve() != root.parent or root.parent.is_symlink():
        raise ValueError("complete-set output parent must not contain symlinks")
    if root == repo or root in repo.parents or repo in root.parents:
        raise ValueError("complete-set outputs must not overlap source checkout")
    if root == protocol_file or root in protocol_file.parents or protocol_file in root.parents:
        raise ValueError("complete-set outputs must not overlap qualification protocol")
    from .meco_plan_a_ubp import _protected_paths
    if any(not _nonoverlap(root, protected)
            for protected in _protected_paths(protocol_file, protocol_file)):
        raise ValueError("complete-set attempt overlaps a protected campaign namespace")
    c_root = root / STAGE_C
    c_descriptor = describe_campaign(protocol_path=protocol_file,
        scenario_id=scenario_id, category=category, seeds=seeds,
        device=device, output_root=c_root, repo_root=repo)
    outputs = _expected_outputs(root)
    if set(outputs["artifacts"]) != set(MECO_TRACKS):
        raise AssertionError("internal artifact map must cover the exact five tracks")
    body = {"schema": SCHEMA, "source": c_descriptor["source"],
        "qualification_protocol": c_descriptor["qualification_protocol"],
        "scenario_id": scenario_id, "category": category,
        "split": "train", "seeds": list(seeds), "device": device,
        "tracks": list(MECO_TRACKS), "source_dataset_contract": {
            "producer_campaign_id": c_descriptor["campaign_id"],
            "split": "train", "protocol_id": protocol_id,
            "qualification_protocol_sha256": protocol_sha,
            "reuse_for": ["C1", "C2", "A1", "A2", "A3"]},
        "c_campaign_descriptor": c_descriptor,
        "plan_a_ubp": {"source_dataset": outputs["source_dataset"],
            "qualification_protocol": str(protocol_file), "output_dir": str(root / STAGE_A),
            "device": device},
        "outputs": outputs,
        "readiness": {"artifact_set_complete": False,
            "phase2_ready": False, "phase2_complete": False,
            "qualification_eligible": False}}
    body["attempt_number"] = _attempt_number(root.name)
    body["descriptor_sha256"] = _digest(body)
    return body


def _attempt_number(name: str) -> int:
    match = re.fullmatch(r"attempt-(\d{4,})", name)
    if match is None or int(match.group(1)) < 1:
        raise ValueError("output root must be a numbered attempt directory, e.g. attempt-0001")
    return int(match.group(1))


def validate_descriptor(descriptor: dict, *, repo_root: Path | None = None) -> dict:
    if not isinstance(descriptor, dict) or descriptor.get("schema") != SCHEMA:
        raise ValueError("unsupported complete-set descriptor")
    digest = descriptor.get("descriptor_sha256")
    body = {key: value for key, value in descriptor.items() if key != "descriptor_sha256"}
    if digest != _digest(body):
        raise ValueError("complete-set descriptor digest mismatch")
    repo = Path(repo_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    if repo != Path(__file__).resolve().parents[1]:
        raise ValueError("coordinator source root must be the current clean checkout")
    commit, source_sha = _source_snapshot(repo)
    if descriptor.get("source") != {"commit": commit, "source_tree_sha256": source_sha}:
        raise ValueError("complete-set descriptor source identity is stale")
    protocol_path, protocol_doc, protocol_id, protocol_sha = _protocol(
        Path(descriptor["qualification_protocol"]["path"]))
    expected_protocol = {"path": str(protocol_path), "id": protocol_id,
                         "sha256": protocol_sha}
    if descriptor.get("qualification_protocol") != expected_protocol:
        raise ValueError("complete-set descriptor protocol bytes/identity are stale")
    root = Path(descriptor.get("outputs", {}).get("root", "")).absolute()
    if (not root.is_absolute() or root.name != f"attempt-{descriptor.get('attempt_number', 0):04d}"
            or descriptor.get("attempt_number") != _attempt_number(root.name) or
            descriptor.get("outputs") != _expected_outputs(root)):
        raise ValueError("complete-set output destinations are not the frozen layout")
    from .meco_plan_a_ubp import _protected_paths
    if any(not _nonoverlap(root, protected)
            for protected in _protected_paths(protocol_path, protocol_path)):
        raise ValueError("complete-set attempt overlaps a protected campaign namespace")
    if descriptor.get("split") != "train" or descriptor.get("tracks") != list(MECO_TRACKS):
        raise ValueError("complete-set descriptor must bind train data and all five tracks")
    c = descriptor.get("c_campaign_descriptor")
    if (not isinstance(c, dict) or c.get("schema") != C_SCHEMA or
            c.get("source") != descriptor["source"] or
            c.get("qualification_protocol") != expected_protocol or
            c.get("scenario_id") != descriptor.get("scenario_id") or
            c.get("category") != descriptor.get("category") or
            c.get("split") != "train" or c.get("seeds") != descriptor.get("seeds") or
            c.get("device") != descriptor.get("device") or
            c.get("tracks") != ["C1", "C2"] or
            c.get("outputs") != {"root": str(root / STAGE_C),
                "dataset": descriptor["outputs"]["source_dataset"],
                "C1": descriptor["outputs"]["artifacts"]["C1"],
                "C2": descriptor["outputs"]["artifacts"]["C2"],
                "report": descriptor["outputs"]["c_campaign_report"]}):
        raise ValueError("embedded C campaign binding differs from complete-set descriptor")
    c_body = {key: value for key, value in c.items() if key != "descriptor_sha256"}
    if c.get("descriptor_sha256") != _digest(c_body):
        raise ValueError("embedded C campaign descriptor digest mismatch")
    if (c.get("source") != {"commit": commit, "source_tree_sha256": source_sha} or
            c.get("qualification_protocol") != expected_protocol or
            c.get("scenario_id") != descriptor.get("scenario_id") or
            c.get("category") != descriptor.get("category") or
            c.get("seeds") != descriptor.get("seeds") or
            c.get("device") != descriptor.get("device") or
            any(c.get(key) != value for key, value in C_FIXED.items())):
        raise ValueError("embedded C campaign configuration is stale or outside frozen contract")
    allowed = []
    matching = [case for case in protocol_doc.get("cases", []) if
        case.get("scenario_id") == descriptor.get("scenario_id") and
        case.get("category") == descriptor.get("category") and case.get("split") == "train"]
    for case in matching:
        case_seeds = case.get("seeds")
        if not isinstance(case_seeds, list) or any(type(seed) is not int or seed < 0
                for seed in case_seeds) or len(set(case_seeds)) != len(case_seeds):
            raise ValueError("qualification protocol train seeds are malformed")
        allowed.extend(case_seeds)
    if (not matching or len(set(allowed)) != len(allowed) or
            not isinstance(descriptor.get("seeds"), list) or not descriptor["seeds"] or
            any(type(seed) is not int or seed < 0 for seed in descriptor["seeds"]) or
            len(set(descriptor["seeds"])) != len(descriptor["seeds"]) or
            not set(descriptor["seeds"]) <= set(allowed)):
        raise ValueError("complete-set seeds are not exact unique members of the frozen train split")
    _safe_device(descriptor.get("device"))
    expected_campaign_id = _sha(_canonical({"commit": commit, "protocol": protocol_sha,
        "scenario": descriptor["scenario_id"], "category": descriptor["category"],
        "seeds": descriptor["seeds"], "device": descriptor["device"],
        "output_root": str(root / STAGE_C)}).encode())[:20]
    expected_c = {"schema": C_SCHEMA, "campaign_id": expected_campaign_id,
        "source": descriptor["source"], "qualification_protocol": expected_protocol,
        "scenario_id": descriptor["scenario_id"], "category": descriptor["category"],
        "split": "train", "seeds": descriptor["seeds"], "tracks": ["C1", "C2"],
        "device": descriptor["device"], **C_FIXED,
        "outputs": {"root": str(root / STAGE_C),
            "dataset": descriptor["outputs"]["source_dataset"],
            "C1": descriptor["outputs"]["artifacts"]["C1"],
            "C2": descriptor["outputs"]["artifacts"]["C2"],
            "report": descriptor["outputs"]["c_campaign_report"]},
        "phase2_ready": False, "qualification_eligible": False}
    expected_c["descriptor_sha256"] = _digest(expected_c)
    if c != expected_c:
        raise ValueError("embedded C campaign does not rederive from frozen inputs")
    plan_a = descriptor.get("plan_a_ubp", {})
    if plan_a != {"source_dataset": descriptor["outputs"]["source_dataset"],
            "qualification_protocol": str(protocol_path),
            "output_dir": str(root / STAGE_A), "device": descriptor.get("device")}:
        raise ValueError("Plan A U+B+P stage binding differs from the shared train source")
    if (descriptor.get("source_dataset_contract") != {
            "producer_campaign_id": c["campaign_id"], "split": "train",
            "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
            "reuse_for": ["C1", "C2", "A1", "A2", "A3"]} or
            descriptor.get("readiness") != {"artifact_set_complete": False,
                "phase2_ready": False, "phase2_complete": False,
                "qualification_eligible": False}):
        raise ValueError("complete-set source contract or readiness claims are malformed")
    return descriptor


def validate_complete_set_manifest(manifest: dict, descriptor: dict) -> dict:
    """Check complete-set identity and exact warm-start mapping without GPU use."""
    validate_descriptor(descriptor)
    if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or
            manifest.get("status") != "complete_validated_artifact_set" or
            manifest.get("descriptor_sha256") != descriptor["descriptor_sha256"] or
            manifest.get("attempt_number") != descriptor.get("attempt_number") or
            manifest.get("source") != descriptor["source"] or
            manifest.get("qualification_protocol") != descriptor["qualification_protocol"] or
            manifest.get("split") != "train" or manifest.get("seeds") != descriptor["seeds"] or
            manifest.get("scenario_id") != descriptor["scenario_id"] or
            manifest.get("category") != descriptor["category"] or
            manifest.get("device") != descriptor["device"] or
            manifest.get("tracks") != list(MECO_TRACKS) or
            manifest.get("readiness") != {"artifact_set_complete": True,
                "phase2_ready": False, "phase2_complete": False,
                "qualification_eligible": False} or
            manifest.get("claims") != {"simulator_outcome_qualified": False,
                "training_readiness_proven": False}):
        raise ValueError("complete-set manifest identity/readiness contract mismatch")
    artifacts = manifest.get("artifacts")
    expected_paths = descriptor["outputs"]["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(MECO_TRACKS):
        raise ValueError("complete-set manifest must contain exactly five artifacts")
    mapping = manifest.get("warm_start_artifacts")
    if not isinstance(mapping, dict) or set(mapping) != set(MECO_TRACKS):
        raise ValueError("complete-set warm-start mapping must contain all five tracks")
    for track in MECO_TRACKS:
        item = artifacts[track]
        if (not isinstance(item, dict) or item.get("path") != expected_paths[track] or
                mapping.get(track) != expected_paths[track] or
                not isinstance(item.get("sha256"), str) or
                re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None):
            raise ValueError(f"{track} artifact path/hash/warm-start mapping is malformed")
        path = _regular_nonsymlink(Path(expected_paths[track]))
        if _sha_file(path) != item["sha256"]:
            raise ValueError(f"{track} artifact bytes changed after validation")
    expected_sources = source_identity()
    native_identity = expected_sources.get("_native_fuel_extension")
    native_identity_sha = _digest(native_identity) if isinstance(native_identity, dict) else None
    runtime_source_sha = _digest(expected_sources)
    source_binding, c_validators = _load_c_stage(descriptor,
        expected_sources=expected_sources)
    a_validators = _load_a_stage(descriptor, expected_sources=expected_sources,
                                 source_binding=source_binding)
    source_record = manifest.get("source_dataset")
    if (not isinstance(source_record, dict) or
            source_record.get("path") != descriptor["outputs"]["source_dataset"] or
            source_record.get("dataset_sha256") != source_binding["dataset_sha256"] or
            source_record.get("manifest_sha256") != source_binding["dataset_manifest_sha256"] or
            source_record.get("record_count") != source_binding["record_count"]):
        raise ValueError("complete-set manifest source dataset record is not revalidated")
    stages = manifest.get("stages")
    c_stage = stages.get("C") if isinstance(stages, dict) else None
    a_stage = stages.get("A") if isinstance(stages, dict) else None
    if (not isinstance(c_stage, dict) or not isinstance(a_stage, dict) or
            set(stages) != {"C", "A"} or
            c_stage.get("report") != descriptor["outputs"]["c_campaign_report"] or
            c_stage.get("report_sha256") != _sha_file(_regular_nonsymlink(
                Path(descriptor["outputs"]["c_campaign_report"]))) or
            a_stage.get("bundle") != descriptor["outputs"]["a_bundle"] or
            a_stage.get("bundle_sha256") != _sha_file(_regular_nonsymlink(
                Path(descriptor["outputs"]["a_bundle"])) )):
        raise ValueError("complete-set stage report/bundle binding is stale")
    journal_path = Path(descriptor["outputs"]["stage_journal"])
    journal_records = _read_stage_journal(journal_path)
    journal_sha = _sha_file(journal_path)
    latest = {}
    for record in journal_records:
        latest[record["stage"]] = record
    if (manifest.get("stage_journal_sha256") != journal_sha or
            set(latest) != {"C", "A"} or
            any(latest[stage].get("attempt_id") != stages[stage].get("attempt_id") or
                _digest(latest[stage]) != stages[stage].get("receipt_sha256")
                for stage in ("C", "A"))):
        raise ValueError("complete-set manifest does not bind the immutable stage journal")
    _validate_stage_receipt_bindings(latest, descriptor, c_stage, a_stage,
        source_binding, c_validators, a_validators)
    if c_stage.get("validators") != c_validators or a_stage.get("validators") != a_validators:
        raise ValueError("complete-set stage validator receipts differ from live validation")
    expected_receipts = {}
    for track in ("C1", "C2"):
        receipt = c_validators[track]
        expected_receipts[track] = {"path": descriptor["outputs"]["artifacts"][track],
            "sha256": receipt["artifact_sha256"], **receipt}
    expected_receipts.update(a_validators)
    if (any(expected_receipts[track].get("source_identity_sha256") != runtime_source_sha
            for track in MECO_TRACKS) or
            any(expected_receipts[track].get("native_extension_identity_sha256") !=
                native_identity_sha for track in MECO_TRACKS)):
        raise ValueError("five artifacts do not share one runtime/native source identity")
    if manifest["artifacts"] != expected_receipts:
        raise ValueError("complete-set artifact receipts differ from revalidated outputs")
    if manifest.get("warm_start_artifacts") != {
            track: expected_receipts[track]["path"] for track in MECO_TRACKS}:
        raise ValueError("complete-set supervisor warm-start map is incomplete or stale")
    from .meco_training_manifest import source_identity_projection
    projection_sha = _digest(source_identity_projection(expected_sources))
    if (manifest.get("runtime_source_identity_sha256") != runtime_source_sha or
            manifest.get("runtime_source_projection_sha256") != projection_sha or
            manifest.get("native_extension_identity_sha256") != native_identity_sha):
        raise ValueError("complete-set runtime/native source identity binding is stale")
    if (manifest.get("source_dataset") != {"path": descriptor["outputs"]["source_dataset"],
            "dataset_sha256": source_binding["dataset_sha256"],
            "manifest_sha256": source_binding["dataset_manifest_sha256"],
            "record_count": source_binding["record_count"]}):
        raise ValueError("complete-set source dataset/protocol/hash claim is stale")
    return manifest


def export_warm_start_artifacts(manifest: dict, descriptor: dict) -> dict[str, str]:
    """Return the exact five-track paths after revalidating the complete seal."""
    validate_complete_set_manifest(manifest, descriptor)
    mapping = manifest.get("warm_start_artifacts")
    if not isinstance(mapping, dict) or set(mapping) != set(MECO_TRACKS):
        raise ValueError("complete-set seal has no exact five-track warm-start mapping")
    return {track: mapping[track] for track in MECO_TRACKS}


def validate_training_manifest_warm_starts(training_manifest: dict,
        complete_manifest: dict, descriptor: dict) -> dict:
    """Check a training manifest retained every sealed path/hash/source/protocol binding."""
    export_warm_start_artifacts(complete_manifest, descriptor)
    from .meco_training_manifest import (validate_campaign_manifest,
        source_identity_projection)
    validation = validate_campaign_manifest(training_manifest)
    if validation.get("valid") is not True:
        raise ValueError("training campaign manifest did not validate")
    warm = training_manifest.get("warm_starts")
    if not isinstance(warm, dict) or set(warm) != set(MECO_TRACKS):
        raise ValueError("training manifest warm starts must enumerate exactly five tracks")
    expected_projection_sha = _digest(source_identity_projection(source_identity()))
    if training_manifest.get("source", {}).get("identity_sha256") != expected_projection_sha:
        raise ValueError("training manifest source projection differs from validated artifacts")
    for track in MECO_TRACKS:
        item = warm[track]
        artifact = complete_manifest["artifacts"][track]
        if (item.get("present") is not True or item.get("binding_valid") is not True or
                item.get("path") != artifact["path"] or item.get("sha256") != artifact["sha256"] or
                item.get("artifact_schema") != artifact["artifact_schema"] or
                item.get("source_identity_sha256") != artifact["source_identity_sha256"] or
                item.get("protocol_id") != complete_manifest["qualification_protocol"]["id"] or
                item.get("protocol_sha256") != complete_manifest["qualification_protocol"]["sha256"]):
            raise ValueError(f"training manifest did not preserve {track} warm-start binding")
    return {"valid": True, "tracks": list(MECO_TRACKS),
        "qualification_eligible": False,
        "phase2_ready": False}


def _atomic_json(path: Path, value: dict):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temp).unlink(missing_ok=True)


def _replace_json(path: Path, value: dict):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def _append_stage_receipt(path: Path, receipt: dict) -> str:
    """Append and fsync one immutable stage receipt; return its canonical hash."""
    path = Path(path)
    encoded = (_canonical(receipt) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT |
        getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return _sha(encoded.rstrip(b"\n"))


def _read_stage_journal(path: Path) -> list[dict]:
    path = _regular_nonsymlink(path)
    records = []
    attempt_ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if (not isinstance(value, dict) or value.get("schema") != SCHEMA + ".stage-receipt" or
                not isinstance(value.get("attempt_id"), str) or
                value["attempt_id"] in attempt_ids or value.get("status") != "validated"):
            raise ValueError("stage journal contains a malformed, duplicate, or unvalidated receipt")
        attempt_ids.add(value["attempt_id"])
        records.append(value)
    return records


def _validate_stage_receipt_bindings(latest: dict, descriptor: dict, c_stage: dict,
        a_stage: dict, source_binding: dict, c_validators: dict,
        a_validators: dict) -> None:
    c_record, a_record = latest["C"], latest["A"]
    if (c_record.get("inputs", {}).get("protocol") != descriptor["qualification_protocol"] or
            c_record.get("inputs", {}).get("seeds") != descriptor["seeds"] or
            c_record.get("inputs", {}).get("scenario_id") != descriptor["scenario_id"] or
            c_record.get("inputs", {}).get("category") != descriptor["category"] or
            c_record.get("inputs", {}).get("device") != descriptor["device"] or
            c_record.get("outputs", {}).get("source_dataset") != {
                "path": descriptor["outputs"]["source_dataset"],
                "sha256": source_binding["dataset_sha256"]} or
            c_record.get("outputs", {}).get("source_manifest") != {
                "path": descriptor["outputs"]["source_manifest"],
                "sha256": source_binding["dataset_manifest_sha256"]} or
            c_record.get("outputs", {}).get("report") != {
                "path": descriptor["outputs"]["c_campaign_report"],
                "sha256": c_stage["report_sha256"]} or
            c_record.get("outputs", {}).get("artifacts") != {
                track: {"path": descriptor["outputs"]["artifacts"][track],
                    "sha256": c_validators[track]["artifact_sha256"]}
                for track in ("C1", "C2")} or
            c_record.get("validator_summary") != c_validators or
            a_record.get("inputs", {}).get("source_dataset") != {
                "path": descriptor["outputs"]["source_dataset"],
                "sha256": source_binding["dataset_sha256"]} or
            a_record.get("inputs", {}).get("source_manifest") != {
                "path": descriptor["outputs"]["source_manifest"],
                "sha256": source_binding["dataset_manifest_sha256"]} or
            a_record.get("inputs", {}).get("protocol") != descriptor["qualification_protocol"] or
            a_record.get("inputs", {}).get("scenario_id") != descriptor["scenario_id"] or
            a_record.get("inputs", {}).get("category") != descriptor["category"] or
            a_record.get("inputs", {}).get("seeds") != descriptor["seeds"] or
            a_record.get("inputs", {}).get("device") != descriptor["device"] or
            a_record.get("outputs", {}).get("bundle") != {
                "path": descriptor["outputs"]["a_bundle"],
                "sha256": a_stage["bundle_sha256"]} or
            a_record.get("outputs", {}).get("artifacts") != {
                track: {"path": descriptor["outputs"]["artifacts"][track],
                    "sha256": a_validators[track]["sha256"]}
                for track in ("A1", "A2", "A3")} or
            a_record.get("validator_summary") != a_validators):
        raise ValueError("immutable stage journal receipts differ from revalidated stage bindings")


def _record_or_reuse_stage_receipt(path: Path, receipt: dict) -> tuple[str, str]:
    """Preserve an identical validated receipt on restart; append only new identity."""
    records = _read_stage_journal(path) if Path(path).exists() else []
    stage = receipt["stage"]
    prior = [item for item in records if item.get("stage") == stage]
    candidate = {key: value for key, value in receipt.items() if key != "attempt_id"}
    if prior:
        latest = prior[-1]
        comparable = {key: value for key, value in latest.items() if key != "attempt_id"}
        if comparable == candidate:
            return latest["attempt_id"], _digest(latest)
    receipt = dict(receipt)
    receipt["attempt_id"] = uuid.uuid4().hex
    return receipt["attempt_id"], _append_stage_receipt(Path(path), receipt)


def _load_c_stage(descriptor: dict, *, expected_sources: dict) -> tuple[dict, dict]:
    from .meco_supervised import (_read_counterfactual_dataset,
        validate_counterfactual_warm_start_artifact)
    outputs = descriptor["outputs"]
    protocol = Path(descriptor["qualification_protocol"]["path"])
    dataset = Path(outputs["source_dataset"])
    _regular_nonsymlink(dataset)
    _regular_nonsymlink(Path(outputs["source_manifest"]))
    rows, dataset_sha, manifest_sha, _protocol_doc, protocol_id, protocol_sha, dataset_manifest = \
        _read_counterfactual_dataset(dataset, qualification_protocol=protocol,
                                     expected_sources=expected_sources)
    if (dataset_manifest.get("split") != "train" or
            dataset_manifest.get("seeds") != descriptor["seeds"] or
            dataset_manifest.get("scenario_id") != descriptor["scenario_id"] or
            dataset_manifest.get("category") != descriptor["category"] or
            dataset_manifest.get("device", {}).get("device") != descriptor["device"] or
            protocol_id != descriptor["qualification_protocol"]["id"] or
            protocol_sha != descriptor["qualification_protocol"]["sha256"]):
        raise ValueError("shared source dataset differs from the frozen train descriptor")
    if dataset_manifest.get("device", {}).get("device") != descriptor["device"]:
        raise ValueError("shared dataset collection device differs from indexed descriptor device")
    report_path = _regular_nonsymlink(Path(outputs["c_campaign_report"]))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("status") != "artifacts_validated" or
            report.get("phase2_ready") is not False or
            report.get("qualification_eligible") is not False or
            report.get("descriptor_sha256") != descriptor["c_campaign_descriptor"]["descriptor_sha256"]):
        raise ValueError("C campaign is not a complete validated stage")
    stage_rows = report.get("stages")
    if (not isinstance(stage_rows, list) or
            [row.get("track") for row in stage_rows if isinstance(row, dict)
             and row.get("stage") == "fit_and_validate"] != ["C1", "C2"]):
        raise ValueError("C campaign report does not prove both exact track validators passed")
    collection_rows = [row for row in stage_rows if isinstance(row, dict)
        and row.get("stage") == "collection"]
    if (len(collection_rows) != 1 or
            collection_rows[0].get("record_count") != len(rows) or
            collection_rows[0].get("dataset_sha256") != dataset_sha or
            collection_rows[0].get("manifest_sha256") != manifest_sha):
        raise ValueError("C report collection receipt differs from strict dataset validation")
    validators = {}
    for track in ("C1", "C2"):
        artifact = _regular_nonsymlink(Path(outputs["artifacts"][track]))
        payload = validate_counterfactual_warm_start_artifact(track, artifact,
            qualification_protocol=protocol, expected_sources=expected_sources,
            expected_prediction_mode="cv")
        if payload.get("dataset_sha256") != dataset_sha:
            raise ValueError(f"{track} does not bind the shared C train dataset")
        if (payload.get("dataset_manifest_sha256") != manifest_sha or
                payload.get("split") != "train" or
                payload.get("scenario_id") != descriptor["scenario_id"] or
                payload.get("category") != descriptor["category"] or
                payload.get("seeds") != descriptor["seeds"] or
                payload.get("protocol_id") != protocol_id or
                payload.get("qualification_protocol_sha256") != protocol_sha or
                payload.get("collection_device", {}).get("device") != descriptor["device"] or
                payload.get("training_device") != payload.get("collection_device")):
            raise ValueError(f"{track} artifact task/device/protocol identity differs from descriptor")
        validators[track] = {"artifact_sha256": _sha_file(artifact),
            "dataset_sha256": dataset_sha, "dataset_manifest_sha256": manifest_sha,
            "split": "train", "scenario_id": descriptor["scenario_id"],
            "category": descriptor["category"], "seeds": descriptor["seeds"],
            "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
            "source_identity_sha256": payload["source_identity_sha256"],
            "native_extension_identity_sha256": (_digest(expected_sources["_native_fuel_extension"])
                if isinstance(expected_sources.get("_native_fuel_extension"), dict) else None),
            "artifact_schema": payload["schema"],
            "device": descriptor["device"], "updates": payload["updates"],
            "non_tied_pair_count": payload["non_tied_pair_count"]}
        report_row = next(row for row in stage_rows if isinstance(row, dict)
            and row.get("stage") == "fit_and_validate" and row.get("track") == track)
        if (report_row.get("status") != "passed" or
                report_row.get("artifact_sha256") != validators[track]["artifact_sha256"] or
                report_row.get("dataset_sha256") != dataset_sha or
                report_row.get("updates") != payload["updates"] or
                report_row.get("non_tied_pair_count") != payload["non_tied_pair_count"]):
            raise ValueError(f"{track} campaign receipt differs from strict artifact validation")
    return {"dataset_sha256": dataset_sha, "dataset_manifest_sha256": manifest_sha,
        "record_count": len(rows), "protocol_id": protocol_id,
        "qualification_protocol_sha256": protocol_sha}, validators


def _load_a_stage(descriptor: dict, *, expected_sources: dict,
                  source_binding: dict) -> dict:
    from .meco_supervised import validate_plan_a_joint_ubp_artifact
    outputs = descriptor["outputs"]
    protocol = Path(descriptor["qualification_protocol"]["path"])
    bundle_path = _regular_nonsymlink(Path(outputs["a_bundle"]))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle_artifacts = bundle.get("artifacts")
    if (bundle.get("schema") != "meco-plan-a-joint-ubp-bundle-v1" or
            bundle.get("status") != "complete_validated_artifact_bundle" or
            bundle.get("qualification_eligible") is not False or
            bundle.get("phase2_ready") is not False or
            bundle.get("phase2_complete") is not False or
            bundle.get("binding", {}).get("source_dataset_sha256") != source_binding["dataset_sha256"] or
            bundle.get("binding", {}).get("source_manifest_sha256") != source_binding["dataset_manifest_sha256"] or
            not isinstance(bundle_artifacts, list) or
            [item.get("track") for item in bundle_artifacts if isinstance(item, dict)] !=
                ["A1", "A2", "A3"]):
        raise ValueError("Plan A U+B+P bundle is incomplete or not bound to the shared source")
    artifacts = {}
    for track in ("A1", "A2", "A3"):
        path = _regular_nonsymlink(Path(outputs["artifacts"][track]))
        payload = validate_plan_a_joint_ubp_artifact(track, path,
            qualification_protocol=protocol, expected_sources=expected_sources,
            expected_prediction_mode="cv")
        if (payload.get("source_dataset_sha256") != source_binding["dataset_sha256"] or
                payload.get("source_manifest_sha256") != source_binding["dataset_manifest_sha256"] or
                payload.get("split") != "train" or
                payload.get("scenario_id") != descriptor["scenario_id"] or
                payload.get("category") != descriptor["category"] or
                payload.get("seeds") != descriptor["seeds"] or
                payload.get("protocol_id") != descriptor["qualification_protocol"]["id"] or
                payload.get("qualification_protocol_sha256") !=
                    descriptor["qualification_protocol"]["sha256"] or
                payload.get("collection_device", {}).get("device") != descriptor["device"] or
                payload.get("training_device") != payload.get("collection_device")):
            raise ValueError(f"{track} does not bind the shared C train dataset")
        artifacts[track] = {"path": str(path), "sha256": _sha_file(path),
            "source_dataset_sha256": payload["source_dataset_sha256"],
            "source_manifest_sha256": payload["source_manifest_sha256"],
            "split": "train", "scenario_id": descriptor["scenario_id"],
            "category": descriptor["category"], "seeds": descriptor["seeds"],
            "protocol_id": descriptor["qualification_protocol"]["id"],
            "qualification_protocol_sha256": descriptor["qualification_protocol"]["sha256"],
            "source_identity_sha256": payload["source_identity_sha256"],
            "native_extension_identity_sha256": (_digest(expected_sources["_native_fuel_extension"])
                if isinstance(expected_sources.get("_native_fuel_extension"), dict) else None),
            "artifact_schema": payload["schema"],
            "device": descriptor["device"],
            "updates": payload["updates"]}
        bundle_item = next(item for item in bundle_artifacts if item["track"] == track)
        if (bundle_item.get("path") != path.name or
                bundle_item.get("sha256") != artifacts[track]["sha256"]):
            raise ValueError(f"{track} bundle entry differs from its validated artifact")
        bundle_item = next(item for item in bundle_artifacts if item["track"] == track)
        if (bundle_item.get("path") != path.name or
                bundle_item.get("sha256") != artifacts[track]["sha256"]):
            raise ValueError(f"{track} bundle entry differs from its validated artifact")
    return artifacts


def execute_complete_set(descriptor: dict, *, repo_root: Path | None = None,
        c_execute=None, a_execute=None, lease_factory=None) -> dict:
    """Run/resume C and Plan A, sealing only after five validators pass.

    Output paths stay pathname-based for the existing stage APIs. The pinned
    directory descriptors and ownership/permission checks detect namespace
    substitution at every stage boundary and before coordinator writes. As
    with other same-account project files, a hostile process running under the
    coordinator's own UID is trusted and may still race a pathname operation.
    """
    validate_descriptor(descriptor, repo_root=repo_root)
    outputs = descriptor["outputs"]
    root = Path(outputs["root"])
    state_path, manifest_path = Path(outputs["state"]), Path(outputs["manifest"])
    if root.exists():
        if root.is_symlink():
            raise ValueError("existing coordinator root lacks a safe matching resume state")
        state_path = _regular_nonsymlink(state_path)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (state.get("schema") != SCHEMA or
                state.get("descriptor_sha256") != descriptor["descriptor_sha256"]):
            raise ValueError("existing coordinator state is not bound to this descriptor")
        if manifest_path.exists():
            return validate_complete_set_manifest(
                json.loads(_regular_nonsymlink(manifest_path).read_text(encoding="utf-8")), descriptor)
    elif manifest_path.exists():
        raise ValueError("complete-set manifest exists without its attempt directory")

    from .meco_phase2_campaign import execute_campaign
    from .meco_supervised import _read_counterfactual_dataset
    from .curriculum_training import source_identity
    if lease_factory is None:
        from .meco_device_lease import lease_devices
        lease_factory = lease_devices
    expected_sources = source_identity()
    c_root = Path(descriptor["c_campaign_descriptor"]["outputs"]["root"])
    c_report_path = Path(outputs["c_campaign_report"])

    # Acquire the exclusive device lease before reserving the numbered attempt
    # directory or its mutable state journal. Nested APIs borrow this lease.
    with lease_factory([descriptor["device"]],
            owner=f"phase2-five-track-{descriptor['attempt_number']:04d}"):
        validate_descriptor(descriptor, repo_root=repo_root)
        _reject_symlink_path(root.parent, allow_missing_leaf=False)
        _reject_symlink_path(root)
        _verify_output_namespace(root.parent)
        if not root.parent.is_dir():
            raise ValueError("complete-set output parent disappeared before lease acquisition")
        parent_fd = _open_pinned_directory(root.parent)
        try:
            if root.exists():
                if root.is_symlink() or not root.is_dir():
                    raise ValueError("existing complete-set root is not a safe directory")
                root_fd = os.open(root.name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                    getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            else:
                os.mkdir(root.name, mode=0o700, dir_fd=parent_fd)
                root_fd = os.open(root.name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                    getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            root_identity = os.fstat(root_fd)
        except BaseException:
            os.close(parent_fd)
            raise
        # Keep both directory descriptors alive until this execution returns.
        # CPython finalizes these guards on exceptional exits as well.
        class _FDGuard:
            pass
        parent_guard, root_guard = _FDGuard(), _FDGuard()
        parent_finalizer = weakref.finalize(parent_guard, os.close, parent_fd)
        root_finalizer = weakref.finalize(root_guard, os.close, root_fd)
        def check_pinned_root():
            current_parent = root.parent.stat(follow_symlinks=False)
            pinned_parent = os.fstat(parent_fd)
            root_entry = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
            if (current_parent.st_dev != pinned_parent.st_dev or
                    current_parent.st_ino != pinned_parent.st_ino or
                    root_entry.st_dev != root_identity.st_dev or
                    root_entry.st_ino != root_identity.st_ino or
                    not stat.S_ISDIR(root_entry.st_mode)):
                raise ValueError("output parent/root was replaced after pinning")
            _reject_symlink_path(root, allow_missing_leaf=False)
            current = root.stat(follow_symlinks=False)
            if (current.st_dev != root_identity.st_dev or
                    current.st_ino != root_identity.st_ino):
                raise ValueError("complete-set output root changed after it was pinned")
        if root.exists():
            state = json.loads(_regular_nonsymlink(state_path).read_text(encoding="utf-8"))
            if (state.get("schema") != SCHEMA or
                    state.get("descriptor_sha256") != descriptor["descriptor_sha256"]):
                raise ValueError("attempt state changed while acquiring device lease")
        else:
            state = {"schema": SCHEMA,
                "descriptor_sha256": descriptor["descriptor_sha256"],
                "attempt_number": descriptor["attempt_number"],
                "status": "running", "completed_stages": []}
            check_pinned_root()
            _atomic_json(state_path, state)
            check_pinned_root()

        _reject_symlink_path(c_root)
        c_complete = c_report_path.is_file() and c_root.is_dir()
        if c_complete:
            source_binding, c_validators = _load_c_stage(descriptor,
                expected_sources=expected_sources)
        else:
            if c_root.exists() or c_root.is_symlink():
                raise ValueError("partial C campaign exists; refusing to overwrite or infer completion")
            borrowed_lease = lambda devices, owner=None: nullcontext()
            if c_execute is None:
                check_pinned_root()
                execute_campaign(descriptor["c_campaign_descriptor"],
                    lease_factory=borrowed_lease)
            else:
                check_pinned_root()
                c_execute(descriptor["c_campaign_descriptor"])
            check_pinned_root()
            source_binding, c_validators = _load_c_stage(descriptor,
                expected_sources=expected_sources)
            c_attempt_id = uuid.uuid4().hex
            c_receipt = {"schema": SCHEMA + ".stage-receipt", "stage": "C",
                "attempt_id": c_attempt_id, "status": "validated",
                "inputs": {"protocol": {"path": descriptor["qualification_protocol"]["path"],
                    "id": descriptor["qualification_protocol"]["id"],
                    "sha256": descriptor["qualification_protocol"]["sha256"]},
                    "seeds": descriptor["seeds"], "scenario_id": descriptor["scenario_id"],
                    "category": descriptor["category"], "device": descriptor["device"]},
                "outputs": {"source_dataset": {"path": outputs["source_dataset"],
                    "sha256": source_binding["dataset_sha256"]},
                    "source_manifest": {"path": outputs["source_manifest"],
                    "sha256": source_binding["dataset_manifest_sha256"]},
                    "report": {"path": outputs["c_campaign_report"],
                    "sha256": _sha_file(c_report_path)},
                    "artifacts": {track: {"path": outputs["artifacts"][track],
                        "sha256": c_validators[track]["artifact_sha256"]}
                        for track in ("C1", "C2")}},
                "validator_summary": c_validators}
            check_pinned_root()
            c_attempt_id, c_receipt_sha = _record_or_reuse_stage_receipt(
                Path(outputs["stage_journal"]), c_receipt)
            state.setdefault("stage_receipts", {})["C"] = {
                "attempt_id": c_attempt_id, "receipt_sha256": c_receipt_sha}
            check_pinned_root()
            state["completed_stages"] = ["C1", "C2"]
            state["c_campaign_id"] = descriptor["c_campaign_descriptor"]["campaign_id"]
            _replace_json(state_path, state)
            check_pinned_root()
        if c_complete:
            c_attempt_id = uuid.uuid4().hex
            c_receipt = {"schema": SCHEMA + ".stage-receipt", "stage": "C",
                "attempt_id": c_attempt_id, "status": "validated",
                "inputs": {"protocol": {"path": descriptor["qualification_protocol"]["path"],
                    "id": descriptor["qualification_protocol"]["id"],
                    "sha256": descriptor["qualification_protocol"]["sha256"]},
                    "seeds": descriptor["seeds"], "scenario_id": descriptor["scenario_id"],
                    "category": descriptor["category"], "device": descriptor["device"]},
                "outputs": {"source_dataset": {"path": outputs["source_dataset"],
                    "sha256": source_binding["dataset_sha256"]},
                    "source_manifest": {"path": outputs["source_manifest"],
                    "sha256": source_binding["dataset_manifest_sha256"]},
                    "report": {"path": outputs["c_campaign_report"],
                    "sha256": _sha_file(c_report_path)},
                    "artifacts": {track: {"path": outputs["artifacts"][track],
                        "sha256": c_validators[track]["artifact_sha256"]}
                        for track in ("C1", "C2")}},
                "validator_summary": c_validators}
            c_attempt_id, c_receipt_sha = _record_or_reuse_stage_receipt(
                Path(outputs["stage_journal"]), c_receipt)
            state.setdefault("stage_receipts", {})["C"] = {
                "attempt_id": c_attempt_id, "receipt_sha256": c_receipt_sha}
            _replace_json(state_path, state)
            check_pinned_root()

        a_root = Path(descriptor["plan_a_ubp"]["output_dir"])
        a_bundle_path = Path(outputs["a_bundle"])
        _reject_symlink_path(a_root)
        a_complete = a_bundle_path.is_file() and a_root.is_dir()
        if a_complete:
            a_artifacts = _load_a_stage(descriptor, expected_sources=expected_sources,
                                        source_binding=source_binding)
        else:
            if a_root.exists() or a_root.is_symlink():
                raise ValueError("partial Plan A bundle exists; refusing to overwrite or infer completion")
            from .meco_plan_a_ubp import orchestrate_plan_a_ubp, _load_adapters
            runner = a_execute or orchestrate_plan_a_ubp
            if a_execute is None:
                adapters = _load_adapters()
                adapters["lease_devices"] = borrowed_lease
                check_pinned_root()
                runner(Path(outputs["source_dataset"]),
                    Path(descriptor["qualification_protocol"]["path"]), a_root,
                    device=descriptor["device"], _adapters=adapters)
            else:
                check_pinned_root()
                runner(Path(outputs["source_dataset"]),
                    Path(descriptor["qualification_protocol"]["path"]), a_root,
                    device=descriptor["device"])
            check_pinned_root()
            a_artifacts = _load_a_stage(descriptor, expected_sources=expected_sources,
                                        source_binding=source_binding)
            a_attempt_id = uuid.uuid4().hex
            a_receipt = {"schema": SCHEMA + ".stage-receipt", "stage": "A",
                "attempt_id": a_attempt_id, "status": "validated",
                "inputs": {"source_dataset": {"path": outputs["source_dataset"],
                    "sha256": source_binding["dataset_sha256"]},
                    "source_manifest": {"path": outputs["source_manifest"],
                    "sha256": source_binding["dataset_manifest_sha256"]},
                    "protocol": {"path": descriptor["qualification_protocol"]["path"],
                    "id": descriptor["qualification_protocol"]["id"],
                    "sha256": descriptor["qualification_protocol"]["sha256"]},
                    "scenario_id": descriptor["scenario_id"],
                    "category": descriptor["category"], "seeds": descriptor["seeds"],
                    "device": descriptor["device"]},
                "outputs": {"bundle": {"path": outputs["a_bundle"],
                    "sha256": _sha_file(a_bundle_path)},
                    "artifacts": {track: {"path": outputs["artifacts"][track],
                        "sha256": a_artifacts[track]["sha256"]}
                        for track in ("A1", "A2", "A3")}},
                "validator_summary": a_artifacts}
            check_pinned_root()
            a_attempt_id, a_receipt_sha = _record_or_reuse_stage_receipt(
                Path(outputs["stage_journal"]), a_receipt)
            state.setdefault("stage_receipts", {})["A"] = {
                "attempt_id": a_attempt_id, "receipt_sha256": a_receipt_sha}
            check_pinned_root()
            state["completed_stages"] = ["C1", "C2", "A1", "A2", "A3"]
            _replace_json(state_path, state)
            check_pinned_root()
        if a_complete:
            a_attempt_id = uuid.uuid4().hex
            a_receipt = {"schema": SCHEMA + ".stage-receipt", "stage": "A",
                "attempt_id": a_attempt_id, "status": "validated",
                "inputs": {"source_dataset": {"path": outputs["source_dataset"],
                    "sha256": source_binding["dataset_sha256"]},
                    "source_manifest": {"path": outputs["source_manifest"],
                    "sha256": source_binding["dataset_manifest_sha256"]},
                    "protocol": {"path": descriptor["qualification_protocol"]["path"],
                    "id": descriptor["qualification_protocol"]["id"],
                    "sha256": descriptor["qualification_protocol"]["sha256"]},
                    "scenario_id": descriptor["scenario_id"],
                    "category": descriptor["category"], "seeds": descriptor["seeds"],
                    "device": descriptor["device"]},
                "outputs": {"bundle": {"path": outputs["a_bundle"],
                    "sha256": _sha_file(a_bundle_path)},
                    "artifacts": {track: {"path": outputs["artifacts"][track],
                        "sha256": a_artifacts[track]["sha256"]}
                        for track in ("A1", "A2", "A3")}},
                "validator_summary": a_artifacts}
            a_attempt_id, a_receipt_sha = _record_or_reuse_stage_receipt(
                Path(outputs["stage_journal"]), a_receipt)
            state.setdefault("stage_receipts", {})["A"] = {
                "attempt_id": a_attempt_id, "receipt_sha256": a_receipt_sha}
            _replace_json(state_path, state)
            check_pinned_root()

        artifact_records = {}
        for track in MECO_TRACKS:
            path = Path(outputs["artifacts"][track])
            if track.startswith("C"):
                receipt = c_validators[track]
                record = {"path": str(path), "sha256": receipt["artifact_sha256"], **receipt}
            else:
                record = a_artifacts[track]
            if not _regular_nonsymlink(path) or _sha_file(path) != record["sha256"]:
                raise ValueError(f"{track} artifact changed after validation")
            artifact_records[track] = record
        rows, current_dataset_sha, current_manifest_sha, _p, protocol_id, protocol_sha, data_manifest = \
            _read_counterfactual_dataset(Path(outputs["source_dataset"]),
                qualification_protocol=Path(descriptor["qualification_protocol"]["path"]),
                expected_sources=expected_sources)
        if (current_dataset_sha != source_binding["dataset_sha256"] or
                current_manifest_sha != source_binding["dataset_manifest_sha256"] or
                data_manifest.get("seeds") != descriptor["seeds"] or
                protocol_id != descriptor["qualification_protocol"]["id"] or
                protocol_sha != descriptor["qualification_protocol"]["sha256"]):
            raise ValueError("shared source dataset/protocol changed before complete-set publication")
        complete = {"schema": SCHEMA, "status": "complete_validated_artifact_set",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "descriptor_sha256": descriptor["descriptor_sha256"],
            "attempt_number": descriptor["attempt_number"],
            "source": descriptor["source"],
            "runtime_source_identity_sha256": _digest(expected_sources),
            "runtime_source_projection_sha256": _digest(
                __import__("frc_defense.meco_training_manifest",
                    fromlist=["source_identity_projection"]).source_identity_projection(expected_sources)),
            "native_extension_identity_sha256": (_digest(expected_sources["_native_fuel_extension"])
                if isinstance(expected_sources.get("_native_fuel_extension"), dict) else None),
            "qualification_protocol": descriptor["qualification_protocol"],
            "split": "train", "scenario_id": descriptor["scenario_id"],
            "category": descriptor["category"], "seeds": descriptor["seeds"],
            "device": descriptor["device"], "tracks": list(MECO_TRACKS),
            "source_dataset": {"path": outputs["source_dataset"],
                "dataset_sha256": source_binding["dataset_sha256"],
                "manifest_sha256": source_binding["dataset_manifest_sha256"],
                "record_count": source_binding["record_count"]},
            "stage_journal_sha256": _sha_file(Path(outputs["stage_journal"])),
            "stages": {"C": {"attempt_id": c_attempt_id,
                    "receipt_sha256": c_receipt_sha,
                    "report": outputs["c_campaign_report"],
                    "report_sha256": _sha_file(Path(outputs["c_campaign_report"])),
                    "validators": c_validators},
                "A": {"attempt_id": a_attempt_id,
                    "receipt_sha256": a_receipt_sha,
                    "bundle": outputs["a_bundle"],
                    "bundle_sha256": _sha_file(Path(outputs["a_bundle"])),
                    "validators": a_artifacts}},
            "artifacts": artifact_records,
            "warm_start_artifacts": {track: artifact_records[track]["path"]
                for track in MECO_TRACKS},
            "readiness": {"artifact_set_complete": True,
                "phase2_ready": False, "phase2_complete": False,
                "qualification_eligible": False},
            "claims": {"simulator_outcome_qualified": False,
                "training_readiness_proven": False}}
        validate_complete_set_manifest(complete, descriptor)
        check_pinned_root()
        _atomic_json(manifest_path, complete)
        check_pinned_root()
        state["status"] = "complete_validated_artifact_set"
        _replace_json(state_path, state)
        check_pinned_root()
        return complete

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    describe = sub.add_parser("describe")
    describe.add_argument("--protocol", type=Path, required=True)
    describe.add_argument("--scenario", required=True)
    describe.add_argument("--category", required=True)
    describe.add_argument("--seeds", type=int, nargs="+", required=True)
    describe.add_argument("--device", required=True)
    describe.add_argument("--output-root", type=Path, required=True)
    describe.add_argument("--descriptor", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("descriptor", type=Path)
    execute = sub.add_parser("execute")
    execute.add_argument("descriptor", type=Path)
    export = sub.add_parser("warm-start-map")
    export.add_argument("descriptor", type=Path)
    export.add_argument("complete-set-manifest", type=Path)
    export.add_argument("--training-manifest", type=Path,
        help="also verify a frozen training manifest retained all five bindings")
    args = parser.parse_args(argv)
    if args.command == "describe":
        value = describe_complete_set(protocol_path=args.protocol,
            scenario_id=args.scenario, category=args.category, seeds=args.seeds,
            device=args.device, output_root=args.output_root)
        descriptor_path = Path(args.descriptor).expanduser().absolute()
        output_root = Path(value["outputs"]["root"]).absolute()
        protocol_file = Path(value["qualification_protocol"]["path"]).resolve()
        repo = Path(__file__).resolve().parents[1]
        if (not _nonoverlap(descriptor_path, output_root) or
                not _nonoverlap(descriptor_path, repo) or
                not _nonoverlap(descriptor_path, protocol_file) or
                descriptor_path.exists() or descriptor_path.is_symlink() or
                not descriptor_path.parent.is_dir()):
            raise ValueError("descriptor destination must be new and outside protected campaign paths")
        _atomic_json(descriptor_path, value)
        print(json.dumps(value, indent=2, sort_keys=True))
        return value
    descriptor = json.loads(args.descriptor.read_text(encoding="utf-8"))
    validate_descriptor(descriptor)
    if args.command == "validate":
        result = {"valid": True, "descriptor_sha256": descriptor["descriptor_sha256"]}
    elif args.command == "warm-start-map":
        complete = json.loads(args.complete_set_manifest.read_text(encoding="utf-8"))
        result = {"warm_start_artifacts": export_warm_start_artifacts(complete, descriptor)}
        if args.training_manifest:
            training_manifest = json.loads(args.training_manifest.read_text(encoding="utf-8"))
            result["training_manifest_binding"] = validate_training_manifest_warm_starts(
                training_manifest, complete, descriptor)
    else:
        result = execute_complete_set(descriptor)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
