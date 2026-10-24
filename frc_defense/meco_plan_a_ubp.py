"""Offline orchestration and publication for joint Plan A U+B+P warm starts.

This module only coordinates the existing collectors, fitter, and validators. It
does not change their label or training contracts. Calling the orchestration
function performs simulator collection and GPU fitting; its unit tests inject
offline adapters and never invoke those operations.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import stat
from typing import Any


SCHEMA = "meco-plan-a-joint-ubp-bundle-v1"
TRACKS = ("A1", "A2", "A3")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(path: Path) -> tuple[dict[str, Any], str]:
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    raw = sidecar.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"manifest must be an object: {sidecar}")
    return value, hashlib.sha256(raw).hexdigest()


def _load_adapters():
    """Load real operations lazily so importing this module stays offline."""
    from .meco_counterfactual import (collect_plan_a_legal_nodes_v1,
        collect_plan_a_p_routes_v3)
    from .meco_supervised import (warm_start_plan_a_joint_ubp_v1,
        validate_plan_a_joint_ubp_artifact)
    from .meco_device_lease import lease_devices
    return {
        "collect_ub": collect_plan_a_legal_nodes_v1,
        "collect_p": collect_plan_a_p_routes_v3,
        "fit": warm_start_plan_a_joint_ubp_v1,
        "validate": validate_plan_a_joint_ubp_artifact,
        "lease_devices": lease_devices,
        "rebind": _rebind_artifact_dataset_paths,
        "prepare": _prepare_checkpoint_for_publication,
        "write_reports": _write_bundle_reports,
    }


def _rebind_artifact_dataset_paths(checkpoint: Path, *, legal_dataset: Path,
        source_dataset: Path, p_dataset: Path) -> None:
    """Rewrite staged absolute paths after atomic directory publication."""
    import torch
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("joint Plan A checkpoint must be a weights-only mapping")
    payload["legal_dataset_path"] = str(Path(legal_dataset).resolve())
    payload["source_dataset_path"] = str(Path(source_dataset).resolve())
    payload["p_dataset_path"] = str(Path(p_dataset).resolve())
    temporary = checkpoint.with_name(f".{checkpoint.name}.{secrets.token_hex(8)}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _verify_artifact_dataset_bindings(checkpoint: Path, expected: dict[str, str]) -> None:
    import torch
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("joint Plan A checkpoint must be a weights-only mapping")
    keys = ("legal_dataset_sha256", "legal_manifest_sha256",
        "source_dataset_sha256", "source_manifest_sha256",
        "p_dataset_sha256", "p_manifest_sha256",
        "legal_dataset_path", "source_dataset_path", "p_dataset_path")
    if any(payload.get(key) != expected.get(key) for key in keys):
        raise ValueError("rebound Plan A checkpoint does not preserve staged input hashes/paths")


def _prepare_checkpoint_for_publication(track: str, checkpoint: Path, *,
        staged_legal: Path, staged_source: Path, staged_p: Path,
        final_legal: Path, final_source: Path, final_p: Path,
        expected_hashes: dict[str, str], qualification_protocol: Path,
        expected_sources: dict, expected_prediction_mode: str,
        rebind, validate) -> None:
    """Validate staged bytes, then bind the publishable checkpoint to final paths."""
    checkpoint = Path(checkpoint)
    validation_copy = checkpoint.with_name(
        f".{checkpoint.name}.validation-{secrets.token_hex(8)}.tmp")
    try:
        shutil.copyfile(checkpoint, validation_copy)
        rebind(validation_copy, legal_dataset=staged_legal,
            source_dataset=staged_source, p_dataset=staged_p)
        validate(track, validation_copy,
            qualification_protocol=qualification_protocol,
            expected_sources=expected_sources,
            expected_prediction_mode=expected_prediction_mode)
    finally:
        validation_copy.unlink(missing_ok=True)
    rebind(checkpoint, legal_dataset=final_legal,
        source_dataset=final_source, p_dataset=final_p)
    expected = dict(expected_hashes)
    expected.update(legal_dataset_path=str(Path(final_legal).resolve()),
        source_dataset_path=str(Path(final_source).resolve()),
        p_dataset_path=str(Path(final_p).resolve()))
    _verify_artifact_dataset_bindings(checkpoint, expected)


def _write_bundle_reports(staging: Path, bundle: dict[str, Any], report: dict[str, Any]) -> None:
    (staging / "bundle.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    (staging / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")


def _fsync_tree(directory_fd: int) -> None:
    """Durably flush regular files and directories without following symlinks."""
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
        file_flags |= os.O_NOFOLLOW
    for name in os.listdir(directory_fd):
        item_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(item_stat.st_mode):
            child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
            try:
                _fsync_tree(child_fd)
                os.fsync(child_fd)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(item_stat.st_mode):
            file_fd = os.open(name, file_flags, dir_fd=directory_fd)
            try:
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
        else:
            raise ValueError(f"bundle staging contains a non-regular output: {name}")


def _fsync_directory(fd: int) -> None:
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.EROFS):
            raise


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _overlaps(first: Path, second: Path) -> bool:
    first, second = _absolute_lexical(first), _absolute_lexical(second)
    return (first == second or first in second.parents or second in first.parents)


def _reject_symlink_ancestors(path: Path) -> None:
    path = _absolute_lexical(path)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"output path has a symlink ancestor: {current}")


def _protected_paths(source_dataset: Path, qualification_protocol: Path) -> list[Path]:
    from .meco_training_manifest import PINNED_LIVE_ROOTS, ROOT
    return [Path(source_dataset).resolve(),
        Path(source_dataset).resolve().with_suffix(Path(source_dataset).suffix + ".manifest.json"),
        Path(qualification_protocol).resolve(),
        *(Path(root).resolve() for root in PINNED_LIVE_ROOTS), Path(ROOT).resolve()]


def _preflight_output(output_dir: Path, source_dataset: Path,
        qualification_protocol: Path) -> tuple[Path, Path]:
    """Reject unsafe namespaces before acquiring a device or creating anything."""
    output_dir = _absolute_lexical(output_dir)
    _reject_symlink_ancestors(output_dir.parent)
    claim_path = output_dir.parent / f".{output_dir.name}.claim"
    for protected in _protected_paths(source_dataset, qualification_protocol):
        if _overlaps(output_dir, protected) or _overlaps(claim_path, protected):
            raise ValueError(f"bundle output/staging namespace overlaps protected path: {protected}")
    if os.path.lexists(output_dir):
        raise FileExistsError(f"bundle output already exists: {output_dir}")
    if os.path.lexists(claim_path):
        raise FileExistsError(f"bundle output is already claimed: {output_dir}")
    if not output_dir.parent.is_dir():
        raise ValueError("bundle output parent must already exist as a directory")
    return output_dir, claim_path


def _open_directory_chain_nofollow(path: Path) -> int:
    """Open an absolute directory path component by component without symlinks."""
    path = _absolute_lexical(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
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


def _directory_path_matches_fd(path: Path, fd: int) -> bool:
    try:
        reopened = _open_directory_chain_nofollow(path)
    except OSError:
        return False
    try:
        expected, actual = os.fstat(fd), os.fstat(reopened)
        return expected.st_dev == actual.st_dev and expected.st_ino == actual.st_ino
    finally:
        os.close(reopened)


def _exclusive_claim_at(parent_fd: int, name: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "purpose": "meco-plan-a-ubp-bundle"}, stream,
                sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        os.unlink(name, dir_fd=parent_fd)
        raise


def _create_staging_at(parent_fd: int, output_name: str) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    for _ in range(10):
        name = f".{output_name}.staging-{secrets.token_hex(8)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            return name, os.open(name, flags, dir_fd=parent_fd)
        except BaseException:
            os.rmdir(name, dir_fd=parent_fd)
            raise
    raise FileExistsError("could not claim a fresh staging directory name")


def _rename_noreplace_at(parent_fd: int, source_name: str, destination_name: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication is unsupported")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(parent_fd, os.fsencode(source_name), parent_fd,
        os.fsencode(destination_name), 1)  # RENAME_NOREPLACE
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(f"bundle destination appeared before publication: {destination_name}")
        raise OSError(error, os.strerror(error), destination_name)


def _remove_tree_at(parent_fd: int, name: str) -> None:
    """Remove a claimed directory through pinned descriptors without following links."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    child_fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        for entry in os.listdir(child_fd):
            entry_stat = os.stat(entry, dir_fd=child_fd, follow_symlinks=False)
            if stat.S_ISDIR(entry_stat.st_mode):
                _remove_tree_at(child_fd, entry)
            else:
                os.unlink(entry, dir_fd=child_fd)
    finally:
        os.close(child_fd)
    os.rmdir(name, dir_fd=parent_fd)


def _target_counts(dataset: Path) -> dict[str, int]:
    u = b = 0
    by_seed: dict[int, dict[str, int]] = {}
    for line in dataset.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        seed = row.get("seed")
        counts = by_seed.setdefault(seed, {"u_targets": 0, "b_preferences": 0})
        u_here = sum(1 for target in row.get("node_u_targets", [])
            if target.get("status") == "available" and target.get("u_target") is not None)
        b_here = sum(1 for preference in row.get("behavior_preferences", [])
            if preference.get("status") == "comparable")
        u += u_here
        b += b_here
        counts["u_targets"] += u_here
        counts["b_preferences"] += b_here
    return {"u_targets": u, "b_preferences": b,
        "seed_counts": {str(seed): counts for seed, counts in sorted(by_seed.items(), key=lambda item: str(item[0]))}}


def _p_target_counts(dataset: Path) -> dict[str, int]:
    labels = comparable = groups = 0
    by_seed: dict[int, dict[str, int]] = {}
    for line in dataset.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        seed = row.get("seed")
        counts = by_seed.setdefault(seed, {"route_groups": 0, "oracle_labels": 0,
            "non_tied_route_pairs": 0})
        for group in row.get("plan_a_p_route_groups", []):
            groups += 1
            outcomes = group.get("outcomes", [])
            labels += len(outcomes)
            counts["route_groups"] += 1
            counts["oracle_labels"] += len(outcomes)
            unique: dict[str, float] = {}
            for outcome in outcomes:
                route = outcome.get("native_route", {})
                oracle = route.get("independent_cost_oracle", {})
                trace = route.get("route_trace_sha256")
                cost = oracle.get("route_cost_grid_edges")
                if isinstance(trace, str) and isinstance(cost, (int, float)):
                    unique.setdefault(trace, float(cost))
            costs = list(unique.values())
            pairs = sum(1 for i, left in enumerate(costs)
                for right in costs[i + 1:] if left != right)
            comparable += pairs
            counts["non_tied_route_pairs"] += pairs
    return {"route_groups": groups, "oracle_labels": labels,
        "non_tied_route_pairs": comparable,
        "seed_counts": {str(seed): counts for seed, counts in sorted(by_seed.items(), key=lambda item: str(item[0]))}}


def _check_shared_binding(source: dict[str, Any], ub: dict[str, Any],
        p: dict[str, Any], *, source_sha: str, source_manifest_sha: str,
        protocol_sha: str, p_cost_scale: float) -> dict[str, Any]:
    expected = {
        "split": "train",
        "seeds": source.get("seeds"),
        "scenario_id": source.get("scenario_id"),
        "category": source.get("category"),
        "protocol_id": source.get("protocol_id"),
        "qualification_protocol_sha256": protocol_sha,
        "sources": source.get("sources"),
        "source_identity_sha256": source.get("source_identity_sha256"),
        "device": source.get("device"),
        "prediction_mode": source.get("prediction_mode"),
        "prediction_contract": source.get("prediction_contract"),
    }
    if not isinstance(expected["seeds"], list) or not expected["seeds"]:
        raise ValueError("frozen source manifest must bind its train seeds")
    if source.get("split") != "train":
        raise ValueError("Plan A orchestration requires a train-split source dataset")
    if not isinstance(source.get("sources"), dict) or not isinstance(source.get("device"), dict):
        raise ValueError("source manifest lacks source or device identity")
    # The legal-node manifest does not duplicate seed/scenario headers; its
    # rows are bound and checked against those source fields separately.
    for name, manifest, keys in (("U+B", ub, tuple(k for k in expected if k not in
            ("seeds", "scenario_id", "category"))), ("P", p, tuple(expected))):
        for key in keys:
            value = expected[key]
            if manifest.get(key) != value:
                raise ValueError(f"{name} manifest differs from immutable source binding: {key}")
        if manifest.get("source_dataset_sha256", manifest.get("source_counterfactual_dataset_sha256")) != source_sha:
            raise ValueError(f"{name} manifest binds different source dataset bytes")
        if manifest.get("source_manifest_sha256", manifest.get("source_counterfactual_manifest_sha256")) != source_manifest_sha:
            raise ValueError(f"{name} manifest binds different source manifest bytes")
    if p.get("p_intervention_scale") != p_cost_scale:
        raise ValueError("P collector did not preserve the frozen intervention scale")
    if p.get("seeds") != expected["seeds"]:
        raise ValueError("P collector did not preserve the exact frozen train seed list")
    if ub.get("expert_goal_source") is not None:
        raise ValueError("orchestrated U collection must not claim expert U supervision")
    return {key: expected[key] for key in (
        "split", "seeds", "scenario_id", "category", "protocol_id",
        "qualification_protocol_sha256", "sources", "source_identity_sha256",
        "device", "prediction_mode", "prediction_contract")} | {
            "source_dataset_sha256": source_sha,
            "source_manifest_sha256": source_manifest_sha,
            "p_cost_scale": p_cost_scale,
        }


def orchestrate_plan_a_ubp(source_dataset: Path, qualification_protocol: Path,
        output_dir: Path, *, device: str, p_cost_scale: float = 1.0,
        intervention_maps: int = 3, updates: int = 10, learning_rate: float = 1e-4,
        minimum_u_targets: int = 1, minimum_b_preferences: int = 1,
        minimum_p_labels: int = 2, minimum_p_pairs: int = 1,
        minimum_selected_node_coverage: float = 0.01,
        minimum_p_coverage: float = 0.01, _adapters=None) -> dict[str, Any]:
    """Collect and fit one bound U+B+P campaign, publishing only a complete bundle.

    The output directory must not already exist. Intermediate datasets and
    checkpoints stay in a sibling staging directory; a final directory rename
    publishes the bundle only after all three artifacts validate.
    """
    source_dataset = Path(source_dataset).resolve()
    qualification_protocol = Path(qualification_protocol).resolve()
    output_dir, claim_path = _preflight_output(Path(output_dir), source_dataset,
        qualification_protocol)
    if not source_dataset.is_file() or not qualification_protocol.is_file():
        raise ValueError("source dataset and qualification protocol must exist")
    if (not isinstance(device, str) or not device.startswith("cuda:") or
            isinstance(p_cost_scale, bool) or not isinstance(p_cost_scale, (int, float)) or
            not math.isfinite(p_cost_scale) or p_cost_scale <= 0 or
            type(intervention_maps) is not int or not 1 <= intervention_maps <= 8 or
            type(updates) is not int or updates < 1 or
            isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or
            not math.isfinite(learning_rate) or learning_rate <= 0 or
            type(minimum_u_targets) is not int or minimum_u_targets < 1 or
            type(minimum_b_preferences) is not int or minimum_b_preferences < 1 or
            type(minimum_p_labels) is not int or minimum_p_labels < 2 or
            type(minimum_p_pairs) is not int or minimum_p_pairs < 1 or
            not math.isfinite(minimum_selected_node_coverage) or
            not 0 < minimum_selected_node_coverage <= 1 or
            not math.isfinite(minimum_p_coverage) or not 0 < minimum_p_coverage <= 1):
        raise ValueError("invalid Plan A U+B+P orchestration configuration")
    source_sha = _sha256(source_dataset)
    source_manifest, source_manifest_sha = _manifest(source_dataset)
    if source_manifest.get("dataset_sha256") != source_sha:
        raise ValueError("source dataset bytes do not match their immutable manifest digest")
    protocol_sha = _sha256(qualification_protocol)
    if source_manifest.get("qualification_protocol_sha256") != protocol_sha:
        raise ValueError("source dataset does not bind the exact qualification protocol bytes")
    adapters = _load_adapters() if _adapters is None else _adapters
    # Device ownership covers the full collect -> validate -> fit -> publish
    # interval. In particular, no claim, staging directory, or output is made
    # until after the shared exclusive device lease has been acquired.
    lease_factory = adapters.get("lease_devices")
    if not callable(lease_factory):
        raise RuntimeError("shared device lease adapter is required")
    lease = lease_factory([device], owner="meco-plan-a-ubp-orchestrator")
    parent_fd = _open_directory_chain_nofollow(output_dir.parent)
    try:
        with lease:
            claim_created = False
            staging_name = None
            staging_fd = None
            published = False
            try:
                if not _directory_path_matches_fd(output_dir.parent, parent_fd):
                    raise RuntimeError("bundle output parent changed while waiting for device lease")
                claim_name = claim_path.name
                _exclusive_claim_at(parent_fd, claim_name)
                claim_created = True
                staging_name, staging_fd = _create_staging_at(parent_fd, output_dir.name)
                # Keep intermediate operations attached to the pinned directory
                # inode; a path-based parent swap cannot redirect these writes.
                staging = Path(f"/proc/self/fd/{staging_fd}")
                for protected in _protected_paths(source_dataset, qualification_protocol):
                    if _overlaps(Path(os.readlink(f"/proc/self/fd/{staging_fd}")), protected):
                        raise ValueError(f"staging directory overlaps protected path: {protected}")
                ub_path = staging / "plan-a-ub-legal-nodes.jsonl"
                p_path = staging / "plan-a-p-routes-oracle-v3.jsonl"
                adapters["collect_ub"](ub_path, dataset=source_dataset,
                    qualification_protocol=qualification_protocol, device=device)
                if _sha256(source_dataset) != source_sha or _manifest(source_dataset)[1] != source_manifest_sha:
                    raise RuntimeError("immutable source dataset changed during U+B collection")
                adapters["collect_p"](p_path, dataset=source_dataset,
                    qualification_protocol=qualification_protocol, p_cost_scale=float(p_cost_scale),
                    intervention_maps=intervention_maps, device=device)
                if _sha256(source_dataset) != source_sha or _manifest(source_dataset)[1] != source_manifest_sha:
                    raise RuntimeError("immutable source dataset changed during P collection")
                ub_manifest, ub_manifest_sha = _manifest(ub_path)
                p_manifest, p_manifest_sha = _manifest(p_path)
                if ub_manifest.get("dataset_sha256") != _sha256(ub_path):
                    raise ValueError("U+B dataset bytes do not match their manifest digest")
                if p_manifest.get("dataset_sha256") != _sha256(p_path):
                    raise ValueError("P dataset bytes do not match their manifest digest")
                binding = _check_shared_binding(source_manifest, ub_manifest, p_manifest,
                    source_sha=source_sha, source_manifest_sha=source_manifest_sha,
                    protocol_sha=protocol_sha, p_cost_scale=float(p_cost_scale))
                ub_counts = _target_counts(ub_path)
                p_counts = _p_target_counts(p_path)
                expected_seed_keys = {str(seed) for seed in source_manifest["seeds"]}
                ub_rows = [json.loads(line) for line in ub_path.read_text(encoding="utf-8").splitlines() if line]
                if any(row.get("scenario_id") != source_manifest.get("scenario_id") or
                        row.get("category") != source_manifest.get("category") for row in ub_rows):
                    raise ValueError("U+B records differ from immutable source scenario/category")
                if {str(row.get("seed")) for row in ub_rows} != expected_seed_keys:
                    raise ValueError("U+B records do not cover the exact frozen source train seeds")
                if set(ub_counts["seed_counts"]) != expected_seed_keys or set(p_counts["seed_counts"]) != expected_seed_keys:
                    raise ValueError("U+B/P target records do not cover the exact frozen source train seeds")
                if ub_counts["u_targets"] < minimum_u_targets:
                    raise ValueError("collected U targets are missing or below the configured minimum")
                if ub_counts["b_preferences"] < minimum_b_preferences:
                    raise ValueError("collected B preferences are missing or below the configured minimum")
                if any(counts["u_targets"] < 1 or counts["b_preferences"] < 1
                        for counts in ub_counts["seed_counts"].values()):
                    raise ValueError("each frozen train seed must have sampled U and B targets")
                if not isinstance(ub_manifest.get("selected_node_coverage_fraction"), (int, float)):
                    # The collector publishes per-row sampled coverage, not a global
                    # exhaustive-coverage claim. Aggregate only those sampled rows.
                    rows = ub_rows
                    coverage = [row.get("selected_node_coverage_fraction") for row in rows]
                    coverage = [float(x) for x in coverage if isinstance(x, (int, float))]
                    sampled_node_coverage = sum(coverage) / len(coverage) if coverage else 0.0
                else:
                    sampled_node_coverage = float(ub_manifest["selected_node_coverage_fraction"])
                if sampled_node_coverage < minimum_selected_node_coverage:
                    raise ValueError("sampled legal-node coverage is below the configured minimum")
                if p_counts["oracle_labels"] < minimum_p_labels:
                    raise ValueError("P oracle labels are missing or below the configured minimum")
                if p_counts["non_tied_route_pairs"] < minimum_p_pairs:
                    raise ValueError("P dataset has too few non-tied oracle route pairs")
                if any(counts["oracle_labels"] < 2 or counts["non_tied_route_pairs"] < 1
                        for counts in p_counts["seed_counts"].values()):
                    raise ValueError("each frozen train seed must have sampled non-tied P route supervision")
                # This is the sampled executed-path evidence only; it is not exhaustive
                # legal-grid or all-route coverage.
                sampled_p_coverage = sum(1 for counts in p_counts["seed_counts"].values()
                    if counts["non_tied_route_pairs"] > 0) / len(expected_seed_keys)
                if sampled_p_coverage < minimum_p_coverage:
                    raise ValueError("sampled P route-group coverage is below the configured minimum")
                artifacts = []
                for track in TRACKS:
                    if (_sha256(ub_path) != ub_manifest["dataset_sha256"] or
                            _sha256(p_path) != p_manifest["dataset_sha256"] or
                            _manifest(ub_path)[1] != ub_manifest_sha or
                            _manifest(p_path)[1] != p_manifest_sha):
                        raise RuntimeError("collected Plan A dataset was modified before fitting")
                    checkpoint = staging / f"{track.lower()}-plan-a-joint-ubp.pt"
                    adapters["fit"](track, ub_path, source_dataset, p_path, checkpoint,
                        qualification_protocol=qualification_protocol, updates=updates,
                        learning_rate=learning_rate, device=device,
                        minimum_coverage_fraction=minimum_p_coverage)
                    if (_sha256(ub_path) != ub_manifest["dataset_sha256"] or
                            _sha256(p_path) != p_manifest["dataset_sha256"] or
                            _manifest(ub_path)[1] != ub_manifest_sha or
                            _manifest(p_path)[1] != p_manifest_sha):
                        raise RuntimeError("collected Plan A dataset changed during fitting")
                    artifacts.append({"track": track, "path": checkpoint.name,
                        "sha256": _sha256(checkpoint)})
                if _sha256(source_dataset) != source_sha or _manifest(source_dataset)[1] != source_manifest_sha:
                    raise RuntimeError("immutable source dataset changed during Plan A fitting")
                pinned_parent_path = Path(os.readlink(f"/proc/self/fd/{parent_fd}"))
                final_ub_path = pinned_parent_path / output_dir.name / ub_path.name
                final_p_path = pinned_parent_path / output_dir.name / p_path.name
                expected_hashes = {
                    "legal_dataset_sha256": ub_manifest["dataset_sha256"],
                    "legal_manifest_sha256": ub_manifest_sha,
                    "source_dataset_sha256": source_sha,
                    "source_manifest_sha256": source_manifest_sha,
                    "p_dataset_sha256": p_manifest["dataset_sha256"],
                    "p_manifest_sha256": p_manifest_sha,
                }
                for artifact in artifacts:
                    checkpoint = staging / artifact["path"]
                    adapters["prepare"](artifact["track"], checkpoint,
                        staged_legal=ub_path, staged_source=source_dataset,
                        staged_p=p_path, final_legal=final_ub_path,
                        final_source=source_dataset, final_p=final_p_path,
                        expected_hashes=expected_hashes,
                        qualification_protocol=qualification_protocol,
                        expected_sources=source_manifest["sources"],
                        expected_prediction_mode=source_manifest["prediction_mode"],
                        rebind=adapters["rebind"], validate=adapters["validate"])
                    artifact["sha256"] = _sha256(checkpoint)
                if not _directory_path_matches_fd(output_dir.parent, parent_fd):
                    raise RuntimeError("bundle output parent changed before bundle marker write")
                bundle = {
                    "schema": SCHEMA,
                    "status": "complete_validated_artifact_bundle",
                    "qualification_eligible": False,
                    "phase2_ready": False,
                    "phase2_complete": False,
                    "expert_u_claim": False,
                    "exhaustive_legal_node_coverage_claim": False,
                    "exhaustive_p_route_or_legal_grid_coverage_claim": False,
                    "binding": binding,
                    "configuration": {"device": device, "p_cost_scale": float(p_cost_scale),
                        "intervention_maps": intervention_maps, "updates": updates,
                        "learning_rate": learning_rate},
                    "inputs": {
                        "source_dataset": str(source_dataset), "source_dataset_sha256": source_sha,
                        "source_manifest_sha256": source_manifest_sha,
                        "qualification_protocol": str(qualification_protocol),
                        "qualification_protocol_sha256": protocol_sha,
                    },
                    "collected_datasets": [
                        {"kind": "sampled_legal_node_u_b", "path": ub_path.name,
                            "dataset_sha256": _sha256(ub_path), "manifest_sha256": ub_manifest_sha,
                            "target_counts": ub_counts,
                            "sampled_selected_node_coverage_fraction": sampled_node_coverage,
                            "coverage_semantics": "deterministic sampled feasible reachable nodes; not exhaustive"},
                        {"kind": "sampled_executed_p_routes_oracle_v3", "path": p_path.name,
                            "dataset_sha256": _sha256(p_path), "manifest_sha256": p_manifest_sha,
                            "target_counts": p_counts,
                            "sampled_route_group_coverage_fraction": sampled_p_coverage,
                            "coverage_semantics": "sampled executed route groups and oracle labels; not exhaustive routes or legal grid"},
                    ],
                    "artifacts": artifacts,
                }
                bundle_path = staging / "bundle.json"
                serialized = json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n"
                bundle_path.write_text(serialized, encoding="utf-8")
                report = {"schema": SCHEMA, "status": bundle["status"],
                    "phase2_ready": False, "phase2_complete": False,
                    "qualification_eligible": False,
                    "binding": binding, "artifact_count": len(artifacts),
                    "artifacts": artifacts}
                adapters["write_reports"](staging, bundle, report)
                _fsync_tree(staging_fd)
                if not _directory_path_matches_fd(output_dir.parent, parent_fd):
                    raise RuntimeError("bundle output parent changed before publication")
                bundle_sha = _sha256(bundle_path)
                _rename_noreplace_at(parent_fd, staging_name, output_dir.name)
                published = True
                _fsync_directory(parent_fd)
                if not _directory_path_matches_fd(output_dir.parent, parent_fd):
                    raise RuntimeError("bundle output parent changed during publication")
                return {"path": str(output_dir / "bundle.json"),
                    "bundle_sha256": bundle_sha,
                    "artifact_count": len(artifacts), "status": bundle["status"],
                    "phase2_ready": False, "phase2_complete": False}
            except BaseException:
                if staging_name is not None:
                    try:
                        _remove_tree_at(parent_fd,
                            output_dir.name if published else staging_name)
                    except FileNotFoundError:
                        pass
                raise
            finally:
                if claim_created:
                    try:
                        os.unlink(claim_path.name, dir_fd=parent_fd)
                    except FileNotFoundError:
                        pass
                if staging_fd is not None:
                    os.close(staging_fd)
    finally:
        os.close(parent_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--qualification-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True, help="explicit indexed CUDA device")
    parser.add_argument("--p-cost-scale", type=float, default=1.0)
    parser.add_argument("--intervention-maps", type=int, default=3)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args(argv)
    print(json.dumps(orchestrate_plan_a_ubp(args.source_dataset,
        args.qualification_protocol, args.output_dir, device=args.device,
        p_cost_scale=args.p_cost_scale, intervention_maps=args.intervention_maps,
        updates=args.updates, learning_rate=args.learning_rate), indent=2))


if __name__ == "__main__":
    main()
