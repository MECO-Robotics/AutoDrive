"""Offline descriptor and explicit executor for matched C1/C2 Phase 2 work.

Importing and describing campaigns are metadata-only. Simulation and fitting
are imported lazily and run only through :func:`execute_campaign`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from datetime import datetime, timezone

SCHEMA = "meco-c1-c2-phase2-campaign-v1"
FIXED = {
    "split": "train", "prediction_mode": "cv", "candidate_k": 16,
    "horizon_ticks": 160, "action_ticks": 12, "max_c_candidates": 8,
    "max_a_pairs": 8, "gamma_per_macro": 0.993,
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _source_snapshot(root: Path) -> tuple[str, str]:
    """Return clean HEAD and a pure-filesystem digest; imports no runtime code."""
    root = Path(root).resolve()
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot establish source repository identity") from exc
    if dirty:
        raise ValueError("source checkout must be clean before campaign description")
    entries = []
    for path in sorted((root / "frc_defense").rglob("*")):
        if path.is_file() and path.suffix in (".py", ".cpp", ".cu", ".h", ".cuh", ".json", ".hip", ".hpp", ".cc"):
            rel = path.relative_to(root).as_posix()
            if rel in ("frc_defense/csrc/fuel_physics_hip_kernel.hip", "frc_defense/csrc/fuel_sleep_hip_kernel.hip"):
                continue
            entries.append((rel, _sha(path.read_bytes())))
    return commit, _sha(_canonical(entries).encode())


def _runtime_source_identity():
    from .curriculum_training import source_identity
    return source_identity()


def _protocol(path: Path):
    supplied = Path(path).expanduser().absolute()
    if supplied.is_symlink():
        raise ValueError("qualification protocol must be a regular non-symlink file")
    path = supplied.resolve(strict=True)
    if not path.is_file():
        raise ValueError("qualification protocol must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("qualification protocol is not valid JSON") from exc
    if not isinstance(document, dict) or document.get("protocol_version") != "meco-qualification-v1":
        raise ValueError("unsupported qualification protocol")
    body = {key: val for key, val in document.items() if key != "protocol_id"}
    protocol_id = _sha(_canonical(body).encode())
    if document.get("protocol_id") != protocol_id:
        raise ValueError("qualification protocol ID does not match its contents")
    return path, document, protocol_id, _sha(raw)


def _safe_device(device):
    if not isinstance(device, str) or re.fullmatch(r"cuda:(?:0|[1-9][0-9]*)", device) is None:
        raise ValueError("device must be an explicit indexed CUDA device such as cuda:0")
    return device


def _nonoverlap(a: Path, b: Path) -> bool:
    return not (a == b or a in b.parents or b in a.parents)


def _reject_symlink_ancestors(path: Path):
    cursor = path
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValueError("campaign output path contains a symlink")
        cursor = cursor.parent


def describe_campaign(*, protocol_path: Path, scenario_id: str, category: str,
                      seeds: list[int], device: str, output_root: Path,
                      repo_root: Path | None = None) -> dict:
    """Build a no-write, no-lease campaign descriptor from frozen inputs."""
    repo = Path(repo_root or _repo_root()).resolve(strict=True)
    commit, source_sha = _source_snapshot(repo)
    protocol_path, protocol, protocol_id, protocol_sha = _protocol(protocol_path)
    if not isinstance(scenario_id, str) or not scenario_id or not isinstance(category, str) or not category:
        raise ValueError("scenario and category are required")
    matching = [row for row in protocol.get("cases", []) if
        row.get("scenario_id") == scenario_id and row.get("category") == category and row.get("split") == "train"]
    if not matching:
        raise ValueError("frozen protocol has no matching train case")
    allowed = []
    for row in matching:
        values = row.get("seeds")
        if not isinstance(values, list) or not values or any(type(seed) is not int or seed < 0 for seed in values):
            raise ValueError("protocol train seed set is malformed")
        if len(set(values)) != len(values):
            raise ValueError("protocol contains duplicate train seeds")
        allowed.extend(values)
    if len(set(allowed)) != len(allowed):
        raise ValueError("protocol has overlapping train cases for this scenario/category")
    if (not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds) or not set(seeds) <= set(allowed)):
        raise ValueError("campaign seeds must be unique exact members of this train split")
    device = _safe_device(device)
    root = Path(output_root).expanduser().absolute()
    _reject_symlink_ancestors(root.parent)
    if root.exists() or root.is_symlink():
        raise ValueError("campaign output root already exists or is a symlink")
    if not _nonoverlap(root, protocol_path) or not _nonoverlap(root, repo):
        raise ValueError("campaign output root must not overlap source checkout or protocol")
    if root.parent.exists():
        ancestor = root.parent.resolve()
        if not _nonoverlap(ancestor, protocol_path) or not _nonoverlap(ancestor, repo):
            raise ValueError("campaign output parent overlaps source checkout or protocol")
    campaign_id = _sha(_canonical({"commit": commit, "protocol": protocol_sha,
        "scenario": scenario_id, "category": category, "seeds": seeds,
        "device": device, "output_root": str(root)}).encode())[:20]
    descriptor = {"schema": SCHEMA, "campaign_id": campaign_id,
        "source": {"commit": commit, "source_tree_sha256": source_sha},
        "qualification_protocol": {"path": str(protocol_path), "id": protocol_id,
            "sha256": protocol_sha},
        "scenario_id": scenario_id, "category": category,
        "split": "train", "seeds": list(seeds), "tracks": ["C1", "C2"],
        "device": device, **FIXED,
        "outputs": {"root": str(root), "dataset": str(root / "matched-prefix.jsonl"),
            "C1": str(root / "C1.pt"), "C2": str(root / "C2.pt"),
            "report": str(root / "campaign-report.json")},
        "phase2_ready": False, "qualification_eligible": False}
    descriptor["descriptor_sha256"] = _sha(_canonical(descriptor).encode())
    return descriptor


def validate_descriptor(descriptor: dict, *, repo_root: Path | None = None) -> dict:
    """Revalidate descriptor and current immutable source/protocol/path claims."""
    if not isinstance(descriptor, dict) or descriptor.get("schema") != SCHEMA:
        raise ValueError("unsupported Phase 2 campaign descriptor")
    expected_digest = descriptor.get("descriptor_sha256")
    body = {key: val for key, val in descriptor.items() if key != "descriptor_sha256"}
    if expected_digest != _sha(_canonical(body).encode()):
        raise ValueError("campaign descriptor digest mismatch")
    repo = Path(repo_root or _repo_root()).resolve(strict=True)
    commit, source_sha = _source_snapshot(repo)
    if descriptor.get("source") != {"commit": commit, "source_tree_sha256": source_sha}:
        raise ValueError("campaign source identity is stale")
    protocol_path, _, protocol_id, protocol_sha = _protocol(Path(descriptor["qualification_protocol"]["path"]))
    if descriptor.get("qualification_protocol") != {"path": str(protocol_path), "id": protocol_id, "sha256": protocol_sha}:
        raise ValueError("campaign qualification protocol binding is stale")
    expected = {"split": "train", "tracks": ["C1", "C2"], **FIXED}
    if any(descriptor.get(key) != value for key, value in expected.items()):
        raise ValueError("campaign does not match the frozen C1/C2 protocol")
    rebuilt = describe_campaign(protocol_path=protocol_path,
        scenario_id=descriptor.get("scenario_id"), category=descriptor.get("category"),
        seeds=descriptor.get("seeds"), device=descriptor.get("device"),
        output_root=Path(descriptor["outputs"]["root"]), repo_root=repo)
    if rebuilt != descriptor:
        raise ValueError("campaign descriptor does not match rederived protocol, seed, or path claims")
    return descriptor


def write_descriptor(descriptor: dict, path: Path) -> Path:
    """Atomically write a validated descriptor without creating campaign outputs."""
    validate_descriptor(descriptor)
    path = Path(path).expanduser().absolute()
    _reject_symlink_ancestors(path.parent)
    if (not _nonoverlap(path, Path(descriptor["outputs"]["root"])) or
            not _nonoverlap(path, Path(descriptor["qualification_protocol"]["path"])) or
            not _nonoverlap(path, _repo_root().resolve())):
        raise ValueError("descriptor output must not overlap campaign, protocol, or source paths")
    if path.exists() or path.is_symlink():
        raise ValueError("descriptor output already exists or is a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(descriptor, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _atomic_report(path: Path, report: dict):
    data = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, temp_name = tempfile.mkstemp(prefix=".campaign-report-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def execute_campaign(descriptor: dict, *, lease_factory=None, collect_fn=None,
                     fit_fn=None, validate_fn=None) -> dict:
    """Explicitly collect once, fit C1 then C2, and validate both artifacts."""
    validate_descriptor(descriptor)
    root = Path(descriptor["outputs"]["root"])
    if root.exists() or root.is_symlink():
        raise ValueError("campaign root already exists; refusing overwrite")
    if lease_factory is None:
        from .meco_device_lease import lease_devices
        lease_factory = lease_devices
    if collect_fn is None or fit_fn is None or validate_fn is None:
        from .meco_counterfactual import collect_counterfactual_v2
        from .meco_supervised import (warm_start_counterfactual_v2,
            validate_counterfactual_warm_start_artifact)
        collect_fn = collect_fn or collect_counterfactual_v2
        fit_fn = fit_fn or warm_start_counterfactual_v2
        validate_fn = validate_fn or validate_counterfactual_warm_start_artifact
    protocol_path = Path(descriptor["qualification_protocol"]["path"])
    report = {"schema": SCHEMA, "campaign_id": descriptor["campaign_id"],
        "descriptor_sha256": descriptor["descriptor_sha256"], "status": "running",
        "source": descriptor["source"],
        "qualification_protocol": descriptor["qualification_protocol"],
        "scenario_id": descriptor["scenario_id"], "category": descriptor["category"],
        "seeds": descriptor["seeds"], "device": descriptor["device"],
        "phase2_ready": False, "qualification_eligible": False,
        "started_at": datetime.now(timezone.utc).isoformat(), "stages": []}
    # Lease is held before creating the root. Revalidate after acquisition so
    # source/protocol/symlink changes during the wait cannot authorize writes.
    with lease_factory([descriptor["device"]], owner=f"phase2-{descriptor['campaign_id']}"):
        validate_descriptor(descriptor)
        root.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_ancestors(root.parent)
        if root.parent.resolve() != root.parent:
            raise ValueError("campaign output parent changed through a symlink")
        validate_descriptor(descriptor)
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
        _atomic_report(root / "campaign-report.json", report)
        try:
            outputs = descriptor["outputs"]
            collected = collect_fn(Path(outputs["dataset"]),
                scenario_id=descriptor["scenario_id"], category=descriptor["category"],
                seeds=list(descriptor["seeds"]), qualification_protocol=protocol_path,
                device=descriptor["device"], horizon_ticks=descriptor["horizon_ticks"],
                action_ticks=descriptor["action_ticks"], candidate_k=descriptor["candidate_k"],
                prediction_mode=descriptor["prediction_mode"],
                max_c_candidates=descriptor["max_c_candidates"],
                max_a_pairs=descriptor["max_a_pairs"], gamma_per_macro=descriptor["gamma_per_macro"])
            dataset = Path(outputs["dataset"])
            dataset_manifest = dataset.with_suffix(dataset.suffix + ".manifest.json")
            manifest = json.loads(dataset_manifest.read_text())
            expected_sources = _runtime_source_identity()
            dataset_sha = _sha(dataset.read_bytes()) if dataset.is_file() else None
            if (not dataset.is_file() or not dataset_manifest.is_file() or
                    manifest.get("dataset_sha256") != dataset_sha or
                    collected.get("dataset_sha256") != dataset_sha or
                    manifest.get("record_count") != collected.get("record_count") or
                    type(manifest.get("record_count")) is not int or manifest["record_count"] < 1 or
                    manifest.get("protocol_id") != descriptor["qualification_protocol"]["id"] or
                    manifest.get("qualification_protocol_sha256") != descriptor["qualification_protocol"]["sha256"] or
                    manifest.get("split") != "train" or manifest.get("scenario_id") != descriptor["scenario_id"] or
                    manifest.get("category") != descriptor["category"] or manifest.get("seeds") != descriptor["seeds"] or
                    manifest.get("candidate_k") != 16 or manifest.get("prediction_mode") != "cv" or
                    manifest.get("horizon_physics_ticks") != 160 or manifest.get("max_macro_ticks") != 12 or
                    manifest.get("max_plan_c_descriptors_per_prefix") != 8 or
                    manifest.get("max_plan_a_pairs_per_prefix") != 8 or manifest.get("gamma_per_macro") != 0.993):
                raise ValueError("collected dataset manifest does not match frozen campaign descriptor")
            if (manifest.get("sources") != expected_sources or
                    manifest.get("source_identity_sha256") != _sha(_canonical(expected_sources).encode())):
                raise ValueError("collected dataset source identity does not match the current runtime")
            report["stages"].append({"stage": "collection", "status": "passed",
                "record_count": manifest["record_count"], "dataset_sha256": dataset_sha,
                "manifest_sha256": _sha(dataset_manifest.read_bytes())})
            _atomic_report(root / "campaign-report.json", report)
            for track in ("C1", "C2"):
                output = Path(outputs[track])
                fit = fit_fn(track, dataset, output, qualification_protocol=protocol_path,
                    device=descriptor["device"])
                if not output.is_file():
                    raise RuntimeError(f"{track} fitter returned without writing its artifact")
                validated = validate_fn(track, output, qualification_protocol=protocol_path,
                    expected_sources=expected_sources, expected_prediction_mode="cv")
                if validated.get("dataset_sha256") != _sha(dataset.read_bytes()):
                    raise ValueError(f"{track} artifact does not bind the collected dataset")
                report["stages"].append({"stage": "fit_and_validate", "track": track,
                    "status": "passed", "artifact_sha256": _sha(output.read_bytes()),
                    "dataset_sha256": validated["dataset_sha256"],
                    "non_tied_pair_count": validated["non_tied_pair_count"],
                    "updates": validated["updates"]})
                _atomic_report(root / "campaign-report.json", report)
            current_commit, current_source_sha = _source_snapshot(_repo_root())
            if descriptor["source"] != {"commit": current_commit,
                    "source_tree_sha256": current_source_sha}:
                raise RuntimeError("source identity changed during Phase 2 campaign")
            report["status"] = "artifacts_validated"
            report["phase2_ready"] = False
            report["qualification_eligible"] = False
            report["completed_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_report(root / "campaign-report.json", report)
            return report
        except BaseException as exc:
            report["status"] = "failed"
            report["phase2_ready"] = False
            report["qualification_eligible"] = False
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            report["failed_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_report(root / "campaign-report.json", report)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    describe = commands.add_parser("describe")
    describe.add_argument("--protocol", type=Path, required=True)
    describe.add_argument("--scenario", required=True)
    describe.add_argument("--category", required=True)
    describe.add_argument("--seeds", type=int, nargs="+", required=True)
    describe.add_argument("--device", required=True)
    describe.add_argument("--output-root", type=Path, required=True)
    describe.add_argument("--descriptor", type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("descriptor", type=Path)
    execute = commands.add_parser("execute")
    execute.add_argument("descriptor", type=Path)
    args = parser.parse_args(argv)
    if args.command == "describe":
        value = describe_campaign(protocol_path=args.protocol, scenario_id=args.scenario,
            category=args.category, seeds=args.seeds, device=args.device, output_root=args.output_root)
        if args.descriptor:
            write_descriptor(value, args.descriptor)
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        descriptor = json.loads(args.descriptor.read_text())
        if args.command == "validate":
            validate_descriptor(descriptor)
            print(json.dumps({"valid": True, "campaign_id": descriptor["campaign_id"]}))
        else:
            print(json.dumps(execute_campaign(descriptor), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
