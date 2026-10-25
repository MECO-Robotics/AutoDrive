"""Frozen multi-case Phase 2 descriptors and dataset contracts.

This module is metadata-only until ``execute_multicase`` is explicitly called.
It composes the reviewed one-case complete-set coordinator and never changes
the one-case descriptor or receipt formats.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from .meco_phase2_campaign import (_protocol, _source_snapshot, _sha,
    _canonical, _safe_device, FIXED as C_FIXED)
from .curriculum_training import source_identity
from .meco_tracks import MECO_TRACKS

SCHEMA = "meco-phase2-multi-case-five-track-v1"
PREDICTION_MODE = C_FIXED["prediction_mode"]
ACTION_TICKS = C_FIXED["action_ticks"]
HORIZON_TICKS = C_FIXED["horizon_ticks"]
CANDIDATE_K = C_FIXED["candidate_k"]


def _digest(value) -> str:
    return _sha(_canonical(value).encode())


def _write_private_json(path: Path, value: dict) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise


def _key(row: dict) -> tuple[str, str]:
    return row["scenario_id"], row["category"]


def _check_output_namespace(root: Path, protected: tuple[Path, ...], *, allow_existing=False) -> None:
    root = Path(root).expanduser().absolute()
    cursor = root
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValueError("multi-case output path contains a symlink")
        cursor = cursor.parent
    if root.is_symlink() or (root.exists() and not allow_existing):
        raise ValueError("multi-case output root must not exist")
    if root.exists() and not root.is_dir():
        raise ValueError("multi-case output root must be a directory")
    for target in protected:
        target = Path(target).resolve()
        if root == target or root in target.parents or target in root.parents:
            raise ValueError("multi-case output root overlaps a protected path")
    if root.parent.exists() and root.parent.resolve() != root.parent:
        raise ValueError("multi-case output parent resolves through a symlink")


def describe_multicase(*, protocol_path: Path, cases: list[dict], device: str,
        output_root: Path, repo_root: Path | None = None,
        _allow_existing_output=False) -> dict:
    """Describe ordered train cases without creating outputs or acquiring devices.

    ``cases`` contains ``scenario_id``, ``category``, and optional ``seeds``.
    Omitting seeds requests the complete frozen train seed set for that case.
    """
    repo = Path(repo_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    commit, source_sha = _source_snapshot(repo)
    protocol_file, protocol, protocol_id, protocol_sha = _protocol(protocol_path)
    if not isinstance(cases, list) or not cases:
        raise ValueError("at least one train case is required")
    indexed = {}
    for row in protocol.get("cases", []):
        if not isinstance(row, dict) or row.get("split") != "train":
            continue
        key = (row.get("scenario_id"), row.get("category"))
        if not all(isinstance(item, str) and item for item in key):
            raise ValueError("frozen train case key is malformed")
        seeds = row.get("seeds")
        if (not isinstance(seeds, list) or not seeds or
                any(type(seed) is not int or seed < 0 for seed in seeds) or
                len(set(seeds)) != len(seeds)):
            raise ValueError("frozen train case seed set is malformed")
        if key in indexed:
            raise ValueError("frozen protocol contains duplicate train case keys")
        indexed[key] = row
    normalized = []
    seen = set()
    for item in cases:
        if not isinstance(item, dict):
            raise ValueError("each requested case must be an object")
        key = (item.get("scenario_id"), item.get("category"))
        if key in seen:
            raise ValueError("requested train cases must be unique")
        seen.add(key)
        frozen = indexed.get(key)
        if frozen is None:
            raise ValueError("requested case is not in the frozen train protocol")
        allowed = frozen["seeds"]
        seeds = item.get("seeds", allowed)
        if (not isinstance(seeds, list) or not seeds or
                any(type(seed) is not int or seed < 0 for seed in seeds) or
                len(set(seeds)) != len(seeds) or not set(seeds) <= set(allowed)):
            raise ValueError("requested seeds must be unique members of the frozen train case")
        normalized.append({"scenario_id": key[0], "category": key[1],
            "seeds": list(seeds), "protocol_seeds": list(allowed),
            "full_case_coverage": seeds == allowed})
    device = _safe_device(device)
    root = Path(output_root).expanduser().absolute()
    _check_output_namespace(root, (repo, protocol_file),
        allow_existing=_allow_existing_output)
    # The current protocol's full train closure is an exact set of case keys.
    full_closure = seen == set(indexed)
    full_seed_coverage = all(item["full_case_coverage"] for item in normalized)
    coverage = "full_train_protocol_closure" if full_closure and full_seed_coverage else "partial_train_coverage"
    shards = []
    for index, item in enumerate(normalized, start=1):
        shard_root = root / "shards" / f"case-{index:04d}" / "attempt-0001"
        shards.append({"case_index": index, "case_key": [item["scenario_id"], item["category"]],
            "seeds": item["seeds"], "full_case_coverage": item["full_case_coverage"],
            "output_root": str(shard_root),
            "retry_policy": "new-numbered-attempt-directory"})
    outputs = {"root": str(root),
        "descriptor": str(root / "multi-case-descriptor.json"),
        "manifest": str(root / "multi-case-manifest.json"),
        "combined": {key: str(root / "combined" / filename) for key, filename in {
            "trajectory": "trajectory.jsonl", "ub": "plan-a-ub.jsonl",
            "p": "plan-a-p.jsonl"}.items()},
        "bundle": {track: str(root / "bundle" / f"{track}.pt") for track in MECO_TRACKS}}
    body = {"schema": SCHEMA, "source": {"commit": commit, "source_tree_sha256": source_sha},
        "runtime_source_identity": source_identity(),
        "qualification_protocol": {"path": str(protocol_file), "id": protocol_id,
            "sha256": protocol_sha}, "ordered_cases": normalized,
        "case_set_sha256": _digest([[item["scenario_id"], item["category"], item["seeds"]]
            for item in normalized]), "coverage": {"status": coverage,
            "requested_case_count": len(normalized), "protocol_train_case_count": len(indexed),
            "full_train_protocol_closure": coverage == "full_train_protocol_closure"},
        "prediction_mode": PREDICTION_MODE, "candidate_k": CANDIDATE_K,
        "action_ticks": ACTION_TICKS, "horizon_ticks": HORIZON_TICKS,
        "device": device, "tracks": list(MECO_TRACKS), "shards": shards,
        "outputs": outputs, "phase2_ready": False, "phase2_complete": False,
        "qualification_eligible": False}
    body["descriptor_sha256"] = _digest(body)
    return body


def validate_multicase_descriptor(descriptor: dict, *, repo_root: Path | None = None) -> dict:
    if not isinstance(descriptor, dict) or descriptor.get("schema") != SCHEMA:
        raise ValueError("unsupported multi-case Phase 2 descriptor")
    body = {key: value for key, value in descriptor.items() if key != "descriptor_sha256"}
    if descriptor.get("descriptor_sha256") != _digest(body):
        raise ValueError("multi-case descriptor digest mismatch")
    repo = Path(repo_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    commit, source_sha = _source_snapshot(repo)
    if descriptor.get("source") != {"commit": commit, "source_tree_sha256": source_sha}:
        raise ValueError("multi-case source identity is stale")
    if descriptor.get("runtime_source_identity") != source_identity():
        raise ValueError("multi-case runtime source identity is stale")
    protocol_file, protocol, protocol_id, protocol_sha = _protocol(
        Path(descriptor["qualification_protocol"]["path"]))
    if descriptor.get("qualification_protocol") != {"path": str(protocol_file),
            "id": protocol_id, "sha256": protocol_sha}:
        raise ValueError("multi-case protocol binding is stale")
    if (descriptor.get("tracks") != list(MECO_TRACKS) or
            descriptor.get("prediction_mode") != PREDICTION_MODE or
            descriptor.get("candidate_k") != CANDIDATE_K or
            descriptor.get("action_ticks") != ACTION_TICKS or
            descriptor.get("horizon_ticks") != HORIZON_TICKS or
            _safe_device(descriptor.get("device")) != descriptor.get("device")):
        raise ValueError("multi-case common training contract changed")
    expected = describe_multicase(protocol_path=protocol_file,
        cases=[{"scenario_id": row["scenario_id"], "category": row["category"],
            "seeds": row["seeds"]} for row in descriptor["ordered_cases"]],
        device=descriptor["device"], output_root=Path(descriptor["outputs"]["root"]),
        repo_root=repo, _allow_existing_output=True)
    if expected != descriptor:
        raise ValueError("multi-case descriptor differs from rederived frozen protocol claims")
    return descriptor


def case_coverage(expected_cases: list[dict], rows: list[dict], *, seed_field="seed") -> dict:
    """Return strict per-case expected/observed seed and row counts."""
    expected = {}
    for case in expected_cases:
        key = (case["scenario_id"], case["category"])
        if key in expected:
            raise ValueError("duplicate expected case key")
        expected[key] = set(case["seeds"])
    observed = {key: {"seeds": set(), "rows": 0} for key in expected}
    identities = set()
    for row in rows:
        key = (row.get("scenario_id"), row.get("category"))
        seed = row.get(seed_field)
        if key not in expected or type(seed) is not int or seed not in expected[key]:
            raise ValueError("dataset row is outside the declared case/seed set")
        identity = (key[0], key[1], seed, row.get("decision_index"))
        if identity in identities:
            raise ValueError("duplicate case-seed-decision identity")
        identities.add(identity)
        observed[key]["seeds"].add(seed)
        observed[key]["rows"] += 1
    report = {}
    for key, seeds in expected.items():
        seen = observed[key]["seeds"]
        if seen != seeds:
            raise ValueError(f"case {key} seed coverage is incomplete")
        report[f"{key[0]}::{key[1]}"] = {"expected_seed_count": len(seeds),
            "observed_seed_count": len(seen), "row_count": observed[key]["rows"],
            "expected_seeds": sorted(seeds), "observed_seeds": sorted(seen)}
    return report


def combine_jsonl_shards(shards: list[dict], output: Path, *, expected_cases: list[dict],
        row_validator=None) -> dict:
    """Combine same-schema JSONL shards in case/seed/decision order.

    Original shard bytes are never modified. Repeated labels inside a row are
    retained. The caller supplies the target-family validator before pooling.
    """
    if not shards or len(shards) != len(expected_cases):
        raise ValueError("combined target requires one shard for each declared case")
    expected_keys = [(row["scenario_id"], row["category"]) for row in expected_cases]
    records = []
    seen_keys = set()
    for case_index, (expected, shard) in enumerate(zip(expected_cases, shards)):
        path = Path(shard).absolute()
        if path.is_symlink() or not path.is_file():
            raise ValueError("target shard must be a regular non-symlink file")
        with path.open(encoding="utf-8") as stream:
            local = []
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("target shard row must be an object")
                if row_validator is not None:
                    row_validator(row)
                if (row.get("scenario_id"), row.get("category")) != expected_keys[case_index]:
                    raise ValueError("target shard contains rows from a different case")
                identity = (row.get("scenario_id"), row.get("category"),
                    row.get("seed", row.get("episode_seed")), row.get("decision_index"))
                if identity in seen_keys:
                    raise ValueError("target shards contain duplicate decision identity")
                seen_keys.add(identity)
                local.append(row)
        if not local:
            raise ValueError("target shard is empty")
        records.extend(local)
    # Stable semantic order; original order breaks ties only when a row family
    # intentionally contains multiple labels at the same branch identity.
    case_order = {key: index for index, key in enumerate(expected_keys)}
    records.sort(key=lambda row: (case_order[(row["scenario_id"], row["category"])],
        row.get("seed", row.get("episode_seed")), row.get("decision_index"),
        _canonical(row)))
    case_coverage(expected_cases, records,
        seed_field="seed" if "seed" in records[0] else "episode_seed")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("combined target output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"),
        allow_nan=False) + "\n").encode() for row in records)
    fd, temp_name = tempfile.mkstemp(prefix="." + output.name + ".", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_name, output)
        os.unlink(temp_name)
    finally:
        Path(temp_name).unlink(missing_ok=True)
    return {"path": str(output), "sha256": _sha(payload),
        "record_count": len(records), "case_coverage": case_coverage(expected_cases, records,
            seed_field="seed" if "seed" in records[0] else "episode_seed")}


def validate_case_receipt(receipt: dict, shard: dict, descriptor: dict,
        *, target_validators: dict[str, callable] | None = None) -> dict:
    """Recompute case-shard evidence before a resume may adopt it."""
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA + ".case-receipt":
        raise ValueError("case shard has no recognized validation receipt")
    if (receipt.get("descriptor_sha256") != descriptor["descriptor_sha256"] or
            receipt.get("case_key") != shard["case_key"] or
            receipt.get("seeds") != shard["seeds"] or
            receipt.get("source") != descriptor["source"] or
            receipt.get("qualification_protocol") != descriptor["qualification_protocol"] or
            receipt.get("device") != descriptor["device"]):
        raise ValueError("case receipt does not bind the frozen descriptor")
    datasets = receipt.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {"trajectory", "ub", "p"}:
        raise ValueError("case receipt must bind C trajectory, U+B, and P target datasets")
    expected_key = tuple(shard["case_key"])
    cross_source = None
    target_validators = target_validators or {}
    for family in ("trajectory", "ub", "p"):
        entry = datasets[family]
        path = Path(entry.get("path", "")).absolute()
        manifest_path = Path(entry.get("manifest_path", "")).absolute()
        for value in (path, manifest_path):
            cursor = value
            while cursor != cursor.parent:
                if cursor.is_symlink():
                    raise ValueError("case target path contains a symlink")
                cursor = cursor.parent
        if not path.is_file() or not manifest_path.is_file():
            raise ValueError(f"case {family} dataset or manifest is missing")
        raw = path.read_bytes()
        manifest_raw = manifest_path.read_bytes()
        if (_sha(raw) != entry.get("sha256") or
                _sha(manifest_raw) != entry.get("manifest_sha256")):
            raise ValueError(f"case {family} bytes differ from their receipt")
        manifest = json.loads(manifest_raw)
        manifest_key = (manifest.get("scenario_id"), manifest.get("category"))
        device = manifest.get("device")
        device_name = device.get("device") if isinstance(device, dict) else device
        if (manifest.get("split") != "train" or
                (family != "ub" and manifest_key != expected_key) or
                (family == "ub" and manifest_key not in ((None, None), expected_key)) or
                (family != "ub" and manifest.get("seeds") != shard["seeds"]) or
                (family == "ub" and manifest.get("seeds") not in (None, shard["seeds"])) or
                manifest.get("protocol_id") != descriptor["qualification_protocol"]["id"] or
                manifest.get("qualification_protocol_sha256") != descriptor["qualification_protocol"]["sha256"] or
                manifest.get("sources") != descriptor["runtime_source_identity"] or
                manifest.get("source_identity_sha256") != _digest(descriptor["runtime_source_identity"]) or
                device_name != descriptor["device"] or
                manifest.get("prediction_mode") != descriptor["prediction_mode"] or
                manifest.get("record_count") != len([line for line in raw.splitlines() if line.strip()]) or
                manifest.get("dataset_sha256") != _sha(raw)):
            raise ValueError(f"case {family} manifest has a stale or mismatched binding")
        if family in target_validators:
            target_validators[family](path, manifest)
        if family == "trajectory":
            cross_source = {"path": str(path), "sha256": _sha(raw),
                "manifest_sha256": _sha(manifest_raw)}
        elif (manifest.get("source_dataset_sha256", manifest.get("source_counterfactual_dataset_sha256")) != cross_source["sha256"] or
                manifest.get("source_manifest_sha256", manifest.get("source_counterfactual_manifest_sha256")) != cross_source["manifest_sha256"]):
            raise ValueError(f"case {family} target does not bind the exact trajectory shard")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        for row in rows:
            if ((row.get("scenario_id"), row.get("category")) != expected_key or
                    row.get("seed") not in shard["seeds"] or
                    row.get("protocol_id") != descriptor["qualification_protocol"]["id"] or
                    row.get("qualification_protocol_sha256") != descriptor["qualification_protocol"]["sha256"] or
                    row.get("source_identity_sha256") != _digest(descriptor["runtime_source_identity"]) or
                    row.get("device") != device):
                raise ValueError(f"case {family} rows drift from their source manifest")
    return receipt


def execute_multicase(descriptor: dict, *, collect_case, combine_target,
        fit_combined, validate_track, target_validators=None,
        lease_factory=None, device_identity_verifier=None) -> dict:
    """Run serial shards and pooled fitting under one verified device lease."""
    validate_multicase_descriptor(descriptor)
    root = Path(descriptor["outputs"]["root"])
    _check_output_namespace(root, (Path(descriptor["qualification_protocol"]["path"]),
        Path(__file__).resolve().parents[1]))
    if lease_factory is None:
        from .meco_device_lease import lease_devices
        lease_factory = lease_devices
    if device_identity_verifier is None:
        from .meco_gpu_attestation import verify_post_lease_wx9100
        device_identity_verifier = verify_post_lease_wx9100
    if not callable(lease_factory) or not callable(device_identity_verifier):
        raise ValueError("exclusive lease and indexed WX9100 identity verifier are required")
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError("multi-case campaign root is not a safe directory")
    root.parent.mkdir(parents=True, exist_ok=True)
    expected_cases = descriptor["ordered_cases"]
    shard_receipts = []
    with lease_factory([descriptor["device"]],
            owner="phase2-multicase-" + descriptor["descriptor_sha256"][:16]) as lease:
        attestation = device_identity_verifier(lease, [descriptor["device"]])
        if (not isinstance(attestation, dict) or
                attestation.get("logical_lease", {}).get("devices") != [descriptor["device"]]):
            raise ValueError("lease attestation does not bind the requested indexed device")
        validate_multicase_descriptor(descriptor)
        if not root.exists():
            root.mkdir(mode=0o700)
        root_stat = root.stat(follow_symlinks=False)
        if root_stat.st_uid != os.geteuid() or root_stat.st_mode & 0o077:
            raise ValueError("multi-case attempt root must be coordinator-owned and private")
        descriptor_path = Path(descriptor["outputs"]["descriptor"])
        encoded_descriptor = json.dumps(descriptor, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if descriptor_path.exists():
            if descriptor_path.is_symlink() or descriptor_path.read_text(encoding="utf-8") != encoded_descriptor:
                raise ValueError("existing descriptor marker differs from the current descriptor")
        else:
            descriptor_path.write_text(encoded_descriptor, encoding="utf-8")
        attempt_root = root / "shards"
        attempt_root.mkdir(mode=0o700, exist_ok=True)
        if attempt_root.stat(follow_symlinks=False).st_mode & 0o077:
            raise ValueError("multi-case shard directory must be private")
        for shard in descriptor["shards"]:
            case_dir = attempt_root / f"case-{shard['case_index']:04d}"
            case_dir.mkdir(mode=0o700, exist_ok=True)
            if case_dir.stat(follow_symlinks=False).st_mode & 0o077:
                raise ValueError("case shard directory must be private")
            attempts = sorted((item for item in case_dir.iterdir()
                if re.fullmatch(r"attempt-[0-9]{4,}", item.name)), key=lambda item: item.name)
            adopted = None
            for existing in attempts:
                if existing.is_symlink() or not existing.is_dir():
                    raise ValueError("existing shard attempt is not a regular directory")
                existing_stat = existing.stat(follow_symlinks=False)
                if existing_stat.st_uid != os.geteuid() or existing_stat.st_mode & 0o077:
                    raise ValueError("existing shard attempt must be coordinator-owned and private")
                receipt_path = existing / "case-receipt.json"
                if not receipt_path.is_file() or receipt_path.is_symlink():
                    continue
                receipt_stat = receipt_path.stat(follow_symlinks=False)
                if receipt_stat.st_uid != os.geteuid() or receipt_stat.st_mode & 0o077:
                    raise ValueError("existing shard receipt must be coordinator-owned and private")
                try:
                    candidate = json.loads(receipt_path.read_text(encoding="utf-8"))
                    validate_case_receipt(candidate, shard, descriptor,
                        target_validators=target_validators)
                    adopted = candidate
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                break
            if adopted is not None:
                shard_receipts.append(adopted)
                continue
            next_number = max([int(item.name.split("-")[1]) for item in attempts] + [0]) + 1
            case_root = case_dir / f"attempt-{next_number:04d}"
            case_root.mkdir(mode=0o700)
            one_case = {key: value for key, value in shard.items()
                if key in ("case_index", "case_key", "seeds", "full_case_coverage")}
            one_case.update({"source": descriptor["source"],
                "qualification_protocol": descriptor["qualification_protocol"],
                "device": descriptor["device"], "prediction_mode": descriptor["prediction_mode"],
                "candidate_k": descriptor["candidate_k"], "action_ticks": descriptor["action_ticks"],
                "horizon_ticks": descriptor["horizon_ticks"],
                "descriptor_sha256": descriptor["descriptor_sha256"],
                "device_identity_attestation": attestation, "output_root": str(case_root)})
            receipt = collect_case(one_case, case_root)
            validate_case_receipt(receipt, shard, descriptor,
                target_validators=target_validators)
            receipt_path = case_root / "case-receipt.json"
            if receipt_path.exists() or receipt_path.is_symlink():
                raise ValueError("case runner pre-created coordinator receipt path")
            _write_private_json(receipt_path, receipt)
            shard_receipts.append(receipt)

    combined = {}
    for family in ("trajectory", "ub", "p"):
        target_paths = [receipt["datasets"][family]["path"] for receipt in shard_receipts]
        target_path = Path(descriptor["outputs"]["combined"][family])
        combined[family] = combine_target(family, target_paths, target_path,
            expected_cases=expected_cases, descriptor=descriptor)
        path = Path(combined[family]["path"])
        if not path.is_file() or path.is_symlink() or _sha(path.read_bytes()) != combined[family].get("sha256"):
            raise ValueError(f"combined {family} corpus does not match returned digest")
    expected_combined = {family: {"path": combined[family]["path"],
        "sha256": combined[family]["sha256"], "manifest_sha256": combined[family].get("manifest_sha256")}
        for family in combined}
    if combined["ub"].get("source_dataset_sha256") != combined["trajectory"]["sha256"] or combined["p"].get("source_dataset_sha256") != combined["trajectory"]["sha256"]:
        raise ValueError("combined U+B and P targets do not bind pooled trajectory bytes")
    artifact_paths = {track: Path(descriptor["outputs"]["bundle"][track]) for track in MECO_TRACKS}
    fit_receipt = fit_combined(combined, artifact_paths, descriptor=descriptor)
    if not isinstance(fit_receipt, dict) or fit_receipt.get("descriptor_sha256") != descriptor["descriptor_sha256"] or fit_receipt.get("combined_inputs") != expected_combined:
        raise ValueError("combined fitter receipt is not bound to current pooled target bytes")
    validator_receipts = {}
    for track in MECO_TRACKS:
        validation = validate_track(track, artifact_paths[track], combined, descriptor=descriptor)
        if (not isinstance(validation, dict) or validation.get("track") != track or
                validation.get("descriptor_sha256") != descriptor["descriptor_sha256"] or
                validation.get("case_set_sha256") != descriptor["case_set_sha256"] or
                validation.get("combined_inputs") != expected_combined):
            raise ValueError(f"{track} validator did not bind pooled data and case set")
        validator_receipts[track] = validation
    manifest = {"schema": SCHEMA + ".bundle",
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "case_set_sha256": descriptor["case_set_sha256"],
        "source": descriptor["source"],
        "qualification_protocol": descriptor["qualification_protocol"],
        "coverage": descriptor["coverage"], "combined_inputs": expected_combined,
        "shard_receipts": [{"case_key": receipt["case_key"], "seeds": receipt["seeds"],
            "receipt_sha256": _digest(receipt)} for receipt in shard_receipts],
        "artifacts": {track: {"track": track, "path": str(artifact_paths[track]),
            "sha256": _sha(artifact_paths[track].read_bytes())} for track in MECO_TRACKS},
        "validator_receipts": validator_receipts,
        "phase2_ready": False, "phase2_complete": False,
        "qualification_eligible": False}
    validate_five_track_manifest(manifest, descriptor,
        validator_receipts=validator_receipts, artifact_paths=artifact_paths)
    manifest_path = Path(descriptor["outputs"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp = manifest_path.with_name("." + manifest_path.name + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True,
        allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, manifest_path)
    return manifest


def validate_five_track_manifest(manifest: dict, descriptor: dict,
        *, validator_receipts: dict[str, dict], artifact_paths: dict[str, Path]) -> dict:
    """Validate a combined-data five-track receipt table without fabricating receipts."""
    validate_multicase_descriptor(descriptor)
    if manifest.get("schema") != SCHEMA + ".bundle" or manifest.get("descriptor_sha256") != descriptor["descriptor_sha256"] or manifest.get("case_set_sha256") != descriptor["case_set_sha256"]:
        raise ValueError("multi-case bundle descriptor/case-set binding mismatch")
    if manifest.get("coverage") != descriptor["coverage"] or manifest.get("qualification_protocol") != descriptor["qualification_protocol"]:
        raise ValueError("multi-case bundle coverage or protocol binding mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(MECO_TRACKS) or set(validator_receipts) != set(MECO_TRACKS) or set(artifact_paths) != set(MECO_TRACKS):
        raise ValueError("multi-case bundle must contain exactly five validated tracks")
    for track in MECO_TRACKS:
        path = Path(artifact_paths[track])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or unsafe {track} artifact")
        record, receipt = artifacts[track], validator_receipts[track]
        if (record.get("track") != track or record.get("sha256") != _sha(path.read_bytes()) or
                record.get("sha256") != receipt.get("artifact_sha256") or
                receipt.get("track") != track or receipt.get("case_set_sha256") != descriptor["case_set_sha256"] or
                receipt.get("descriptor_sha256") != descriptor["descriptor_sha256"]):
            raise ValueError(f"{track} validator receipt does not bind current artifact and case set")
    if manifest.get("phase2_ready") is not False or manifest.get("phase2_complete") is not False or manifest.get("qualification_eligible") is not False:
        raise ValueError("multi-case training bundle cannot claim readiness or qualification")
    return manifest


def export_training_manifest_warm_starts(manifest: dict, descriptor: dict) -> dict[str, str]:
    """Forward all five track artifact paths and preserve the frozen case set."""
    if manifest.get("descriptor_sha256") != descriptor.get("descriptor_sha256"):
        raise ValueError("training manifest export requires the exact multi-case descriptor")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(MECO_TRACKS):
        raise ValueError("training manifest export requires all five tracks")
    result = {track: artifacts[track]["path"] for track in MECO_TRACKS}
    result["_case_set_sha256"] = descriptor["case_set_sha256"]
    result["_coverage_status"] = descriptor["coverage"]["status"]
    return result
