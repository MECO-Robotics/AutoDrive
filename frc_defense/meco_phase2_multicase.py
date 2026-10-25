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
from contextlib import nullcontext

from .meco_phase2_campaign import (_protocol, _source_snapshot, _sha,
    _canonical, _safe_device, FIXED as C_FIXED)
from .curriculum_training import source_identity
from .meco_tracks import MECO_TRACKS, prediction_contract

SCHEMA = "meco-phase2-multi-case-five-track-v1"
PREDICTION_MODE = C_FIXED["prediction_mode"]
ACTION_TICKS = C_FIXED["action_ticks"]
HORIZON_TICKS = C_FIXED["horizon_ticks"]
CANDIDATE_K = C_FIXED["candidate_k"]
A_SETTINGS = {"p_cost_scale": 1.0, "intervention_maps": 3,
    "updates": 10, "learning_rate": 1e-4, "route_margin": 0.02,
    "map_regularization": 0.001, "minimum_coverage_fraction": 0.01}


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
    from .meco_gpu_attestation import EXPECTED_IDENTITIES
    from .meco_training_manifest import HOST_GPU_INVENTORY, HOST_GPU_INVENTORY_SHA256
    if device not in EXPECTED_IDENTITIES:
        raise ValueError("multi-case Phase 2 requires a pinned indexed WX9100 device")
    inventory_sha = _sha(HOST_GPU_INVENTORY.read_bytes())
    if inventory_sha != HOST_GPU_INVENTORY_SHA256:
        raise ValueError("pinned WX9100 inventory digest differs from the reviewed value")
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
        "prediction_mode": PREDICTION_MODE,
        "prediction_contract": prediction_contract(PREDICTION_MODE),
        "c_settings": C_FIXED,
        "candidate_k": CANDIDATE_K,
        "action_ticks": ACTION_TICKS, "horizon_ticks": HORIZON_TICKS,
        "a_settings": A_SETTINGS, "device": device,
        "device_identity_requirement": {**EXPECTED_IDENTITIES[device],
            "inventory_path": str(HOST_GPU_INVENTORY),
            "inventory_sha256": inventory_sha},
        "tracks": list(MECO_TRACKS), "shards": shards,
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
    from .meco_gpu_attestation import EXPECTED_IDENTITIES
    from .meco_training_manifest import HOST_GPU_INVENTORY, HOST_GPU_INVENTORY_SHA256
    inventory_sha = _sha(HOST_GPU_INVENTORY.read_bytes())
    if descriptor.get("device") not in EXPECTED_IDENTITIES:
        raise ValueError("multi-case Phase 2 requires a pinned indexed WX9100 device")
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
            descriptor.get("prediction_contract") != prediction_contract(PREDICTION_MODE) or
            descriptor.get("c_settings") != C_FIXED or
            descriptor.get("a_settings") != A_SETTINGS or
            descriptor.get("device_identity_requirement") != {**EXPECTED_IDENTITIES.get(descriptor.get("device"), {}),
                "inventory_path": str(HOST_GPU_INVENTORY), "inventory_sha256": inventory_sha} or
            inventory_sha != HOST_GPU_INVENTORY_SHA256 or
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
    if output.is_symlink():
        raise ValueError("combined target output is a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"),
        allow_nan=False) + "\n").encode() for row in records)
    if output.exists():
        if not output.is_file() or output.read_bytes() != payload:
            raise ValueError("existing combined target bytes differ from rederived shard contents")
        return {"path": str(output), "sha256": _sha(payload),
            "record_count": len(records), "case_coverage": case_coverage(expected_cases, records,
                seed_field="seed" if "seed" in records[0] else "episode_seed")}
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


def combine_target_shards(family: str, shards: list[Path], output: Path, *,
        expected_cases: list[dict], descriptor: dict, source_binding=None,
        source_shards=None) -> dict:
    """Pool one validated target family and seal a case-level sidecar manifest."""
    if family not in ("trajectory", "ub", "p") or len(shards) != len(expected_cases):
        raise ValueError("unsupported target family or incomplete shard list")
    if family in ("ub", "p") and (not isinstance(source_shards, list) or
            len(source_shards) != len(expected_cases)):
        raise ValueError("Plan A pooled targets require ordered per-case source shard bindings")
    shard_manifests = []
    case_source_bindings = []
    for index, (case, path) in enumerate(zip(expected_cases, shards)):
        path = Path(path)
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        if path.is_symlink() or manifest_path.is_symlink():
            raise ValueError("target shard or sidecar is a symlink")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = (case["scenario_id"], case["category"])
        if (manifest.get("split") != "train" or
                (family != "ub" and (manifest.get("scenario_id"), manifest.get("category")) != key) or
                (family != "ub" and manifest.get("seeds") != case["seeds"]) or
                manifest.get("protocol_id") != descriptor["qualification_protocol"]["id"] or
                manifest.get("qualification_protocol_sha256") != descriptor["qualification_protocol"]["sha256"] or
                manifest.get("sources") != descriptor["runtime_source_identity"] or
                manifest.get("source_identity_sha256") != _digest(descriptor["runtime_source_identity"]) or
                (manifest.get("device", {}).get("device") if isinstance(manifest.get("device"), dict)
                 else manifest.get("device")) != descriptor["device"] or
                manifest.get("prediction_mode") != descriptor["prediction_mode"] or
                manifest.get("dataset_sha256") != _sha(path.read_bytes())):
            raise ValueError(f"{family} shard manifest differs from its descriptor or bytes")
        if family in ("ub", "p"):
            source_entry = source_shards[index]
            source_path = Path(source_entry["path"])
            source_manifest_path = Path(source_entry["manifest_path"])
            source_sha = _sha(source_path.read_bytes())
            source_manifest_sha = _sha(source_manifest_path.read_bytes())
            if (manifest.get("source_dataset_sha256",
                    manifest.get("source_counterfactual_dataset_sha256")) != source_sha or
                    manifest.get("source_manifest_sha256",
                    manifest.get("source_counterfactual_manifest_sha256")) != source_manifest_sha):
                raise ValueError(f"{family} shard no longer binds its exact per-case trajectory bytes")
            case_source_bindings.append({"case_key": [case["scenario_id"], case["category"]],
                "source_dataset_path": str(source_path), "source_dataset_sha256": source_sha,
                "source_manifest_path": str(source_manifest_path),
                "source_manifest_sha256": source_manifest_sha})
        shard_manifests.append({"case_key": list(key), "seeds": case["seeds"],
            "path": str(path), "manifest_path": str(manifest_path),
            "dataset_sha256": _sha(path.read_bytes()),
            "manifest_sha256": _sha(manifest_path.read_bytes()),
            "record_count": manifest.get("record_count"),
            "schema": manifest.get("schema"), "target_schema": manifest.get("target_schema")})
    common_fields = ("schema", "target_schema", "protocol_id",
        "qualification_protocol_sha256", "sources", "source_identity_sha256",
        "device", "prediction_mode", "prediction_contract")
    reference = json.loads(Path(shards[0]).with_suffix(Path(shards[0]).suffix + ".manifest.json").read_text())
    for path in shards[1:]:
        current = json.loads(Path(path).with_suffix(Path(path).suffix + ".manifest.json").read_text())
        if any(current.get(key) != reference.get(key) for key in common_fields):
            raise ValueError(f"{family} shards have incompatible target/source contracts")
    combined = combine_jsonl_shards(shards, output, expected_cases=expected_cases)
    manifest = {"schema": SCHEMA + ".combined-target",
        "target_family": family, "dataset_schema": reference["schema"],
        "target_schema": reference["target_schema"], "split": "train",
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "case_set_sha256": descriptor["case_set_sha256"],
        "source": descriptor["source"],
        "qualification_protocol": descriptor["qualification_protocol"],
        "sources": reference["sources"],
        "source_identity_sha256": reference["source_identity_sha256"],
        "device": reference["device"], "prediction_mode": reference["prediction_mode"],
        "prediction_contract": reference["prediction_contract"],
        "case_coverage": combined["case_coverage"],
        "record_count": combined["record_count"], "dataset_sha256": combined["sha256"],
        "shard_manifests": shard_manifests}
    if family in ("ub", "p"):
        if not isinstance(source_binding, dict) or not source_binding.get("path"):
            raise ValueError("Plan A pooled target requires the exact combined trajectory binding")
        manifest.update({"source_dataset_path": source_binding["path"],
            "source_dataset_sha256": source_binding["sha256"],
            "source_manifest_sha256": source_binding["manifest_sha256"],
            "case_source_bindings": case_source_bindings})
    # Keep shared family settings only when every shard agrees. This preserves
    # repeat protocols, P scale/oracle version and target policies verbatim.
    for key in sorted(set.intersection(*(set(json.loads(Path(path).with_suffix(
            Path(path).suffix + ".manifest.json").read_text())) for path in shards))):
        if key not in manifest and all(json.loads(Path(path).with_suffix(
                Path(path).suffix + ".manifest.json").read_text()).get(key) ==
                reference.get(key) for path in shards):
            manifest[key] = reference[key]
    manifest_path = Path(output).with_suffix(Path(output).suffix + ".manifest.json")
    if manifest_path.exists():
        if manifest_path.is_symlink() or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"existing combined {family} manifest differs from recomputed provenance")
    else:
        _write_private_json(manifest_path, manifest)
    return {"path": str(output), "sha256": combined["sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha(manifest_path.read_bytes()),
        "record_count": combined["record_count"],
        "case_coverage": combined["case_coverage"],
        "source_dataset_sha256": manifest.get("source_dataset_sha256"),
        "source_manifest_sha256": manifest.get("source_manifest_sha256")}


def _validate_case_coverage_summary(summary: dict, cases: list[dict]) -> None:
    if not isinstance(summary, dict) or set(summary) != {
            f"{case['scenario_id']}::{case['category']}" for case in cases}:
        raise ValueError("combined target sidecar omits or adds a declared case")
    for case in cases:
        value = summary[f"{case['scenario_id']}::{case['category']}"]
        if (value.get("expected_seeds") != sorted(case["seeds"]) or
                value.get("observed_seeds") != sorted(case["seeds"]) or
                value.get("expected_seed_count") != len(case["seeds"]) or
                value.get("observed_seed_count") != len(case["seeds"]) or
                type(value.get("row_count")) is not int or value["row_count"] < 1):
            raise ValueError("combined target per-case seed or row coverage is incomplete")


def validate_case_receipt(receipt: dict, shard: dict, descriptor: dict,
        *, target_validators: dict[str, callable] | None = None,
        require_complete_set=False) -> dict:
    """Recompute case-shard evidence before a resume may adopt it."""
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA + ".case-receipt":
        raise ValueError("case shard has no recognized validation receipt")
    datasets = receipt.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {"trajectory", "ub", "p"}:
        raise ValueError("case receipt dataset family map is malformed")
    for family, entry in datasets.items():
        if (not isinstance(entry, dict) or
                any(not isinstance(entry.get(key), str) or not entry[key]
                    for key in ("path", "sha256", "manifest_path", "manifest_sha256"))):
            raise ValueError(f"case receipt {family} dataset binding is malformed")
    if (receipt.get("descriptor_sha256") != descriptor["descriptor_sha256"] or
            receipt.get("case_key") != shard["case_key"] or
            receipt.get("seeds") != shard["seeds"] or
            receipt.get("source") != descriptor["source"] or
            receipt.get("qualification_protocol") != descriptor["qualification_protocol"] or
            receipt.get("device") != descriptor["device"]):
        raise ValueError("case receipt does not bind the frozen descriptor")
    expected_key = tuple(shard["case_key"])
    if require_complete_set:
        from .meco_phase2_complete_set import validate_complete_set_manifest
        one_case = receipt.get("one_case_descriptor")
        manifest_value = receipt.get("complete_set_manifest_path")
        attempt_value = receipt.get("attempt_root")
        if (not isinstance(one_case, dict) or not isinstance(manifest_value, str) or
                not manifest_value or not isinstance(attempt_value, str) or not attempt_value):
            raise ValueError("case receipt lacks the existing complete-set validator evidence")
        one_case_outputs = one_case.get("outputs")
        if not isinstance(one_case_outputs, dict):
            raise ValueError("case receipt complete-set descriptor outputs are malformed")
        manifest_path = Path(manifest_value)
        attempt_root = Path(attempt_value).absolute()
        expected_attempt_parent = Path(descriptor["outputs"]["root"]) / "shards" / f"case-{shard['case_index']:04d}"
        if (not manifest_path.is_file() or manifest_path.is_symlink() or
                one_case_outputs.get("root") != str(attempt_root) or
                attempt_root.parent != expected_attempt_parent or
                re.fullmatch(r"attempt-[0-9]{4,}", attempt_root.name) is None or
                one_case_outputs.get("manifest") != str(manifest_path) or
                _sha(manifest_path.read_bytes()) != receipt.get("complete_set_manifest_sha256") or
                one_case.get("scenario_id") != expected_key[0] or
                one_case.get("category") != expected_key[1] or
                one_case.get("seeds") != shard["seeds"] or one_case.get("device") != descriptor["device"] or
                one_case.get("source") != descriptor["source"] or
                one_case.get("qualification_protocol") != descriptor["qualification_protocol"]):
            raise ValueError("case receipt lacks the existing complete-set validator evidence")
        complete_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not isinstance(complete_manifest, dict) or
                receipt.get("complete_set_manifest") != complete_manifest):
            raise ValueError("case receipt summary differs from the current complete-set manifest")
        validate_complete_set_manifest(complete_manifest, one_case)
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
        if not isinstance(manifest, dict):
            raise ValueError(f"case {family} manifest must be an object")
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
            if not isinstance(row, dict):
                raise ValueError(f"case {family} dataset rows must be objects")
            identity_ok = ((row.get("scenario_id"), row.get("category")) == expected_key and
                row.get("seed") in shard["seeds"])
            metadata_ok = (family == "ub" or (
                row.get("protocol_id") == descriptor["qualification_protocol"]["id"] and
                row.get("qualification_protocol_sha256") == descriptor["qualification_protocol"]["sha256"] and
                row.get("source_identity_sha256") == _digest(descriptor["runtime_source_identity"]) and
                row.get("device") == device))
            if not identity_ok or not metadata_ok:
                raise ValueError(f"case {family} rows drift from their source manifest")
    return receipt


def collect_case_complete_set(case: dict, output_root: Path, *, lease_attestation=None) -> dict:
    """Use the reviewed one-case coordinator to collect and validate a shard."""
    from .meco_phase2_complete_set import describe_complete_set, execute_complete_set
    protocol = case["qualification_protocol"]
    one_case = describe_complete_set(protocol_path=Path(protocol["path"]),
        scenario_id=case["case_key"][0], category=case["case_key"][1],
        seeds=list(case["seeds"]), device=case["device"],
        output_root=Path(output_root))
    attestation = lease_attestation or case["device_identity_attestation"]
    execute_complete_set(one_case,
        lease_factory=lambda *_args, **_kwargs: nullcontext(),
        device_identity_verifier=lambda _lease, _devices: attestation)
    outputs = one_case["outputs"]
    manifest_path = Path(outputs["manifest"])
    complete_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = {
        "trajectory": Path(outputs["source_dataset"]),
        "ub": Path(one_case["plan_a_ubp"]["output_dir"]) / "plan-a-ub-legal-nodes.jsonl",
        "p": Path(one_case["plan_a_ubp"]["output_dir"]) / "plan-a-p-routes-oracle-v3.jsonl"}
    record = {"schema": SCHEMA + ".case-receipt",
        "descriptor_sha256": case["descriptor_sha256"],
        "case_key": list(case["case_key"]), "seeds": list(case["seeds"]),
        "source": case["source"], "qualification_protocol": case["qualification_protocol"],
        "device": case["device"], "one_case_descriptor": one_case,
        "attempt_root": str(output_root),
        "complete_set_manifest_path": str(manifest_path),
        "complete_set_manifest_sha256": _sha(manifest_path.read_bytes()),
        "complete_set_manifest": complete_manifest,
        "datasets": {}}
    for family, path in datasets.items():
        sidecar = path.with_suffix(path.suffix + ".manifest.json")
        record["datasets"][family] = {"path": str(path),
            "sha256": _sha(path.read_bytes()), "manifest_path": str(sidecar),
            "manifest_sha256": _sha(sidecar.read_bytes())}
    return record


def fit_multicase_warm_starts(combined, artifact_paths, *, descriptor, shard_receipts):
    """Fit all five native composite artifacts from strict original case shards."""
    from . import meco_supervised
    from .curriculum_training import source_identity
    validate_multicase_descriptor(descriptor)
    combined = _canonical_combined_inputs(combined)
    if not isinstance(shard_receipts, list) or len(shard_receipts) != len(descriptor["ordered_cases"]):
        raise ValueError("composite fitter requires every ordered case receipt")
    c_shards, a_shards = [], []
    for case, receipt in zip(descriptor["ordered_cases"], shard_receipts):
        datasets = receipt.get("datasets", {})
        if receipt.get("case_key") != [case["scenario_id"], case["category"]] or receipt.get("seeds") != case["seeds"]:
            raise ValueError("fitter shard receipt differs from frozen case ordering")
        c_shards.append({"path": datasets["trajectory"]["path"]})
        a_shards.append({"case_key": [case["scenario_id"], case["category"]],
            "seeds": list(case["seeds"]), "source_dataset": datasets["trajectory"]["path"],
            "legal_dataset": datasets["ub"]["path"], "p_dataset": datasets["p"]["path"]})
    expected_cases = descriptor["ordered_cases"]
    sources = source_identity()
    for track in MECO_TRACKS:
        output = Path(artifact_paths[track])
        if track in ("C1", "C2"):
            meco_supervised.warm_start_counterfactual_composite_v1(track, c_shards,
                output, expected_cases=expected_cases,
                qualification_protocol=Path(descriptor["qualification_protocol"]["path"]),
                updates=descriptor["c_settings"]["updates"],
                learning_rate=descriptor["c_settings"]["learning_rate"],
                device=descriptor["device"], descriptor_sha256=descriptor["descriptor_sha256"],
                combined_inputs=combined["trajectory"])
        else:
            meco_supervised.warm_start_plan_a_joint_ubp_v1(track, None, None, None,
                output, qualification_protocol=Path(descriptor["qualification_protocol"]["path"]),
                updates=descriptor["a_settings"]["updates"],
                learning_rate=descriptor["a_settings"]["learning_rate"],
                device=descriptor["device"],
                route_margin=descriptor["a_settings"]["route_margin"],
                map_regularization=descriptor["a_settings"]["map_regularization"],
                minimum_coverage_fraction=descriptor["a_settings"]["minimum_coverage_fraction"],
                composite_case_shards=a_shards,
                composite_descriptor_sha256=descriptor["descriptor_sha256"],
                composite_combined_inputs=combined)
        payload = meco_supervised.validate_warm_start_artifact(track, output,
            qualification_protocol=Path(descriptor["qualification_protocol"]["path"]),
            expected_sources=sources, expected_prediction_mode=descriptor["prediction_mode"])
        if (payload.get("case_set_sha256") != descriptor["case_set_sha256"] or
                payload.get("descriptor_sha256") != descriptor["descriptor_sha256"] or
                payload.get("track") != track):
            raise ValueError(f"{track} composite fitter did not preserve the frozen case set")
    return {"descriptor_sha256": descriptor["descriptor_sha256"],
        "case_set_sha256": descriptor["case_set_sha256"],
        "combined_inputs": combined,
        "fit_schema": SCHEMA + ".native-composite-fit-v1",
        "case_shard_bindings": [{"case_key": receipt["case_key"],
            "seeds": receipt["seeds"], "receipt_sha256": _digest(receipt)}
            for receipt in shard_receipts]}


def validate_multicase_warm_start_track(track, artifact_path, combined, *, descriptor):
    """Validate one concrete composite artifact and seal a five-track receipt."""
    from . import meco_supervised
    from .curriculum_training import source_identity
    combined = _canonical_combined_inputs(combined)
    if track not in MECO_TRACKS:
        raise ValueError("unknown MECO track")
    payload = meco_supervised.validate_warm_start_artifact(track, Path(artifact_path),
        qualification_protocol=Path(descriptor["qualification_protocol"]["path"]),
        expected_sources=source_identity(), expected_prediction_mode=descriptor["prediction_mode"])
    digest = _sha(Path(artifact_path).read_bytes())
    if (payload.get("track") != track or
            payload.get("case_set_sha256") != descriptor["case_set_sha256"] or
            payload.get("descriptor_sha256") != descriptor["descriptor_sha256"]):
        raise ValueError(f"{track} composite artifact does not bind the declared cases")
    return {"track": track, "artifact_sha256": digest,
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "case_set_sha256": descriptor["case_set_sha256"],
        "combined_inputs": combined,
        "artifact_schema": payload.get("schema"),
        "phase2_ready": False, "phase2_complete": False,
        "qualification_eligible": False}


def _canonical_combined_inputs(combined):
    """Project production combiner records to the immutable four-field contract."""
    if not isinstance(combined, dict) or set(combined) != {"trajectory", "ub", "p"}:
        raise ValueError("combined inputs must contain trajectory, U+B and P families")
    result = {}
    for family, item in combined.items():
        if not isinstance(item, dict):
            raise ValueError(f"combined {family} input is malformed")
        required = ("path", "sha256", "manifest_path", "manifest_sha256")
        projected = {key: item.get(key) for key in required}
        if any(not isinstance(projected[key], str) or not projected[key] for key in required):
            raise ValueError(f"combined {family} input lacks its exact data/manifest hash binding")
        result[family] = projected
    return result


def execute_multicase(descriptor: dict, *, collect_case=None, combine_target=None,
        fit_combined=None, validate_track=None, target_validators=None,
        lease_factory=None, device_identity_verifier=None) -> dict:
    """Run serial shards and pooled fitting under one verified device lease."""
    validate_multicase_descriptor(descriptor)
    root = Path(descriptor["outputs"]["root"])
    _check_output_namespace(root, (Path(descriptor["qualification_protocol"]["path"]),
        Path(__file__).resolve().parents[1]), allow_existing=True)
    if lease_factory is None:
        from .meco_device_lease import lease_devices
        lease_factory = lease_devices
    if device_identity_verifier is None:
        from .meco_gpu_attestation import verify_post_lease_wx9100
        device_identity_verifier = verify_post_lease_wx9100
    if not callable(lease_factory) or not callable(device_identity_verifier):
        raise ValueError("exclusive lease and indexed WX9100 identity verifier are required")
    if collect_case is None:
        collect_case = collect_case_complete_set
    production_fitter = fit_combined is None
    production_validator = validate_track is None
    if production_fitter:
        fit_combined = fit_multicase_warm_starts
    if production_validator:
        validate_track = validate_multicase_warm_start_track
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
            _write_private_json(descriptor_path, descriptor)
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
                        target_validators=target_validators, require_complete_set=True)
                    adopted = candidate
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                break
            if adopted is not None:
                shard_receipts.append(adopted)
                continue
            next_number = max([int(item.name.split("-")[1]) for item in attempts] + [0]) + 1
            case_root = case_dir / f"attempt-{next_number:04d}"
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
                target_validators=target_validators, require_complete_set=True)
            receipt_path = case_root / "case-receipt.json"
            if receipt_path.exists() or receipt_path.is_symlink():
                raise ValueError("case runner pre-created coordinator receipt path")
            _write_private_json(receipt_path, receipt)
            shard_receipts.append(receipt)

        combined = {}
        for family in ("trajectory", "ub", "p"):
            target_paths = [receipt["datasets"][family]["path"] for receipt in shard_receipts]
            target_path = Path(descriptor["outputs"]["combined"][family])
            source_binding = (None if family == "trajectory" else {
                "path": combined["trajectory"]["path"],
                "sha256": combined["trajectory"]["sha256"],
                "manifest_sha256": combined["trajectory"]["manifest_sha256"]})
            runner = combine_target or combine_target_shards
            combined[family] = runner(family, target_paths, target_path,
                expected_cases=expected_cases, descriptor=descriptor,
                source_binding=source_binding,
                source_shards=(None if family == "trajectory" else [
                    receipt["datasets"]["trajectory"] for receipt in shard_receipts]))
            path = Path(combined[family]["path"])
            if not path.is_file() or path.is_symlink() or _sha(path.read_bytes()) != combined[family].get("sha256"):
                raise ValueError(f"combined {family} corpus does not match returned digest")
            manifest_path = Path(combined[family].get("manifest_path", ""))
            if (not manifest_path.is_file() or manifest_path.is_symlink() or
                    _sha(manifest_path.read_bytes()) != combined[family].get("manifest_sha256")):
                raise ValueError(f"combined {family} corpus manifest is missing or stale")
            pooled_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (pooled_manifest.get("schema") != SCHEMA + ".combined-target" or
                    pooled_manifest.get("target_family") != family or
                    pooled_manifest.get("descriptor_sha256") != descriptor["descriptor_sha256"] or
                    pooled_manifest.get("case_set_sha256") != descriptor["case_set_sha256"] or
                    pooled_manifest.get("dataset_sha256") != combined[family]["sha256"] or
                    pooled_manifest.get("record_count") != combined[family].get("record_count") or
                    pooled_manifest.get("qualification_protocol") != descriptor["qualification_protocol"]):
                raise ValueError(f"combined {family} corpus manifest is not descriptor-bound")
            _validate_case_coverage_summary(pooled_manifest.get("case_coverage"), expected_cases)
            combined[family]["case_coverage"] = pooled_manifest["case_coverage"]
            if family == "trajectory":
                combined[family]["manifest_sha256"] = _sha(manifest_path.read_bytes())
        expected_combined = _canonical_combined_inputs(combined)
        target_coverage = {family: combined[family]["case_coverage"] for family in combined}
        if combined["ub"].get("source_dataset_sha256") != combined["trajectory"]["sha256"] or combined["p"].get("source_dataset_sha256") != combined["trajectory"]["sha256"]:
            raise ValueError("combined U+B and P targets do not bind pooled trajectory bytes")
        artifact_paths = {track: Path(descriptor["outputs"]["bundle"][track]) for track in MECO_TRACKS}
        for path in artifact_paths.values():
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fit_receipt = (fit_combined(expected_combined, artifact_paths, descriptor=descriptor,
            shard_receipts=shard_receipts) if production_fitter else
            fit_combined(expected_combined, artifact_paths, descriptor=descriptor))
        if not isinstance(fit_receipt, dict) or fit_receipt.get("descriptor_sha256") != descriptor["descriptor_sha256"] or fit_receipt.get("combined_inputs") != expected_combined:
            raise ValueError("combined fitter receipt is not bound to current pooled target bytes")
        validator_receipts = {}
        for track in MECO_TRACKS:
            validation = validate_track(track, artifact_paths[track], expected_combined, descriptor=descriptor)
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
            "target_coverage": target_coverage,
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
    combined = manifest.get("combined_inputs")
    target_coverage = manifest.get("target_coverage")
    if (not isinstance(combined, dict) or set(combined) != {"trajectory", "ub", "p"} or
            not isinstance(target_coverage, dict) or set(target_coverage) != {"trajectory", "ub", "p"}):
        raise ValueError("multi-case manifest omits a combined target family or coverage table")
    pooled_manifests = {}
    for family in ("trajectory", "ub", "p"):
        item = combined[family]
        data_path, sidecar_path = Path(item["path"]), Path(item["manifest_path"])
        if (data_path.is_symlink() or sidecar_path.is_symlink() or
                not data_path.is_file() or not sidecar_path.is_file() or
                _sha(data_path.read_bytes()) != item.get("sha256") or
                _sha(sidecar_path.read_bytes()) != item.get("manifest_sha256")):
            raise ValueError(f"combined {family} bytes changed after bundle publication")
        pooled = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (pooled.get("schema") != SCHEMA + ".combined-target" or
                pooled.get("target_family") != family or
                pooled.get("descriptor_sha256") != descriptor["descriptor_sha256"] or
                pooled.get("case_set_sha256") != descriptor["case_set_sha256"] or
                pooled.get("dataset_sha256") != item["sha256"] or
                pooled.get("qualification_protocol") != descriptor["qualification_protocol"] or
                pooled.get("case_coverage") != target_coverage[family]):
            raise ValueError(f"combined {family} manifest binding or coverage changed")
        _validate_case_coverage_summary(target_coverage[family], descriptor["ordered_cases"])
        pooled_manifests[family] = pooled
    trajectory_shards = pooled_manifests["trajectory"].get("shard_manifests")
    if not isinstance(trajectory_shards, list) or len(trajectory_shards) != len(descriptor["ordered_cases"]):
        raise ValueError("pooled trajectory manifest lost ordered source shard identities")
    expected_source_links = [{"case_key": shard["case_key"],
        "source_dataset_path": shard["path"],
        "source_dataset_sha256": shard["dataset_sha256"],
        "source_manifest_path": shard["manifest_path"],
        "source_manifest_sha256": shard["manifest_sha256"]} for shard in trajectory_shards]
    for family in ("ub", "p"):
        pooled = pooled_manifests[family]
        if (pooled.get("case_source_bindings") != expected_source_links or
                pooled.get("source_dataset_sha256") != combined["trajectory"]["sha256"] or
                pooled.get("source_manifest_sha256") != combined["trajectory"]["manifest_sha256"]):
            raise ValueError(f"combined {family} source links differ from ordered trajectory shards")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(MECO_TRACKS) or set(validator_receipts) != set(MECO_TRACKS) or set(artifact_paths) != set(MECO_TRACKS):
        raise ValueError("multi-case bundle must contain exactly five validated tracks")
    for track in MECO_TRACKS:
        path = Path(artifact_paths[track])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or unsafe {track} artifact")
        record, receipt = artifacts[track], validator_receipts[track]
        if (record.get("track") != track or Path(record.get("path", "")).resolve() != path.resolve() or
                record.get("sha256") != _sha(path.read_bytes()) or
                record.get("sha256") != receipt.get("artifact_sha256") or
                receipt.get("track") != track or receipt.get("case_set_sha256") != descriptor["case_set_sha256"] or
                receipt.get("descriptor_sha256") != descriptor["descriptor_sha256"]):
            raise ValueError(f"{track} validator receipt does not bind current artifact and case set")
    if manifest.get("phase2_ready") is not False or manifest.get("phase2_complete") is not False or manifest.get("qualification_eligible") is not False:
        raise ValueError("multi-case training bundle cannot claim readiness or qualification")
    return manifest


def validate_multicase_warm_start_bundle(descriptor_path, bundle_path, *,
        expected_descriptor_sha256=None, expected_descriptor_file_sha256=None,
        expected_bundle_file_sha256=None, expected_case_set_sha256=None,
        artifact_paths=None) -> dict:
    """Re-open and revalidate the published multi-case data and five artifacts.

    This is the production manifest-consumption boundary. It deliberately
    repeats expensive CPU-side validation and does not attest runtime devices
    or elevate any training/qualification gate.
    """
    from . import meco_supervised
    from .curriculum_training import source_identity

    descriptor_path = Path(descriptor_path).expanduser()
    bundle_path = Path(bundle_path).expanduser()
    for path in (descriptor_path, bundle_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("descriptor and bundle must be regular files")
    descriptor_path = descriptor_path.resolve(strict=True)
    bundle_path = bundle_path.resolve(strict=True)
    descriptor_raw, bundle_raw = descriptor_path.read_bytes(), bundle_path.read_bytes()
    if (expected_descriptor_file_sha256 is not None and
            _sha(descriptor_raw) != expected_descriptor_file_sha256):
        raise ValueError("descriptor bytes differ from the training binding")
    if expected_bundle_file_sha256 is not None and _sha(bundle_raw) != expected_bundle_file_sha256:
        raise ValueError("bundle bytes differ from the training binding")
    descriptor, manifest = json.loads(descriptor_raw), json.loads(bundle_raw)
    if not isinstance(descriptor, dict) or not isinstance(manifest, dict):
        raise ValueError("descriptor and bundle must be JSON objects")
    validate_multicase_descriptor(descriptor)
    if (expected_descriptor_sha256 is not None and
            descriptor.get("descriptor_sha256") != expected_descriptor_sha256):
        raise ValueError("descriptor digest differs from the training binding")
    if (expected_case_set_sha256 is not None and
            descriptor.get("case_set_sha256") != expected_case_set_sha256):
        raise ValueError("case-set digest differs from the training binding")
    if descriptor.get("outputs", {}).get("descriptor") != str(descriptor_path) or \
            descriptor.get("outputs", {}).get("manifest") != str(bundle_path):
        raise ValueError("descriptor output paths do not identify the bound files")

    artifact_records = manifest.get("artifacts")
    if not isinstance(artifact_records, dict) or set(artifact_records) != set(MECO_TRACKS):
        raise ValueError("published bundle must identify exactly five artifacts")
    paths = artifact_paths or {track: Path(item["path"])
        for track, item in artifact_records.items()}
    paths = {track: Path(path).expanduser() for track, path in paths.items()}
    if set(paths) != set(MECO_TRACKS) or any(path.is_symlink() or not path.is_file()
            for path in paths.values()):
        raise ValueError("five-track artifacts must be regular non-symlink files")
    paths = {track: path.resolve(strict=True) for track, path in paths.items()}
    validate_five_track_manifest(manifest, descriptor,
        validator_receipts=manifest.get("validator_receipts", {}), artifact_paths=paths)
    canonical = _canonical_combined_inputs(manifest["combined_inputs"])
    for family in ("trajectory", "ub", "p"):
        expected_data = Path(descriptor["outputs"]["combined"][family]).resolve(strict=True)
        expected_sidecar = expected_data.with_suffix(expected_data.suffix + ".manifest.json")
        if (Path(canonical[family]["path"]).resolve(strict=True) != expected_data or
                Path(canonical[family]["manifest_path"]).resolve(strict=True) != expected_sidecar):
            raise ValueError(f"combined {family} files differ from the frozen descriptor output paths")

    # Revalidate original one-case receipts and prove that each pooled family
    # was recomputed from the exact ordered shards named by those receipts.
    pooled = {}
    case_reports = {f"{c['scenario_id']}::{c['category']}": {
        "case_key": [c["scenario_id"], c["category"]], "seeds": list(c["seeds"]),
        "target_families": {}} for c in descriptor["ordered_cases"]}
    receipt_rows = manifest.get("shard_receipts")
    if not isinstance(receipt_rows, list) or len(receipt_rows) != len(descriptor["shards"]):
        raise ValueError("bundle omits complete ordered case receipts")
    sidecars = {family: json.loads(Path(canonical[family]["manifest_path"]).read_text(encoding="utf-8"))
        for family in ("trajectory", "ub", "p")}
    receipts = []
    for index, (shard, row) in enumerate(zip(descriptor["shards"], receipt_rows)):
        if (not isinstance(row, dict) or row.get("case_key") != shard["case_key"] or
                row.get("seeds") != shard["seeds"]):
            raise ValueError("bundle case receipt order or identity differs from descriptor")
        entries = {}
        for family in ("trajectory", "ub", "p"):
            shard_entries = sidecars[family].get("shard_manifests")
            if not isinstance(shard_entries, list) or len(shard_entries) != len(descriptor["shards"]):
                raise ValueError(f"combined {family} sidecar has incomplete shard bindings")
            entry = shard_entries[index]
            if (entry.get("case_key") != shard["case_key"] or entry.get("seeds") != shard["seeds"]):
                raise ValueError(f"combined {family} sidecar case order differs from descriptor")
            entries[family] = entry
        receipt_path = Path(entries["trajectory"]["path"]).parent / "case-receipt.json"
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise ValueError("original case receipt is missing or unsafe")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if _digest(receipt) != row.get("receipt_sha256"):
            raise ValueError("original case receipt differs from the published receipt digest")
        validate_case_receipt(receipt, shard, descriptor, require_complete_set=True)
        receipts.append(receipt)
        for target, actual in entries.items():
            source = receipt["datasets"][target]
            if any(actual.get(k) != source.get(src_k) for k, src_k in (
                    ("path", "path"), ("manifest_path", "manifest_path"),
                    ("dataset_sha256", "sha256"), ("manifest_sha256", "manifest_sha256"))):
                raise ValueError(f"combined {target} shard binding differs from its case receipt")
            case_reports["::".join(shard["case_key"])]["target_families"][target] = {
                "dataset_sha256": source["sha256"], "manifest_sha256": source["manifest_sha256"],
                "record_count": json.loads(Path(source["manifest_path"]).read_text()).get("record_count")}

    for family in ("trajectory", "ub", "p"):
        item = canonical[family]
        fd, temp_name = tempfile.mkstemp(prefix="." + Path(item["path"]).name + ".revalidate-",
            dir=Path(item["path"]).parent)
        os.close(fd)
        temp = Path(temp_name)
        temp.unlink()
        try:
            result = combine_jsonl_shards(
                [Path(r["datasets"][family]["path"]) for r in receipts], temp,
                expected_cases=descriptor["ordered_cases"])
            if result["sha256"] != item["sha256"]:
                raise ValueError(f"combined {family} bytes do not equal rederived shard bytes")
            pooled[family] = {"record_count": result["record_count"],
                "case_coverage": result["case_coverage"]}
        finally:
            temp.unlink(missing_ok=True)

    track_reports = {}
    protocol_path = Path(descriptor["qualification_protocol"]["path"])
    for track in MECO_TRACKS:
        path = paths[track]
        payload = meco_supervised.validate_warm_start_artifact(track, path,
            qualification_protocol=protocol_path, expected_sources=source_identity(),
            expected_prediction_mode=descriptor["prediction_mode"])
        if (payload.get("case_set_sha256") != descriptor["case_set_sha256"] or
                payload.get("descriptor_sha256") != descriptor["descriptor_sha256"] or
                payload.get("track") != track):
            raise ValueError(f"{track} artifact does not bind the exact descriptor and case set")
        expected_inputs = (canonical["trajectory"] if track.startswith("C") else canonical)
        payload_inputs = payload.get("combined_input", payload.get("combined_inputs"))
        if payload_inputs != expected_inputs:
            raise ValueError(f"{track} artifact does not bind the published combined corpus")
        track_reports[track] = {"artifact_sha256": _sha(path.read_bytes()),
            "artifact_schema": payload.get("schema"),
            "case_coverage": payload.get("case_coverage", payload.get("case_coverage_by_case"))}

    return {"schema": SCHEMA + ".production-validation-receipt-v1",
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "case_set_sha256": descriptor["case_set_sha256"],
        "bundle_manifest_sha256": _sha(bundle_raw), "combined_inputs": pooled,
        "cases": case_reports, "tracks": track_reports,
        "training_readiness_proven": False, "phase2_complete": False,
        "qualification_eligible": False}


def export_training_manifest_warm_starts(manifest: dict, descriptor: dict) -> dict[str, str]:
    """Forward all five track artifact paths and preserve the frozen case set."""
    if manifest.get("descriptor_sha256") != descriptor.get("descriptor_sha256"):
        raise ValueError("training manifest export requires the exact multi-case descriptor")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(MECO_TRACKS):
        raise ValueError("training manifest export requires all five tracks")
    # This mapping is directly accepted by the training manifest's exact-five
    # path field. Case-set binding stays in the bundle manifest, not this map.
    return {track: artifacts[track]["path"] for track in MECO_TRACKS}


def export_training_manifest_binding(manifest: dict, descriptor: dict) -> dict:
    """Export and byte-bind the descriptor/bundle case set for training manifests."""
    if (manifest.get("descriptor_sha256") != descriptor.get("descriptor_sha256") or
            manifest.get("case_set_sha256") != descriptor.get("case_set_sha256")):
        raise ValueError("training manifest binding requires the exact multi-case bundle")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(MECO_TRACKS):
        raise ValueError("training manifest binding requires all five tracks")
    descriptor_path = Path(descriptor["outputs"]["descriptor"])
    bundle_path = Path(descriptor["outputs"]["manifest"])
    if not descriptor_path.is_file() or not bundle_path.is_file():
        raise ValueError("training manifest binding requires published descriptor and bundle files")
    if json.loads(bundle_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("training manifest export differs from the published bundle bytes")
    return {"schema": SCHEMA + ".training-binding",
        "descriptor_path": str(descriptor_path),
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "descriptor_file_sha256": _sha(descriptor_path.read_bytes()),
        "bundle_manifest_path": str(bundle_path),
        "bundle_manifest_sha256": _sha(bundle_path.read_bytes()),
        "case_set_sha256": descriptor["case_set_sha256"],
        "artifact_sha256_by_track": {track: artifacts[track]["sha256"]
            for track in MECO_TRACKS}}
