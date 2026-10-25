"""Offline contract tests for source-bound Phase 0 freeze and split contracts."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from frc_defense.meco_baseline_freeze import (
    AUTHORITATIVE_FIELD_SOURCES,
    PARTITION_NAMES,
    PARTITION_WORKFLOWS,
    canonical_sha256,
    compile_baseline_freeze,
    compile_partition_contract,
    validate_baseline_freeze,
    validate_partition_contract,
)


SOURCE_PLAN = "docs/source-plan.md"


def _init_repo(root: Path) -> None:
    paths = {SOURCE_PLAN}
    for required in AUTHORITATIVE_FIELD_SOURCES.values():
        paths.update(required)
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic source fixture: {relative}\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Fixture"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "fixture@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "synthetic contract fixture"], check=True)


def _runtime_identity() -> dict[str, str]:
    digest = hashlib.sha256(b"synthetic fixture only").hexdigest()
    return {
        "simulator_id": "synthetic-fixture-simulator",
        "simulator_version": "synthetic-fixture-version",
        "simulator_build_sha256": digest,
        "python_version": "synthetic-fixture-python",
        "torch_version": "synthetic-fixture-torch",
        "backend": "synthetic-fixture-backend",
        "backend_version": "synthetic-fixture-backend-version",
        "device_id": "synthetic-fixture-device",
        "dependency_lock_sha256": digest,
    }


def _field_records(root: Path) -> dict[str, dict]:
    records = {}
    for field, paths in AUTHORITATIVE_FIELD_SOURCES.items():
        if field == "scoring_observation_semantics":
            value = {
                "simulator_metric": "fuel_score_count",
                "metric_semantics": "scored_fuel_piece_count",
                "official_match_points_available": False,
            }
        else:
            value = {"fixture_only": True, "field": field}
        records[field] = {
            "value": value,
            "selector": f"synthetic_fixture.{field}",
            "sources": [
                {
                    "path": path,
                    "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(),
                }
                for path in paths
            ],
        }
    return records


def _valid_partitions() -> dict[str, dict[str, list]]:
    return {
        "train": {
            "seeds": [1, 2], "defender_families": ["train_policy"],
            "layouts": ["train_layout"], "start_conditions": ["train_start"],
        },
        "calibration": {
            "seeds": [3, 4], "defender_families": ["cal_policy"],
            "layouts": ["cal_layout"], "start_conditions": ["cal_start"],
        },
        "validation": {
            "seeds": [5, 6], "defender_families": ["validation_policy"],
            "layouts": ["validation_layout"], "start_conditions": ["validation_start"],
        },
        "final_test": {
            "seeds": [7, 8], "defender_families": ["heldout_policy"],
            "layouts": ["heldout_layout"], "start_conditions": ["heldout_start"],
        },
    }


def _valid_uses() -> dict[str, list[str]]:
    return {name: sorted(PARTITION_WORKFLOWS[name]) for name in PARTITION_NAMES}


def test_baseline_freeze_is_bound_to_clean_source_runtime_and_field_sources(tmp_path):
    _init_repo(tmp_path)
    freeze = compile_baseline_freeze(
        tmp_path,
        source_plan_path=SOURCE_PLAN,
        runtime_identity=_runtime_identity(),
        field_records=_field_records(tmp_path),
    )

    assert freeze["qualification_eligible"] is False
    assert freeze == compile_baseline_freeze(
        tmp_path,
        source_plan_path=SOURCE_PLAN,
        runtime_identity=_runtime_identity(),
        field_records=_field_records(tmp_path),
    )
    assert validate_baseline_freeze(freeze, tmp_path) == freeze

    changed = dict(freeze)
    changed["runtime"] = dict(freeze["runtime"], device_id="different-device")
    with pytest.raises(ValueError, match="runtime identity digest"):
        validate_baseline_freeze(changed, tmp_path)


def test_baseline_freeze_rejects_missing_runtime_field_and_dirty_source(tmp_path):
    _init_repo(tmp_path)
    runtime = _runtime_identity()
    runtime.pop("device_id")
    with pytest.raises(ValueError, match="runtime identity"):
        compile_baseline_freeze(
            tmp_path,
            source_plan_path=SOURCE_PLAN,
            runtime_identity=runtime,
            field_records=_field_records(tmp_path),
        )

    (tmp_path / "frc_defense/tensor_3v3.py").write_text("dirty\n")
    with pytest.raises(ValueError, match="clean source checkout"):
        compile_baseline_freeze(
            tmp_path,
            source_plan_path=SOURCE_PLAN,
            runtime_identity=_runtime_identity(),
            field_records=_field_records(tmp_path),
        )


def test_baseline_freeze_rejects_missing_source_field_or_changed_source_hash(tmp_path):
    _init_repo(tmp_path)
    records = _field_records(tmp_path)
    del records["legal_connectivity"]
    with pytest.raises(ValueError, match="baseline fields must be complete"):
        compile_baseline_freeze(
            tmp_path,
            source_plan_path=SOURCE_PLAN,
            runtime_identity=_runtime_identity(),
            field_records=records,
        )

    records = _field_records(tmp_path)
    records["field_geometry"]["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source digest mismatch"):
        compile_baseline_freeze(
            tmp_path,
            source_plan_path=SOURCE_PLAN,
            runtime_identity=_runtime_identity(),
            field_records=records,
        )


def test_scoring_field_cannot_claim_official_match_points_or_convert_fuel_counts(tmp_path):
    _init_repo(tmp_path)
    records = _field_records(tmp_path)
    records["scoring_observation_semantics"]["value"] = {
        "simulator_metric": "match_points",
        "metric_semantics": "official_match_points",
        "official_match_points_available": True,
    }
    with pytest.raises(ValueError, match="FUEL-piece-count-only"):
        compile_baseline_freeze(
            tmp_path,
            source_plan_path=SOURCE_PLAN,
            runtime_identity=_runtime_identity(),
            field_records=records,
        )

    records = _field_records(tmp_path)
    records["scoring_observation_semantics"]["value"]["official_match_points_available"] = 0
    with pytest.raises(ValueError, match="FUEL-piece-count-only"):
        compile_baseline_freeze(
            tmp_path,
            source_plan_path=SOURCE_PLAN,
            runtime_identity=_runtime_identity(),
            field_records=records,
        )


def test_partition_contract_is_deterministic_disjoint_and_final_test_isolated():
    baseline_digest = hashlib.sha256(b"synthetic freeze").hexdigest()
    contract = compile_partition_contract(
        baseline_freeze_sha256=baseline_digest,
        partitions=_valid_partitions(),
        allowed_uses=_valid_uses(),
    )
    assert contract["qualification_eligible"] is False
    assert contract == compile_partition_contract(
        baseline_freeze_sha256=baseline_digest,
        partitions=_valid_partitions(),
        allowed_uses=_valid_uses(),
    )
    assert validate_partition_contract(
        contract, baseline_freeze_sha256=baseline_digest) == contract
    with pytest.raises(ValueError, match="does not bind"):
        validate_partition_contract(
            contract,
            baseline_freeze_sha256=hashlib.sha256(b"different freeze").hexdigest(),
        )

    changed = dict(contract)
    changed["partition_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest or canonical form"):
        validate_partition_contract(changed, baseline_freeze_sha256=baseline_digest)


@pytest.mark.parametrize("axis", ["seeds", "defender_families", "layouts", "start_conditions"])
def test_partition_contract_rejects_cross_partition_leakage(axis):
    partitions = _valid_partitions()
    partitions["final_test"][axis] = list(partitions["train"][axis])
    with pytest.raises(ValueError, match="leaks across"):
        compile_partition_contract(
            baseline_freeze_sha256=hashlib.sha256(b"baseline").hexdigest(),
            partitions=partitions,
            allowed_uses=_valid_uses(),
        )


def test_partition_contract_rejects_final_test_use_for_training_or_margin_review():
    uses = _valid_uses()
    uses["final_test"] = ["final_evaluation", "training"]
    with pytest.raises(ValueError, match="preregistered policy"):
        compile_partition_contract(
            baseline_freeze_sha256=hashlib.sha256(b"baseline").hexdigest(),
            partitions=_valid_partitions(),
            allowed_uses=uses,
        )
