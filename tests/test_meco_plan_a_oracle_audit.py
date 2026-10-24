import hashlib
import json

import pytest

from frc_defense.meco_counterfactual import (DATASET_SCHEMA,
    TARGET_SCHEMA, PLAN_A_LEGAL_NODE_DATASET_SCHEMA, PLAN_A_LEGAL_NODE_TARGET_SCHEMA)
from frc_defense.meco_counterfactual import _artifact_identity_sha256, _hash
from frc_defense.meco_plan_a_oracle_audit import audit_plan_a_legal_nodes


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False) + "\n").encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path):
    sources = {"tracked_sources": ["source.py"], "_native_fuel_extension": {"status": "present"}}
    source_identity = _sha(json.dumps(sources, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode())
    protocol = {"protocol_version": "meco-qualification-v1", "master_seed": 7,
        "cases": [{"scenario_id": "pickup_simple", "category": "single",
            "split": "train", "seeds": [17]}]}
    protocol_id = _sha(json.dumps(protocol, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode())
    protocol["protocol_id"] = protocol_id
    protocol_bytes = _bytes(protocol)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(protocol_bytes)
    source_row = {"scenario_id": "pickup_simple", "category": "single", "seed": 17,
        "decision_index": 3, "branch_state": {"snapshot_id": "snap"},
        "predecision_simulator_state_sha256": "state", "support_digest": "support",
        "source_identity_sha256": source_identity}
    source_bytes = _bytes(source_row)
    source_dataset = tmp_path / "source.jsonl"
    source_dataset.write_bytes(source_bytes)
    source_manifest = {"schema": DATASET_SCHEMA, "target_schema": TARGET_SCHEMA,
        "split": "train",
        "records_have_prefix_replay_parity": True,
        "native_extension_identity_schema": 1,
        "native_extension_identity": sources["_native_fuel_extension"],
        "artifact_identity_sha256": _artifact_identity_sha256(source_identity,
            sources["_native_fuel_extension"]),
        "record_count": 1, "dataset_sha256": _sha(source_bytes),
        "protocol_id": protocol_id, "qualification_protocol_sha256": _sha(protocol_bytes),
        "qualification_protocol_file": protocol_path.name, "scenario_id": "pickup_simple",
        "category": "single", "seeds": [17], "sources": sources,
        "source_identity_sha256": source_identity, "device": {"device": "cuda:0"}}
    source_manifest_bytes = _bytes(source_manifest)
    source_dataset.with_suffix(".jsonl.manifest.json").write_bytes(source_manifest_bytes)
    legal_row = {"schema": PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
        "target_schema": PLAN_A_LEGAL_NODE_TARGET_SCHEMA,
        "scenario_id": "pickup_simple", "category": "single", "seed": 17,
        "decision_index": 3,
        "source_identity_sha256": source_identity, "snapshot_id": "snap",
        "state_sha256": "state", "support_digest": "support",
        "feasible_node_ids": [0, 1, 2, 3], "selected_node_ids": [0, 2],
        "canonical_poses": [{"node_id": i, "heading_bin": i % 2, "region_id": i // 2}
            for i in range(4)],
        "full_pair_layout": [{"pair_index": i, "node_id": node,
            "behavior_id": behavior, "descriptor_id": f"d-{i}", "feasible": True}
            for i, (node, behavior) in enumerate([(0, 1), (1, 2), (1, 1),
                (2, 2), (3, 1)])],
        "node_u_targets": [
            {"node_id": 0, "status": "available", "u_target": 1.,
                "feasible_behavior_ids": [1],
                "available_behavior_ids": [1],
                "censored_feasible_behavior_ids": [], "node_inclusion_probability": .5,
                "unavailable_behavior_ids": [0, 2, 3, 4],
                "node_inverse_probability_weight": 2.,
                "behavior_repeat_return_variances": {"1": 1.5}},
            {"node_id": 2, "status": "unavailable", "u_target": None,
                "feasible_behavior_ids": [2],
                "available_behavior_ids": [],
                "censored_feasible_behavior_ids": [2], "node_inclusion_probability": .5,
                "unavailable_behavior_ids": [0, 1, 3, 4],
                "node_inverse_probability_weight": 2.,
                "behavior_repeat_return_variances": {}}],
        "pairs": [{"pair_index": 0, "node_id": 0, "behavior_id": 1, "descriptor_id": "d-0",
            "status": "available", "gross_horizon_return": 1.,
            "descriptor_inclusion_probability": .25,
            "descriptor_inverse_probability_weight": 4.,
            "repeat_returns": [{"status": "available", "gross_horizon_return": 1.},
                {"status": "available", "gross_horizon_return": 1.}]},
            {"pair_index": 3, "node_id": 2, "behavior_id": 2, "descriptor_id": "d-3",
            "status": "unavailable",
            "descriptor_inclusion_probability": 1.,
            "descriptor_inverse_probability_weight": 1.,
            "repeat_returns": [{"status": "unavailable"}, {"status": "unavailable"}]}],
        "behavior_preferences": [{"status": "comparable"}, {"status": "tie"},
            {"status": "uncertain"}]}
    legal_row["full_pair_layout_sha256"] = _hash(legal_row["full_pair_layout"])
    legal_row["selection_digest"] = _hash({"protocol_id": protocol_id,
        "key": ("pickup_simple", "single", 17, 3), "support_digest": "support",
        "selected_node_ids": [0, 2], "selected_pair_indices": [0, 3]})
    legal_bytes = _bytes(legal_row)
    legal_dataset = tmp_path / "legal.jsonl"
    legal_dataset.write_bytes(legal_bytes)
    legal_manifest = {"schema": PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
        "target_schema": PLAN_A_LEGAL_NODE_TARGET_SCHEMA, "split": "train",
        "record_count": 1, "dataset_sha256": _sha(legal_bytes),
        "source_dataset_sha256": _sha(source_bytes),
        "source_manifest_sha256": _sha(source_manifest_bytes),
        "source_dataset": source_dataset.name,
        "protocol_id": protocol_id, "qualification_protocol_sha256": _sha(protocol_bytes),
        "qualification_protocol_file": protocol_path.name, "sources": sources,
        "source_identity_sha256": source_identity, "device": {"device": "cuda:0"}}
    legal_dataset.with_suffix(".jsonl.manifest.json").write_bytes(_bytes(legal_manifest))
    return legal_dataset, source_dataset, protocol_path


def test_offline_audit_summarizes_coverage_labels_and_weights_without_claiming_replay(tmp_path):
    args = _fixture(tmp_path)
    report = audit_plan_a_legal_nodes(*args)
    assert report["coverage"]["sampled_legal_nodes"] == 2
    assert report["coverage"]["feasible_legal_nodes"] == 4
    assert report["coverage"]["sampled_heading_region_counts"] == {"0:0": 1, "0:1": 1}
    assert report["coverage"]["feasible_node_behavior_counts"] == {"1": 3, "2": 2}
    assert report["coverage"]["sampled_node_behavior_counts"] == {"1": 1, "2": 1}
    assert report["u_summary"]["available_node_count"] == 1
    assert report["u_summary"]["censored_node_count"] == 1
    assert report["u_summary"]["repeat_available_count"] == 2
    assert report["u_summary"]["per_behavior_repeat_return_variance"]["1"]["mean"] == 1.5
    assert report["b_summary"] == {"preference_count": 3, "comparable_count": 1,
        "tie_count": 1, "uncertain_count": 1}
    assert report["sampling_weights"]["node"]["effective_sample_size"] == 2.
    assert report["sampling_weights"]["descriptor"]["effective_sample_size"] == 1.4705882352941178
    assert report["evidence_status"] == {"replay_validated": False,
        "empirical_oracle_established": False, "warm_start_artifact_produced": False}


def test_audit_rejects_modified_dataset_bytes(tmp_path):
    args = _fixture(tmp_path)
    args[0].write_bytes(args[0].read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest bytes"):
        audit_plan_a_legal_nodes(*args)


def test_audit_rejects_heldout_seed_even_with_updated_dataset_digest(tmp_path):
    args = _fixture(tmp_path)
    row = json.loads(args[0].read_text())
    row["seed"] = 999
    data = _bytes(row)
    args[0].write_bytes(data)
    manifest_path = args[0].with_suffix(".jsonl.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_sha256"] = _sha(data)
    manifest_path.write_bytes(_bytes(manifest))
    with pytest.raises(ValueError, match="out-of-train"):
        audit_plan_a_legal_nodes(*args)
