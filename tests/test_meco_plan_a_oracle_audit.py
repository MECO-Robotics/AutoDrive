import hashlib
import json

import pytest

from frc_defense.meco_counterfactual import (DATASET_SCHEMA,
    TARGET_SCHEMA, PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA,
    PLAN_A_EXPERT_GOAL_SOURCE_TARGET_SCHEMA, PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
    PLAN_A_LEGAL_NODE_TARGET_SCHEMA, PLAN_A_U_ORACLE_VERSION,
    PLAN_A_U_ORACLE_SEMANTICS, CONTINUATION_POLICY,
    _plan_a_u_evidence_sha256)
from frc_defense.meco_counterfactual import _artifact_identity_sha256, _hash
from frc_defense.meco_plan_a_oracle_audit import audit_plan_a_legal_nodes


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False) + "\n").encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path, *, expert=True):
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
    expert_path = tmp_path / "expert.jsonl"
    expert_sha = expert_manifest_sha = None
    if expert:
        expert_row = {"scenario_id": "pickup_simple", "category": "single",
            "seed": 17, "decision_index": 3, "base_record": source_row,
            "snapshot_id": "snap", "state_sha256": "state", "support_digest": "support",
            "execution_failure": False, "selected_goal": {"node_id": 2}}
        expert_bytes = _bytes(expert_row)
        expert_path.write_bytes(expert_bytes)
        expert_manifest_bytes = _bytes({"schema": PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA,
            "target_schema": PLAN_A_EXPERT_GOAL_SOURCE_TARGET_SCHEMA, "split": "train",
            "base_dataset_sha256": _sha(source_bytes),
            "base_manifest_sha256": _sha(source_manifest_bytes), "protocol_id": protocol_id,
            "qualification_protocol_sha256": _sha(protocol_bytes), "sources": sources,
            "source_identity_sha256": source_identity, "device": {"device": "cuda:0"},
            "record_count": 1, "dataset_sha256": _sha(expert_bytes)})
        expert_path.with_suffix(".jsonl.manifest.json").write_bytes(expert_manifest_bytes)
        expert_sha, expert_manifest_sha = _sha(expert_bytes), _sha(expert_manifest_bytes)
    selected_nodes = [0, 2] if expert else [0]
    expert_node_pi = 1. if expert else .5
    expert_node_weight = 1. if expert else 2.
    legal_row = {"schema": PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
        "target_schema": PLAN_A_LEGAL_NODE_TARGET_SCHEMA,
        "u_oracle_version": PLAN_A_U_ORACLE_VERSION,
        "scenario_id": "pickup_simple", "category": "single", "seed": 17,
        "decision_index": 3,
        "source_identity_sha256": source_identity, "snapshot_id": "snap",
        "state_sha256": "state", "support_digest": "support",
        "feasible_node_ids": list(range(8)), "selected_node_ids": selected_nodes,
        "canonical_poses": [{"node_id": i, "heading_bin": i % 2, "region_id": (i % 4) // 2}
            for i in range(8)],
        "full_pair_layout": [{"pair_index": i, "node_id": node,
            "behavior_id": behavior, "descriptor_id": f"d-{i}", "feasible": True}
            for i, (node, behavior) in enumerate([(0, 1), (0, 1), (0, 2),
                (0, 3), (0, 4), (1, 2), (1, 1), (2, 2), (3, 1),
                (4, 1), (5, 2), (6, 2), (7, 1)])],
        "node_u_targets": [
            {"node_id": 0, "status": "available", "u_target": 1.75,
                "feasible_behavior_ids": [1, 2, 3, 4],
                "available_behavior_ids": [1, 2, 3, 4],
                "censored_feasible_behavior_ids": [], "node_inclusion_probability": 1.,
                "unavailable_behavior_ids": [0],
                "node_inverse_probability_weight": 1.,
                "behavior_repeat_return_variances": {"1": 0., "2": 0.,
                    "3": 0., "4": 1.}, "u_oracle_version": PLAN_A_U_ORACLE_VERSION},
            *([{"node_id": 2, "status": "unavailable", "u_target": None,
                "feasible_behavior_ids": [2],
                "available_behavior_ids": [],
                "censored_feasible_behavior_ids": [2], "node_inclusion_probability": expert_node_pi,
                "unavailable_behavior_ids": [0, 1, 3, 4],
                "node_inverse_probability_weight": expert_node_weight,
                "behavior_repeat_return_variances": {},
                "u_oracle_version": PLAN_A_U_ORACLE_VERSION}] if expert else []),
        ],
        "root_node": 0, "root_node_behavior_supported": True,
        "node_inclusion_probabilities": ({"0": 1., "2": expert_node_pi}
            if expert else {"0": 1.}),
        "stratum_population_counts": {"0:0": 2, "1:0": 2, "0:1": 2, "1:1": 2},
        "stratum_selected_counts": {"0:0": 1, "1:0": 0,
            "0:1": int(expert), "1:1": 0},
        "pairs": [], "behavior_preferences": []}
    outcome_returns = {1: [2., 2.], 2: [1., 1.], 3: [2., 2.], 4: [1., 3.]}
    for behavior, values in outcome_returns.items():
        pair_index = {1: 0, 2: 2, 3: 3, 4: 4}[behavior]
        population = 2 if behavior == 1 else 1
        legal_row["pairs"].append({"pair_index": pair_index, "node_id": 0,
            "behavior_id": behavior, "descriptor_id": f"d-{pair_index}",
            "status": "available", "gross_horizon_return": sum(values) / 2,
            "behavior_descriptor_support_count": population,
            "descriptor_inclusion_probability": 1 / population,
            "descriptor_inverse_probability_weight": population,
            "repeat_returns": [{"status": "available", "gross_horizon_return": value,
                "repeat_seed": int.from_bytes(hashlib.sha256(
                    f"legal-node-repeat:17:{repeat}".encode()).digest()[:4], "big")}
                for repeat, value in enumerate(values)]})
    if expert:
        legal_row["pairs"].append({"pair_index": 7, "node_id": 2,
        "behavior_id": 2, "descriptor_id": "d-7", "status": "unavailable",
        "behavior_descriptor_support_count": 1,
        "descriptor_inclusion_probability": 1., "descriptor_inverse_probability_weight": 1.,
        "repeat_outcomes": [{"status": "unavailable", "repeat_seed": int.from_bytes(
            hashlib.sha256(f"legal-node-repeat:17:{repeat}".encode()).digest()[:4], "big")}
            for repeat in range(2)]})
    for target in legal_row["node_u_targets"]:
        target["u_evidence_sha256"] = _plan_a_u_evidence_sha256(
            "support", target, legal_row["pairs"])
    for offset, left in enumerate(sorted(outcome_returns)):
        for right in sorted(outcome_returns)[offset + 1:]:
            differences = [a - b for a, b in zip(outcome_returns[left], outcome_returns[right])]
            signs = [0 if value == 0 else 1 if value > 0 else -1 for value in differences]
            status = ("comparable" if len(set(signs)) == 1 and signs[0] != 0 else
                "tie" if all(value == 0 for value in differences) else "uncertain")
            mean_diff = sum(differences) / 2
            legal_row["behavior_preferences"].append({"node_id": 0,
                "left_behavior_id": left, "right_behavior_id": right,
                "left_mean_return": sum(outcome_returns[left]) / 2,
                "right_mean_return": sum(outcome_returns[right]) / 2,
                "paired_repeat_differences": differences,
                "paired_repeat_difference_variance": sum((value - mean_diff) ** 2
                    for value in differences) / 2,
                "preferred_behavior_id": (left if signs[0] > 0 else right)
                    if status == "comparable" else None,
                "status": status})
    legal_row["full_pair_layout_sha256"] = _hash(legal_row["full_pair_layout"])
    legal_row["selection_digest"] = _hash({"protocol_id": protocol_id,
        "key": ("pickup_simple", "single", 17, 3), "support_digest": "support",
        "selected_node_ids": selected_nodes,
        "selected_pair_indices": [0, 2, 3, 4, *([7] if expert else [])]})
    legal_bytes = _bytes(legal_row)
    legal_dataset = tmp_path / "legal.jsonl"
    legal_dataset.write_bytes(legal_bytes)
    legal_manifest = {"schema": PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
        "target_schema": PLAN_A_LEGAL_NODE_TARGET_SCHEMA, "split": "train",
        "u_oracle_version": PLAN_A_U_ORACLE_VERSION,
        "u_oracle_semantics": PLAN_A_U_ORACLE_SEMANTICS,
        "continuation_policy": CONTINUATION_POLICY,
        "record_count": 1, "dataset_sha256": _sha(legal_bytes),
        "source_dataset_sha256": _sha(source_bytes),
        "source_manifest_sha256": _sha(source_manifest_bytes),
        "source_dataset": source_dataset.name,
        "protocol_id": protocol_id, "qualification_protocol_sha256": _sha(protocol_bytes),
        "qualification_protocol_file": protocol_path.name, "sources": sources,
        "source_identity_sha256": source_identity, "device": {"device": "cuda:0"}}
    legal_manifest["expert_goal_source"] = expert_path.name if expert else None
    legal_manifest["expert_goal_source_sha256"] = expert_sha
    legal_manifest["expert_goal_source_manifest_sha256"] = expert_manifest_sha
    legal_dataset.with_suffix(".jsonl.manifest.json").write_bytes(_bytes(legal_manifest))
    return legal_dataset, source_dataset, protocol_path


def _rewrite_legal(args, mutate):
    dataset, _, _ = args
    row = json.loads(dataset.read_text())
    mutate(row)
    data = _bytes(row)
    dataset.write_bytes(data)
    manifest_path = dataset.with_suffix(".jsonl.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_sha256"] = _sha(data)
    manifest_path.write_bytes(_bytes(manifest))


def test_offline_audit_summarizes_coverage_labels_and_weights_without_claiming_replay(tmp_path):
    args = _fixture(tmp_path)
    report = audit_plan_a_legal_nodes(*args)
    assert report["coverage"]["sampled_legal_nodes"] == 2
    assert report["coverage"]["feasible_legal_nodes"] == 8
    assert report["coverage"]["sampled_heading_region_counts"] == {"0:0": 1, "0:1": 1}
    assert report["coverage"]["feasible_node_behavior_counts"] == {
        "1": 5, "2": 5, "3": 1, "4": 1}
    assert report["coverage"]["sampled_node_behavior_counts"] == {
        "1": 1, "2": 2, "3": 1, "4": 1}
    assert report["u_summary"]["available_node_count"] == 1
    assert report["u_summary"]["censored_node_count"] == 1
    assert report["u_summary"]["repeat_available_count"] == 8
    assert report["u_summary"]["per_behavior_repeat_return_variance"]["4"]["mean"] == 1.
    assert report["b_summary"] == {"preference_count": 6, "comparable_count": 2,
        "tie_count": 1, "uncertain_count": 3}
    assert report["sampling_weights"]["node"]["effective_sample_size"] == 2.
    assert report["sampling_weights"]["descriptor"]["effective_sample_size"] == 4.5
    assert report["evidence_status"] == {"replay_validated": False,
        "empirical_oracle_established": False, "warm_start_artifact_produced": False}


def test_audit_counts_replay_bound_expert_goal_as_a_certainty_unit(tmp_path):
    report = audit_plan_a_legal_nodes(*_fixture(tmp_path, expert=True))
    assert report["provenance"]["expert_goal_source"]["dataset_sha256"]
    assert report["sampling_weights"]["node"]["effective_sample_size"] == 2.


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


def test_audit_recomputes_behavior_variance_from_repeats(tmp_path):
    args = _fixture(tmp_path)
    _rewrite_legal(args, lambda row: row["node_u_targets"][0][
        "behavior_repeat_return_variances"].update({"4": 999.}))
    with pytest.raises(ValueError, match="repeat variance differs"):
        audit_plan_a_legal_nodes(*args)


def test_audit_binds_u_target_to_exact_matched_rollout_returns(tmp_path):
    args = _fixture(tmp_path)
    _rewrite_legal(args, lambda row: row["pairs"][0]["repeat_returns"][0].update(
        {"gross_horizon_return": 4.}))
    with pytest.raises(ValueError, match="evidence digest|mean differs"):
        audit_plan_a_legal_nodes(*args)


def test_audit_rejects_rewritten_u_evidence_digest(tmp_path):
    args = _fixture(tmp_path)
    _rewrite_legal(args, lambda row: row["node_u_targets"][0].update(
        {"u_evidence_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="evidence digest"):
        audit_plan_a_legal_nodes(*args)


def test_audit_rejects_unpaired_repeat_seed(tmp_path):
    args = _fixture(tmp_path)
    _rewrite_legal(args, lambda row: row["pairs"][0]["repeat_returns"][0].update(
        {"repeat_seed": 0}))
    with pytest.raises(ValueError, match="repeat seed"):
        audit_plan_a_legal_nodes(*args)


def test_audit_requires_explicit_u_oracle_semantics(tmp_path):
    args = _fixture(tmp_path)
    manifest_path = args[0].with_suffix(".jsonl.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["u_oracle_semantics"] = "expert optimality"
    manifest_path.write_bytes(_bytes(manifest))
    with pytest.raises(ValueError, match="manifest bytes"):
        audit_plan_a_legal_nodes(*args)


def test_audit_requires_exact_feasible_censoring_inventory(tmp_path):
    args = _fixture(tmp_path)
    _rewrite_legal(args, lambda row: row["node_u_targets"][1][
        "censored_feasible_behavior_ids"].clear())
    with pytest.raises(ValueError, match="inventories are inconsistent"):
        audit_plan_a_legal_nodes(*args)


@pytest.mark.parametrize("mutation", ["remove", "wrong_status"])
def test_audit_requires_exact_b_pair_coverage_and_repeat_status(tmp_path, mutation):
    args = _fixture(tmp_path)
    def mutate(row):
        if mutation == "remove":
            row["behavior_preferences"].pop()
        else:
            row["behavior_preferences"][0]["status"] = "tie"
    _rewrite_legal(args, mutate)
    with pytest.raises(ValueError, match="B preference"):
        audit_plan_a_legal_nodes(*args)


def test_audit_binds_descriptor_probability_to_full_support_population(tmp_path):
    args = _fixture(tmp_path)
    def mutate(row):
        pair = next(item for item in row["pairs"] if item["pair_index"] == 0)
        pair["descriptor_inclusion_probability"] = .25
        pair["descriptor_inverse_probability_weight"] = 4.
    _rewrite_legal(args, mutate)
    with pytest.raises(ValueError, match="descriptor inclusion"):
        audit_plan_a_legal_nodes(*args)


@pytest.mark.parametrize("node_id,pi,weight,expected_message", [
    (2, .25, 4., "node inclusion probability"),
    (0, .5, 2., "node inclusion probability"),
])
def test_audit_recomputes_node_inclusion_from_strata_and_certainty_units(
        tmp_path, node_id, pi, weight, expected_message):
    args = _fixture(tmp_path)
    def mutate(row):
        target = next(item for item in row["node_u_targets"] if item["node_id"] == node_id)
        target["node_inclusion_probability"] = pi
        target["node_inverse_probability_weight"] = weight
        row["node_inclusion_probabilities"][str(node_id)] = pi
    _rewrite_legal(args, mutate)
    with pytest.raises(ValueError, match=expected_message):
        audit_plan_a_legal_nodes(*args)


def test_audit_rejects_coherently_rewritten_non_sampled_node(tmp_path):
    args = _fixture(tmp_path, expert=False)
    assert audit_plan_a_legal_nodes(*args)["coverage"]["sampled_legal_nodes"] == 1

    def inject_node_two(row):
        row["selected_node_ids"] = [0, 2]
        row["node_inclusion_probabilities"]["2"] = .5
        row["stratum_selected_counts"]["0:1"] = 1
        row["node_u_targets"].append({"node_id": 2, "status": "unavailable",
            "u_target": None, "feasible_behavior_ids": [2],
            "available_behavior_ids": [], "censored_feasible_behavior_ids": [2],
            "node_inclusion_probability": .5, "unavailable_behavior_ids": [0, 1, 3, 4],
            "node_inverse_probability_weight": 2.,
            "behavior_repeat_return_variances": {}})
        row["pairs"].append({"pair_index": 7, "node_id": 2, "behavior_id": 2,
            "descriptor_id": "d-7", "status": "unavailable",
            "behavior_descriptor_support_count": 1,
            "descriptor_inclusion_probability": 1.,
            "descriptor_inverse_probability_weight": 1.,
            "repeat_outcomes": [{"status": "unavailable"}, {"status": "unavailable"}]})
        row["selection_digest"] = _hash({"protocol_id": json.loads(
            args[2].read_text())["protocol_id"],
            "key": ("pickup_simple", "single", 17, 3), "support_digest": "support",
            "selected_node_ids": [0, 2], "selected_pair_indices": [0, 2, 3, 4, 7]})

    _rewrite_legal(args, inject_node_two)
    with pytest.raises(ValueError, match="deterministic capped round-robin sample"):
        audit_plan_a_legal_nodes(*args)


def test_audit_rejects_feasible_node_missing_from_layout_after_digest_rewrite(tmp_path):
    args = _fixture(tmp_path, expert=False)

    def remove_node_seven_layout_rows(row):
        row["full_pair_layout"] = [pair for pair in row["full_pair_layout"]
            if pair["node_id"] != 7]
        for index, pair in enumerate(row["full_pair_layout"]):
            pair["pair_index"] = index
        row["full_pair_layout_sha256"] = _hash(row["full_pair_layout"])

    _rewrite_legal(args, remove_node_seven_layout_rows)
    with pytest.raises(ValueError, match="feasible node inventory differs"):
        audit_plan_a_legal_nodes(*args)
