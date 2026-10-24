import hashlib
import json

import pytest

from frc_defense import meco_counterfactual as counterfactual
from frc_defense import meco_plan_a_preferences_v2 as preferences
from frc_defense import meco_supervised as supervised


def _write(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_jsonl(path, rows):
    raw = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _fixture(tmp_path, monkeypatch):
    key = {"scenario_id": "pickup_simple", "category": "single", "seed": 11,
        "decision_index": 0}
    sources = {"curriculum": "source"}
    protocol_path = tmp_path / "protocol.json"
    _write(protocol_path, {"protocol_id": "test"})
    protocol = {"protocol_id": "test"}
    protocol_id, protocol_sha = "frozen-protocol", "p" * 64
    monkeypatch.setattr(supervised, "_load_protocol",
        lambda _path: (protocol, protocol_id, protocol_sha, protocol_path))
    monkeypatch.setattr(supervised, "_protocol_train_seeds", lambda *_args: {11})
    monkeypatch.setattr("frc_defense.curriculum_training.source_identity", lambda: sources)
    state_sha, support_digest, snapshot_id = "a" * 64, "d" * 64, "snapshot-11"
    source_row = {**key, "schema": counterfactual.DATASET_SCHEMA,
        "target_schema": counterfactual.TARGET_SCHEMA, "protocol_id": protocol_id,
        "qualification_protocol_sha256": protocol_sha,
        "source_identity_sha256": supervised._canonical_hash(sources),
        "device": {"device": "cuda:0"}, "branch_state": {"snapshot_id": snapshot_id},
        "predecision_simulator_state_sha256": state_sha,
        "support_digest": support_digest,
        "teacher_selected_goal": {"node_id": 3}}
    source = tmp_path / "source.jsonl"
    source_sha = _write_jsonl(source, [source_row])
    source_manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    source_manifest_sha = _write(source_manifest_path, {"schema": counterfactual.DATASET_SCHEMA})
    source_manifest = {"device": {"device": "cuda:0"}}
    monkeypatch.setattr(supervised, "_read_counterfactual_dataset", lambda *_args, **_kwargs:
        ([source_row], source_sha, source_manifest_sha, protocol, protocol_id, protocol_sha,
            source_manifest))
    legal_row = {**key, "schema": counterfactual.PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
        "target_schema": counterfactual.PLAN_A_LEGAL_NODE_TARGET_SCHEMA,
        "snapshot_id": snapshot_id, "state_sha256": state_sha,
        "support_digest": support_digest, "feasible_node_ids": [1, 2, 3],
        "canonical_poses": [{"node_id": node, "x_m": float(node), "y_m": 2.0,
            "heading_bin": 0} for node in (1, 2, 3)],
        "pairs": [{"node_id": 1, "behavior_id": 1, "pair_index": 0, "status": "available"},
            {"node_id": 1, "behavior_id": 2, "pair_index": 1, "status": "available"},
            {"node_id": 2, "behavior_id": 1, "pair_index": 2, "status": "available"}],
        "node_u_targets": []}
    legal = tmp_path / "legal.jsonl"
    legal_sha = _write_jsonl(legal, [legal_row])
    legal_manifest_path = legal.with_suffix(legal.suffix + ".manifest.json")
    legal_manifest = {"schema": counterfactual.PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
        "target_schema": counterfactual.PLAN_A_LEGAL_NODE_TARGET_SCHEMA,
        "dataset_sha256": legal_sha, "source_dataset_sha256": source_sha,
        "source_manifest_sha256": source_manifest_sha, "protocol_id": protocol_id,
        "qualification_protocol_sha256": protocol_sha, "split": "train",
        "sources": sources, "source_identity_sha256": supervised._canonical_hash(sources),
        "device": {"device": "cuda:0"}, "record_count": 1}
    legal_manifest_sha = _write(legal_manifest_path, legal_manifest)
    monkeypatch.setattr(supervised, "_read_and_replay_plan_a_legal_node_dataset",
        lambda *_args, **_kwargs: ([{"row": legal_row}], legal_manifest,
            legal_sha, legal_manifest_sha))
    authored = tmp_path / "authored.jsonl"
    _write_jsonl(authored, [{**key, "goal_preference_tiers": [[1], [2, 3]],
        "behavior_preferences": [{"node_id": 1, "left_behavior_id": 1,
            "right_behavior_id": 2, "preferred_behavior_id": 2}]}])
    authority = tmp_path / "authority.json"
    _write(authority, {"source_id": "declared-panel"})
    return authored, source, legal, protocol_path, authority, legal_row, legal_manifest, source_row


def _materialize(paths, output):
    authored, source, legal, protocol, authority = paths[:5]
    return preferences.materialize_plan_a_preferences_v2(authored_rows=authored,
        source_dataset=source, legal_node_dataset=legal, qualification_protocol=protocol,
        authority_json=authority, output=output)


def test_v2_binds_independent_goal_ranking_and_keeps_numeric_u_unavailable(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "preferences.jsonl"
    result = _materialize(paths, output)
    row = json.loads(output.read_text())
    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    assert result["rows"] == 1
    assert row["goal_preference_tiers"] == [[1], [2, 3]]
    assert row["goal_preferences"] == [
        {"left_node_id": 1, "right_node_id": 2, "preferred_node_id": 1},
        {"left_node_id": 1, "right_node_id": 3, "preferred_node_id": 1}]
    assert row["behavior_preferences"][0]["preferred_behavior_id"] == 2
    assert row["u_numeric_status"] == "unavailable" and row["u_numeric_target"] is None
    assert manifest["u_numeric_target"] is None and manifest["replay_validation_required"] is True
    assert paths[7]["teacher_selected_goal"]["node_id"] == 3


@pytest.mark.parametrize("mutate, message", [
    (lambda row: row.update(goal_preference_tiers=[[1], [99]]), "outside replayed feasible"),
    (lambda row: row.update(extra="unsupported"), "unsupported or missing fields"),
    (lambda row: row.update(behavior_preferences=[{"node_id": 1, "left_behavior_id": 1,
        "right_behavior_id": 99, "preferred_behavior_id": 1}]), "infeasible or unavailable"),
])
def test_v2_rejects_unsupported_preferences(tmp_path, monkeypatch, mutate, message):
    paths = list(_fixture(tmp_path, monkeypatch))
    authored = json.loads(paths[0].read_text())
    mutate(authored)
    _write_jsonl(paths[0], [authored])
    with pytest.raises(ValueError, match=message):
        _materialize(paths, tmp_path / "bad.jsonl")


def test_v2_replay_loader_checks_bound_rows_and_protocol(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    _materialize(paths, tmp_path / "preferences.jsonl")
    validated, manifest, *_ = preferences.load_plan_a_preferences_v2(
        tmp_path / "preferences.jsonl", paths[2], paths[1],
        qualification_protocol=paths[3], expected_sources={"curriculum": "source"},
        authority_json=paths[4], device="cpu", track=None)
    assert len(validated) == 1
    assert manifest["target_schema"] == preferences.TARGET_SCHEMA
    assert validated[0][0]["u_numeric_status"] == "unavailable"


def test_v2_rejects_heldout_seed(tmp_path, monkeypatch):
    paths = list(_fixture(tmp_path, monkeypatch))
    monkeypatch.setattr(supervised, "_protocol_train_seeds", lambda *_args: {12})
    with pytest.raises(ValueError, match="heldout or undeclared"):
        _materialize(paths, tmp_path / "heldout.jsonl")
