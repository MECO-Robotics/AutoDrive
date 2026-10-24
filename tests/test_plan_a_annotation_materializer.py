import hashlib
import json

import pytest

from frc_defense import meco_counterfactual as counterfactual
from frc_defense import meco_plan_a_annotations as materializer
from frc_defense import meco_supervised as supervised


def _write(path, data):
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _json(path, value):
    return _write(path, (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())


def _jsonl(path, rows):
    return _write(path, b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows))


def _fixture(tmp_path, monkeypatch):
    scenario, category, seed, decision = "pickup_simple", "single", 103, 0
    key = {"scenario_id": scenario, "category": category, "seed": seed, "decision_index": decision}
    protocol_path = tmp_path / "protocol.json"
    protocol = {"protocol_id": "fake"}
    monkeypatch.setattr(supervised, "_load_protocol", lambda _p: (protocol, "protocol-id", "p" * 64, protocol_path))
    monkeypatch.setattr(supervised, "_protocol_train_seeds", lambda *_args: {seed})
    base = tmp_path / "base.jsonl"
    base_record = {**key, "payload": "base", "branch_state": {"snapshot_id": "snap"},
        "predecision_simulator_state_sha256": "a" * 64}
    base_sha = _jsonl(base, [base_record])
    base_manifest = base.with_suffix(base.suffix + ".manifest.json")
    base_manifest_sha = _json(base_manifest, {"schema": "base"})
    desc = "d" * 64
    goal = {"node_id": 7, "x_m": 1.0, "y_m": 2.0, "heading_bin": 3,
            "descriptor_sha256": desc}
    source = tmp_path / "expert.jsonl"
    source_selected = {"node_id": 7, "behavior_id": 1, "action": 0, "descriptor_sha256": desc}
    source_sha = _jsonl(source, [{**key, "schema": counterfactual.PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA,
        "target_schema": counterfactual.PLAN_A_EXPERT_GOAL_SOURCE_TARGET_SCHEMA,
        "selected_goal": source_selected, "snapshot_id": "snap", "state_sha256": "a" * 64,
        "teacher_selected_goal_descriptor_sha256": desc, "executed_action": 0,
        "execution_failure": False, "support_digest": "r" * 64, "base_record": base_record}])
    source_manifest = source.with_suffix(source.suffix + ".manifest.json")
    sources = {"fixture": "source"}
    source_manifest_sha = _json(source_manifest, {
        "schema": counterfactual.PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA,
        "target_schema": counterfactual.PLAN_A_EXPERT_GOAL_SOURCE_TARGET_SCHEMA,
        "dataset_sha256": source_sha, "split": "train", "protocol_id": "protocol-id",
        "qualification_protocol_sha256": "p" * 64, "records_have_replay_bound_selected_goal": True,
        "base_dataset": base.name, "base_dataset_sha256": base_sha,
        "base_manifest_sha256": base_manifest_sha, "sources": sources,
        "source_identity_sha256": supervised._canonical_hash(sources), "record_count": 1})
    assert source_manifest_sha
    legal = tmp_path / "legal.jsonl"
    legal_sha = _jsonl(legal, [{**key,
        "schema": counterfactual.PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
        "target_schema": counterfactual.PLAN_A_LEGAL_NODE_TARGET_SCHEMA,
        "snapshot_id": "snap", "state_sha256": "a" * 64,
        "support_digest": "r" * 64,
        "canonical_poses": [{k: goal[k] for k in ("node_id", "x_m", "y_m", "heading_bin")}],
        "feasible_node_ids": [7], "pairs": [{"node_id": 7, "behavior_id": 1, "status": "available"}],
        "node_u_targets": [{"node_id": 7, "status": "unavailable",
            "feasible_behavior_ids": [1], "available_behavior_ids": [],
            "censored_feasible_behavior_ids": []}]}])
    legal_manifest = legal.with_suffix(legal.suffix + ".manifest.json")
    _json(legal_manifest, {"schema": counterfactual.PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
        "target_schema": counterfactual.PLAN_A_LEGAL_NODE_TARGET_SCHEMA, "dataset_sha256": legal_sha,
        "source_dataset_sha256": base_sha, "source_manifest_sha256": base_manifest_sha,
        "expert_goal_source_sha256": source_sha, "expert_goal_source_manifest_sha256": source_manifest_sha,
        "protocol_id": "protocol-id", "qualification_protocol_sha256": "p" * 64,
        "split": "train", "sources": sources, "record_count": 1,
        "source_identity_sha256": supervised._canonical_hash(sources)})
    authored = tmp_path / "authored.jsonl"
    row = {**key, "selected_goal": goal, "executed_action": 0, "selected_behavior_id": 1,
        "censored_feasible_behavior_ids": [], "u_status": "unavailable", "u_target": None}
    _jsonl(authored, [row])
    authority = tmp_path / "authority.json"
    _json(authority, {"source_id": "declared-review-panel", "notes": "declared only"})
    return authored, source, legal, protocol_path, authority


def _run(paths, output):
    authored, source, legal, protocol, authority = paths
    return materializer.materialize_plan_a_annotations(authored_rows=authored,
        source_dataset=source, legal_node_dataset=legal, qualification_protocol=protocol,
        authority_json=authority, output=output)


def test_materializer_writes_deterministic_loader_schema_and_disclaimer(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "annotations.jsonl"
    first = _run(paths, out)
    bytes1, manifest1 = out.read_bytes(), out.with_suffix(out.suffix + ".manifest.json").read_bytes()
    second = _run(paths, out)
    assert bytes1 == out.read_bytes() and manifest1 == out.with_suffix(out.suffix + ".manifest.json").read_bytes()
    assert first["dataset_sha256"] == second["dataset_sha256"]
    row = json.loads(bytes1)
    manifest = json.loads(manifest1)
    assert row["schema"] == supervised.PLAN_A_EXPERT_GOAL_DATASET_SCHEMA
    assert row["u_status"] == "unavailable" and row["u_target"] is None
    assert manifest["replay_validation_required"] is True
    assert manifest["authority_is_authenticated_identity"] is False
    assert manifest["authority_proves_expertise_or_optimality"] is False


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r.update(seed=999), "exactly cover source rows"),
    (lambda r: r.update(executed_action=2), "mismatch replay-bound"),
    (lambda r: r.pop("selected_goal"), "incomplete fields"),
    (lambda r: r.update(selected_goal={"node_id": 99}), "mismatch replay-bound"),
    (lambda r: r.update(u_status="censored", u_target=0.0), "unavailable/censored U"),
    (lambda r: r.update(u_status="available", u_target=float("nan")), "non-finite JSON number"),
    (lambda r: r.update(rule="minimum_route_cost"), "unsupported/incomplete fields"),
])
def test_materializer_rejects_invalid_authored_rows(tmp_path, monkeypatch, mutate, message):
    paths = list(_fixture(tmp_path, monkeypatch))
    row = json.loads(paths[0].read_text())
    mutate(row)
    _jsonl(paths[0], [row])
    with pytest.raises(ValueError, match=message):
        _run(paths, tmp_path / "out.jsonl")


def test_materializer_rejects_duplicate_and_extra_identities(tmp_path, monkeypatch):
    paths = list(_fixture(tmp_path, monkeypatch))
    row = json.loads(paths[0].read_text())
    _jsonl(paths[0], [row, row])
    with pytest.raises(ValueError, match="duplicate authored"):
        _run(paths, tmp_path / "duplicate.jsonl")
    extra = {**row, "seed": 104}
    _jsonl(paths[0], [extra])
    with pytest.raises(ValueError, match="extra=1"):
        _run(paths, tmp_path / "extra.jsonl")


def test_materializer_rejects_heldout_seed_and_nonfinite_json(tmp_path, monkeypatch):
    paths = list(_fixture(tmp_path, monkeypatch))
    monkeypatch.setattr(supervised, "_protocol_train_seeds", lambda *_args: {104})
    with pytest.raises(ValueError, match="heldout or undeclared"):
        _run(paths, tmp_path / "heldout.jsonl")
    paths = list(_fixture(tmp_path, monkeypatch))
    row = json.loads(paths[0].read_text())
    row["u_status"], row["u_target"] = "available", 1.0
    # NaN is rejected by the JSON parser before packaging.
    paths[0].write_text(json.dumps(row, allow_nan=True) + "\n")
    with pytest.raises(ValueError, match="non-finite|finite"):
        _run(paths, tmp_path / "nan.jsonl")
