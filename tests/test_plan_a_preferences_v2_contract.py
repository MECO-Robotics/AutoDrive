import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

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


def test_signed_choice_flows_through_v2_replay_and_fitter(tmp_path, monkeypatch):
    real_legal_reader = supervised._read_and_replay_plan_a_legal_node_dataset
    paths = _fixture(tmp_path, monkeypatch)
    authored, source, legal, protocol_path, authority_path, legal_row, legal_manifest, source_row = paths
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(serialization.Encoding.Raw,
        serialization.PublicFormat.Raw)
    protocol = {"protocol_id": "test", "plan_a_expert_authority_keys": {
        "pinned-human-key": public.hex()}}
    protocol_id, protocol_sha = "frozen-protocol", "p" * 64
    monkeypatch.setattr(supervised, "_load_protocol",
        lambda _path: (protocol, protocol_id, protocol_sha, protocol_path))
    sources = {"curriculum": "source"}
    monkeypatch.setattr("frc_defense.curriculum_training.source_identity", lambda: sources)
    source_manifest_path = source.with_suffix(".jsonl.manifest.json")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    source_manifest = {"schema": counterfactual.DATASET_SCHEMA,
        "device": {"device": "cpu"}, "split": "train",
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "sources": sources, "source_identity_sha256": supervised._canonical_hash(sources),
        "dataset_sha256": source_sha}
    source_manifest_sha = _write(source_manifest_path, source_manifest)

    # Bind the human pick to a choice that differs from the deterministic
    # minimum-route pair (node 2 / behavior 1).
    legal_row["pairs"].append({"node_id": 1, "behavior_id": 2,
        "pair_index": 3, "status": "available"})
    _write_jsonl(legal, [legal_row])
    legal_bytes = legal.read_bytes()
    legal_sha = hashlib.sha256(legal_bytes).hexdigest()
    legal_manifest.update({"u_oracle_version": counterfactual.PLAN_A_U_ORACLE_VERSION,
        "u_oracle_semantics": counterfactual.PLAN_A_U_ORACLE_SEMANTICS,
        "prediction_mode": "current",
        "prediction_contract": supervised.prediction_contract("current"),
        "continuation_policy": counterfactual.CONTINUATION_POLICY,
        "source_dataset_sha256": source_sha,
        "source_manifest_sha256": source_manifest_sha,
        "device": {"device": "cpu"}, "sources": sources,
        "source_identity_sha256": supervised._canonical_hash(sources),
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "split": "train"})
    legal_manifest["dataset_sha256"] = legal_sha
    legal_manifest_sha = _write(legal.with_suffix(".jsonl.manifest.json"), legal_manifest)
    state_sha, snapshot = source_row["predecision_simulator_state_sha256"], "snapshot-11"
    choice = {"schema": preferences.SIGNED_CHOICE_SCHEMA,
        "scenario_id": "pickup_simple", "category": "single", "seed": 11,
        "decision_index": 0, "snapshot_id": snapshot, "state_sha256": state_sha,
        "support_digest": "d" * 64, "node_id": 1, "behavior_id": 2,
        "baseline_node_id": 2, "baseline_behavior_id": 1}
    sidecar = tmp_path / "signed-choices.jsonl"
    _write_jsonl(sidecar, [choice])
    sidecar_bytes = sidecar.read_bytes()
    source_manifest_bytes = source_manifest_path.read_bytes()
    from frc_defense import meco_plan_a_expert_choice as signed
    signed_payload = signed.authority_payload(key_id="pinned-human-key",
        author_id="human-7", basis="selected feasible action in reviewed replay",
        sidecar_sha256=hashlib.sha256(sidecar_bytes).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256(source_manifest_bytes).hexdigest(),
        replay_sha256=legal_sha,
        replay_manifest_sha256=hashlib.sha256(
            legal.with_suffix(".jsonl.manifest.json").read_bytes()).hexdigest())
    signed_authority = {"schema": signed.SCHEMA, "payload": signed_payload,
        "signature_ed25519": private.sign(signed.canonical_bytes(signed_payload)).hex()}
    Path(str(sidecar) + ".authority.json").write_text(json.dumps(signed_authority))
    _write(authority_path, {"source_id": "legacy-source-metadata"})

    monkeypatch.setattr(supervised, "_read_counterfactual_dataset",
        lambda *_args, **_kwargs: ([source_row], source_sha, source_manifest_sha,
            protocol, protocol_id, protocol_sha, source_manifest))

    out = tmp_path / "signed-preferences.jsonl"
    preferences.materialize_plan_a_preferences_v2(source_dataset=source,
        legal_node_dataset=legal, qualification_protocol=protocol_path,
        authority_json=authority_path, output=out, signed_choice_sidecar=sidecar)
    materialized = json.loads(out.read_text())
    assert materialized["signed_human_choice"]["node_id"] == 1
    assert materialized["goal_preference_tiers"] == []
    assert materialized["goal_preferences"] == []
    assert materialized["action_pair_preferences"][0]["preferred_node_id"] == 1
    assert materialized["action_pair_preferences"][0]["right_node_id"] == 2

    support = SimpleNamespace(valid_mask=torch.tensor([True, True, True, True]),
        pair_node_ids=torch.tensor([1, 1, 2, 1]),
        behavior_ids=torch.tensor([1, 2, 1, 2]),
        descriptor_ids=torch.tensor([-1, -1, -1, -1]),
        action_ids=torch.tensor([0, 1, 0, 1]),
        route_cost=torch.tensor([float("inf"), 4.0, 1.0, float("inf")]),
        goal_xy=torch.tensor([[0., 0.], [1., 1.], [2., 2.], [3., 3.]]),
        goal_heading=torch.zeros(4), support_digest="d" * 64)
    observation = SimpleNamespace(raster=[0.], raster_shape=(1, 1), globals=[0.] * 7)
    fake_env = SimpleNamespace(close=lambda: None,
        sim=SimpleNamespace(field_length=10., field_width=5.))
    fake_graph = SimpleNamespace()
    def fake_replay(*_args, **_kwargs):
        return {"env": fake_env, "graph": fake_graph, "support": support,
            "observation": observation}
    monkeypatch.setattr(counterfactual, "replay_counterfactual_branch", fake_replay)
    raw_legal = legal.read_bytes()
    monkeypatch.setattr(counterfactual, "collect_plan_a_legal_nodes_v1",
        lambda output, **_kwargs: Path(output).write_bytes(raw_legal))
    monkeypatch.setattr("frc_defense.meco_track_training._graph_inputs",
        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(supervised, "_read_and_replay_plan_a_legal_node_dataset",
        real_legal_reader)
    protocol["plan_a_expert_authority_keys"] = {}
    with pytest.raises(ValueError, match="no pinned Plan A expert authority keys"):
        preferences.load_plan_a_preferences_v2(out, legal, source,
            qualification_protocol=protocol_path, expected_sources=sources,
            authority_json=authority_path, device="cpu", track=None)
    protocol["plan_a_expert_authority_keys"] = {"pinned-human-key": public.hex()}
    validated, *_ = preferences.load_plan_a_preferences_v2(out, legal, source,
        qualification_protocol=protocol_path, expected_sources=sources,
        authority_json=authority_path, device="cpu", track=None)
    assert len(validated) == 1
    assert validated[0][0]["action_pair_preferences"][0]["preferred_behavior_id"] == 2

    # Rewriting the unsigned preference rows and manifest must not let an
    # attacker change the signed selected-vs-baseline target or add labels.
    tampered = tmp_path / "tampered-preferences.jsonl"
    tampered_row = dict(materialized)
    tampered_row.update({
        "goal_preference_tiers": [[2], [1]],
        "goal_preferences": [{"left_node_id": 2, "right_node_id": 1,
            "preferred_node_id": 2}],
        "behavior_preferences": [{"node_id": 1, "left_behavior_id": 1,
            "right_behavior_id": 2, "preferred_behavior_id": 1}],
        "action_pair_preferences": [{"left_node_id": 2, "left_behavior_id": 1,
            "right_node_id": 1, "right_behavior_id": 2,
            "preferred_node_id": 2, "preferred_behavior_id": 1}],
    })
    tampered_sha = _write_jsonl(tampered, [tampered_row])
    tampered_manifest_path = tampered.with_suffix(".jsonl.manifest.json")
    tampered_manifest = json.loads(out.with_suffix(".jsonl.manifest.json").read_text())
    tampered_manifest["dataset_sha256"] = tampered_sha
    _write(tampered_manifest_path, tampered_manifest)
    with pytest.raises(ValueError,
            match="signed action-pair labels differ from signed choice and replayed baseline"):
        preferences.load_plan_a_preferences_v2(tampered, legal, source,
            qualification_protocol=protocol_path, expected_sources=sources,
            authority_json=authority_path, device="cpu", track=None)

    # Exercise the actual fitter and weights-only artifact validator on CPU.
    class TinyTrack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.utility = torch.nn.Parameter(torch.zeros(1, 4))
            self.scores = torch.nn.Parameter(torch.zeros(4))
            self.p_head = torch.nn.Linear(1, 2)
            with torch.no_grad():
                self.p_head.weight.zero_()
                self.p_head.bias.copy_(torch.tensor([0., -20.]))

        def forward(self, **_kwargs):
            return {"utility": self.utility, "scores": self.scores}

    monkeypatch.setattr(supervised, "_prepare_native_gpu_runtime", lambda *_args: None)
    monkeypatch.setattr("frc_defense.meco_device.require_indexed_cuda_device",
        lambda device, **_kwargs: device)
    monkeypatch.setattr(supervised, "make_meco_track", lambda *_args, **_kwargs: TinyTrack())
    monkeypatch.setattr("frc_defense.meco_plan_a_policy.gather_pose_behavior_logits",
        lambda _track, raw, *_args, **_kwargs: raw["scores"])
    monkeypatch.setattr(supervised, "_initialize_unlabelled_plan_a_penalty_head",
        lambda *_args: None)
    monkeypatch.setattr(supervised, "_plan_a_p_output_parameters",
        lambda model, _track: (model.p_head, torch.ones(2)))
    monkeypatch.setattr(counterfactual, "_device_identity", lambda device: {"device": device})
    result = supervised.warm_start_plan_a_preferences_v2("A2", out, legal, source,
        authority_path, tmp_path / "fitted.pt", qualification_protocol=protocol_path,
        updates=2, learning_rate=1e-2, device="cpu")
    assert result["action_pairs"] == 1
    artifact = torch.load(tmp_path / "fitted.pt", map_location="cpu", weights_only=True)
    assert artifact["action_pair_count"] == 1
    assert artifact["annotated_record_count"] == 1


def test_signed_choice_rejects_untrusted_protocol_key(tmp_path, monkeypatch):
    # The protocol, rather than an editable sidecar or authority file, is the
    # verifier's trust anchor. No key in it means fail closed.
    protocol = {"protocol_id": "test", "plan_a_expert_authority_keys": {}}
    # Verify that a key outside the qualification protocol is not accepted.
    with pytest.raises(ValueError, match="no pinned Plan A expert authority keys"):
        preferences._protocol_trusted_keys(protocol)
