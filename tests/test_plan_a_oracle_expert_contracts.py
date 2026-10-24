import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from frc_defense import meco_counterfactual as counterfactual
from frc_defense import meco_supervised as supervised


def test_plan_a_p_oracle_integral_uses_grid_edge_units_for_axis_and_diagonal_segments():
    grid = [[1.0, 2.0], [3.0, 4.0]]

    horizontal = counterfactual._plan_a_p_oracle_integral(
        grid, [(0.25, 0.25), (1.75, 0.25)], 1.0)
    vertical = counterfactual._plan_a_p_oracle_integral(
        grid, [(0.25, 0.25), (0.25, 1.75)], 1.0)
    diagonal = counterfactual._plan_a_p_oracle_integral(
        grid, [(0.25, 0.25), (1.75, 1.75)], 1.0)

    assert horizontal == pytest.approx(3.0)
    assert vertical == pytest.approx(2.25)
    assert diagonal == pytest.approx(3.75 * 2 ** 0.5)


def test_plan_a_p_oracle_integral_uses_high_side_cell_for_boundary_aligned_segment():
    grid = [[1.0, 2.0], [3.0, 4.0]]

    cost = counterfactual._plan_a_p_oracle_integral(
        grid, [(1.0, 0.2), (1.0, 1.8)], 1.0)

    assert cost == pytest.approx(5.6)


def test_plan_a_p_oracle_grid_includes_base_once_and_scales_effective_p_once():
    env = SimpleNamespace(planner=SimpleNamespace(
        _bump_cost=torch.tensor([[2.0]], dtype=torch.float64)),
        _meco_oracle_p_scale=2.0)
    intervention = torch.tensor([[3.0]], dtype=torch.float64)

    traversability, base, effective = counterfactual._plan_a_p_oracle_grid(env, intervention)
    total = counterfactual._plan_a_p_oracle_integral(effective, [(0.1, 0.2), (0.9, 0.2)], 1.0)
    base_cost = counterfactual._plan_a_p_oracle_integral(base, [(0.1, 0.2), (0.9, 0.2)], 1.0)
    p_cost = counterfactual._plan_a_p_oracle_integral(
        [[2.0 * float(intervention[0, 0])]], [(0.1, 0.2), (0.9, 0.2)], 1.0)

    assert traversability == [[1.0]]
    assert base == [[2.0]]
    assert effective == [[8.0]]
    assert total == pytest.approx(6.4)
    assert total == pytest.approx(base_cost + p_cost)


def _fake_plan_a_support():
    return SimpleNamespace(
        valid_mask=torch.tensor([True, True]),
        pair_node_ids=torch.tensor([0, 1]),
        behavior_ids=torch.tensor([0, 1]),
        descriptor_ids=torch.tensor([-1, -1]),
        action_ids=torch.tensor([7, 0]),
        route_cost=torch.tensor([4.0, 1.0]),
        goal_xy=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        goal_heading=torch.tensor([0.0, 0.0]),
        support_digest="d" * 64,
    )


def test_expert_goal_collector_uses_min_route_and_never_labels_u(tmp_path, monkeypatch):
    base = {"scenario_id": "pickup_simple", "category": "single", "seed": 103,
        "decision_index": 0, "macro_ticks": 2,
        "branch_state": {"snapshot_id": "snapshot-1"},
        "predecision_simulator_state_sha256": "s" * 64}
    base_manifest = {"device": {"device": "cuda:0"}, "split": "train"}
    support = _fake_plan_a_support()
    env = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(counterfactual, "replay_counterfactual_branch",
        lambda *_args, **_kwargs: {"env": env, "support": support})
    monkeypatch.setattr(counterfactual, "_reward_step",
        lambda *_args, **_kwargs: (3.0, False, False, [{"action": 0}], None, None,
                                   False, False, False, False))
    monkeypatch.setattr("frc_defense.meco_supervised._read_counterfactual_dataset",
        lambda *_args, **_kwargs: ([base], "a" * 64, "b" * 64, {}, "protocol-id",
                                   "c" * 64, base_manifest))
    monkeypatch.setattr("frc_defense.curriculum_training.source_identity", lambda: {"src": "ok"})
    monkeypatch.setattr("frc_defense.meco_device.require_indexed_cuda_device",
        lambda device, **_kwargs: device)

    output = tmp_path / "expert-goals.jsonl"
    counterfactual.collect_plan_a_expert_goal_source_v1(
        output, dataset=tmp_path / "base.jsonl",
        qualification_protocol=tmp_path / "protocol.json", device="cuda:0")
    row = json.loads(output.read_text().splitlines()[0])
    manifest = json.loads(output.with_suffix(output.suffix + ".manifest.json").read_text())

    assert row["selected_pair_index"] == 1
    assert row["selected_goal"]["node_id"] == 1
    assert row["selected_goal"]["selection_rule"] == (
        "minimum finite route_cost; canonical pair-index tie break")
    assert row["u_status"] == "unavailable"
    assert row["u_target"] is None
    assert row["u_unavailable_reason"] == "no independent expert goal-value oracle"
    assert "expert_label" not in row and "globally_optimal" not in row
    assert "no U optimality claim" in manifest["selection_semantics"]


def _synthetic_plan_a_p_oracle_dataset(tmp_path, monkeypatch, corruption=None):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}")
    protocol, protocol_id, protocol_sha = {}, "protocol-id", "p" * 64
    monkeypatch.setattr(supervised, "_load_protocol",
        lambda _path: (protocol, protocol_id, protocol_sha, protocol_path))
    monkeypatch.setattr(supervised, "_protocol_train_seeds",
        lambda *_args: {1})
    monkeypatch.setattr(counterfactual, "_device_identity",
        lambda device: {"device": str(device)})
    monkeypatch.setattr(counterfactual, "_require_native_extension_manifest_identity",
        lambda _manifest: None)
    monkeypatch.setattr(counterfactual, "_verify_native_fuel_extension",
        lambda _identity: None)

    expected_sources = {"source": "synthetic"}
    source_identity_sha = supervised._canonical_hash(expected_sources)
    descriptor = {"descriptor_sha256": "d" * 64, "action": 7,
        "goal_xy_m": [1.0, 0.0]}

    def make_outcome(index):
        intervention_id = "P0" if index == 0 else "probe-1"
        intervention_seed = None if index == 0 else probe_seed
        route_points = [(0.1, 0.2), (0.9 if index == 0 else 0.7, 0.2)]
        route_sha = supervised._canonical_hash(route_points)
        route_snapshot = {"tick_index": 1, "route_points_m": route_points,
            "route_sha256": route_sha, "p_map_applied": index != 0}
        trace = [route_snapshot]
        trace_sha = supervised._canonical_hash([route_sha])
        base_grid, effective_grid = [[2.0]], ([[2.0]] if index == 0 else [[4.0]])
        resolution = 1.0
        cost = counterfactual._plan_a_p_oracle_integral(
            effective_grid, route_points, resolution)
        route_sha_digest = "m" * 64
        route = {"route_points_m": route_points, "route_sha256": route_sha,
            "route_trace": trace, "route_trace_sha256": trace_sha,
            "map_applied": index != 0, "selected_action": 7,
            "selected_goal_xy_m": [1.0, 0.0],
            "selected_goal_descriptor_sha256": descriptor["descriptor_sha256"],
            "selected_goal_refresh_forced": True,
            "map_scale": None if index == 0 else 1.0,
            "map_sha256": route_sha_digest}
        oracle = {"version": counterfactual.PLAN_A_P_ROUTE_ORACLE_VERSION,
            "implementation": "frc_defense.meco_counterfactual._plan_a_p_oracle_integral",
            "grid_shape": [1, 1], "resolution_m": resolution,
            "base_traversability_cost_grid": [[1.0]],
            "base_bump_cost_grid": base_grid,
            "effective_cost_grid": effective_grid,
            "grid_inputs_sha256": supervised._canonical_hash({
                "traversability": [[1.0]], "base": base_grid,
                "effective": effective_grid, "resolution_m": resolution}),
            "route_snapshot_costs_grid_edges": [cost],
            "route_cost_grid_edges": cost, "route_trace_sha256": trace_sha,
            "selected_goal_descriptor_sha256": descriptor["descriptor_sha256"],
            "intervention_map_sha256": route_sha_digest,
            "intervention_scale": 0.0 if index == 0 else 1.0}
        route["independent_cost_oracle"] = oracle
        return {"gross_horizon_return": 1.0, "intervention_id": intervention_id,
            "intervention_seed": intervention_seed,
            "continuation_policy": counterfactual.CONTINUATION_POLICY,
            "discount_semantics": counterfactual.COUNTERFACTUAL_DISCOUNT_SEMANTICS,
            "native_route": route}

    seed_base = int.from_bytes(hashlib.sha256(
        b"1:0:0:p-route-v1").digest()[:8], "little") & ((1 << 63) - 1)
    # Keep the seed within the signed 63-bit domain used by the collector.
    probe_seed = (seed_base + 1) & ((1 << 63) - 1)
    outcomes = [make_outcome(0), make_outcome(1)]
    outcomes[1]["intervention_seed"] = probe_seed
    row = {"schema": counterfactual.PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA,
        "target_schema": counterfactual.PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA,
        "scenario_id": "pickup_simple", "category": "single",
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "source_identity_sha256": source_identity_sha, "device": {"device": "cuda:0"},
        "gamma_per_macro": 0.99,
        "discount_semantics": counterfactual.COUNTERFACTUAL_DISCOUNT_SEMANTICS,
        "max_macro_ticks": 10, "horizon_physics_ticks": 10,
        "seed": 1, "decision_index": 0,
        "prefix_transcript": [], "prefix_transcript_sha256": supervised._canonical_hash([]),
        "prefix_sha256": supervised._canonical_hash([]), "branch_state": {},
        "plan_a_p_route_groups": [{"descriptor": descriptor, "outcomes": outcomes}]}
    path = tmp_path / "p-routes.jsonl"
    payload = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    path.write_bytes(payload)
    native_identity = {"status": "present"}
    manifest = {"schema": counterfactual.PLAN_A_P_ROUTE_ORACLE_DATASET_SCHEMA,
        "target_schema": counterfactual.PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA,
        "source_counterfactual_schema": counterfactual.DATASET_SCHEMA,
        "p_intervention_generator": "seeded_smooth_positive_xy_maps_v1",
        "p_oracle_version": counterfactual.PLAN_A_P_ROUTE_ORACLE_VERSION,
        "plan_a_p_target": counterfactual.PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA,
        "plan_a_p_oracle_target": counterfactual.PLAN_A_P_ROUTE_ORACLE_TARGET_SCHEMA,
        "plan_a_p_supervision_semantics":
            "non_tied sampled executed-path cost preference; lower independent oracle cost is preferred",
        "plan_a_p_prior": 0.0, "plan_a_p_unlabelled_reason": None,
        "records_have_exact_weighted_route_replay": True,
        "source_runtime_device_bound": True, "split": "train",
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "qualification_protocol_file": protocol_path.name,
        "sources": expected_sources, "source_identity_sha256": source_identity_sha,
        "device": {"device": "cuda:0"},
        "continuation_policy": counterfactual.CONTINUATION_POLICY,
        "p_intervention_maps_per_descriptor": 1, "p_intervention_scale": 1.0,
        "source_counterfactual_dataset_sha256": "a" * 64,
        "source_counterfactual_manifest_sha256": "b" * 64,
        "native_extension_identity": native_identity,
        "native_extension_identity_schema": 1,
        "artifact_identity_sha256": counterfactual._artifact_identity_sha256(
            source_identity_sha, native_identity),
        "seeds": [1], "scenario_id": "pickup_simple", "category": "single",
        "gamma_per_macro": 0.99,
        "discount_semantics": counterfactual.COUNTERFACTUAL_DISCOUNT_SEMANTICS,
        "max_macro_ticks": 10, "horizon_physics_ticks": 10,
        "record_count": 1, "route_group_count": 1,
        "dataset_sha256": hashlib.sha256(payload).hexdigest()}
    if corruption == "oracle_digest":
        row["plan_a_p_route_groups"][0]["outcomes"][0]["native_route"]\
            ["independent_cost_oracle"]["grid_inputs_sha256"] = "x" * 64
    elif corruption == "trace_digest":
        row["plan_a_p_route_groups"][0]["outcomes"][0]["native_route"]\
            ["route_trace_sha256"] = "x" * 64
    elif corruption == "trace_point":
        route = row["plan_a_p_route_groups"][0]["outcomes"][0]["native_route"]
        route["route_trace"][0]["route_points_m"][0] = [0.2, 0.2]
        route["route_points_m"][0] = [0.2, 0.2]
        route["route_sha256"] = supervised._canonical_hash(route["route_points_m"])
    if corruption is not None:
        payload = (json.dumps(row, separators=(",", ":")) + "\n").encode()
        path.write_bytes(payload)
        manifest["dataset_sha256"] = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps(manifest))
    return path, protocol_path, expected_sources


def test_plan_a_p_v3_consumer_accepts_bound_oracle_trace(tmp_path, monkeypatch):
    path, protocol_path, sources = _synthetic_plan_a_p_oracle_dataset(tmp_path, monkeypatch)

    rows, *_ = supervised._read_plan_a_p_route_dataset(
        path, qualification_protocol=protocol_path, expected_sources=sources)

    assert rows[0]["plan_a_p_route_groups"][0]["outcomes"][0]["native_route"]\
        ["independent_cost_oracle"]["route_cost_grid_edges"] == pytest.approx(1.6)


@pytest.mark.parametrize("corruption, message", (
    ("oracle_digest", "independent Plan A P route-cost oracle provenance"),
    ("trace_digest", "invalid Plan A P executed-route outcome"),
    ("trace_point", "invalid P native route trace snapshot"),
))
def test_plan_a_p_v3_consumer_rejects_oracle_digest_or_trace_tampering(
        tmp_path, monkeypatch, corruption, message):
    path, protocol_path, sources = _synthetic_plan_a_p_oracle_dataset(
        tmp_path, monkeypatch, corruption=corruption)

    with pytest.raises(ValueError, match=message):
        supervised._read_plan_a_p_route_dataset(
            path, qualification_protocol=protocol_path, expected_sources=sources)


def _write_jsonl(path, rows):
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                      for row in rows).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _synthetic_expert_annotation_inputs(tmp_path, monkeypatch, *, missing=None):
    scenario, category, seed, decision = "pickup_simple", "single", 103, 0
    snapshot, state_sha, support_digest = "snapshot-1", "a" * 64, "r" * 64
    descriptor_sha = "d" * 64
    selected_goal = {"node_id": 7, "x_m": 1.0, "y_m": 2.0,
        "heading_bin": 3, "descriptor_sha256": descriptor_sha}
    base_record = {"scenario_id": scenario, "category": category, "seed": seed,
        "decision_index": decision, "macro_ticks": 2,
        "branch_state": {"snapshot_id": snapshot},
        "predecision_simulator_state_sha256": state_sha,
        "support_digest": support_digest}
    base_path = tmp_path / "base.jsonl"
    base_sha = _write_jsonl(base_path, [base_record])
    base_manifest_path = base_path.with_suffix(base_path.suffix + ".manifest.json")
    base_manifest_path.write_text("{}\n")
    base_manifest_sha = hashlib.sha256(base_manifest_path.read_bytes()).hexdigest()
    source_manifest = {"scenario_id": scenario, "category": category}
    protocol, protocol_id, protocol_sha = {}, "protocol-id", "p" * 64
    monkeypatch.setattr(supervised, "_read_counterfactual_dataset",
        lambda *_args, **_kwargs: ([base_record], base_sha, base_manifest_sha,
            protocol, protocol_id, protocol_sha, source_manifest))
    monkeypatch.setattr(supervised, "_protocol_train_seeds", lambda *_args: {seed})

    goal_row = {"schema": counterfactual.PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA,
        "target_schema": counterfactual.PLAN_A_EXPERT_GOAL_SOURCE_TARGET_SCHEMA,
        "scenario_id": scenario, "category": category, "seed": seed,
        "decision_index": decision, "base_record": base_record,
        "snapshot_id": snapshot, "state_sha256": state_sha,
        "support_digest": support_digest}
    expert_path = tmp_path / "expert-goals.jsonl"
    expert_sha = _write_jsonl(expert_path, [goal_row])
    expert_manifest = {"schema": counterfactual.PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA,
        "target_schema": counterfactual.PLAN_A_EXPERT_GOAL_SOURCE_TARGET_SCHEMA,
        "base_dataset_schema": counterfactual.DATASET_SCHEMA,
        "base_dataset_sha256": base_sha, "base_manifest_sha256": base_manifest_sha,
        "base_dataset": base_path.name, "sources": {"source": "synthetic"},
        "source_identity_sha256": supervised._canonical_hash({"source": "synthetic"}),
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "dataset_sha256": expert_sha, "support_order": supervised.SUPPORT_ORDER,
        "split": "train", "records_have_replay_bound_selected_goal": True,
        "record_count": 1}
    expert_manifest_path = expert_path.with_suffix(expert_path.suffix + ".manifest.json")
    expert_manifest_path.write_text(json.dumps(expert_manifest, sort_keys=True) + "\n")
    expert_manifest_sha = hashlib.sha256(expert_manifest_path.read_bytes()).hexdigest()
    monkeypatch.setattr(counterfactual, "_replay_plan_a_expert_goal_source_row",
        lambda *_args: {"teacher_selected_goal_descriptor_sha256": descriptor_sha,
            "selected_goal": selected_goal, "teacher_selected_action": 0,
            "teacher_selected_behavior": 1, "support_digest": support_digest,
            "snapshot_id": snapshot, "simulator_state_sha256": state_sha})

    legal_row = {"scenario_id": scenario, "category": category, "seed": seed,
        "decision_index": decision, "support_digest": support_digest,
        "canonical_poses": [{"node_id": 7, "x_m": 1.0, "y_m": 2.0,
                             "heading_bin": 3}],
        "feasible_node_ids": [7],
        "pairs": [{"node_id": 7, "behavior_id": 1, "status": "available"}],
        "node_u_targets": [{"node_id": 7, "status": "unavailable",
            "feasible_behavior_ids": [1], "available_behavior_ids": [],
            "censored_feasible_behavior_ids": []}]}
    legal_path = tmp_path / "legal.jsonl"
    legal_sha = _write_jsonl(legal_path, [legal_row])
    legal_manifest = {"schema": counterfactual.PLAN_A_LEGAL_NODE_DATASET_SCHEMA,
        "target_schema": counterfactual.PLAN_A_LEGAL_NODE_TARGET_SCHEMA,
        "split": "train", "source_dataset_sha256": base_sha,
        "source_manifest_sha256": base_manifest_sha, "protocol_id": protocol_id,
        "qualification_protocol_sha256": protocol_sha,
        "support_order": supervised.SUPPORT_ORDER,
        "sources": {"source": "synthetic"},
        "source_identity_sha256": supervised._canonical_hash({"source": "synthetic"}),
        "dataset_sha256": legal_sha, "record_count": 1,
        "expert_goal_source_sha256": expert_sha,
        "expert_goal_source_manifest_sha256": expert_manifest_sha}
    legal_manifest_path = legal_path.with_suffix(legal_path.suffix + ".manifest.json")
    legal_manifest_path.write_text(json.dumps(legal_manifest, sort_keys=True) + "\n")
    legal_manifest_sha = hashlib.sha256(legal_manifest_path.read_bytes()).hexdigest()
    monkeypatch.setattr(supervised, "_read_and_replay_plan_a_legal_node_dataset",
        lambda *_args, **_kwargs: ([{"row": legal_row}], None, legal_sha, None))

    annotation_source = tmp_path / "annotation-authority.txt"
    annotation_source.write_text("synthetic source authority\n")
    annotation = {"schema": supervised.PLAN_A_EXPERT_GOAL_DATASET_SCHEMA,
        "target_schema": supervised.PLAN_A_EXPERT_GOAL_TARGET_SCHEMA,
        "annotation_source_id": "annotator-1", "scenario_id": scenario,
        "category": category, "seed": seed, "decision_index": decision,
        "snapshot_id": snapshot, "state_sha256": state_sha,
        "support_digest": support_digest,
        "executed_goal_descriptor_sha256": descriptor_sha,
        "executed_action": 0, "selected_behavior_id": 1,
        "selected_goal": selected_goal, "censored_feasible_behavior_ids": [],
        "u_status": "unavailable", "u_target": None}
    if missing == "u_status":
        annotation.pop("u_status")
    if missing == "selected_goal":
        annotation.pop("selected_goal")
    annotation_path = tmp_path / "annotations.jsonl"
    annotation_sha = _write_jsonl(annotation_path, [annotation])
    annotation_manifest = {"schema": supervised.PLAN_A_EXPERT_GOAL_DATASET_SCHEMA,
        "target_schema": supervised.PLAN_A_EXPERT_GOAL_TARGET_SCHEMA,
        "target_semantics": supervised.PLAN_A_EXPERT_GOAL_SEMANTICS,
        "split": "train", "source_dataset_sha256": base_sha,
        "source_manifest_sha256": base_manifest_sha,
        "expert_goal_source_sha256": expert_sha,
        "legal_node_dataset_sha256": legal_sha,
        "legal_node_manifest_sha256": legal_manifest_sha,
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "sources": {"source": "synthetic"},
        "source_identity_sha256": supervised._canonical_hash({"source": "synthetic"}),
        "dataset_sha256": annotation_sha,
        "annotation_source": {"source_id": "annotator-1",
            "sha256": hashlib.sha256(annotation_source.read_bytes()).hexdigest()},
        "support_order": supervised.SUPPORT_ORDER, "record_count": 1}
    if missing == "authority":
        annotation_manifest.pop("annotation_source")
    annotation_path.with_suffix(annotation_path.suffix + ".manifest.json").write_text(
        json.dumps(annotation_manifest, sort_keys=True) + "\n")
    return annotation_path, legal_path, expert_path, tmp_path / "protocol.json", {
        "source": "synthetic"}, annotation_source


def _load_synthetic_expert_annotations(monkeypatch, paths):
    annotation, legal, source, protocol, sources, authority = paths
    return supervised.load_plan_a_expert_goal_annotations(
        annotation, legal, source, qualification_protocol=protocol,
        expected_sources=sources, annotation_source_path=authority, device="cuda:0")


def test_expert_goal_consumer_keeps_u_unavailable_and_requires_explicit_b_authority(
        tmp_path, monkeypatch):
    loaded, *_ = _load_synthetic_expert_annotations(
        monkeypatch, _synthetic_expert_annotation_inputs(tmp_path, monkeypatch))

    assert loaded[0]["u_mask"] is False
    assert loaded[0]["u_target"] is None
    assert loaded[0]["b_mask"] is True
    assert loaded[0]["b_target"] == 1


@pytest.mark.parametrize("missing, message", (
    ("u_status", "U target status"),
    ("selected_goal", "explicit selected legal goal annotation"),
    ("authority", "annotation manifest/source/protocol binding mismatch"),
))
def test_expert_goal_consumer_fails_closed_without_authority_or_u_contract(
        tmp_path, monkeypatch, missing, message):
    paths = _synthetic_expert_annotation_inputs(tmp_path, monkeypatch, missing=missing)

    with pytest.raises(ValueError, match=message):
        _load_synthetic_expert_annotations(monkeypatch, paths)
