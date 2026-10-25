import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from frc_defense import meco_counterfactual as counterfactual
from frc_defense import meco_supervised as supervised


def _synthetic_row():
    protocol_id = "protocol-test"
    repeat_protocol = {"repeat_protocol_id": "repeat-protocol-test"}
    outcomes = []
    for candidate_index, (candidate_id, values) in enumerate((
            ("left", (4.0, 5.0)), ("right", (2.0, 3.0)))):
        repeats = []
        for repeat_index, gross in enumerate(values):
            repeats.append({
                "gross_macro_return": gross / 2,
                "gross_horizon_return": gross,
                "discounted_horizon_return": gross,
                "continuation_policy": counterfactual.CONTINUATION_POLICY,
                "continuation_macro_horizon": 1,
                "discount_semantics": counterfactual.COUNTERFACTUAL_DISCOUNT_SEMANTICS,
                "terminal": False,
                "repeat_index": repeat_index,
                "repeat_seed": counterfactual.plan_c_repeat_seed(
                    protocol_id=protocol_id, scenario_id="scenario",
                    category="category", seed=7, decision_index=2,
                    repeat_index=repeat_index),
                "repeat_rng_provenance": {"generator_count": 2,
                    "seed_manifest_sha256": "a" * 64},
            })
        outcomes.append(counterfactual._plan_c_repeated_outcome(
            {"candidate_id": candidate_id, "candidate_index": candidate_index}, repeats))
    return {
        "plan_c_target_schema": counterfactual.PLAN_C_REPEAT_TARGET_SCHEMA,
        "plan_c_repeat_protocol_id": repeat_protocol["repeat_protocol_id"],
        "plan_c_counterfactuals": outcomes,
        "plan_c_pairwise_labels": counterfactual._plan_c_pairwise_labels(outcomes),
    }, repeat_protocol, protocol_id


def test_repeat_seed_is_domain_separated_and_matched_across_candidates():
    base = {"protocol_id": "p", "scenario_id": "s", "category": "c",
            "seed": 4, "decision_index": 3}
    first = counterfactual.plan_c_repeat_seed(**base, repeat_index=0)
    assert first == counterfactual.plan_c_repeat_seed(**base, repeat_index=0)
    assert first != counterfactual.plan_c_repeat_seed(**base, repeat_index=1)
    assert first != counterfactual.plan_c_repeat_seed(**{**base, "protocol_id": "other"},
                                                       repeat_index=0)


def test_environment_generators_are_reseeded_and_provenance_is_stable():
    env = SimpleNamespace(sim=SimpleNamespace(generator=torch.Generator(device="cpu")))
    first = counterfactual._seed_plan_c_environment_generators(env, 123)
    first_draw = torch.rand(4, generator=env.sim.generator)
    second = counterfactual._seed_plan_c_environment_generators(env, 123)
    second_draw = torch.rand(4, generator=env.sim.generator)
    assert first == second
    assert torch.equal(first_draw, second_draw)
    assert first["generator_count"] == 1

    counterfactual._seed_plan_c_environment_generators(env, 456)
    assert not torch.equal(first_draw, torch.rand(4, generator=env.sim.generator))

    with pytest.raises(RuntimeError, match="no simulator-local Torch generators"):
        counterfactual._seed_plan_c_environment_generators(SimpleNamespace(value=1), 123)


def test_pair_statistics_abstain_when_matched_repeat_signs_disagree():
    stable = counterfactual.plan_c_repeat_statistics([5, 7], [2, 3])
    assert stable["stable_preference"] is True
    assert stable["preferred_side"] == "left"
    assert stable["repeat_differences"] == [3., 4.]
    assert stable["sample_variance"] == pytest.approx(0.5)
    assert stable["standard_error"] == pytest.approx(0.5)

    unstable = counterfactual.plan_c_repeat_statistics([5, 2], [3, 4])
    assert unstable["stable_preference"] is False
    assert unstable["preferred_side"] is None

    with pytest.raises(ValueError, match="every finite matched repeat"):
        counterfactual.plan_c_repeat_statistics([3], [2, 1])


def test_repeated_label_validator_accepts_complete_matched_contract():
    row, repeat_protocol, protocol_id = _synthetic_row()
    assert supervised._validate_plan_c_repeated_labels(row,
        repeat_protocol=repeat_protocol, protocol_id=protocol_id,
        scenario_id="scenario", category="category", seed=7,
        decision_index=2)
    assert supervised._plan_c_stable_pair_indices(
        row["plan_c_counterfactuals"], row["plan_c_pairwise_labels"]) == [(0, 1, True)]


@pytest.mark.parametrize("corruption", [
    "missing_repeat", "wrong_seed", "variance", "label", "generator_provenance"])
def test_repeated_label_validator_fails_closed_on_incomplete_or_forged_evidence(corruption):
    row, repeat_protocol, protocol_id = _synthetic_row()
    row = copy.deepcopy(row)
    if corruption == "missing_repeat":
        row["plan_c_counterfactuals"][0]["repeat_returns"].pop()
    elif corruption == "wrong_seed":
        row["plan_c_counterfactuals"][0]["repeat_returns"][1]["repeat_seed"] += 1
    elif corruption == "variance":
        row["plan_c_counterfactuals"][0]["repeat_sample_variance_gross_horizon_return"] += 1
    elif corruption == "generator_provenance":
        row["plan_c_counterfactuals"][0]["repeat_returns"][1][
            "repeat_rng_provenance"]["seed_manifest_sha256"] = "b" * 64
    else:
        row["plan_c_pairwise_labels"][0]["preferred_id"] = "right"
    with pytest.raises(ValueError):
        supervised._validate_plan_c_repeated_labels(row,
            repeat_protocol=repeat_protocol, protocol_id=protocol_id,
            scenario_id="scenario", category="category", seed=7,
            decision_index=2)


def test_unstable_pairs_are_excluded_from_fitter_preferences():
    outcomes = [
        {"descriptor": {"candidate_id": "a", "candidate_index": 8}},
        {"descriptor": {"candidate_id": "b", "candidate_index": 3}},
        {"descriptor": {"candidate_id": "c", "candidate_index": 10}},
    ]
    labels = [
        {"left_id": "a", "right_id": "b", "stable_preference": True,
         "preferred_id": "a"},
        {"left_id": "a", "right_id": "c", "stable_preference": False,
         "preferred_id": None},
        {"left_id": "b", "right_id": "c", "stable_preference": True,
         "preferred_id": "c"},
    ]
    assert supervised._plan_c_stable_pair_indices(outcomes, labels) == [
        (8, 3, True), (3, 10, False)]


def test_collector_uses_matched_repeat_seeds_for_every_candidate(monkeypatch, tmp_path):
    """Exercise collector seed plumbing with every simulator boundary faked."""
    identity = {"status": "synthetic-test-only"}
    monkeypatch.setattr(supervised, "_load_protocol",
        lambda path: ({}, "protocol-test", "a" * 64, Path(path)))
    monkeypatch.setattr(supervised, "_protocol_train_seeds", lambda *args: {7})
    monkeypatch.setattr(counterfactual, "get_scenario",
        lambda _: SimpleNamespace(required_categories=("category",)))
    monkeypatch.setattr(counterfactual, "_counterfactual_rng_scope", lambda _: nullcontext())
    monkeypatch.setattr(counterfactual, "make_curriculum_env",
        lambda *args, **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(counterfactual, "build_legal_pose_graph", lambda *args, **kwargs: object())
    monkeypatch.setattr("frc_defense.meco_track_training._reset_episode",
        lambda *args: None)
    monkeypatch.setattr(counterfactual, "_native_fuel_extension_identity", lambda: identity)
    monkeypatch.setattr(counterfactual, "_verify_native_fuel_extension", lambda _: None)
    monkeypatch.setattr(counterfactual, "_device_identity", lambda _: {"device": "cpu"})
    monkeypatch.setattr("frc_defense.meco_device.require_indexed_cuda_device",
        lambda *args, **kwargs: "cpu")
    monkeypatch.setattr("frc_defense.fuel_physics_hip.prepare_native_fuel_extension",
        lambda _: None)
    sources = {"_native_fuel_extension": identity}
    monkeypatch.setattr("frc_defense.curriculum_training.source_identity", lambda: sources)

    def fake_prefix(*args, **kwargs):
        index = len(prefix_calls)
        prefix_calls.append(index)
        eligibility = {"eligible": True, "skip_reasons": []}
        eligibility["metadata_sha256"] = counterfactual._hash(eligibility)
        return {"snapshot_id": f"snapshot-{index}", "macro_ticks": 1,
            "simulator_state_sha256": "1" * 64, "observation_sha256": "2" * 64,
            "safe_observation_sha256": "3" * 64, "candidate_digest": "4" * 64,
            "candidate_replay_digest": "5" * 64, "support_digest": "6" * 64,
            "teacher_action": 0, "branch_eligibility": eligibility,
            "candidate_descriptors": [
                {"candidate_index": i, "candidate_id": f"c{i}", "action": i,
                 "goal_xy_m": [float(i), 0.], "descriptor_sha256": str(i) * 64}
                for i in range(2)],
            "plan_a_descriptors": [
                {"pair_index": i, "action": i, "goal_xy_m": [float(i), 0.],
                 "goal_heading_rad": 0., "descriptor_sha256": str(i + 7) * 64}
                for i in range(2)]}

    prefix_calls = []
    monkeypatch.setattr(counterfactual, "_prefix_snapshot", fake_prefix)
    monkeypatch.setattr(counterfactual, "_reward_step",
        lambda env, action, ticks: (1., False, False, [action] * ticks, 0., 0., False))
    branch_calls = []
    def fake_branch(**kwargs):
        if kwargs["kind"] == "C":
            branch_calls.append((kwargs["decision_index"], kwargs["descriptor"]["candidate_id"],
                                 kwargs["repeat_seed"]))
        return {"gross_macro_return": 1., "gross_horizon_return": 1.,
            "discounted_horizon_return": 1.,
            "continuation_policy": counterfactual.CONTINUATION_POLICY,
            "continuation_macro_horizon": 1,
            "discount_semantics": counterfactual.COUNTERFACTUAL_DISCOUNT_SEMANTICS,
            "terminal": False, "repeat_rng_provenance": {"generator_count": 1,
                "seed_manifest_sha256": "b" * 64}}
    monkeypatch.setattr(counterfactual, "_branch_return", fake_branch)

    out = tmp_path / "synthetic.jsonl"
    counterfactual.collect_counterfactual_v2(out, scenario_id="scenario",
        category="category", seeds=[7], qualification_protocol=tmp_path / "protocol.json",
        horizon_ticks=2, action_ticks=1, candidate_k=8, max_c_candidates=2,
        max_a_pairs=2)

    assert len(branch_calls) == 8
    by_repeat = {}
    for decision, candidate, repeat_seed in branch_calls:
        by_repeat.setdefault((decision, repeat_seed), set()).add(candidate)
    assert len(by_repeat) == 4
    assert all(candidates == {"c0", "c1"} for candidates in by_repeat.values())
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        seeds_by_candidate = [
            [repeat["repeat_seed"] for repeat in outcome["repeat_returns"]]
            for outcome in row["plan_c_counterfactuals"]]
        assert len(seeds_by_candidate) == 2
        assert seeds_by_candidate[0] == seeds_by_candidate[1]


@pytest.mark.parametrize("track", ["C1", "C2"])
def test_c_fitter_rejects_legacy_single_run_manifest(monkeypatch, tmp_path, track):
    sources = {"synthetic": "source"}
    manifest = {"device": {"device": "cpu"}, "plan_c_target_schema": None}
    monkeypatch.setattr(supervised, "_prepare_native_gpu_runtime", lambda _: None)
    monkeypatch.setattr("frc_defense.curriculum_training.source_identity", lambda: sources)
    monkeypatch.setattr("frc_defense.meco_counterfactual._device_identity",
        lambda _: {"device": "cpu"})
    monkeypatch.setattr(supervised, "_read_counterfactual_dataset",
        lambda *args, **kwargs: ([], "a" * 64, "b" * 64, "c" * 64,
                                 "protocol", "d" * 64, manifest))
    with pytest.raises(ValueError, match="rejects legacy single-rollout"):
        supervised.warm_start_counterfactual_v2(track, tmp_path / "legacy.jsonl",
            tmp_path / "weights.pt", qualification_protocol=tmp_path / "protocol.json",
            device="cpu")


@pytest.mark.parametrize("field", [
    "plan_c_repeat_protocol", "plan_c_repeat_variance_note", "plan_c_stable_pair_count",
    "plan_c_abstained_pair_count", "pair_count_semantics"])
def test_repeated_artifact_rejects_tampered_repeat_provenance(monkeypatch, tmp_path, field):
    from frc_defense.meco_candidates import CANDIDATE_BOUNDARY
    from frc_defense.meco_plan_a_policy import SUPPORT_ORDER
    from frc_defense.meco_tracks import make_meco_track

    protocol = {"protocol_id": "protocol-test"}
    protocol_path = tmp_path / "protocol.json"
    sources = {"synthetic": "source"}
    device = {"device": "cpu"}
    base = {"schema": supervised.COUNTERFACTUAL_PLAN_C_REPEATED_WEIGHTS_SCHEMA,
        "schema_version": 1, "artifact_kind": "weights_only_plan_c_repeated_pairwise_v1",
        "track": "C1", "model_schema_version": 1,
        "model": make_meco_track("C1", candidate_features=supervised.MECO_CANDIDATE_RAW_FEATURE_COUNT).state_dict(),
        "dataset_sha256": "a" * 64, "dataset_schema": "meco-counterfactual-trajectory-v3",
        "dataset_manifest_sha256": "b" * 64, "records_have_prefix_replay_parity": True,
        "source_runtime_device_bound": True, "state_digest_schema": "meco-full-environment-state-v2",
        "target_schema": "meco-matched-prefix-gross-return-v3",
        "label_provenance": "within-prefix matched repeated simulator gross-horizon returns; stable paired sign only",
        "target_semantics": "within_prefix_repeated_matched_stable_pairwise_preference_from_gross_horizon_return",
        "scoring_semantics": "raw_candidate_utility; route_and_risk_costs_excluded_for_gross_return_target",
        "plan_a_p_target": None, "plan_a_p_prior": 0., "plan_a_p_mode": None,
        "candidate_raw_feature_count": supervised.MECO_CANDIDATE_RAW_FEATURE_COUNT,
        "candidate_model_input_width": supervised.MECO_CANDIDATE_MODEL_INPUT_WIDTH,
        "candidate_feature_names": list(tuple(supervised.SAFE_ACTION_FEATURES) + tuple(supervised.EXTRA_FEATURES)),
        "prediction_mode": "cv", "prediction_contract": supervised.prediction_contract("cv"),
        "candidate_boundary": CANDIDATE_BOUNDARY, "candidate_k": 8,
        "max_plan_c_descriptors_per_prefix": 2, "max_plan_a_pairs_per_prefix": 2,
        "route_cost_points_per_second": supervised.ROUTE_COST_POINTS_PER_SECOND,
        "plan_a_support_order": SUPPORT_ORDER,
        "plan_a_p_unlabelled_reason": "simulator executor does not execute P-modified routes",
        "horizon_physics_ticks": 1, "max_macro_ticks": 1, "gamma_per_macro": 1.,
        "discount_semantics": supervised.COUNTERFACTUAL_DISCOUNT_SEMANTICS,
        "continuation_policy": counterfactual.CONTINUATION_POLICY,
        "plan_c_target_schema": counterfactual.PLAN_C_REPEAT_TARGET_SCHEMA,
        "plan_c_repeat_protocol": counterfactual.plan_c_repeat_protocol("protocol-test"),
        "plan_c_label_policy": "stable_same_strict_sign_all_paired_repeats_else_abstain",
        "plan_c_repeat_variance_note": "R=2 is a bounded replication policy, not precise variance calibration",
        "plan_c_stable_pair_count": 1, "plan_c_abstained_pair_count": 0,
        "pair_count_semantics": "non_tied_pair_count counts stable Plan C labels; tied_pair_count counts all abstentions",
        "source_identity_sha256": supervised._canonical_hash(sources), "sources": sources,
        "collection_device": device, "training_device": device, "protocol_id": "protocol-test",
        "qualification_protocol_sha256": "c" * 64, "qualification_protocol_file": "protocol.json",
        "split": "train", "scenario_id": "scenario", "category": "category", "seeds": [7],
        "optimizer_state_included": False, "resume_compatible": False, "updates": 1,
        "learning_rate": 1e-4, "non_tied_pair_count": 1, "tied_pair_count": 0,
        "mean_loss_by_epoch": [0.5]}
    monkeypatch.setattr(supervised.torch, "load", lambda *args, **kwargs: base)
    monkeypatch.setattr(supervised, "_load_protocol",
        lambda _: (protocol, "protocol-test", "c" * 64, protocol_path))
    monkeypatch.setattr(supervised, "_protocol_train_seeds", lambda *args: {7})
    bad = copy.deepcopy(base)
    bad[field] = {"tampered": True}
    monkeypatch.setattr(supervised.torch, "load", lambda *args, **kwargs: bad)
    with pytest.raises(ValueError, match="identity/schema mismatch"):
        supervised.validate_counterfactual_warm_start_artifact("C1", tmp_path / "artifact.pt",
            qualification_protocol=protocol_path, expected_sources=sources)
