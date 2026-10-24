import json

import pytest
import torch

from frc_defense.learning_contract import CandidateTable, ObservationSnapshot
from frc_defense.meco_candidates import RouteFeasibleCandidates
from frc_defense.meco_track_training import (_candidate_replay_fingerprints,
    _candidate_replay_digest, _policy_inputs_for_row, _tensor_evidence,
    _ppo_update)
from frc_defense.meco_track_training import MecoTrainConfig
from frc_defense.meco_tracks import (categorical_from_candidate_logits,
    make_meco_track, candidate_digest, score_candidates)


def _row_and_model():
    torch.manual_seed(42)
    observation = ObservationSnapshot(snapshot_id="snapshot-1", episode_id="episode-1",
        sequence=1, timestamp=0., raster_shape=(6, 1, 1), raster=(0.,) * 6,
        globals=(0.,) * 7, capture_monotonic=1.)
    table = CandidateTable(snapshot_id="snapshot-1",
        candidate_ids=("candidate-a", "candidate-b"),
        features=((.2, .4), (.7, .1)), valid_mask=(True, True),
        feature_names=("feature-a", "feature-b"), route_costs=(0., 0.))
    candidate_set = RouteFeasibleCandidates(observation=observation, table=table,
        fallback_candidate_id="candidate-a", action_indices=(0, 1),
        route_cost_points=torch.zeros(2), risk_cost_points=torch.zeros(2),
        route_records=({}, {}), candidate_recall={})
    model = make_meco_track("C1", candidate_features=2).eval()
    row = {"episode_id": "episode-1", "raster": torch.zeros(1, 6, 1, 1),
        "globals": torch.zeros(1, 7), "graph": None,
        "candidate_set": candidate_set, "candidate_digest": candidate_digest(table),
        "route_cost": torch.zeros(1, 2), "risk_cost": torch.zeros(1, 2),
        "hidden_before": None, "action": torch.tensor([0]),
        "rollout_candidate_ids": list(table.candidate_ids),
        "rollout_valid_mask": list(table.valid_mask),
        "inference_provenance": {"status": "local_model"}}
    row["candidate_replay_digest"] = _candidate_replay_digest(table,
        row["route_cost"], row["risk_cost"])
    replay = _policy_inputs_for_row("C1", model, row, None, None)
    row["rollout_logits"] = replay["logits"].detach().clone()
    row["logprob"] = categorical_from_candidate_logits(
        replay["logits"]).log_prob(row["action"]).reshape(()).detach()
    row["hidden_next"] = replay["hidden"].detach().clone()
    row["candidate_replay_digest"] = _candidate_replay_digest(table,
        row["route_cost"], row["risk_cost"])
    row["replay_fingerprints"] = _candidate_replay_fingerprints(row, model)
    row.update(reward=torch.tensor(0.), value=torch.tensor(0.), done=False,
        execution_failure=False)
    return row, model


def test_c1_replay_artifact_reproduces_exact_logits_and_detects_input_mutation(
        tmp_path):
    row, model = _row_and_model()
    original_globals = _tensor_evidence(row["globals"])
    expected_logprob = float(row["logprob"])
    row["globals"] = row["globals"] + .25
    config = MecoTrainConfig(track="C1")

    with pytest.raises(RuntimeError, match="frozen rollout inputs changed"):
        _ppo_update("C1", model, None, None, [row], config,
            torch.tensor(0.), update_number=7, diagnostic_output=tmp_path)

    diagnostic_path = next(tmp_path.glob("ppo-logprob-replay-failure-C1-*.json"))
    diagnostic = json.loads(diagnostic_path.read_text())
    assert diagnostic["update"] == 7
    assert diagnostic["row_index"] == 0
    assert diagnostic["episode_id"] == "episode-1"
    assert diagnostic["expected_logprob"] == expected_logprob
    assert diagnostic["actual_replay_logprob"] != expected_logprob
    assert diagnostic["max_logits_delta"] > 0.
    assert diagnostic["chosen_action"] == 0
    assert diagnostic["rollout_fingerprints"]["globals"] == original_globals
    assert diagnostic["replay_fingerprints"]["globals"] != original_globals
    assert diagnostic["replay_inference_state"] is None

    replay_payload = torch.load(diagnostic["replay_tensor_artifact"],
                                map_location="cpu", weights_only=False)
    replay_model = make_meco_track("C1", candidate_features=2).eval()
    replay_model.load_state_dict(replay_payload["model_state_dict"])
    replay_row = replay_payload["row"]
    exact = _policy_inputs_for_row("C1", replay_model, replay_row,
                                   replay_row["hidden_before"], None)
    exact_logprob = categorical_from_candidate_logits(exact["logits"]).log_prob(
        replay_row["action"]).reshape(())
    assert torch.equal(exact["logits"], replay_payload["replay_logits"])
    assert torch.equal(exact_logprob, replay_payload["replay_logprob"])

    mutated = dict(replay_row)
    mutated["globals"] = mutated["globals"] + 1.
    assert (_tensor_evidence(mutated["globals"])["sha256"] !=
            _tensor_evidence(replay_row["globals"])["sha256"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA device required")
def test_c1_accelerator_forward_restores_determinism_and_keeps_gradients():
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    model = make_meco_track("C1", candidate_features=2).to(device).train()
    row, _ = _row_and_model()
    table = row["candidate_set"].table
    raster = row["raster"].to(device)
    globals_ = row["globals"].to(device)

    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        scored = score_candidates("C1", model, raster=raster,
            globals_=globals_, table=table)
        assert not torch.are_deterministic_algorithms_enabled()
        scored["logits"][0, 0].backward()
        assert any(parameter.grad is not None and
                   bool(torch.isfinite(parameter.grad).all()) and
                   bool(parameter.grad.abs().sum() > 0)
                   for parameter in model.parameters())

        torch.use_deterministic_algorithms(True)
        score_candidates("C1", model, raster=raster,
            globals_=globals_, table=table)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previous)
