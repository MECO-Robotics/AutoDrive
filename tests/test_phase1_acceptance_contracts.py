from types import SimpleNamespace

import torch

from frc_defense.learning_contract import CandidateTable, ObservationSnapshot
from frc_defense.meco_track_training import _route_cost_tensor
from frc_defense.meco_tracks import (
    DecisionRequest,
    candidate_policy_logits,
    choose_candidate,
    make_scorer_snapshot,
    new_meco_episode_id,
)


def test_candidate_policy_logits_subtracts_each_cost_once_and_masks_invalid_rows():
    gross = torch.tensor([[100.0, 80.0, 70.0]])
    valid = torch.tensor([[True, True, False]])
    route = torch.tensor([[10.0, 2.0, 1_000_000.0]])
    risk = torch.tensor([[3.0, 4.0, 1_000_000.0]])
    path = torch.tensor([[5.0, 6.0, 1_000_000.0]])

    logits = candidate_policy_logits(gross, valid, route, risk, path)

    assert torch.equal(logits[0, :2], torch.tensor([82.0, 68.0]))
    assert torch.isneginf(logits[0, 2])


def test_route_cost_tensor_converts_eta_seconds_to_points_once():
    candidates = SimpleNamespace(route_records=(
        {"eta_s": 2.5}, {"eta_s": None}, {"eta_s": 4.0},
    ))

    costs = _route_cost_tensor(candidates, 3.2, device="cpu")

    assert torch.allclose(costs, torch.tensor([[8.0, 0.0, 12.8]]))


def _decision_request(valid_mask=(True, True, False)):
    episode_id = new_meco_episode_id("acceptance")
    snapshot_id = f"{episode_id}:7"
    observation = ObservationSnapshot(snapshot_id, episode_id, 7, 10.0,
                                     (1, 1, 1), (0.0,), (0.0,) * 7)
    candidates = CandidateTable(snapshot_id, ("fallback", "tie_first", "invalid"),
                                ((0.0,), (1.0,), (2.0,)), valid_mask,
                                ("feature",))
    request = DecisionRequest(observation, candidates, 10.0, 0.2,
                              "fallback", "acceptance-model")
    return observation, candidates, request


def test_choose_candidate_excludes_invalid_highest_score_and_uses_first_tie():
    observation, candidates, request = _decision_request()
    output = make_scorer_snapshot("acceptance-model", observation, candidates,
                                  (5.0, 5.0, 1000.0), 10.0)

    assert choose_candidate(request, output) == ("fallback", "model")


def test_choose_candidate_uses_deterministic_fallback_when_output_missing():
    _, _, request = _decision_request()

    assert choose_candidate(request, None) == ("fallback", "deterministic_fallback")
