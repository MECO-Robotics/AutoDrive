import torch

from frc_defense.learning_contract import CandidateTable, ObservationSnapshot, ScorerSnapshot
from frc_defense.meco_tracks import (
    DecisionRequest,
    MECO_TRACKS,
    choose_candidate,
    make_meco_track,
    sample_candidate_action,
    score_candidates,
)


def table(*, graph=False):
    names = ("goal_x_m", "goal_y_m", "heading_bin", "behavior_index",
             "field_length_m", "field_width_m")
    rows = ((1., 1., 0., 0., 16.5, 8.1), (2., 2., 1., 1., 16.5, 8.1))
    if graph:
        names += ("legal_node_index",)
        rows = tuple(row + (float(i),) for i, row in enumerate(rows))
    return CandidateTable("e:1", ("fallback", "other"), rows,
                          (True, True), names, (None, None))


def test_five_architecture_outputs_and_candidate_logits():
    torch.manual_seed(4)
    raster = torch.zeros(1, 6, 25, 51)
    globals_ = torch.zeros(1, 7)
    for track in MECO_TRACKS:
        candidates = table(graph=track in ("A2", "A3"))
        model = make_meco_track(track, candidate_features=len(candidates.feature_names))
        graph = None
        if track in ("A2", "A3"):
            nodes = 8
            graph = {"node_features": torch.zeros(1, nodes, 9),
                     "edges": torch.arange(nodes).view(1, nodes, 1),
                     "edge_mask": torch.ones(1, nodes, 1, dtype=torch.bool),
                     "region_ids": torch.arange(nodes).view(1, nodes) % 64}
        result = score_candidates(track, model, raster=raster, globals_=globals_,
                                  table=candidates, graph=graph)
        assert result["logits"].shape == (1, 2)
        action, logprob, entropy = sample_candidate_action(result["logits"])
        assert action.shape == logprob.shape == entropy.shape == (1,)


def test_stale_mismatched_and_nonfinite_outputs_use_valid_fallback():
    observation = ObservationSnapshot("e:1", "e", 1, 2., (1, 1, 1), (0.,), (0.,) * 7)
    candidates = CandidateTable("e:1", ("fallback", "other"), ((0.,), (1.,)),
                                (True, True), ("feature",), (None, None))
    request = DecisionRequest(observation, candidates, 2., .1, "fallback", "model-v1")
    assert choose_candidate(request, None) == ("fallback", "deterministic_fallback")
    stale = ScorerSnapshot("model-v1", "e:1", candidates.candidate_ids,
                           (0., 5.), 1.8)
    assert choose_candidate(request, stale) == ("fallback", "deterministic_fallback")
    wrong_model = ScorerSnapshot("other-model", "e:1", candidates.candidate_ids,
                                 (0., 5.), 2.)
    assert choose_candidate(request, wrong_model) == ("fallback", "deterministic_fallback")
    fresh = ScorerSnapshot("model-v1", "e:1", candidates.candidate_ids,
                           (0., 5.), 2.)
    assert choose_candidate(request, fresh) == ("other", "model")

