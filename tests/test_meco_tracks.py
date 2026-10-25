import torch
import pytest
import threading
import time
from types import SimpleNamespace

from frc_defense.learning_contract import CandidateTable, ObservationSnapshot
from frc_defense.meco_tracks import (
    DecisionRequest,
    MECO_TRACKS,
    choose_candidate,
    make_meco_track,
    make_scorer_snapshot,
    new_meco_episode_id,
    sample_candidate_action,
    score_candidates,
    _gather_graph_candidates,
    SE2MessageBlock,
    _AsyncInferenceBroker,
    SAFE_ACTION_FEATURES,
    EXTRA_FEATURES,
)


def table(*, graph=False):
    names = tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)
    row_values = []
    for index in range(2):
        values = {
            "action_index": index,
            "sensor_track_buffer_index": None,
            "goal_x_m": float(index + 1),
            "goal_y_m": float(index + 1),
            "heading_bin": index,
            "behavior_index": index,
            "field_length_m": 16.5,
            "field_width_m": 8.1,
            "target_distance_m": None,
            "nearby_sensed_track_count": None,
            "track_confidence": None,
            "track_age_s": None,
            "opponent_distance_m": None,
            "route_time_s": None,
            "legal_node_index": float(index) if graph else None,
            "route_distance_m": None,
            "risk_penalty_points": None,
            "target_heading_rad": None,
            "is_continue": float(index == 0),
            "commitment_switch_cost": None,
        }
        row_values.append(tuple(values[name] for name in names))
    return CandidateTable("e:1", ("fallback", "other"), tuple(row_values),
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
            node_features = torch.zeros(1, nodes, 9)
            node_features[0, :, 8] = torch.arange(nodes) % 8
            graph = {"node_features": node_features,
                     "edges": torch.arange(nodes).view(1, nodes, 1),
                     "edge_mask": torch.ones(1, nodes, 1, dtype=torch.bool),
                     "region_ids": torch.arange(nodes).view(1, nodes) % 64}
            if track == "A2":
                graph["coarse_edges"] = torch.arange(64).view(1, 64, 1)
                graph["coarse_edge_mask"] = torch.zeros(1, 64, 1, dtype=torch.bool)
        result = score_candidates(track, model, raster=raster, globals_=globals_,
                                  table=candidates, graph=graph)
        assert result["logits"].shape == (1, 2)
        action, logprob, entropy = sample_candidate_action(result["logits"])
        assert action.shape == logprob.shape == entropy.shape == (1,)
        if track in ("A2", "A3"):
            assert int(result["raw"]["occupied_regions"].sum()) == nodes
            assert torch.isfinite(result["raw"]["region_tokens"]).all()


@pytest.mark.parametrize("track", ("C1", "C2"))
@pytest.mark.parametrize("malformation", ("name", "order", "width"))
def test_c_tracks_reject_malformed_candidate_schema(track, malformation):
    candidates = table()
    names = list(candidates.feature_names)
    rows = [list(row) for row in candidates.features]
    if malformation == "name":
        names[0] = "unexpected_feature"
    elif malformation == "order":
        names[0], names[1] = names[1], names[0]
    else:
        rows[0].pop()
    if malformation == "width":
        malformed = object.__new__(CandidateTable)
        for field, value in (
                ("snapshot_id", candidates.snapshot_id),
                ("candidate_ids", candidates.candidate_ids),
                ("features", tuple(tuple(row) for row in rows)),
                ("valid_mask", candidates.valid_mask),
                ("feature_names", candidates.feature_names),
                ("route_costs", candidates.route_costs),
                ("schema_version", candidates.schema_version),
                ("context_digest", candidates.context_digest)):
            object.__setattr__(malformed, field, value)
    else:
        malformed = CandidateTable(
            candidates.snapshot_id, candidates.candidate_ids,
            tuple(tuple(row) for row in rows), candidates.valid_mask,
            tuple(names), candidates.route_costs)
    model = make_meco_track(track, candidate_features=len(candidates.feature_names))
    with pytest.raises(ValueError, match="registered shared feature schema"):
        score_candidates(track, model, raster=torch.zeros(1, 6, 25, 51),
                          globals_=torch.zeros(1, 7), table=malformed)


def test_stale_mismatched_and_nonfinite_outputs_use_valid_fallback():
    episode_id = new_meco_episode_id("e")
    snapshot_id = f"{episode_id}:1"
    observation = ObservationSnapshot(snapshot_id, episode_id, 1, 2., (1, 1, 1), (0.,), (0.,) * 7)
    candidates = CandidateTable(snapshot_id, ("fallback", "other"), ((0.,), (1.,)),
                                (True, True), ("feature",), (None, None))
    request = DecisionRequest(observation, candidates, 2., .1, "fallback", "model-v1")
    assert choose_candidate(request, None) == ("fallback", "deterministic_fallback")
    stale = make_scorer_snapshot("model-v1", observation, candidates, (0., 5.), 1.8)
    assert choose_candidate(request, stale) == ("fallback", "deterministic_fallback")
    wrong_model = make_scorer_snapshot("other-model", observation, candidates, (0., 5.), 2.)
    assert choose_candidate(request, wrong_model) == ("fallback", "deterministic_fallback")
    fresh = make_scorer_snapshot("model-v1", observation, candidates, (0., 5.), 2.)
    assert choose_candidate(request, fresh) == ("other", "model")
    changed_table = CandidateTable(snapshot_id, candidates.candidate_ids, ((9.,), (1.,)),
                                   candidates.valid_mask, candidates.feature_names,
                                   candidates.route_costs)
    changed_request = DecisionRequest(observation, changed_table, 2., .1,
                                      "fallback", "model-v1")
    assert choose_candidate(changed_request, fresh) == ("fallback", "deterministic_fallback")
    changed_observation = ObservationSnapshot(snapshot_id, episode_id, 1, 2.,
                                              (1, 1, 1), (1.,), (0.,) * 7)
    changed_request = DecisionRequest(changed_observation, candidates, 2., .1,
                                      "fallback", "model-v1")
    assert choose_candidate(changed_request, fresh) == ("fallback", "deterministic_fallback")


def test_inference_broker_discards_result_completed_after_expiry_and_fallback_is_safe():
    started = threading.Event()
    release = threading.Event()
    adapter = SimpleNamespace(_generation_id="generation-1", _request_sequence=1,
        checkpoint_sha256="a" * 64, protocol_id="protocol-1", track="C1")

    def infer(_self, request, _inputs, *, now):
        started.set()
        release.wait(timeout=2)
        return SimpleNamespace(completed_monotonic=time.monotonic(),
                               outputs={"candidate_utility": torch.tensor([0., 5.])})

    adapter._infer_sync = infer.__get__(adapter, SimpleNamespace)
    broker = _AsyncInferenceBroker(adapter)
    now = time.monotonic()
    broker_request = SimpleNamespace(generation_id="generation-1", request_sequence=1,
        checkpoint_sha256="a" * 64, protocol_id="protocol-1", model_id="meco:C1",
        expires_monotonic=now + 0.1)
    try:
        handle = broker.submit(broker_request, {})
        assert handle is not None
        assert started.wait(timeout=1)
        time.sleep(0.12)
        release.set()
        deadline = time.monotonic() + 1
        while not handle["done"] and time.monotonic() < deadline:
            time.sleep(0.001)
        assert handle["done"]
        broker_response = broker.poll(broker_request, handle)
        assert broker_response is None
        assert broker.counters()["late_responses"] == 1
    finally:
        release.set()
        broker.close()

    episode_id = new_meco_episode_id("e")
    snapshot_id = f"{episode_id}:1"
    observation = ObservationSnapshot(snapshot_id, episode_id, 1, 2., (1, 1, 1),
                                      (0.,), (0.,) * 7)
    candidates = CandidateTable(snapshot_id, ("fallback", "other"), ((0.,), (1.,)),
                                (True, True), ("feature",), (None, None))
    decision = DecisionRequest(observation, candidates, 2., .1, "fallback", "model-v1")
    assert choose_candidate(decision, broker_response) == (
        "fallback", "deterministic_fallback")


def test_se2_heading_embedding_wraps_and_rejects_bad_bins():
    import pytest
    from frc_defense.meco_tracks import _se2_node_inputs

    features = torch.zeros(1, 3, 9)
    features[0, :, 8] = torch.tensor((7., 0., 1.))
    embedded = _se2_node_inputs(features, torch.zeros(1, 7))[0, :, 8:10]
    # Bin 7 and bin 0 are one heading step apart under circular encoding.
    assert torch.allclose((embedded[0] - embedded[1]).norm(),
                          (embedded[1] - embedded[2]).norm(), atol=1e-6)
    features[0, 0, 8] = 8.
    with pytest.raises(ValueError, match="heading bins"):
        _se2_node_inputs(features, torch.zeros(1, 7))


def test_a1_rejects_out_of_field_and_nonintegral_pose_candidates():
    import pytest
    from frc_defense.meco_tracks import _gather_dense_candidates

    model = make_meco_track("A1")
    output = model(raster=torch.zeros(1, 6, 25, 51), globals_=torch.zeros(1, 7))
    feature_row = (0., 0., 16.5, 1., 0., 1., 16.5, 8.1,
                   0., 0., 1., 0., None, None)
    invalid = CandidateTable("s", ("bad",), (feature_row,),
                             (True,), ("action_index", "sensor_track_buffer_index", "goal_x_m",
                                       "goal_y_m", "heading_bin", "behavior_index", "field_length_m",
                                       "field_width_m", "target_distance_m", "nearby_sensed_track_count",
                                       "track_confidence", "track_age_s", "opponent_distance_m", "route_time_s"))
    with pytest.raises(ValueError, match="outside"):
        _gather_dense_candidates(output, invalid, (25, 51))
    feature_row = (0., 0., 1., 1., 0.5, 1., 16.5, 8.1,
                   0., 0., 1., 0., None, None)
    invalid = CandidateTable("s", ("bad",), (feature_row,), (True,), invalid.feature_names)
    with pytest.raises(ValueError, match="heading_bin"):
        _gather_dense_candidates(output, invalid, (25, 51))


def test_graph_candidate_indices_are_not_truncated_or_clamped():
    import pytest
    from frc_defense.meco_tracks import _gather_graph_candidates

    output = {"utility": torch.zeros(1, 3), "penalty": torch.zeros(1, 3),
              "behavior_logits": torch.zeros(1, 5)}
    bad = CandidateTable("s", ("bad",), ((1.5, 0.),), (True,),
                         ("legal_node_index", "behavior_index"))
    with pytest.raises(ValueError, match="legal_node_index"):
        _gather_graph_candidates(output, bad)


def test_masked_sampling_returns_valid_action_and_matching_logprob():
    from frc_defense.meco_tracks import candidate_policy_logits

    scores = torch.tensor([[0.25, 100.]])
    mask = torch.tensor([[True, False]])
    logits = candidate_policy_logits(scores, mask)
    action, logprob, entropy = sample_candidate_action(logits)
    assert action.tolist() == [0]
    assert logprob.tolist() == [0.]
    assert entropy.tolist() == [0.]
    torch.manual_seed(12)
    logits = torch.tensor([[0., 0., -torch.inf]])
    action, logprob, entropy = sample_candidate_action(logits)
    assert action.item() in (0, 1)
    assert torch.allclose(logprob, torch.tensor([-torch.log(torch.tensor(2.)).item()]))
    assert torch.allclose(entropy, torch.tensor([torch.log(torch.tensor(2.)).item()]))


def test_transition_mask_blocks_messages_and_heading_wrap_is_continuous():
    from frc_defense.meco_tracks import _se2_node_inputs

    torch.manual_seed(9)
    block = SE2MessageBlock()
    nodes = torch.randn(1, 2, 64)
    edges = torch.tensor([[[1], [0]]])
    edge_mask = torch.zeros(1, 2, 1, dtype=torch.bool)
    out1 = block(nodes, edges, edge_mask)
    changed = nodes.clone()
    changed[0, 1] += 100.
    out2 = block(changed, edges, edge_mask)
    assert torch.allclose(out1[0, 0], out2[0, 0])

    features = torch.zeros(1, 3, 9)
    features[0, :, 8] = torch.tensor((7., 0., 1.))
    headings = _se2_node_inputs(features, torch.zeros(1, 7))[0, :, 8:10]
    assert torch.allclose((headings[0] - headings[1]).norm(),
                          (headings[1] - headings[2]).norm(), atol=1e-6)
