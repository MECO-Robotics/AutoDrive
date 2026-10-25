"""Plan A PPO replay contracts for A1/A2/A3 (synthetic tensors only).

These tests intentionally target an explicitly selected WX9100. They have no
CPU fallback; run only after the training owner releases the assigned device.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from frc_defense.meco_plan_a_policy import PlanAActionSupport, plan_a_support_digest
from frc_defense.meco_track_training import (
    MecoTrainConfig,
    _policy_inputs_for_row,
    _ppo_update,
)
from frc_defense.meco_tracks import (
    HIDDEN,
    categorical_from_candidate_logits,
    make_meco_track,
)


DEVICE = torch.device("cuda:1")


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class ValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(7 + HIDDEN, 1)

    def forward(self, globals_, hidden):
        return self.linear(torch.cat((globals_, hidden), dim=-1))


def _support(graph, valid=(True, False)):
    device = graph.xy.device
    support = PlanAActionSupport(
        node_ids=torch.arange(2, device=device),
        behavior_ids=torch.tensor([0, 1], device=device),
        action_ids=torch.tensor([0, 1], device=device),
        pair_node_ids=torch.tensor([0, 1], device=device),
        descriptor_ids=torch.tensor([-1, -1], device=device),
        valid_mask=torch.tensor(valid, dtype=torch.bool, device=device),
        distances_m=torch.tensor([0., 1.], device=device),
        route_cost=torch.tensor([0., 1.], device=device),
        parent_ids=torch.tensor([0, 0], device=device),
        parent_edge_m=torch.tensor([0., 1.], device=device),
        traversal_order=torch.tensor([0, 1], device=device),
        ancestor_jumps=torch.tensor([[0, 0]], device=device),
        root_node=0,
        root_connector_m=0.,
        fallback_pair=0,
        goal_xy=graph.xy.detach().clone(),
        goal_heading=torch.zeros(2, device=device),
        snapshot_id="synthetic-plan-a-replay",
        support_digest="",
    )
    digest = plan_a_support_digest(support, graph)
    return PlanAActionSupport(**{**support.__dict__, "support_digest": digest})


def _inputs(track):
    torch.manual_seed(42)
    graph = SimpleNamespace(
        node_count=2,
        xy=torch.tensor([[2., 2.], [8., 4.]], device=DEVICE),
        heading_bin=torch.tensor([0, 1], dtype=torch.long, device=DEVICE),
        heading=torch.tensor([0., .785398], device=DEVICE),
        edges=torch.zeros((1, 2, 1), dtype=torch.long, device=DEVICE),
        edge_mask=torch.zeros((1, 2, 1), dtype=torch.bool, device=DEVICE),
        lookup=torch.zeros((16, 8, 8), dtype=torch.long, device=DEVICE),
        nx=16,
        ny=8,
        resolution_m=1.,
    )
    support = _support(graph)
    raster = torch.zeros((1, 6, 16, 16), device=DEVICE)
    globals_ = torch.zeros((1, 7), device=DEVICE)
    model = make_meco_track(track).to(DEVICE).eval()
    graph_inputs = None
    if track in ("A2", "A3"):
        node_features = torch.zeros((1, 2, 9), device=DEVICE)
        node_features[0, :, 6:8] = torch.tensor([[.125, .25], [.5, .5]], device=DEVICE)
        node_features[0, :, 8] = torch.tensor([0., 1.], device=DEVICE)
        graph_inputs = {
            "node_features": node_features,
            "edges": torch.zeros((1, 2, 1), dtype=torch.long, device=DEVICE),
            "edge_mask": torch.zeros((1, 2, 1), dtype=torch.bool, device=DEVICE),
            "region_ids": torch.tensor([[0, 1]], dtype=torch.long, device=DEVICE),
            "coarse_edges": torch.zeros((1, 64, 1), dtype=torch.long, device=DEVICE),
            "coarse_edge_mask": torch.zeros((1, 64, 1), dtype=torch.bool, device=DEVICE),
        }
    config = MecoTrainConfig(track=track, device="cuda:1", p_mode="p0",
                             epochs=1, rollout_steps=1)
    planner = SimpleNamespace(length=16., width=8.)
    row = {
        "support": support,
        "support_digest": support.support_digest,
        "graph_object": graph,
        "raster": raster,
        "globals": globals_,
        "graph": graph_inputs,
        "route_cost_nodes": torch.zeros(2, device=DEVICE),
        "risk_cost_nodes": torch.zeros(2, device=DEVICE),
        "planner_object": planner,
        "hidden_before": None,
        "action": torch.tensor([0], dtype=torch.long, device=DEVICE),
        "reward": torch.tensor(1., device=DEVICE),
        "value": torch.tensor(0., device=DEVICE),
        "done": True,
        "execution_failure": False,
    }
    return row, model, config


def _rollout_logprob(row, model, config):
    with torch.no_grad():
        decision = _policy_inputs_for_row(row["track"], model, row, None, config)
        dist = categorical_from_candidate_logits(decision["logits"])
        action = dist.sample()
        row["action"] = action
        row["logprob"] = dist.log_prob(action).reshape(()).detach()
        row["value"] = torch.zeros((), device=DEVICE)
        row["hidden_before"] = None
        return decision["logits"]


@pytest.mark.parametrize("track", ["A1", "A2", "A3"])
def test_plan_a_masked_support_sampling_and_exact_replay(track):
    row, model, config = _inputs(track)
    row["track"] = track
    logits = _rollout_logprob(row, model, config)
    dist = categorical_from_candidate_logits(logits)
    assert torch.equal(dist.probs[:, 1], torch.zeros_like(dist.probs[:, 1]))
    assert torch.isfinite(row["logprob"])
    assert torch.equal(dist.log_prob(row["action"]).reshape(()), row["logprob"])
    samples = torch.stack([dist.sample() for _ in range(128)])
    assert bool((samples == 0).all())

    value_head = ValueHead().to(DEVICE)
    optimizer = CountingSGD(list(model.parameters()) + list(value_head.parameters()))
    metrics = _ppo_update(track, model, value_head, optimizer, [row], config,
        torch.zeros((), device=DEVICE), update_number=1, diagnostic_output=None)
    assert optimizer.step_count == 1
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))


@pytest.mark.parametrize("track", ["A1", "A2", "A3"])
@pytest.mark.parametrize("mutation", ["policy_input", "support"])
def test_plan_a_replay_mutations_fail_before_optimizer_step(track, mutation, tmp_path):
    row, model, config = _inputs(track)
    row["track"] = track
    _rollout_logprob(row, model, config)
    if mutation == "policy_input":
        row["globals"] = row["globals"] + .125
        expected = "rollout log-probability does not replay"
    else:
        row["support"].valid_mask[0] = False
        row["support"].valid_mask[1] = True
        expected = "frozen support changed"
    value_head = ValueHead().to(DEVICE)
    optimizer = CountingSGD(list(model.parameters()) + list(value_head.parameters()))
    with pytest.raises(RuntimeError, match=expected):
        _ppo_update(track, model, value_head, optimizer, [row], config,
            torch.zeros((), device=DEVICE), update_number=2,
            diagnostic_output=tmp_path)
    assert optimizer.step_count == 0


@pytest.mark.parametrize("track", ["A1", "A2", "A3"])
def test_plan_a_execution_failure_is_critic_only(track):
    row, model, config = _inputs(track)
    row["track"] = track
    _rollout_logprob(row, model, config)
    row["execution_failure"] = True
    row["reward"] = torch.tensor(2., device=DEVICE)
    value_head = ValueHead().to(DEVICE)
    optimizer = CountingSGD(list(model.parameters()) + list(value_head.parameters()))
    metrics = _ppo_update(track, model, value_head, optimizer, [row], config,
        torch.zeros((), device=DEVICE), update_number=3, diagnostic_output=None)
    assert optimizer.step_count == 1
    assert metrics["policy_loss"] == 0.
    assert metrics["entropy"] == 0.
    assert metrics["value_loss"] > 0.
