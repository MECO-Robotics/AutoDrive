import copy

import pytest
import torch

from frc_defense.meco_tracks import SE2MessageBlock


def _original_forward(block, nodes, edges, edge_mask, node_valid):
    edge_valid = (edges >= 0) & (edges < nodes.shape[1])
    safe_index = edges.clamp(0, nodes.shape[1] - 1)
    destination_valid = node_valid.gather(1, safe_index.reshape(nodes.shape[0], -1)).reshape_as(edges)
    source_valid = node_valid[:, :, None]
    status = ~(edge_mask & (~edge_valid | ~destination_valid | ~source_valid)).any()
    safe_mask = edge_mask & edge_valid & destination_valid & source_valid
    safe = torch.where(safe_mask, edges, torch.zeros_like(edges))
    expanded = nodes[:, :, None, :].expand(-1, -1, safe.shape[-1], -1)
    neighbors = expanded.gather(1, safe[..., None].expand(-1, -1, -1, nodes.shape[-1]))
    messages = block.message(neighbors)
    mask = safe_mask[..., None].to(messages.dtype)
    aggregate = (messages * mask).sum(2) / mask.sum(2).clamp_min(1.)
    output = block.norm(nodes + block.update(torch.cat((nodes, aggregate), -1)))
    return torch.where(node_valid[..., None], output, torch.zeros_like(output)), status


@pytest.mark.parametrize("malformed", [False, True])
def test_reused_graph_messages_match_forward_and_gradients(malformed):
    torch.manual_seed(681)
    baseline = SE2MessageBlock(width=16)
    candidate = copy.deepcopy(baseline)
    old_nodes = torch.randn(2, 19, 16, requires_grad=True)
    new_nodes = old_nodes.detach().clone().requires_grad_()
    edges = torch.randint(-1 if malformed else 0, 20 if malformed else 19, (2, 19, 8))
    edge_mask = torch.rand(2, 19, 8) < .7
    node_valid = (torch.rand(2, 19) < .8 if malformed else
                  torch.ones(2, 19, dtype=torch.bool))
    weighting = torch.randn_like(old_nodes)

    expected, old_status = _original_forward(baseline, old_nodes, edges, edge_mask, node_valid)
    actual, new_status = candidate(new_nodes, edges, edge_mask,
                                   node_valid=node_valid, return_status=True)
    assert torch.equal(old_status, new_status)
    assert bool(old_status) is not malformed
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    (expected * weighting).sum().backward()
    (actual * weighting).sum().backward()
    torch.testing.assert_close(new_nodes.grad, old_nodes.grad, rtol=1e-5, atol=1e-6)
    for original, changed in zip(baseline.parameters(), candidate.parameters()):
        torch.testing.assert_close(changed.grad, original.grad, rtol=1e-5, atol=1e-6)
