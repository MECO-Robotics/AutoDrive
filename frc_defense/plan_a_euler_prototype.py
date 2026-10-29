"""Experimental device-only tree scan for Plan A path-integral gradients.

This is not wired into training. Callers must honor the returned validity flag
before publishing any model, optimizer, or RNG state.
"""
import math
import torch


def euler_metadata(parent: torch.Tensor, edge: torch.Tensor, jumps=None):
    """Build an Euler tour using pointer doubling, without device readback."""
    if parent.ndim != 1 or edge.shape != parent.shape or parent.numel() == 0:
        raise ValueError("expected nonempty one-dimensional parent and edge")
    n = parent.numel()
    dev = parent.device
    nodes = torch.arange(n, device=dev, dtype=torch.long)
    in_bounds = (parent >= 0) & (parent < n)
    p = parent.clamp(0, n - 1).long()
    root_mask = p == nodes
    root = root_mask.to(torch.long).argmax()
    # Sorting by parent puts siblings next to one another. Root is a sentinel
    # parent group so it is never treated as its own child.
    keys = torch.where(root_mask, torch.full_like(p, n), p)
    child_order = torch.argsort(keys, stable=True)
    sorted_keys = keys[child_order]
    next_sorted = torch.cat((child_order[1:], torch.full((1,), n, device=dev)))
    same_parent = torch.cat((sorted_keys[1:] == sorted_keys[:-1], torch.zeros(1, device=dev, dtype=torch.bool)))
    next_sibling = torch.full((n,), n, device=dev, dtype=torch.long)
    next_sibling.scatter_(0, child_order, torch.where(same_parent, next_sorted, n))
    first = torch.full((n + 1,), n, device=dev, dtype=torch.long)
    first.scatter_reduce_(0, keys, nodes, reduce="amin", include_self=True)
    # Sort is stable in node order, so amin gives the first sorted sibling.
    down = torch.where(first[:n] < n, first[:n], n + nodes)
    sibling = next_sibling[nodes]
    up = torch.where(sibling < n, sibling, n+p)
    up = torch.where(root_mask, 2*n, up)
    successor = torch.cat((down, up, torch.full((1,), 2*n, device=dev, dtype=torch.long)))
    successor = torch.where(successor < 2*n, successor, 2*n)
    distance = torch.cat((torch.ones(2*n, device=dev, dtype=torch.long), torch.zeros(1, device=dev, dtype=torch.long)))
    for _ in range(math.ceil(math.log2(2*n)) + 1):
        distance = distance + distance[successor]
        successor = successor[successor]
    position = 2*n - distance[:2*n]
    # All event positions must be a permutation, and all paths must terminate.
    valid = (in_bounds.all() & (root_mask.sum() == 1) & (edge[root] == 0)
             & torch.isfinite(edge).all()
             & (successor[:2*n] == 2*n).all()
             & (torch.sort(position).values == torch.arange(2*n, device=dev)).all())
    if jumps is not None:
        if jumps.ndim != 2 or jumps.shape != (max(1, math.ceil(math.log2(n))), n):
            raise ValueError("jump table must cover the complete tree")
        expected = p
        for row in jumps:
            valid = valid & (row == expected).all()
            expected = expected[expected]
    return position.clamp(0, 2*n - 1), root, valid


def subtree_gradient(grad_output: torch.Tensor, edge: torch.Tensor,
                     root_connector: torch.Tensor, metadata):
    """Return gradient and a device validity flag; float64 limits cancellation."""
    position, root, valid = metadata
    n = grad_output.numel()
    if edge.shape != grad_output.shape or position.numel() != 2*n:
        raise ValueError("gradient, edge and metadata shapes disagree")
    # index_copy is race-free for valid metadata; invalid metadata must not be
    # committed and duplicate positions need not yield meaningful gradients.
    values = torch.zeros(2*n, device=grad_output.device, dtype=torch.float64)
    values.index_copy_(0, position[:n], grad_output.double())
    prefix = torch.cat((torch.zeros(1, device=values.device, dtype=values.dtype), values.cumsum(0)))
    subtree = prefix[position[n:] + 1] - prefix[position[:n]]
    result = (subtree * edge.double()).to(grad_output.dtype)
    root_add = grad_output.double().sum() * root_connector.double()
    result = result.scatter_add(0, root.reshape(1), root_add.to(result.dtype).reshape(1))
    return result, valid & torch.isfinite(grad_output).all() & torch.isfinite(root_connector).all()


def subtree_gradient_segmented(grad_output: torch.Tensor, edge: torch.Tensor,
                               root_connector: torch.Tensor, metadata,
                               *, accumulator_dtype=torch.float64):
    """Accumulate dyadic intervals, never subtracting outside subtree values."""
    position, root, valid = metadata
    n = grad_output.numel()
    if edge.shape != grad_output.shape or position.numel() != 2*n:
        raise ValueError("gradient, edge and metadata shapes disagree")
    width = 1 << math.ceil(math.log2(2*n))
    leaves = torch.zeros(width, device=grad_output.device, dtype=accumulator_dtype)
    leaves.index_copy_(0, position[:n], grad_output.to(accumulator_dtype))
    levels = [leaves]
    while levels[-1].numel() > 1:
        levels.append(levels[-1].reshape(-1, 2).sum(dim=1))
    left, right = position[:n], position[n:] + 1
    accum = torch.zeros(n, device=grad_output.device, dtype=accumulator_dtype)
    for level in levels:
        take_left = (left < right) & ((left & 1) != 0)
        left = left + take_left.long()
        take_right = (left < right) & ((right & 1) != 0)
        right = right - take_right.long()
        accum = accum + torch.where(take_left, level[(left-1).clamp(0, level.numel()-1)], 0.)
        accum = accum + torch.where(take_right, level[right.clamp(0, level.numel()-1)], 0.)
        left, right = left // 2, right // 2
    result = (accum * edge.to(accumulator_dtype)).to(grad_output.dtype)
    root_add = grad_output.to(accumulator_dtype).sum() * root_connector.to(accumulator_dtype)
    result = result.scatter_add(0, root.reshape(1), root_add.to(result.dtype).reshape(1))
    return result, valid & torch.isfinite(grad_output).all() & torch.isfinite(root_connector).all()


def graph_path_integral_segmented(node_cost, support):
    """Diagnostic autograd bridge; host flag read means this is not GPU-only."""
    jumps, edge = support.ancestor_jumps, support.parent_edge_m
    if node_cost.ndim != 1 or node_cost.numel() != support.node_ids.numel():
        raise ValueError("node costs must match the frozen legal graph")
    metadata = euler_metadata(jumps[0], edge, jumps)
    expected_root = torch.as_tensor(support.root_node, device=node_cost.device, dtype=torch.long)
    if expected_root.numel() != 1 or not bool(metadata[2] & (metadata[1] == expected_root.reshape(()))):
        raise RuntimeError("invalid tree metadata")

    class _Expand(torch.autograd.Function):
        @staticmethod
        def forward(ctx, initial):
            ctx.save_for_backward(*metadata)
            total = initial
            for parent in jumps:
                total = total + total.index_select(0, parent)
            return total

        @staticmethod
        def backward(ctx, gradient):
            position, root, valid = ctx.saved_tensors
            ones = torch.ones_like(gradient).scatter(0, root.reshape(1), 0.)
            connector = torch.zeros((), device=gradient.device, dtype=gradient.dtype)
            result, valid = subtree_gradient_segmented(gradient, ones, connector, (position, root, valid))
            if not bool(valid):
                raise RuntimeError("invalid segmented path gradient")
            return result

    total = _Expand.apply(node_cost * edge)
    return total + node_cost[support.root_node] * support.root_connector_m


def graph_path_integral_segmented_checked(node_cost, support, flags):
    """Speculate an integral and append device validity checks to ``flags``.

    The caller must discard every speculative optimizer update when any flag
    is false. Parent-edge gradients use ordinary autograd, since the optimized
    backward assumes a frozen zero root edge.
    """
    jumps, edge = support.ancestor_jumps, support.parent_edge_m
    n = node_cost.numel()
    if (node_cost.ndim != 1 or n == 0 or n != support.node_ids.numel()
            or jumps.device != node_cost.device or jumps.dtype != torch.int64
            or jumps.ndim != 2 or jumps.shape[1] != n or jumps.shape[0] == 0
            or edge.device != node_cost.device or edge.dtype != node_cost.dtype
            or edge.shape != node_cost.shape):
        raise ValueError("invalid path integral tensor metadata")
    metadata = euler_metadata(jumps[0], edge, jumps)
    root = torch.as_tensor(support.root_node, device=node_cost.device, dtype=torch.long)
    if root.numel() != 1:
        raise ValueError("root must be scalar")
    root = root.reshape(())
    root_valid = (root >= 0) & (root < n) & (root == metadata[1])
    safe_root = root.clamp(0, n - 1)
    safe_jumps = jumps.clamp(0, n - 1)
    connector = torch.as_tensor(support.root_connector_m, device=node_cost.device, dtype=node_cost.dtype)
    flags.append(metadata[2] & root_valid & torch.isfinite(node_cost).all()
                 & torch.isfinite(connector).all())

    if edge.requires_grad:
        total = node_cost * edge
        for parent in safe_jumps:
            total = total + total.index_select(0, parent)
    else:
        class _ExpandChecked(torch.autograd.Function):
            @staticmethod
            def forward(ctx, initial):
                ctx.save_for_backward(*metadata)
                total = initial
                for parent in safe_jumps:
                    total = total + total.index_select(0, parent)
                return total

            @staticmethod
            def backward(ctx, gradient):
                position, tree_root, valid = ctx.saved_tensors
                ones = torch.ones_like(gradient).scatter(0, tree_root.reshape(1), 0.)
                zero = torch.zeros((), device=gradient.device, dtype=gradient.dtype)
                result, valid = subtree_gradient_segmented(gradient, ones, zero, (position, tree_root, valid))
                flags.append(valid & torch.isfinite(result).all())
                return result

        total = _ExpandChecked.apply(node_cost * edge)
    result = total + node_cost[safe_root] * connector
    flags.append(torch.isfinite(result).all())
    if result.requires_grad:
        def check_output_gradient(gradient):
            flags.append(torch.isfinite(gradient).all())
            return gradient
        result.register_hook(check_output_gradient)
    return result
