#!/usr/bin/env python3
"""Exactness checks for the isolated fuel candidate rank prototype."""
import json
import torch

from frc_defense.tensor_fuel_candidate_rank import candidates, available


def reference(points, robot, free):
    distance = (points - robot[:, None, :]).norm(dim=-1).masked_fill(~free, float("inf"))
    nearest, indices = distance.topk(4, dim=-1, largest=False)
    return indices, torch.isfinite(nearest), nearest


def check(points, robot, free, label):
    expected = reference(points, robot, free)
    active = torch.zeros((points.shape[0],), device=points.device, dtype=torch.bool)
    active[::3] = True
    actual = candidates(points, robot, free, active_mask=active)
    for name, got, want in zip(("indices", "valid", "nearest"), actual, expected):
        if not torch.equal(got, want):
            differing = int((got != want).sum().item())
            raise AssertionError(f"{label}: {name} mismatch in {differing} values")


def check_tie_semantics(points, robot, free):
    """Check determinate tie invariants; ROCm topk tie indices are unstable."""
    expected = reference(points, robot, free)
    active = torch.zeros((points.shape[0],), device=points.device, dtype=torch.bool)
    active[::3] = True
    actual = candidates(points, robot, free, active_mask=active)
    if not torch.equal(actual[1], expected[1]) or not torch.equal(actual[2], expected[2]):
        raise AssertionError("tie case nearest distances/validity mismatch")
    # Row 0 has eight equal finite candidates with k=4. Torch can choose any
    # four of them, and can vary its tie indices across otherwise identical
    # topk launches on this ROCm build.
    tied = actual[0][0].cpu().tolist()
    if len(set(tied)) != 4 or not set(tied).issubset(set(range(8))):
        raise AssertionError("finite tied candidate membership changed")
    # Rows 1..4 contain 0..3 unique valid points. Indices attached to +inf are
    # deliberately ignored: backend topk returns nondeterministic indices for
    # equal infinities, so those indices have no stable Torch behavior to copy.
    for row, count in enumerate((0, 1, 2, 3), start=1):
        if count and not torch.equal(actual[0][row, :count], expected[0][row, :count]):
            raise AssertionError(f"row {row}: finite candidate indices differ")
        if int(actual[1][row].sum().item()) != count:
            raise AssertionError(f"row {row}: expected {count} valid outputs")


def main():
    if not available("cuda"):
        raise RuntimeError("HIP candidate extension unavailable")
    torch.manual_seed(94721)
    n, p = 257, 504
    points = torch.randn((n, p, 2), device="cuda", dtype=torch.float32).contiguous()
    robot = torch.randn((n, 2), device="cuda", dtype=torch.float32).contiguous()
    free = torch.rand((n, p), device="cuda") > 0.2
    # Deliberate symmetric ties and a tie crossing top-4. This asks ATen topk
    # itself to retain its backend-specific tie order.
    robot[0] = 0
    points[0, :8] = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.],
                                  [.6, .8], [-.6, .8], [.8, -.6], [-.8, -.6]], device="cuda")
    free[0].fill_(False)
    free[0, :4] = True
    # Empty and one/two/three valid rows exercise +inf ordering and valid bits.
    for row, count in enumerate((0, 1, 2, 3), start=1):
        free[row].fill_(False)
        free[row, :count] = True
    # Mixed active mask rows must be read-only and reproduce their own data.
    active = torch.zeros((n,), device="cuda", dtype=torch.bool)
    active[::3] = True
    state_before = torch.cuda.get_rng_state().clone()
    check_tie_semantics(points, robot, free)
    if not torch.equal(state_before, torch.cuda.get_rng_state()):
        raise AssertionError("candidate ranking changed CUDA RNG state")

    cases = []
    for seed in range(4):
        torch.manual_seed(seed + 101)
        pnts = torch.randn((n, p, 2), device="cuda")
        rob = torch.randn((n, 2), device="cuda")
        mask = torch.rand((n, p), device="cuda") > (0.05 + seed * 0.2)
        check(pnts, rob, mask, f"random seed {seed}")
        cases.append({"seed": seed, "exact": True})
    # Record tie behavior explicitly; do not treat any one tied index order as
    # a deterministic reference contract.
    tied_values = torch.full((n, p), 1.0, device="cuda")
    tie_patterns = [tied_values.topk(4, dim=-1, largest=False).indices[0].cpu().tolist()
                    for _ in range(8)]
    print(json.dumps({"device": torch.cuda.get_device_name(), "shape": [n, p],
                      "adversarial_ties": "distance/validity and finite membership exact; +inf tie indices unstable",
                      "repeated_finite_tie_index_patterns": tie_patterns,
                      "finite_tie_indices_varied": len({tuple(x) for x in tie_patterns}) > 1,
                      "valid_counts_0_to_3": True,
                      "mixed_active_rows": True, "rng_unchanged": True,
                      "random_cases": cases}, indent=2))


if __name__ == "__main__":
    main()
