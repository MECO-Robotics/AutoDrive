import pytest
import torch

from frc_defense.tensor_adstar import _hip_adstar_extension


@pytest.mark.skipif(not (torch.cuda.is_available() and torch.version.hip),
                    reason="requires the optional ROCm fused AD* kernel")
def test_fused_early_convergence_matches_full_sweeps():
    extension = _hip_adstar_extension()
    if extension is None:
        pytest.skip("optional AD* HIP extension could not be built")

    batch, nx, ny, sweeps = 12, 56, 27, 72
    cells = nx * ny
    for seed in (3, 17, 41):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        blocked = torch.rand((batch, nx, ny), generator=generator, device="cuda") < .18
        goal_x = torch.randint(0, nx, (batch,), generator=generator, device="cuda")
        goal_y = torch.randint(0, ny, (batch,), generator=generator, device="cuda")
        rows = torch.arange(batch, device="cuda")
        blocked[rows, goal_x, goal_y] = False
        value = torch.full((batch, nx, ny), 1.e6, device="cuda")
        value[rows, goal_x, goal_y] = 0.
        bump = torch.where(
            torch.rand((batch, nx, ny), generator=generator, device="cuda") < .12,
            torch.full((batch, nx, ny), 1.12, device="cuda"),
            torch.ones((batch, nx, ny), device="cuda"),
        )
        full, early = torch.empty_like(value), torch.empty_like(value)
        extension.bellman_sweep(value, blocked, bump, full, sweeps, False)
        extension.bellman_sweep(value, blocked, bump, early, sweeps, True)
        torch.cuda.synchronize()
        assert torch.equal(early, full)
