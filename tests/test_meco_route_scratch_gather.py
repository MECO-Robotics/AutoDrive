from types import SimpleNamespace

import torch

from frc_defense import experimental_route_evaluation as routes


def test_scratch_planner_gathers_query_rows_in_order(monkeypatch):
    class FakePlanner:
        def __init__(self, env, **kwargs):
            self.env = env
            self.kwargs = kwargs

    monkeypatch.setattr(routes, "TensorADStar", FakePlanner)
    values = torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6)
    sim = SimpleNamespace(
        velocity=torch.stack((values + 10, values + 20, values + 30), -1),
        omega_limit=values + 40, alpha=values + 50,
        field_length=16., field_width=8., field_colliders=(), obstacles=(),
        swerve=None)
    planner = SimpleNamespace(resolution=.2, max_points=70, sweeps=3,
        avoid_bumps=True, early_convergence=False, footprint_clearance=.1,
        _fused_blocked_hip_enabled=True)
    env = SimpleNamespace(device=torch.device("cpu"), sim=sim, planner=planner,
        policy_pose=torch.stack((values, values + 1, values + 2), -1),
        field_boxes=())
    worlds = torch.tensor([1, 0, 1, 1])
    slots = torch.tensor([3, 4, 3, 0])
    rows = [(SimpleNamespace(world=int(w), slot=int(s)), None)
            for w, s in zip(worlds, slots)]

    for indices in ({}, {"world_index": worlds, "slot_index": slots}):
        scratch = routes._scratch_planner(env, rows, **indices)
        assert scratch.env.n == len(rows)
        assert torch.equal(scratch.env.sim.pose[:, 0], env.policy_pose[worlds, slots])
        assert torch.equal(scratch.env.sim.velocity[:, 0], sim.velocity[worlds, slots])
        assert torch.equal(scratch.env.sim.omega_limit, sim.omega_limit[worlds, slots])
        assert torch.equal(scratch.env.sim.alpha, sim.alpha[worlds, slots])
        assert scratch.footprint_clearance == planner.footprint_clearance
        assert scratch._fused_blocked_hip_enabled is True
