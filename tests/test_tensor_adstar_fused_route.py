from types import SimpleNamespace

import pytest
import torch

import frc_defense.tensor_adstar as adstar_module
from frc_defense.tensor_adstar import TensorADStar, _hip_adstar_extension


class _Box:
    def __init__(self, x, y, length, width, name="wall"):
        self.x, self.y = x, y
        self.length, self.width = length, width
        self.name = name


def _planner(batch, avoid_bumps):
    return _planner_on_device(batch, avoid_bumps, torch.device("cuda"))


def _planner_on_device(batch, avoid_bumps, device):
    boxes = [_Box(.2, 4.1, .4, 8.2), _Box(16.3, 4.1, .4, 8.2),
             _Box(8.25, .2, 16.5, .4), _Box(8.25, 8., 16.5, .4),
             _Box(8.0, 4.1, .5, 1.4, "center_bump")]
    sim = SimpleNamespace(
        pose=torch.zeros((batch, 2, 3), device=device),
        field_length=16.5, field_width=8.2,
        field_colliders=[(b.x, b.y, b.length / 2, b.width / 2) for b in boxes[:4]],
        obstacles=torch.tensor([[5.0, 4.1, .35]], device=device),
    )
    env = SimpleNamespace(device=device, sim=sim, n=batch, field_boxes=boxes)
    return TensorADStar(env, resolution=.3, max_points=72, sweeps=72,
                        avoid_bumps=avoid_bumps, early_convergence=True)


def test_cpu_planner_keeps_torch_fallback():
    planner = _planner_on_device(2, False, torch.device("cpu"))
    start = torch.tensor([[1.2, 2.0], [1.3, 6.0]])
    goal = torch.tensor([[14.8, 2.0], [14.7, 6.0]])
    heading = torch.zeros(2)
    size = torch.full((2,), .8)
    planner.plan(start, goal, heading, size, size, active_mask=None)
    assert torch.all(planner.last_lengths > 1)
    assert torch.isfinite(planner.potential).all()


@pytest.mark.skipif(not (torch.cuda.is_available() and torch.version.hip),
                    reason="requires the optional ROCm fused AD* kernel")
@pytest.mark.parametrize("avoid_bumps,dynamic", [(False, False), (True, False),
                                                   (False, True), (True, True)])
def test_fused_route_matches_torch_and_preserves_inactive_rows(
        avoid_bumps, dynamic, monkeypatch):
    extension = _hip_adstar_extension()
    if extension is None:
        pytest.skip("optional AD* HIP extension could not be built")
    batch = 8
    fused = _planner(batch, avoid_bumps)
    reference = _planner(batch, avoid_bumps)
    generator = torch.Generator(device="cuda").manual_seed(100 + int(avoid_bumps) * 7 + int(dynamic))
    start = torch.empty((batch, 2), device="cuda").uniform_(1.0, 2.3, generator=generator)
    start[:, 1] = torch.empty((batch,), device="cuda").uniform_(1., 7.2, generator=generator)
    goal = torch.empty((batch, 2), device="cuda").uniform_(14.2, 15.4, generator=generator)
    goal[:, 1] = torch.empty((batch,), device="cuda").uniform_(1., 7.2, generator=generator)
    heading = torch.empty((batch,), device="cuda").uniform_(-.3, .3, generator=generator)
    length = torch.full((batch,), .9, device="cuda")
    width = torch.full((batch,), .75, device="cuda")
    speed = torch.linspace(2.5, 4.5, batch, device="cuda")
    friction = torch.linspace(.8, 1.4, batch, device="cuda")
    acceleration = torch.linspace(3., 9., batch, device="cuda")
    defender = torch.stack((torch.full((batch,), 8., device="cuda"),
                            torch.linspace(2., 6., batch, device="cuda")), -1)
    defender_velocity = torch.zeros_like(defender)
    active = torch.tensor([True, False, True, True, False, True, False, True],
                          device="cuda")

    # Make stale state visible: inactive rows must remain byte-for-byte intact.
    for planner in (fused, reference):
        planner.last_path.fill_(37.)
        planner.last_lengths.fill_(19)
        planner.last_intercept.fill_(23.)
        planner.last_intercept_time.fill_(29.)
        planner.last_start.fill_(31.)
        planner.last_goal.fill_(41.)
        planner.last_heading.fill_(43.)
        planner.last_speed_profile.fill_(47.)
        planner.potential.fill_(53.)

    monkeypatch.setattr(adstar_module, "_HIP_ADSTAR_EXTENSION", extension)
    monkeypatch.setattr(adstar_module, "_HIP_ADSTAR_EXTENSION_ATTEMPTED", True)
    fused.plan(start, goal, heading, length, width, speed=speed,
               defender=defender if dynamic else None,
               defender_velocity=defender_velocity if dynamic else None,
               dynamic_defender=dynamic, lateral_friction=friction,
               acceleration=acceleration, active_mask=active)

    monkeypatch.setattr(adstar_module, "_HIP_ADSTAR_EXTENSION", None)
    reference.plan(start, goal, heading, length, width, speed=speed,
                   defender=defender if dynamic else None,
                   defender_velocity=defender_velocity if dynamic else None,
                   dynamic_defender=dynamic, lateral_friction=friction,
                   acceleration=acceleration, active_mask=active)
    torch.cuda.synchronize()

    for name in ("last_path", "last_lengths", "last_intercept", "last_intercept_time",
                 "last_start", "last_goal", "last_heading", "potential"):
        got, want = getattr(fused, name), getattr(reference, name)
        assert torch.equal(got[active], want[active]), name
        assert torch.equal(got[~active], torch.full_like(got[~active], {
            "last_path": 37., "last_lengths": 19, "last_intercept": 23.,
            "last_intercept_time": 29., "last_start": 31., "last_goal": 41.,
            "last_heading": 43., "potential": 53.,
        }[name])), name
    torch.testing.assert_close(fused.last_speed_profile[active],
                               reference.last_speed_profile[active],
                               rtol=2e-5, atol=2e-5)
    assert torch.equal(fused.last_speed_profile[~active],
                       torch.full_like(fused.last_speed_profile[~active], 47.))

    # Exercise the no-compaction path used when all environments share cadence.
    for planner in (fused, reference):
        planner.last_path.zero_()
        planner.last_lengths.zero_()
        planner.last_intercept.zero_()
        planner.last_intercept_time.zero_()
        planner.last_start.zero_()
        planner.last_goal.zero_()
        planner.last_heading.zero_()
        planner.last_speed_profile.zero_()
        planner.potential.zero_()
    monkeypatch.setattr(adstar_module, "_HIP_ADSTAR_EXTENSION", extension)
    fused.plan(start, goal, heading, length, width, speed=speed,
               defender=defender if dynamic else None,
               defender_velocity=defender_velocity if dynamic else None,
               dynamic_defender=dynamic, lateral_friction=friction,
               acceleration=acceleration, active_mask=None)
    monkeypatch.setattr(adstar_module, "_HIP_ADSTAR_EXTENSION", None)
    reference.plan(start, goal, heading, length, width, speed=speed,
                   defender=defender if dynamic else None,
                   defender_velocity=defender_velocity if dynamic else None,
                   dynamic_defender=dynamic, lateral_friction=friction,
                   acceleration=acceleration, active_mask=None)
    torch.cuda.synchronize()
    for name in ("last_path", "last_lengths", "last_intercept", "last_intercept_time",
                 "last_start", "last_goal", "last_heading", "potential"):
        assert torch.equal(getattr(fused, name), getattr(reference, name)), name
    torch.testing.assert_close(fused.last_speed_profile, reference.last_speed_profile,
                               rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not (torch.cuda.is_available() and torch.version.hip),
                    reason="requires the optional ROCm fused AD* kernel")
def test_fused_full_batch_accepts_strided_simulator_start(monkeypatch):
    extension = _hip_adstar_extension()
    if extension is None:
        pytest.skip("optional AD* HIP extension could not be built")
    batch = 8
    planner = _planner(batch, False)
    storage = torch.zeros((batch, 3), device="cuda")
    storage[:, 0] = torch.linspace(1.2, 2.3, batch, device="cuda")
    storage[:, 1] = torch.linspace(1.1, 7.1, batch, device="cuda")
    start = storage[:, :2]
    assert not start.is_contiguous()
    goal = torch.stack((torch.full((batch,), 14.8, device="cuda"),
                        torch.linspace(7.1, 1.1, batch, device="cuda")), -1)
    size = torch.full((batch,), .8, device="cuda")
    heading = torch.zeros(batch, device="cuda")
    calls = []

    class _DispatchSpy:
        def __getattr__(self, name):
            return getattr(extension, name)

        def fused_route(self, *args):
            calls.append(True)
            raise RuntimeError("fused route dispatch reached")

    monkeypatch.setattr(adstar_module, "_HIP_ADSTAR_EXTENSION", _DispatchSpy())
    monkeypatch.setattr(adstar_module, "_HIP_ADSTAR_EXTENSION_ATTEMPTED", True)
    with pytest.raises(RuntimeError, match="fused route dispatch reached"):
        planner.plan(start, goal, heading, size, size, active_mask=None)
    assert calls == [True]
