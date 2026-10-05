"""Parity checks for the optional fused HIP gamepiece step."""
import pytest
import torch

from frc_defense import tensor_gamepieces
from frc_defense.tensor_sim import TensorDefenseEnv


@pytest.mark.skipif(not (torch.cuda.is_available() and torch.version.hip),
                    reason="requires a ROCm/HIP device")
def test_fused_gamepiece_update_matches_cpu_torch_reference(monkeypatch):
    """Compare all mutated gamepiece and match-clock state, including inactive rows."""
    monkeypatch.setattr(tensor_gamepieces, "_ENABLED", True)
    monkeypatch.setattr(tensor_gamepieces, "_ATTEMPTED", False)
    monkeypatch.setattr(tensor_gamepieces, "_EXT", None)

    cpu = TensorDefenseEnv(num_envs=5, task="counter_defense", action_mode="strategic",
                           device="cpu", seed=310, randomize=False, fuel_count=504,
                           max_fuel_capacity=1)
    gpu = TensorDefenseEnv(num_envs=5, task="counter_defense", action_mode="strategic",
                           device="cuda", seed=310, randomize=False, fuel_count=504,
                           max_fuel_capacity=1)
    pieces = cpu.fuel_count
    cpu.piece_active.zero_()
    cpu.piece_owner.fill_(-1)
    cpu.piece_zone.fill_(0)
    cpu.piece_pos.zero_()
    cpu.piece_vel.zero_()
    cpu.sim.pose.zero_()
    cpu.sim.velocity.zero_()
    cpu.sim.length.fill_(.9)
    cpu.sim.width.fill_(.9)
    cpu.match_elapsed.copy_(torch.tensor([19.98, 42., 12., 90., 12.]))
    cpu.match_remaining.copy_(160. - cpu.match_elapsed)
    # Keep robot 0's hub active at the autonomous/teleop boundary so its
    # lowest-index held piece scores at t=20.0 and exercises the auto counter.
    cpu.hub_inactive_first.copy_(torch.tensor([1, 1, 0, 0, 0]))
    cpu._last_strategic_action.copy_(torch.tensor([4, 2, 4, 4, 2]))
    cpu.next_intake_time.zero_()
    cpu.next_score_time.zero_()
    cpu._last_hub_zone.zero_()
    cpu.fuel_acquired_event.fill_(7)
    cpu.fuel_scored_event.fill_(8)
    cpu.fuel_denied_event.fill_(9)
    cpu.fuel_abandoned_event.fill_(10)

    # Row 0 scores the lowest indexed of two held pieces and is still in auto.
    for row, robot, index in ((0, 0, 261), (0, 0, 503), (3, 0, 258)):
        cpu.piece_active[row, index] = True
        cpu.piece_owner[row, index] = robot
        cpu.piece_pos[row, index] = cpu.sim.pose[row, robot, :2]
    hub = cpu.hub_centers[0]
    for row in (0, 3):
        cpu.sim.pose[row, 0, :2] = hub + torch.tensor([-.1, 0.])
        cpu.sim.pose[row, 0, 2] = 0.  # face +X toward own HUB
    # Row 1 is at capacity despite nearby free pieces and must not acquire.
    cpu.sim.pose[1, 0] = torch.tensor([2., 2., 0.])
    cpu.piece_active[1, 259] = True
    cpu.piece_owner[1, 259] = 0
    cpu.piece_active[1, 256:258] = True
    cpu.piece_pos[1, 256] = torch.tensor([2.5, 2.])
    cpu.piece_pos[1, 257] = torch.tensor([2.5, 2.])
    # Row 4 exercises front intake and index-stable ties while under capacity.
    cpu.sim.pose[4, 0] = torch.tensor([2., 2., 0.])
    cpu.piece_active[4, 256:258] = True
    cpu.piece_pos[4, 256] = torch.tensor([2.5, 2.])
    cpu.piece_pos[4, 257] = torch.tensor([2.5, 2.])
    # Row 3 starts inside the inactive hub zone with fuel; denial is entry-latched.
    cpu._last_hub_zone[3, 0] = False
    active = torch.tensor([True, True, False, True, True])
    # Set distinct counters so accidental inactive mutation is observable.
    cpu.fuel_acquisition_count.copy_(torch.arange(10).reshape(5, 2))
    cpu.fuel_score_count.copy_(torch.arange(10).reshape(5, 2) + 10)
    cpu.fuel_denied_count.copy_(torch.arange(10).reshape(5, 2) + 20)
    cpu.fuel_abandoned_count.copy_(torch.arange(10).reshape(5, 2) + 30)

    state_names = (
        "piece_pos", "piece_vel", "piece_active", "piece_owner", "piece_zone",
        "match_elapsed", "match_remaining", "hub_active", "auto_fuel_scores",
        "next_intake_time", "next_score_time", "_last_hub_zone",
        "fuel_acquisition_count", "fuel_score_count", "fuel_denied_count",
        "fuel_abandoned_count", "fuel_acquired_event", "fuel_scored_event",
        "fuel_denied_event", "fuel_abandoned_event",
    )
    kernel_inputs = (
        "hub_inactive_first", "_last_strategic_action", "hub_centers",
        "next_intake_time", "next_score_time", "_last_hub_zone",
        "fuel_acquisition_count", "fuel_score_count", "fuel_denied_count",
        "fuel_abandoned_count", "fuel_acquired_event", "fuel_scored_event",
        "fuel_denied_event", "fuel_abandoned_event", "piece_pos", "piece_vel",
        "piece_active", "piece_owner", "piece_zone", "match_elapsed",
        "match_remaining", "hub_active", "auto_fuel_scores",
    )
    for name in kernel_inputs:
        getattr(gpu, name).copy_(getattr(cpu, name).to("cuda"))
    for name in ("pose", "velocity", "length", "width"):
        getattr(gpu.sim, name).copy_(getattr(cpu.sim, name).to("cuda"))
    active_gpu = active.to("cuda")

    cpu._update_gamepieces_torch(active)
    assert tensor_gamepieces.update_gamepieces(gpu, active_gpu), "HIP extension did not load"
    for name in state_names:
        actual = getattr(gpu, name).cpu()
        expected = getattr(cpu, name)
        if actual.dtype.is_floating_point:
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        else:
            assert torch.equal(actual, expected), name
    assert gpu.piece_active[0, 261].item() is False
    assert gpu.piece_active[0, 503].item() is True
    assert gpu.fuel_scored_event[0, 0].item() == 1
    assert gpu.piece_owner[4, 256].item() == 0
    assert gpu.piece_owner[4, 257].item() == -1
    assert gpu.fuel_acquired_event[4, 0].item() == 1
    assert gpu.fuel_acquired_event[1, 0].item() == 0  # capacity blocks pickup
    assert gpu.fuel_denied_event[3, 0].item() == 1
    assert gpu.fuel_scored_event[3, 0].item() == 0
    assert gpu.fuel_acquired_event[2].tolist() == [7, 7]  # inactive events survive
