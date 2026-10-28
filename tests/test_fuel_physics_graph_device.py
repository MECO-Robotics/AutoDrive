"""CPU environments must not capture CUDA graphs, including on HIP builds."""
from types import SimpleNamespace

import pytest
import torch

from frc_defense import fuel_physics, fuel_physics_runtime


@pytest.mark.parametrize("worlds", [1, 2])
def test_cpu_graph_disabled_on_hip_build(monkeypatch, worlds):
    monkeypatch.setattr(torch.version, "hip", "test-hip-build")
    monkeypatch.setenv("AUTODRIVE_HIP_PHYSICS_GRAPH", "1")
    monkeypatch.setattr(fuel_physics, "FuelPhysics", lambda *args: SimpleNamespace(
        config=SimpleNamespace(radius=.083)))
    env = SimpleNamespace(
        n=worlds, device=torch.device("cpu"), _fuel_radius=.083,
        hub_centers=torch.zeros(2, 2),
        piece_pos=torch.zeros(worlds, 1, 2), piece_vel=torch.zeros(worlds, 1, 2),
        sim=SimpleNamespace(num_robots=6, pose=torch.zeros(worlds, 6, 3),
                            field_length=16., field_width=8.),
    )
    fuel_physics_runtime.initialize_fuel_physics(env)
    assert env.fuel_physics is not None
    assert env._physics_graph_enabled is False
    assert env._physics_graphs == {}
    assert env._physics_graph_command.device.type == "cpu"


def test_whole_tick_graph_signature_tracks_command_buffers_and_swerve():
    sim = SimpleNamespace(swerve=SimpleNamespace(drive_ratio=6.),
                          pose=torch.zeros(1, 2, 3), dt=.02,
                          _base={"speed": 4.})
    physics = SimpleNamespace(config=SimpleNamespace(substeps=10),
                              _hip=SimpleNamespace(buffer=torch.zeros(2)))
    env = SimpleNamespace(sim=sim, piece_active=torch.ones(1, 2, dtype=torch.bool),
                          piece_owner=torch.zeros(1, 2, dtype=torch.long),
                          _physics_graph_command=torch.zeros(1, 2, 3),
                          _physics_graph_active=torch.ones(1, dtype=torch.bool))
    signature = fuel_physics_runtime._whole_tick_graph_signature(env, physics, 64, .02)
    assert signature == fuel_physics_runtime._whole_tick_graph_signature(env, physics, 64, .02)
    env._physics_graph_command = env._physics_graph_command.clone()
    changed = fuel_physics_runtime._whole_tick_graph_signature(env, physics, 64, .02)
    assert signature != changed
    sim.swerve.drive_ratio = 7.
    assert changed != fuel_physics_runtime._whole_tick_graph_signature(env, physics, 64, .02)


def test_experimental_whole_tick_capture_failure_fails_closed(monkeypatch):
    from contextlib import nullcontext
    import sys

    monkeypatch.setitem(sys.modules, "frc_defense.gamepiece_actions",
                        SimpleNamespace(_flush_fuel_score_awards=lambda env: None))
    calls = []
    def fail_step(*args):
        calls.append("captured")
        sim.mutated = True
        raise RuntimeError("capture failed")

    physics = SimpleNamespace(_gpu_qualifying=False, _hip=object(),
        config=SimpleNamespace(adaptive_substeps=True, sleep_islands=False),
        _validate_qualifying_backend=lambda: None, required_substeps=lambda *args: 64)
    sim = SimpleNamespace(dt=.02, step=fail_step)
    env = SimpleNamespace(fuel_physics=physics, n=1, device=torch.device("cpu"), sim=sim,
        piece_active=torch.ones(1, 1, dtype=torch.bool),
        piece_owner=torch.full((1, 1), -1), _physics_graph_enabled=True, _physics_graphs={},
        _physics_graph_command=torch.zeros(1, 2, 3), _physics_graph_active=torch.zeros(1, dtype=torch.bool))
    monkeypatch.setenv("AUTODRIVE_EXPERIMENTAL_FUEL_WHOLE_TICK_GRAPH", "1")
    monkeypatch.setattr(fuel_physics_runtime, "bind_fuel_state", lambda env: physics)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", object)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph: nullcontext())
    with pytest.raises(RuntimeError, match="refusing to advance potentially modified state"):
        fuel_physics_runtime.advance_with_fuel(env, torch.zeros(1, 2, 3), torch.ones(1, dtype=torch.bool))
    assert calls == ["captured"]
    assert sim.mutated
    assert sim.dt == .02
    assert not env._physics_graphs
    with pytest.raises(RuntimeError, match="reconstruct the simulator"):
        fuel_physics_runtime.advance_with_fuel(env, torch.zeros(1, 2, 3), torch.ones(1, dtype=torch.bool))
    assert calls == ["captured"]


def test_experimental_whole_tick_replay_failure_latches_simulator(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "frc_defense.gamepiece_actions",
                        SimpleNamespace(_flush_fuel_score_awards=lambda env: None))
    calls = []
    class FailingGraph:
        def replay(self):
            calls.append("replay")
            sim.mutated = True
            raise RuntimeError("replay failed")
    physics = SimpleNamespace(_gpu_qualifying=False, _hip=object(),
        config=SimpleNamespace(adaptive_substeps=True),
        _validate_qualifying_backend=lambda: None, required_substeps=lambda *args: 64)
    sim = SimpleNamespace(dt=.02)
    env = SimpleNamespace(fuel_physics=physics, n=1, device=torch.device("cpu"), sim=sim,
        piece_active=torch.ones(1, 1, dtype=torch.bool),
        piece_owner=torch.full((1, 1), -1), _physics_graph_enabled=True,
        _physics_graphs={64: (FailingGraph(), 1, "signature")},
        _physics_graph_command=torch.zeros(1, 2, 3), _physics_graph_active=torch.zeros(1, dtype=torch.bool))
    monkeypatch.setattr(fuel_physics_runtime, "bind_fuel_state", lambda env: physics)
    monkeypatch.setattr(fuel_physics_runtime, "_whole_tick_graph_signature", lambda *args: "signature")
    with pytest.raises(RuntimeError, match="graph replay failed"):
        fuel_physics_runtime.advance_with_fuel(env, torch.zeros(1, 2, 3), torch.ones(1, dtype=torch.bool))
    assert sim.mutated and sim.dt == .02
    with pytest.raises(RuntimeError, match="reconstruct the simulator"):
        fuel_physics_runtime.advance_with_fuel(env, torch.zeros(1, 2, 3), torch.ones(1, dtype=torch.bool))
    assert calls == ["replay"]
