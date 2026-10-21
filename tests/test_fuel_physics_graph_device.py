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
