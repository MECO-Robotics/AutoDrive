from types import SimpleNamespace
import math

import pytest

torch = pytest.importorskip("torch")


def _planner(n, device, *, avoid_bumps, with_circles):
    from frc_defense.field import INCH, rebuilt_field, static_collision_boxes
    from frc_defense.tensor_adstar import TensorADStar

    field = rebuilt_field()
    circles = (torch.tensor([[3.05, 2.45, .37], [8.4, 4.03, .61]],
                            dtype=torch.float32, device=device) if with_circles else
               torch.empty((0, 3), dtype=torch.float32, device=device))
    sim = SimpleNamespace(
        pose=torch.zeros((n, 2, 3), dtype=torch.float32, device=device),
        field_length=651.22 * INCH,
        field_width=317.7 * INCH,
        field_colliders=[box.as_tensor() for box in static_collision_boxes(field)],
        obstacles=circles)
    env = SimpleNamespace(device=device, n=n, sim=sim, field_boxes=field)
    planner = TensorADStar(env, resolution=.3, avoid_bumps=avoid_bumps)
    # Force the reference call to exercise the fallback even on HIP by default.
    planner._fused_blocked_hip_enabled = False
    return planner


def _edge_inputs(planner, n, device, seed, with_dynamic):
    generator = torch.Generator(device=device).manual_seed(seed)
    heading = torch.rand((n,), generator=generator, device=device) * (2 * math.pi) - math.pi
    length = torch.rand((n,), generator=generator, device=device) * .45 + .65
    width = torch.rand((n,), generator=generator, device=device) * .35 + .55
    dynamic = None
    if with_dynamic:
        dynamic = torch.zeros((n, 4), dtype=torch.float32, device=device)
        dynamic[:, 0] = torch.rand((n,), generator=generator, device=device) * planner.length
        dynamic[:, 1] = torch.rand((n,), generator=generator, device=device) * planner.width
        dynamic[:, 2:] = .45
    if n >= 8:
        target_x, target_y = planner.x[12].item(), planner.y[12].item()
        planner._boxes = torch.cat((planner._boxes, torch.tensor(
            [[target_x + .60, target_y + .50, .15, .10]], device=device)), 0)
        planner._bumps = torch.tensor([[target_x + .55, target_y, .10, .10]],
                                      dtype=torch.float32, device=device)
        if planner._extra_circles.numel():
            planner._extra_circles = torch.cat((planner._extra_circles,
                torch.tensor([[target_x + .65, target_y, .20]], device=device)), 0)
        heading[:2] = 0.
        length[:2] = torch.tensor([.3, .30000007], device=device)
        width[:2] = .3
        heading[2] = 0.; length[2] = .9; width[2] = 1.0
        heading[6] = 0.; length[6] = .9; width[6] = .8
        if with_dynamic:
            heading[4] = 0.; length[4] = .9; width[4] = .9
            dynamic[4] = torch.tensor([target_x + .9, target_y, .45, .35], device=device)
    return heading, length, width, dynamic


@pytest.mark.parametrize("avoid_bumps", [False, True])
@pytest.mark.parametrize("with_circles", [False, True])
@pytest.mark.parametrize("with_dynamic", [False, True])
def test_fused_blocked_grid_matches_torch_exactly(
        avoid_bumps, with_circles, with_dynamic):
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("fused AD* occupancy requires a HIP device")
    from frc_defense.tensor_adstar import _hip_occupancy_extension

    extension = _hip_occupancy_extension()
    if extension is None:
        pytest.skip("AD* occupancy HIP extension is unavailable")
    device = torch.device("cuda:0")
    n = 32
    planner = _planner(n, device, avoid_bumps=avoid_bumps,
                       with_circles=with_circles)
    heading, length, width, dynamic = _edge_inputs(
        planner, n, device, 8831, with_dynamic)
    expected = planner._blocked(heading, length, width, dynamic)
    planner._fused_blocked_hip_enabled = True
    actual = planner._blocked(heading, length, width, dynamic)
    torch.cuda.synchronize(device)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("lane_y", [.8, 7.27])
def test_trench_route_plans_with_travel_aligned_footprint(lane_y):
    planner = _planner(1, torch.device("cpu"), avoid_bumps=True,
                       with_circles=False)
    trench_x = planner._trench_xs[0].item()
    start = torch.tensor([[3.0, lane_y]])
    goal = torch.tensor([[6.0, lane_y]])
    planner.plan(start, goal, torch.tensor([0.]),
                 torch.tensor([1.1]), torch.tensor([.7]),
                 speed=torch.tensor([4.5]))

    route = planner.last_path[0, :planner.last_lengths[0]]
    assert planner.last_trench_alignment[0]
    assert planner.last_heading[0].item() == pytest.approx(0.)
    assert route.shape[0] > 1
    assert route[:, 0].min() < trench_x < route[:, 0].max()


@pytest.mark.parametrize("lane_y", [.8, 7.27])
def test_trench_lane_alignment_enables_diagonal_chassis(lane_y):
    planner = _planner(1, torch.device("cpu"), avoid_bumps=True,
                       with_circles=False)
    planner.plan(torch.tensor([[3.0, lane_y]]),
                 torch.tensor([[6.0, lane_y]]),
                 torch.tensor([math.pi / 4]), torch.tensor([1.1]),
                 torch.tensor([.7]), speed=torch.tensor([4.5]))

    assert planner.last_trench_alignment[0]
    assert planner.last_heading[0].item() == pytest.approx(0.)


@pytest.mark.parametrize("lane_y", [1.0, 7.0])
def test_trench_alignment_rejects_centers_overlapping_support(lane_y):
    planner = _planner(1, torch.device("cpu"), avoid_bumps=True,
                       with_circles=False)
    planner.plan(torch.tensor([[3.0, lane_y]]),
                 torch.tensor([[6.0, lane_y]]), torch.tensor([0.]),
                 torch.tensor([1.1]), torch.tensor([.7]),
                 speed=torch.tensor([4.5]))

    assert not planner.last_trench_alignment[0]


def test_fused_occupancy_preserves_full_plan_and_command_outputs():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("fused AD* occupancy requires a HIP device")
    from frc_defense.tensor_adstar import _hip_occupancy_extension

    if _hip_occupancy_extension() is None:
        pytest.skip("AD* occupancy HIP extension is unavailable")
    device = torch.device("cuda:0")
    n = 128
    torch_planner = _planner(n, device, avoid_bumps=True, with_circles=True)
    hip_planner = _planner(n, device, avoid_bumps=True, with_circles=True)
    hip_planner._fused_blocked_hip_enabled = True
    generator = torch.Generator(device=device).manual_seed(10471)
    start = torch.rand((n, 2), generator=generator, device=device)
    start[:, 0] = start[:, 0] * 1.2 + .8
    start[:, 1] = start[:, 1] * 1.2 + .8
    goal = torch.stack((torch.full((n,), torch_planner.length, device=device) - start[:, 0],
                        torch.full((n,), torch_planner.width, device=device) - start[:, 1]), -1)
    heading = torch.rand((n,), generator=generator, device=device) * (2 * math.pi) - math.pi
    robot_length = torch.full((n,), .9, device=device)
    robot_width = torch.full((n,), .75, device=device)
    speed = torch.full((n,), 4.5, device=device)
    defender = torch.rand((n, 2), generator=generator, device=device)
    defender[:, 0] *= torch_planner.length
    defender[:, 1] *= torch_planner.width
    defender_velocity = torch.rand((n, 2), generator=generator, device=device) * 2 - 1
    friction = torch.full((n,), 1.2, device=device)
    acceleration = torch.full((n,), 8., device=device)
    args = (start, goal, heading, robot_length, robot_width, speed,
            defender, defender_velocity, True, friction, acceleration)
    torch_planner.plan(*args)
    hip_planner.plan(*args)
    for field in ("potential", "last_path", "last_lengths", "last_intercept",
                  "last_intercept_time", "last_start", "last_goal", "last_heading",
                  "last_speed_profile"):
        assert torch.equal(getattr(torch_planner, field), getattr(hip_planner, field)), field
    position = start + torch.tensor([.2, -.1], device=device)
    velocity = torch.rand((n, 2), generator=generator, device=device) * 3 - 1.5
    torch_command, torch_tangent = torch_planner.path_reference(position, velocity, speed)
    hip_command, hip_tangent = hip_planner.path_reference(position, velocity, speed)
    assert torch.equal(hip_command, torch_command)
    assert torch.equal(hip_tangent, torch_tangent)


def test_trajectory_validation_catches_rotation_that_start_heading_misses():
    planner = _planner(1, torch.device("cpu"), avoid_bumps=True,
                       with_circles=False)
    planner._boxes = torch.tensor([[5.0, 5.0, .045, .019]])
    path = torch.tensor([[[5.0, 5.6], [5.001, 5.601]]])
    heading = torch.tensor([0.])
    length = width = torch.tensor([.9])

    assert not planner._footprint_path_clear(path, heading, length, width).item()

    clear_path = torch.tensor([[[5.0, 5.75], [5.001, 5.751]]])
    assert planner._footprint_path_clear(clear_path, heading, length, width).item()


@pytest.mark.parametrize(("start", "goal"), [
    ((2.0, 3.0), (.8, 3.6)),
    ((2.0, 5.0), (.8, 4.5)),
])
def test_routes_around_tower_uprights_validate_the_rotated_chassis(start, goal):
    planner = _planner(1, torch.device("cpu"), avoid_bumps=True,
                       with_circles=False)
    heading = torch.tensor([0.])
    size = torch.tensor([.9])
    planner.plan(torch.tensor([start]), torch.tensor([goal]), heading, size, size,
                 speed=torch.tensor([4.8]))

    route = planner.last_path
    assert (route[0] - route[0, :1]).norm(dim=-1).max() > .2
    assert planner._footprint_path_clear(route, heading, size, size).all()
