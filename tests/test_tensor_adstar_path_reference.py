import pytest

torch = pytest.importorskip("torch")


def _torch_path_reference(path, lengths, goal, profile, position, velocity, speed_limit):
    n = path.shape[0]
    d2 = (path - position[:, None, :]).square().sum(-1)
    index = d2.argmin(-1)
    next_index = (index + 1).clamp_max(71)
    rows = torch.arange(n, device=path.device)
    here = path[rows, index]
    ahead = path[rows, next_index]
    tangent = ahead - here
    tangent = torch.where((lengths <= 1)[:, None], goal - position, tangent)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    normal = torch.stack((-tangent[:, 1], tangent[:, 0]), -1)
    cross = ((position - here) * normal).sum(-1)
    cross_velocity = (velocity * normal).sum(-1)
    lateral = (-3.0 * cross - 4.0 * cross_velocity).clamp(-.75, .75)
    planned = profile[rows, index].clamp(max=speed_limit).clamp_min(.2)
    current = velocity.norm(dim=-1)
    target = (planned - .55 * (current - planned).clamp_min(0)).clamp_min(0)
    command = tangent * target[:, None] + normal * lateral[:, None]
    return command, tangent


def test_fused_path_reference_matches_torch_with_production_strides():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("fused path reference requires a HIP device")

    from frc_defense.tensor_adstar import _hip_path_reference_extension

    extension = _hip_path_reference_extension()
    if extension is None:
        pytest.skip("AD* HIP extension is unavailable")

    device = torch.device("cuda")
    n = 512
    generator = torch.Generator(device=device).manual_seed(4219)
    path = torch.rand((n, 72, 2), device=device, generator=generator)
    lengths = torch.randint(0, 73, (n,), device=device, generator=generator)
    # Include the no-route and one-point cases, plus routes of several lengths.
    lengths[:8] = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)
    valid_points = torch.arange(72, device=device)[None, :] < lengths[:, None]
    path = torch.where(valid_points[:, :, None], path, torch.zeros_like(path))

    state = torch.randn((n, 2, 3), device=device, generator=generator)
    position = state[:, 0, :2]
    velocity = state[:, 1, :2]
    speed_storage = torch.rand((n, 2), device=device, generator=generator) * 4. + .5
    speed_limit = speed_storage[:, 1]
    goal = torch.rand((n, 2), device=device, generator=generator)
    profile = torch.rand((n, 72), device=device, generator=generator) * 5.

    assert position.stride() == (6, 1)
    assert velocity.stride() == (6, 1)
    assert speed_limit.stride() == (2,)
    expected = _torch_path_reference(path, lengths, goal, profile,
                                     position, velocity, speed_limit)
    actual = extension.path_reference(path, lengths, goal, profile,
                                      position, velocity, speed_limit)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual[0], expected[0], rtol=0., atol=2.0e-6)
    torch.testing.assert_close(actual[1], expected[1], rtol=0., atol=2.0e-6)
