from dataclasses import replace

import pytest
import torch

from frc_defense.experimental_route_evaluation import (
    RouteQuery, _scratch_planner, evaluate_routes,
)
from frc_defense.tensor_adstar import TensorADStar
from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


@pytest.fixture
def env():
    torch.set_num_threads(1)
    return TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", control_modes=("none",) * 6,
        randomize=False, fuel_count=96, max_fuel_capacity=8,
        perception_dropout=0.0)


def _live_state(env):
    planner = env.planner
    names = ("potential", "last_path", "last_lengths", "last_intercept",
             "last_intercept_time", "last_start", "last_goal", "last_heading",
             "last_progress", "last_trench_alignment", "last_speed_profile")
    return {name: getattr(planner, name).clone() for name in names}


def test_route_queries_do_not_mutate_live_planner_or_environment(env):
    pose = env.sim.pose[0, 3].clone()
    before_planner = _live_state(env)
    before_pose = env.sim.pose.clone()
    queries = [
        RouteQuery("move", "episode-1:0", 0, 3,
                   (float(pose[0]) + 1.0, float(pose[1]))),
        RouteQuery("stay", "episode-1:0", 0, 3,
                   (float(pose[0]), float(pose[1]))),
    ]

    result = evaluate_routes(env, queries)

    assert [row.candidate_id for row in result] == ["move", "stay"]
    assert result[0].status == "feasible" and result[0].feasible
    assert result[0].distance_m == pytest.approx(1.0, abs=0.15)
    assert result[1].status == "at_goal" and result[1].distance_m == 0.0
    assert torch.equal(env.sim.pose, before_pose)
    assert all(torch.equal(getattr(env.planner, name), value)
               for name, value in before_planner.items())


def test_result_identity_and_order_are_independent_of_batch_order(env):
    pose = env.sim.pose[0, 3]
    rows = [
        RouteQuery("far", "episode-1:0", 0, 3,
                   (float(pose[0]) + 1.0, float(pose[1]))),
        RouteQuery("stay", "episode-1:0", 0, 3,
                   (float(pose[0]), float(pose[1]))),
        RouteQuery("bad", "episode-1:0", 0, 3,
                   (float("nan"), float(pose[1]))),
    ]
    first = evaluate_routes(env, rows)
    second = evaluate_routes(env, [rows[2], rows[0], rows[1]])
    assert [row.candidate_id for row in second] == ["bad", "far", "stay"]
    assert {row.candidate_id: row for row in first} == {
        "bad": second[0], "far": second[1], "stay": second[2]}
    assert first[0].status == "feasible"
    assert first[1].status == "at_goal"
    assert first[2].status == "invalid"


def test_invalid_and_one_point_routes_are_not_reported_as_feasible(env):
    pose = env.sim.pose[0, 3]
    x, y = float(pose[0]), float(pose[1])
    result = evaluate_routes(env, [
        RouteQuery("outside", "episode-1:0", 0, 3,
                   (float(env.sim.field_length) + 1.0, y)),
    ])[0]
    assert result.status == "invalid" and not result.feasible
    assert result.distance_m is None


def test_one_point_planner_output_is_no_path(env, monkeypatch):
    pose = env.sim.pose[0, 3]
    query = RouteQuery("one-point", "episode-1:0", 0, 3,
                       (float(pose[0]) + 1.0, float(pose[1])))

    def one_point(self, start, goal, heading, length, width, **kwargs):
        self.last_path.zero_()
        self.last_path[:, 0] = start
        self.last_lengths.zero_()
        self.last_lengths[:] = 1
        return self.last_path, self.last_lengths, self.last_intercept, self.last_intercept_time

    monkeypatch.setattr(TensorADStar, "plan", one_point)
    result = evaluate_routes(env, [query])[0]
    assert result.status == "no_path"
    assert not result.feasible
    assert result.distance_m is None


def test_batch_requires_one_snapshot_and_valid_intake_flag(env):
    pose = env.sim.pose[0, 3]
    query = RouteQuery("same", "episode-1:0", 0, 3,
                       (float(pose[0]), float(pose[1])))
    with pytest.raises(ValueError, match="single snapshot_id"):
        evaluate_routes(env, [query, replace(query, candidate_id="other",
                                              snapshot_id="episode-1:1")])
    with pytest.raises(ValueError, match="exceeds 64"):
        evaluate_routes(env, [replace(query, candidate_id=str(i)) for i in range(65)])
    assert evaluate_routes(env, [replace(query, intake_extended=True)])[0].status in {
        "at_goal", "no_path"}


def test_intake_extension_matches_controller_planner_footprint(env, monkeypatch):
    pose = env.sim.pose[0, 3]
    x, y = float(pose[0]), float(pose[1])
    seen_lengths = []
    plan = TensorADStar.plan

    def capture(self, start, goal, heading, length, width, **kwargs):
        seen_lengths.append(length.clone())
        return plan(self, start, goal, heading, length, width, **kwargs)

    monkeypatch.setattr(TensorADStar, "plan", capture)
    queries = [
        RouteQuery("body", "episode-1:0", 0, 3, (x + 1.0, y)),
        RouteQuery("intake", "episode-1:0", 0, 3, (x + 1.0, y),
                   intake_extended=True),
    ]
    evaluate_routes(env, queries)

    assert len(seen_lengths) == 1
    assert seen_lengths[0].tolist() == pytest.approx([
        env.sim.length[0, 3].item(), env.sim.length[0, 3].item() + 0.6096])


def test_route_endpoint_must_reach_the_requested_goal(env, monkeypatch):
    pose = env.sim.pose[0, 3]
    query = RouteQuery("endpoint", "episode-1:0", 0, 3,
                       (float(pose[0]) + 1.0, float(pose[1])))
    plan = TensorADStar.plan

    def move_reported_endpoint(self, start, goal, heading, length, width, **kwargs):
        result = plan(self, start, goal, heading, length, width, **kwargs)
        self.last_path[0, self.last_lengths[0] - 1, 0] += 0.2
        return result

    monkeypatch.setattr(TensorADStar, "plan", move_reported_endpoint)
    result = evaluate_routes(env, [query], endpoint_tolerance_m=0.05)[0]
    assert result.status == "no_path"
    assert result.distance_m is None


def test_at_goal_rejects_exact_static_overlap_even_when_raster_cell_is_clear(env):
    env.sim.pose[0, 3, 0] = 13.19
    env.sim.field_colliders[0] = torch.tensor([13.63, 1.35, .02, .02])
    goal = (13.19, 1.35)
    query = RouteQuery("static-overlap", "episode-1:0", 0, 3, goal)
    scratch = _scratch_planner(env, [(query, torch.tensor(goal))])
    heading = env.sim.pose[0, 3, 2:3]
    length, width = env.sim.length[0, 3:4], env.sim.width[0, 3:4]
    blocked = scratch._blocked(heading, length, width)
    gx, gy = scratch._indices(torch.tensor([goal]))
    assert not blocked[0, gx[0], gy[0]]

    result = evaluate_routes(env, [query])[0]

    assert result.status == "no_path" and not result.feasible
    assert result.distance_m is None


def test_at_goal_rejects_exact_observed_robot_circle_overlap(env):
    env.sim.pose[0, 3, 0] = 13.19
    goal = (13.19, 1.35)
    env.opponent_pose[0, 3, :2] = torch.tensor([13.79, 1.35])
    env.opponent_size[0, 3, :2] = torch.tensor([.4, .4])
    env.opponent_valid[0, 3] = True
    query = RouteQuery("dynamic-overlap", "episode-1:0", 0, 3, goal)
    scratch = _scratch_planner(env, [(query, torch.tensor(goal))])
    heading = env.sim.pose[0, 3, 2:3]
    length, width = env.sim.length[0, 3:4], env.sim.width[0, 3:4]
    blocked = scratch._blocked(heading, length, width)
    blocked = scratch._block_robot_obstacles(
        blocked, length, width, env._perceived_robot_obstacles()[3:4])
    gx, gy = scratch._indices(torch.tensor([goal]))
    assert not blocked[0, gx[0], gy[0]]

    result = evaluate_routes(env, [query])[0]

    assert result.status == "no_path" and not result.feasible
    assert result.distance_m is None


def test_blocked_raster_start_can_escape_through_live_planner(env):
    env.sim.pose[0, 3, 0] = 13.19
    env.sim.field_colliders[0] = torch.tensor([12.70, 1.35, .02, .02])
    start = env.sim.pose[0, 3]
    goal = (14.3, float(start[1]))
    query = RouteQuery("escape", "episode-1:0", 0, 3, goal)
    scratch = _scratch_planner(env, [(query, torch.tensor(goal))])
    heading = start[2:3]
    length, width = env.sim.length[0, 3:4], env.sim.width[0, 3:4]
    blocked = scratch._blocked(heading, length, width)
    sx, sy = scratch._indices(start[None, :2])
    assert blocked[0, sx[0], sy[0]]

    result = evaluate_routes(env, [query], endpoint_tolerance_m=.1)[0]

    assert result.status == "feasible" and result.feasible
    assert result.distance_m is not None and result.distance_m > 0
