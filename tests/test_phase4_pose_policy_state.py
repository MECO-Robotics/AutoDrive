"""No-leak contract for physical pose versus policy localization."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from frc_defense.experimental_route_evaluation import RouteQuery, RouteResult, evaluate_routes
from frc_defense.learning_inputs import build_observation_snapshot
from frc_defense import meco_candidates
from frc_defense.meco_candidates import (_candidate_target,
                                         build_route_feasible_candidates,
                                         build_safe_action_candidates)
from frc_defense.meco_plan_a_policy import build_plan_a_support
from frc_defense.curriculum_scenarios import make_curriculum_env
from frc_defense.architecture_inputs import PerceptionInputs


def _env():
    env = make_curriculum_env("pickup_simple", "single", 1, "cpu", 71,
                              horizon=10, fuel_physics=False)
    env.set_focal_pose_estimate((10., 4., 0.))
    # Freeze the sensor state to isolate the pose authority boundary.
    env.track_pos.zero_()
    env.track_vel.zero_()
    env.track_mask.zero_()
    env.track_age.fill_(float("inf"))
    env.track_confidence.zero_()
    env.track_pos[0, 3, :3] = torch.tensor(((10.6, 4.), (10.8, 4.2), (11., 4.)))
    env.track_vel[0, 3, :3].zero_()
    env.track_mask[0, 3, :3] = True
    env.track_age[0, 3, :3] = 0.
    env.track_confidence[0, 3, :3] = 1.
    return env


def _tiny_graph(device):
    # Small connected legal-pose fixture keeps the contract focused on its
    # rooted support logic without constructing the production-sized lattice.
    xy = torch.tensor(((9.8, 4.), (10.2, 4.), (10.6, 4.)), device=device)
    heading = torch.zeros(3, device=device)
    return SimpleNamespace(
        xy=xy, heading=heading,
        heading_bin=torch.zeros(3, device=device, dtype=torch.long),
        edges=torch.tensor([[[1, 0], [0, 2], [1, 0]]], device=device),
        edge_mask=torch.tensor([[[True, False], [True, True], [True, False]]],
                               device=device),
        resolution_m=.4,
        node_count=3)


def _decision_snapshot(env, graph):
    observation = env.observe()
    candidates, valid, _ = env._candidates()
    action_mask, _ = env._action_mask(candidates, valid)
    actions = torch.full((1, 6), 7, device=env.device, dtype=torch.long)
    actions[0, 3] = 0
    controller_target = env._target_for_actions(
        actions, torch.ones((1,), device=env.device, dtype=torch.bool))[0]
    safe = build_safe_action_candidates(
        env, observation, episode_id="pose-contract")
    support = build_plan_a_support(env, graph, observation,
                                   snapshot_id="pose-contract:0")
    target, heading, allowed, intake = _candidate_target(
        env, 0, safe, target_row=safe.table.features[0])
    route = evaluate_routes(env, (RouteQuery(
        candidate_id="route", snapshot_id="pose-contract:0", world=0,
        slot=3, goal_xy=(10.8, 4.)),))[0]
    snapshot = build_observation_snapshot(env, episode_id="pose-contract")
    perception = PerceptionInputs(env, (3,))(observation)
    return {
        "observation": observation.clone(),
        "candidate_indices": candidates.clone(),
        "candidate_valid": valid.clone(),
        "action_mask": action_mask.clone(),
        "controller_target": controller_target.clone(),
        "safe_table": (safe.table.candidate_ids, safe.table.features,
                       safe.table.valid_mask),
        "plan_a_root": support.root_node,
        "plan_a_mask": support.valid_mask.clone(),
        "plan_a_digest": support.support_digest,
        "target": target.clone(),
        "heading": heading,
        "allowed": allowed,
        "intake": intake,
        "route": (route.status, route.feasible, route.distance_m,
                  route.path_points_m),
        "obstacles": env._perceived_robot_obstacles().clone(),
        "route_start": env.policy_pose[0, 3].clone(),
        "meco_globals": snapshot.globals,
        "meco_raster": snapshot.raster,
        "perception_field": perception["field"].clone(),
        "perception_entities": perception["entities"].clone(),
    }


def test_policy_inputs_are_invariant_to_physical_pose_when_estimate_and_sensors_are_fixed():
    env = _env()
    graph = _tiny_graph(env.device)
    estimated = env.policy_pose.clone()
    truth = env.sim.pose.clone()

    env.sim.pose[0, 3] = torch.tensor((2., 1., 2.))
    first = _decision_snapshot(env, graph)
    env.sim.pose[0, 3] = torch.tensor((14., 7., -1.))
    second = _decision_snapshot(env, graph)

    assert torch.equal(env.policy_pose, estimated)
    assert torch.equal(env.sim.pose[0, 3], torch.tensor((14., 7., -1.)))
    assert torch.equal(first["observation"], second["observation"])
    assert torch.equal(first["candidate_indices"], second["candidate_indices"])
    assert torch.equal(first["candidate_valid"], second["candidate_valid"])
    assert torch.equal(first["action_mask"], second["action_mask"])
    assert torch.equal(first["controller_target"], second["controller_target"])
    assert first["safe_table"] == second["safe_table"]
    assert first["plan_a_root"] == second["plan_a_root"]
    assert torch.equal(first["plan_a_mask"], second["plan_a_mask"])
    assert first["plan_a_digest"] == second["plan_a_digest"]
    assert torch.equal(first["target"], second["target"])
    assert first["heading"] == second["heading"]
    assert first["allowed"] == second["allowed"]
    assert first["intake"] == second["intake"]
    assert first["route"] == second["route"]
    assert torch.equal(first["obstacles"], second["obstacles"])
    assert torch.equal(first["route_start"], second["route_start"])
    assert first["meco_globals"] == second["meco_globals"]
    assert first["meco_raster"] == second["meco_raster"]
    assert torch.equal(first["perception_field"], second["perception_field"])
    assert torch.equal(first["perception_entities"], second["perception_entities"])

    # Truth remains available for evaluator/physics output and did change.
    assert not torch.equal(truth[0, 3], env.sim.pose[0, 3])


def test_changing_estimated_pose_changes_policy_geometry_without_moving_truth():
    env = _env()
    graph = _tiny_graph(env.device)
    physical = env.sim.pose.clone()
    before = _decision_snapshot(env, graph)

    env.set_focal_pose_estimate((10.5, 4.5, 0.))
    after = _decision_snapshot(env, graph)

    assert torch.equal(env.sim.pose, physical)
    assert not torch.equal(before["observation"], after["observation"])
    assert not torch.equal(before["controller_target"], after["controller_target"])
    assert before["safe_table"] != after["safe_table"]
    assert not torch.equal(before["perception_field"], after["perception_field"])
    assert not torch.equal(before["perception_entities"], after["perception_entities"])
    assert before["plan_a_root"] != after["plan_a_root"]
    assert not torch.equal(before["target"], after["target"])
    assert not torch.equal(before["route_start"], after["route_start"])


def test_plan_c_eta_uses_estimated_heading_not_physical_heading(monkeypatch):
    env = _env()
    graph = _tiny_graph(env.device)

    def goal_node(_env, _graph, xy, heading, *, intake_extended=False):
        del _env, _graph, intake_extended
        return 0, xy.clone(), int(round((heading % (2 * torch.pi)) * 8 /
                                        (2 * torch.pi))) % 8

    def fixed_routes(_env, queries, **kwargs):
        del _env, kwargs
        return tuple(RouteResult(query.candidate_id, query.snapshot_id,
            query.world, query.slot, "feasible", True, 1.,
            path_points_m=((10., 4.), query.goal_xy)) for query in queries)

    monkeypatch.setattr(meco_candidates, "_goal_node", goal_node)
    monkeypatch.setattr(meco_candidates, "evaluate_routes", fixed_routes)
    env.set_focal_pose_estimate((10., 4., 0.))
    observation = env.observe()
    first = build_route_feasible_candidates(
        env, observation, graph, episode_id="pose-contract", candidate_k=8)
    first_eta = first.table.features[0][13]

    # Alter only physical heading: the Plan C estimate must remain identical.
    env.sim.pose[0, 3, 2] = 1.2
    second = build_route_feasible_candidates(
        env, env.observe(), graph, episode_id="pose-contract", candidate_k=8)
    assert second.table.features == first.table.features
    assert second.table.valid_mask == first.table.valid_mask

    # Changing the estimate changes turn-time ETA while physical state stays fixed.
    truth = env.sim.pose.clone()
    env.set_focal_pose_estimate((10., 4., torch.pi / 2))
    third = build_route_feasible_candidates(
        env, env.observe(), graph, episode_id="pose-contract", candidate_k=8)
    assert torch.equal(env.sim.pose, truth)
    assert third.table.features[0][13] > first_eta


def test_perception_inputs_follow_restored_delayed_sensor_and_pose_frame():
    env = _env()
    builder = PerceptionInputs(env, (3,))
    env.policy_pose[0, 3] = torch.tensor((9.5, 3.5, 0.))
    env.policy_pose[0, 4] = torch.tensor((8.5, 3.5, 0.))
    env.policy_pose[0, 5] = torch.tensor((8.5, 3.5, 0.))
    env.track_pos[0, 3, 0] = torch.tensor((10.25, 3.5))
    env._last_obs = env.observe()
    delayed = env._phase4_capture_sensor_frame(env._last_obs)

    env.policy_pose[0, 3] = torch.tensor((12.5, 5.5, 1.))
    env.policy_pose[0, 4] = torch.tensor((13.5, 5.5, 1.))
    env.policy_pose[0, 5] = torch.tensor((13.5, 5.5, 1.))
    env.track_pos[0, 3, 0] = torch.tensor((12.75, 5.5))
    env._last_obs = env.observe()
    env._phase4_restore_sensor_frame(delayed)
    inputs = builder(env._last_obs)
    torch.testing.assert_close(inputs["scalars"], env._last_obs[:, 3])

    # Entities are fuel slots, focal robot, two teammates, opponent. Locations
    # are relative to the focal estimate and the retained fuel track frame.
    entities = inputs["entities"][0]
    fuel_count = entities.shape[0] - 4
    torch.testing.assert_close(entities[0, :2], torch.tensor((
        .75 / float(builder.scales[0]), 0.)))
    torch.testing.assert_close(entities[fuel_count, :2], torch.zeros(2))
    expected_ally_delta = torch.tensor((-1. / float(builder.scales[0]), 0.))
    torch.testing.assert_close(entities[fuel_count + 1, :2], expected_ally_delta)
    torch.testing.assert_close(entities[fuel_count + 2, :2], expected_ally_delta)
