"""Six-robot, parameter-shared strategic environment for REBUILT 3v3 play.

The established 137-input/8-action ActorCritic is shared across robot slots.
Each row is a robot-centric observation; the reserved second local-track block
now carries the two nearest teammates.  The simulator returns one observation
and reward per robot while match termination and physics remain per world.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from .tensor_sim import TensorVectorizedSimulator, swerve_heading_rate
from .field import ALLIANCE_ZONE_DEPTH, midfield_respawn_points
from ._tensor_3v3_actions import TensorThreeVsThreeActionMixin
from ._tensor_3v3_constants import TEAM_IDS
from ._tensor_3v3_gamepieces import TensorThreeVsThreeGamepieceMixin
from ._tensor_3v3_navigation import TensorThreeVsThreeNavigationMixin
from ._tensor_3v3_observation import (
    ACTION_DIM, NUM_ROBOTS, OBS_DIM, TensorThreeVsThreeObservationMixin,
)


CONTROL_MODES = ("none", "nn", "deterministic")


class TensorThreeVsThreeEnv(
    TensorThreeVsThreeActionMixin, TensorThreeVsThreeObservationMixin,
    TensorThreeVsThreeGamepieceMixin, TensorThreeVsThreeNavigationMixin,
):
    """A batched 3v3 environment with six per-robot controller assignments.

    ``step`` advances one 20 ms physics tick. The caller holds categorical
    strategic actions for the configured 4 Hz policy interval. Rewards and
    observations are agent-shaped ``[world, robot, ...]``; terminal masks are
    world-shaped so all six robots share one match boundary.
    """

    def __init__(self, num_envs=1, device="cuda", seed=0, *, control_modes=None,
                 robot_roles=None, robot_types=None,
                 dt=.02, horizon=8000, randomize=True, fuel_count=504,
                 preloaded_per_robot=0, max_fuel_capacity=60,
                 max_scoring_bps=25.0, turret_fuel_capacity=40,
                 turret_scoring_bps=15.0,
                 teammate_intent_knowledge=True,
                 perception_range=18., perception_fov=360.,
                 perception_dropout=.02, position_noise=.015,
                 velocity_noise=.05, replan_interval=20,
                 normalize_observations=True, fused_sensor_rng=False,
                 field_length=16.54,
                 field_width=8.07, obstacles=()):
        self.n = int(num_envs)
        if self.n < 1:
            raise ValueError("num_envs must be positive")
        if horizon < 1 or dt <= 0:
            raise ValueError("horizon and dt must be positive")
        if not 96 + NUM_ROBOTS * preloaded_per_robot <= fuel_count <= 600:
            raise ValueError("fuel_count must include staging and six robot preloads")
        if not 0 <= preloaded_per_robot <= 8:
            raise ValueError("preloaded_per_robot must be between 0 and 8")
        if not 1 <= max_fuel_capacity <= fuel_count:
            raise ValueError("max_fuel_capacity must be between 1 and fuel_count")
        if not math.isfinite(max_scoring_bps) or max_scoring_bps <= 0:
            raise ValueError("max_scoring_bps must be finite and positive")
        if not 1 <= turret_fuel_capacity <= fuel_count:
            raise ValueError("turret_fuel_capacity must be between 1 and fuel_count")
        if not math.isfinite(turret_scoring_bps) or turret_scoring_bps <= 0:
            raise ValueError("turret_scoring_bps must be finite and positive")
        self.device = torch.device(device)
        self.field_length = float(field_length)
        self.alliance_zone_depth = float(ALLIANCE_ZONE_DEPTH)
        self.dt, self.horizon = float(dt), int(horizon)
        self._world_indices=torch.arange(self.n,device=self.device)
        self._robot_owner_ids=torch.arange(NUM_ROBOTS,device=self.device,dtype=torch.long)
        self.obs_dim, self.action_dim, self.action_kind = OBS_DIM, ACTION_DIM, "categorical"
        self.fuel_count = int(fuel_count)
        self.preloaded_per_robot = int(preloaded_per_robot)
        self.fuel_capacity = int(max_fuel_capacity)
        self.max_scoring_bps = float(max_scoring_bps)
        self.perception_range = float(perception_range)
        self.perception_fov_degrees = float(perception_fov)
        self.perception_fov = math.radians(self.perception_fov_degrees) * .5
        self.perception_dropout = float(perception_dropout)
        self.position_noise, self.velocity_noise = float(position_noise), float(velocity_noise)
        self.replan_interval = max(1, int(replan_interval))
        self._planner_tick_scalar=self.replan_interval-1
        self._planner_tick_aligned=True
        self.normalize_observations = bool(normalize_observations)
        modes = list(control_modes or ("deterministic",) * NUM_ROBOTS)
        if len(modes) != NUM_ROBOTS or any(mode not in CONTROL_MODES for mode in modes):
            raise ValueError("control_modes must assign none, nn, or deterministic to six robots")
        self.control_modes = tuple(modes)
        self.teammate_intent_knowledge=bool(teammate_intent_knowledge)
        self._controlled_mode_mask=torch.tensor(
            [mode!="none" for mode in modes],device=self.device,dtype=torch.bool)
        roles = list(robot_roles or ("offense",) * 3 + ("defense",) * 3)
        if len(roles) != NUM_ROBOTS or any(role not in ("offense", "defense") for role in roles):
            raise ValueError("robot_roles must assign offense or defense to six robots")
        if any(mode == "nn" and role != "defense" for mode, role in zip(modes, roles)):
            raise ValueError("neural control is available only for defense-role robots")
        self.robot_roles = tuple(roles)
        types = list(robot_types or ("dumper",) * NUM_ROBOTS)
        if len(types) != NUM_ROBOTS or any(kind not in ("dumper", "turret") for kind in types):
            raise ValueError("robot_types must assign dumper or turret to six robots")
        self.robot_types = tuple(types)
        self.robot_fuel_capacity_values = tuple(
            int(turret_fuel_capacity if kind == "turret" else max_fuel_capacity)
            for kind in types)
        self.fuel_capacity = max(self.robot_fuel_capacity_values)
        self.max_scoring_bps = max(
            float(turret_scoring_bps if kind == "turret" else max_scoring_bps)
            for kind in types)
        self.robot_score_interval_values = tuple(
            1. / float(turret_scoring_bps if kind == "turret" else max_scoring_bps)
            for kind in types)
        self.robot_fuel_capacities=torch.tensor(
            self.robot_fuel_capacity_values,device=self.device,dtype=torch.long)
        self.robot_score_intervals=torch.tensor(
            self.robot_score_interval_values,device=self.device,dtype=torch.float32)
        self._dumper_mask=torch.tensor([kind=="dumper" for kind in types],
                                       device=self.device,dtype=torch.bool)
        self._turret_mask=~self._dumper_mask
        self._defense_role_mask = torch.tensor(
            [role == "defense" for role in roles], device=self.device, dtype=torch.bool)
        self.agent_train_mask = torch.tensor([mode == "nn" for mode in modes],
                                             device=self.device, dtype=torch.bool)
        self._nn_mode_mask=self.agent_train_mask.clone()
        self._deterministic_mode_mask=torch.tensor([mode=="deterministic" for mode in modes],
                                                   device=self.device,dtype=torch.bool)
        self._has_deterministic_offense=any(
            mode=="deterministic" and role=="offense"
            for mode,role in zip(modes,roles))
        self.team_ids = torch.tensor(TEAM_IDS, device=self.device, dtype=torch.long)
        self._team_robot_ids=tuple(torch.tensor(
            [robot for robot,team in enumerate(TEAM_IDS) if team==alliance],
            device=self.device,dtype=torch.long) for alliance in range(2))
        # Opponent slot membership never changes. Cache the six small index
        # tensors instead of calling CUDA nonzero for each robot every tick.
        self._enemy_robot_ids=tuple(torch.tensor(
            [i for i, team in enumerate(TEAM_IDS) if team != TEAM_IDS[robot]],
            device=self.device,dtype=torch.long) for robot in range(NUM_ROBOTS))
        self._peer_robot_ids=tuple(torch.tensor(
            [i for i in range(NUM_ROBOTS) if i != robot],
            device=self.device,dtype=torch.long) for robot in range(NUM_ROBOTS))
        self._valid_enemy_occluders=tuple(torch.tensor(
            [[peer != robot and enemy != peer for peer in range(NUM_ROBOTS)]
             for enemy in [i for i, team in enumerate(TEAM_IDS)
                           if team != TEAM_IDS[robot]]],
            device=self.device,dtype=torch.bool) for robot in range(NUM_ROBOTS))
        # Ferry shots launch from midfield, with a separate lane per robot to
        # keep the three carriers on each alliance from converging on one spot.
        midfield_x=float(field_length)/2
        ferry_x=midfield_x+torch.where(self.team_ids==0,-.45,.45)
        ferry_lane=torch.tensor((1.35,4.035,6.72,6.72,4.035,1.35),
                                device=self.device,dtype=torch.float32)
        self.ferry_targets=torch.stack((ferry_x,ferry_lane),-1)
        self.field_search_lanes=ferry_lane
        self._raster_waypoint_lookahead=20
        piece_ids=torch.arange(self.fuel_count,device=self.device)
        self._fuel_indices=piece_ids[None]
        self.pass_zone_positions=torch.zeros(
            (self.n,2,self.fuel_count,2),device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed or 0))
        self._perception_rng_seed = int(seed or 0) & 0xffffffff
        self._perception_rng_ticks = torch.zeros(
            self.n, device=self.device, dtype=torch.long)
        self.fused_sensor_rng = bool(fused_sensor_rng)
        self._fused_sensor_rng_used = False
        self._fused_opponent_tracks_used = False

        from .field import (box_observation_radius, bump_boxes, rebuilt_field,
                            static_collision_boxes)
        self.field_boxes = rebuilt_field()
        solids = static_collision_boxes(self.field_boxes)
        self.field_feature_boxes = solids + bump_boxes(self.field_boxes)
        colliders = [box.as_tensor() for box in solids]
        bumps = [box.as_tensor() for box in bump_boxes(self.field_boxes)]
        self.sim = TensorVectorizedSimulator(
            self.n, self.device, seed, dt=self.dt, num_robots=NUM_ROBOTS,
            team_ids=TEAM_IDS, field_length=field_length, field_width=field_width,
            randomize=randomize, obstacles=obstacles, field_colliders=colliders,
            bump_regions=bumps)
        self.raster_sweep_points=torch.tensor(
            self._build_raster_spline_points(),device=self.device,dtype=torch.float32)
        self._raster_cursor=torch.zeros((self.n,NUM_ROBOTS),device=self.device,
                                        dtype=torch.long)
        self._raster_reanchor=torch.ones((self.n,NUM_ROBOTS),device=self.device,
                                         dtype=torch.bool)
        self._avoidance_winner=torch.full(
            (self.n,self.sim._collision_pair_i.numel()),-1,
            device=self.device,dtype=torch.long)
        self._field_scales=torch.tensor((field_length,field_width),device=self.device)
        self._robot_occlusion_radius=.5*torch.sqrt(
            self.sim.length.square()+self.sim.width.square())
        self.randomize = bool(randomize)
        self.drivetrain_config={"name":"built-in illustrative defaults",
            "randomize":self.randomize,"mass":55.0,"robot_length":.9,"robot_width":.9,
            "max_speed":4.5,"max_acceleration":8.0,"max_omega":8.0,"max_alpha":18.0,
            "swerve":self.sim.swerve.__dict__.copy()}
        self.field_feature_obstacles = torch.tensor(
            [(b.x, b.y, box_observation_radius(b)) for b in self.field_feature_boxes],
            device=self.device, dtype=torch.float32).reshape(-1, 3)

        by_name = {b.name: b for b in self.field_boxes}
        self.hub_centers = torch.tensor(
            [(by_name[n].x, by_name[n].y) for n in ("red_hub", "blue_hub")],
            device=self.device, dtype=torch.float32)
        self.hub_active = torch.ones((self.n, 2), device=self.device, dtype=torch.bool)
        self.auto_fuel_scores = torch.zeros((self.n, 2), device=self.device, dtype=torch.long)
        self.fuel_score_count = torch.zeros_like(self.auto_fuel_scores)
        self.fuel_acquisition_count = torch.zeros((self.n, NUM_ROBOTS),
                                                  device=self.device, dtype=torch.long)
        self.fuel_denied_count = torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_acquired_event = torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_scored_event = torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_passed_event = torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_denied_event = torch.zeros_like(self.fuel_acquisition_count)
        self.piece_pos = torch.zeros((self.n, self.fuel_count, 2), device=self.device)
        self.piece_vel = torch.zeros_like(self.piece_pos)
        self.piece_active = torch.zeros((self.n, self.fuel_count), device=self.device,
                                        dtype=torch.bool)
        self.piece_owner = torch.full((self.n, self.fuel_count), -1,
                                      device=self.device, dtype=torch.long)
        self.piece_zone = torch.full_like(self.piece_owner, -1)
        self._pickup_grid_cell_size = 2.0
        self._pickup_grid_nx = math.ceil(float(field_length) / self._pickup_grid_cell_size)
        self._pickup_grid_ny = math.ceil(float(field_width) / self._pickup_grid_cell_size)
        self._pickup_possible_cells = torch.full(
            (self.n, self.fuel_count, 3), -1, device=self.device, dtype=torch.int16)
        self._midfield_respawn_positions = torch.tensor(
            midfield_respawn_points(self.fuel_count, field_length, field_width),
            device=self.device, dtype=self.piece_pos.dtype)
        self._midfield_respawn_cells=torch.zeros(
            (self.n,self.fuel_count,3),device=self.device,dtype=torch.int16)
        self.track_pos = torch.zeros((self.n, NUM_ROBOTS, self.fuel_count, 2), device=self.device)
        self.track_vel = torch.zeros_like(self.track_pos)
        self.track_mask = torch.zeros((self.n, NUM_ROBOTS, self.fuel_count),
                                      device=self.device, dtype=torch.bool)
        self.track_age = torch.full_like(self.track_mask, float("inf"), dtype=torch.float32)
        self.opponent_pose = torch.zeros((self.n, NUM_ROBOTS, 3), device=self.device)
        self.opponent_velocity = torch.zeros_like(self.opponent_pose)
        self.opponent_size = torch.zeros((self.n, NUM_ROBOTS, 3), device=self.device)
        self.opponent_valid = torch.zeros((self.n, NUM_ROBOTS), device=self.device,
                                          dtype=torch.bool)
        self.opponent_age = torch.full((self.n, NUM_ROBOTS), float("inf"), device=self.device)
        self.next_intake = torch.zeros((self.n, NUM_ROBOTS), device=self.device)
        self.next_score = torch.zeros_like(self.next_intake)
        self.next_ferry = torch.zeros_like(self.next_intake)
        self._ferry_committed = torch.zeros(
            (self.n,NUM_ROBOTS),device=self.device,dtype=torch.bool)
        self.last_hub_zone = torch.zeros_like(self.next_intake, dtype=torch.bool)
        self.last_actions = torch.full((self.n, NUM_ROBOTS), 7, device=self.device,
                                       dtype=torch.long)
        self.match_elapsed = torch.zeros(self.n, device=self.device)
        self.match_remaining = torch.full((self.n,), 160., device=self.device)
        self.steps = torch.zeros((self.n,), device=self.device, dtype=torch.long)
        self.planner_ticks = torch.zeros_like(self.steps)

        # AD* treats each robot slot as a batch row and shares the static field grid.
        from .tensor_adstar import TensorADStar
        proxy = SimpleNamespace(n=self.n * NUM_ROBOTS, device=self.device, sim=self.sim,
                                field_boxes=self.field_boxes)
        self.planner = TensorADStar(proxy, avoid_bumps=True)
        self._planner_controller_mode_key=tuple(
            i for i,mode in enumerate(modes) if mode!="none")
        self._planner_controller_modes_dirty=False
        self._route_segment_indices=torch.arange(
            self.planner.max_points-1,device=self.device)
        self._last_obs = torch.zeros((self.n, NUM_ROBOTS, OBS_DIM), device=self.device)
        self.reset(seed=seed)








    def reset(self, seed=None):
        if seed is not None:
            self.generator.manual_seed(int(seed))
            self._perception_rng_seed = int(seed) & 0xffffffff
        self._perception_rng_ticks.zero_()
        # Three legal, separated starting poses per alliance, facing toward midfield.
        starts = torch.tensor([
            [1.0, 1.35, 0.], [2.0, 4.03, 0.], [1.0, 6.70, 0.],
            [15.54, 1.35, math.pi], [14.54, 4.03, math.pi], [15.54, 6.70, math.pi],
        ], device=self.device)
        pose = starts[None].expand(self.n, -1, -1).clone()
        if self.randomize:
            pose[..., 0] += (self._random((self.n, NUM_ROBOTS)) - .5) * .5
            pose[..., 1] += (self._random((self.n, NUM_ROBOTS)) - .5) * .3
            pose[..., 2] += (self._random((self.n, NUM_ROBOTS)) - .5) * .2
        self.sim.reset(seed, pose=pose)
        self._raster_cursor.copy_(self._closest_raster_indices(self.sim.pose[:,:,:2]))
        self._raster_reanchor.fill_(True)
        self.match_elapsed.zero_(); self.match_remaining.fill_(160.)
        self.steps.zero_(); self.planner_ticks.fill_(self.replan_interval-1)
        self._planner_tick_scalar=self.replan_interval-1; self._planner_tick_aligned=True
        self.hub_active.fill_(True); self.auto_fuel_scores.zero_(); self.fuel_score_count.zero_()
        for tensor in (self.fuel_acquisition_count, self.fuel_denied_count,
                       self.fuel_acquired_event, self.fuel_scored_event,
                       self.fuel_passed_event, self.fuel_denied_event,
                       self.next_intake, self.next_score, self.next_ferry,
                       self._ferry_committed,
                       self._avoidance_winner,
                       self.last_hub_zone):
            tensor.zero_()
        self._avoidance_winner.fill_(-1)
        self._randomize_pass_zone_positions()
        self._initialize_fuel()
        self.track_mask.zero_(); self.track_age.fill_(float("inf"))
        self.opponent_valid.zero_(); self.opponent_age.fill_(float("inf"))
        self.opponent_size.zero_()
        self._update_perception(torch.ones(self.n, device=self.device, dtype=torch.bool))
        self.last_actions.fill_(7)
        self._last_obs = self.observe()
        return self._last_obs, {"field_layout": "2026_rebuilt", "robots": NUM_ROBOTS}

    def reset_done(self, mask):
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool).reshape(self.n)
        starts = self.sim.pose.clone()
        base = torch.tensor([
            [1., 1.35, 0.], [2., 4.03, 0.], [1., 6.70, 0.],
            [15.54, 1.35, math.pi], [14.54, 4.03, math.pi], [15.54, 6.70, math.pi],
        ], device=self.device)
        pose = base[None].expand(self.n, -1, -1).clone()
        if self.randomize:
            pose[..., 0] += (self._random((self.n, 6)) - .5) * .5
            pose[..., 1] += (self._random((self.n, 6)) - .5) * .3
        self.sim.reset_done(mask)
        # Keep the counter monotonic across matches so counter-based sensor
        # noise does not repeat from the same episode tick on every reset.
        self.sim.pose.copy_(torch.where(mask[:, None, None], pose, self.sim.pose))
        reset_cursor=self._closest_raster_indices(self.sim.pose[:,:,:2])
        self._raster_cursor.copy_(torch.where(mask[:,None],reset_cursor,self._raster_cursor))
        self._raster_reanchor.copy_(torch.where(
            mask[:,None],torch.ones_like(self._raster_reanchor),self._raster_reanchor))
        self.sim.velocity.copy_(torch.where(mask[:, None, None], torch.zeros_like(self.sim.velocity),
                                            self.sim.velocity))
        for tensor in (self.match_elapsed, self.steps):
            tensor.masked_fill_(mask, 0)
        self.planner_ticks.copy_(torch.where(mask,
            torch.full_like(self.planner_ticks,self.replan_interval-1),self.planner_ticks))
        if bool(mask.all()):
            self._planner_tick_scalar=self.replan_interval-1
            self._planner_tick_aligned=True
        else:
            self._planner_tick_aligned=False
        self.match_remaining.copy_(torch.where(mask, torch.full_like(self.match_remaining, 160.),
                                               self.match_remaining))
        for tensor in (self.auto_fuel_scores, self.fuel_score_count, self.fuel_acquisition_count,
                       self.fuel_denied_count, self.fuel_acquired_event, self.fuel_scored_event,
                       self.fuel_passed_event, self.fuel_denied_event,
                       self.next_intake, self.next_score, self.next_ferry,
                       self._ferry_committed,
                       self._avoidance_winner,
                       self.last_hub_zone):
            tensor[mask] = 0
        self._avoidance_winner[mask] = -1
        self.hub_active[mask] = True
        self._randomize_pass_zone_positions(mask)
        self._initialize_fuel(mask)
        self.track_mask[mask] = False; self.track_age[mask] = float("inf")
        self.opponent_valid[mask] = False; self.opponent_age[mask] = float("inf")
        self.opponent_size[mask] = 0.
        self._update_perception(mask)
        self.last_actions[mask] = 7
        self._last_obs = torch.where(mask[:, None, None], self.observe(), self._last_obs)
        return self._last_obs




    def step(self, actions, active_mask=None, *, capture_observation=True,
             capture_info=True, _active_count=None, _return_info=None):
        full_batch=active_mask is None
        active=(torch.ones(self.n,device=self.device,dtype=torch.bool) if full_batch else
                torch.as_tensor(active_mask,device=self.device,dtype=torch.bool).reshape(self.n))
        if _return_info is not None:
            capture_info=bool(_return_info)
        active_count=(self.n if full_batch else
                      int(_active_count) if _active_count is not None else
                      int(active.sum().item()))
        if active_count == 0:
            obs=self._last_obs
            return obs,torch.zeros((self.n,6),device=self.device),torch.zeros_like(active),torch.zeros_like(active),{}
        self._prepare_raster_cursor(active)
        targets,action,carrying,fuel_delta=self._target_for_actions(actions,active)
        raster_mask=((self._deterministic_mode_mask &
                      ~self._defense_role_mask)[None] &
                     ((action<4)|((action==6)&~carrying)))
        # AD* routes are consumed only by robots with an active controller.
        # Keep stationary robots' cached paths untouched; when a rollout
        # changes controller assignments, the trainer forces a fresh plan.
        planner_robot_mask=self._nn_mode_mask|self._deterministic_mode_mask
        planner_modes_dirty=self._planner_controller_modes_dirty
        self._planner_controller_modes_dirty=False
        # The strategic trainer supplies its horizon-derived active count as
        # a host integer. While it reports the full batch and planner cadence
        # remains aligned, every planner row has the same replan tick; avoid
        # reducing the GPU mask and synchronizing a device boolean each tick.
        full_active_batch=(full_batch or
                           (_active_count is not None and active_count==self.n))
        if full_active_batch and self._planner_tick_aligned:
            self._planner_tick_scalar+=1
            self.planner_ticks.add_(1)
            replan_due=((self._planner_tick_scalar%self.replan_interval)==0 or
                        planner_modes_dirty)
            if replan_due and len(self._planner_controller_mode_key)==NUM_ROBOTS:
                # Preserve the planner's no-compaction fast path when all six
                # robots are controlled.
                flat_active=None
            elif replan_due:
                flat_active=planner_robot_mask[None].expand(self.n,-1).reshape(-1)
            else:
                flat_active=None
        else:
            self._planner_tick_aligned=False
            self.planner_ticks.copy_(torch.where(active,self.planner_ticks+1,self.planner_ticks))
            replan=active&((self.planner_ticks%self.replan_interval)==0)
            if planner_modes_dirty:
                replan=active
            flat_active=(replan[:,None]&planner_robot_mask[None]).reshape(-1)
            replan_due=bool(flat_active.any())
        start=self.sim.pose[:,:,:2].reshape(-1,2); goal=targets.reshape(-1,2)
        heading=self.sim.pose[:,:,2].reshape(-1)
        lengths=self.sim.length.reshape(-1); widths=self.sim.width.reshape(-1)
        speeds=self.sim.speed.reshape(-1); accel=self.sim.accel.reshape(-1)
        lateral=self.sim.lateral_mu.reshape(-1)
        if replan_due:
            robot_pose=self.sim.pose
            robot_radii=.5*torch.sqrt(self.sim.length.square()+self.sim.width.square())
            robot_obstacles=torch.cat((
                robot_pose[:,None,:,:2].expand(-1,NUM_ROBOTS,-1,-1),
                robot_radii[:,None,:,None].expand(-1,NUM_ROBOTS,-1,1)),dim=-1)
            robot_obstacles=robot_obstacles.reshape(-1,NUM_ROBOTS,3)
            # Each route row sees every other robot as an obstacle. Zero radius
            # marks the row's own chassis so its starting cell stays traversable.
            own_radius=torch.eye(NUM_ROBOTS,device=self.device,dtype=torch.bool)
            own_radius=own_radius[None].expand(self.n,-1,-1).reshape(-1,NUM_ROBOTS)
            robot_obstacles[:,:,2].masked_fill_(own_radius,0.)
            waypoint_chains=self.raster_sweep_points[None].expand(
                self.n,-1,-1,-1).reshape(-1,self.raster_sweep_points.shape[1],2)
            self.planner.plan_waypoints(
                start,waypoint_chains,self._raster_cursor.reshape(-1),heading,lengths,widths,
                lookahead=self._raster_waypoint_lookahead,
                fallback_goal=goal,waypoint_mask=raster_mask.reshape(-1),
                speed=speeds,lateral_friction=lateral,acceleration=accel,
                active_mask=flat_active,robot_obstacles=robot_obstacles)
        hub=self.hub_centers[self.team_ids][None]
        hub_delta=hub-self.sim.pose[:,:,:2]
        hub_bearing=torch.atan2(hub_delta[...,1],hub_delta[...,0])
        angle_error=torch.atan2(torch.sin(hub_bearing-self.sim.pose[:,:,2]),
                                torch.cos(hub_bearing-self.sim.pose[:,:,2]))
        omega=swerve_heading_rate(angle_error,self.sim.omega_limit,self.sim.alpha)
        omega=torch.where((action==4)&carrying&self._dumper_mask[None],omega,
                          torch.zeros_like(omega))
        home_zone_center=hub.clone()
        home_zone_center[...,0]=torch.where(
            self.team_ids[None,:]==0,self.alliance_zone_depth*.5,
            self.field_length-self.alliance_zone_depth*.5)
        ferry_vector=home_zone_center-self.sim.pose[:,:,:2]
        ferry_bearing=torch.atan2(ferry_vector[...,1],ferry_vector[...,0])
        ferry_error=torch.atan2(torch.sin(ferry_bearing-self.sim.pose[:,:,2]),
                                torch.cos(ferry_bearing-self.sim.pose[:,:,2]))
        ferrying=(action==6)&carrying&self._dumper_mask[None]
        ferry_omega=swerve_heading_rate(ferry_error,self.sim.omega_limit,self.sim.alpha)
        omega=torch.where(ferrying,ferry_omega,omega)
        deterministic_robot=self._deterministic_mode_mask[None]
        deterministic_offense=(self._deterministic_mode_mask &
                                ~self._defense_role_mask)[None]
        pickup_active=(deterministic_offense &
                       ((action<4)|((action==6)&~carrying)))
        pickup_bearing=torch.atan2(fuel_delta[...,1],fuel_delta[...,0])
        pickup_error=torch.atan2(torch.sin(pickup_bearing-self.sim.pose[:,:,2]),
                                 torch.cos(pickup_bearing-self.sim.pose[:,:,2]))
        pickup_omega=swerve_heading_rate(pickup_error,self.sim.omega_limit,self.sim.alpha)
        omega=torch.where(pickup_active,pickup_omega,omega)
        trench_alignment=self.planner.last_trench_alignment.reshape(self.n,6)
        trench_heading=self.planner.last_heading.reshape(self.n,6)
        trench_error=torch.atan2(torch.sin(trench_heading-self.sim.pose[:,:,2]),
                                 torch.cos(trench_heading-self.sim.pose[:,:,2]))
        trench_omega=swerve_heading_rate(trench_error,self.sim.omega_limit,self.sim.alpha)
        omega=torch.where(trench_alignment,trench_omega,omega)
        available_speed=(self.sim.speed-omega.abs()*self.sim._module_radius).clamp_min(0.)
        command,_=self.planner.path_reference(
            start,self.sim.velocity[:,:,:2].reshape(-1,2),available_speed.reshape(-1))
        command=command.reshape(self.n,6,2)
        command=torch.where((trench_alignment & (trench_error.abs()>.25))[...,None],
                            torch.zeros_like(command),command)
        distance=(targets-self.sim.pose[:,:,:2]).norm(dim=-1,keepdim=True)
        deterministic_scorer = ((self._deterministic_mode_mask &
                                 ~self._defense_role_mask)[None] &
                                (action == 4) & carrying)
        deterministic_picker = pickup_active
        # Raster spline targets stay ahead of the robot, so collectors keep
        # following through waypoint arrival instead of stopping.
        stop_radius = torch.where(deterministic_scorer, .02,
                                  torch.where(deterministic_picker, 0., .3))
        command=torch.where(distance<=stop_radius[...,None],torch.zeros_like(command),command)
        command=torch.where((action==7)[...,None],torch.zeros_like(command),command)
        # Keep following the route while the chassis turns to align the
        # intake; waiting for perfect alignment stalls starts and turnarounds.
        command=self._avoid_robot_contention(command,targets,active)
        commands=torch.cat((command,omega[...,None]),-1)
        enabled=self._controlled_mode_mask
        commands=commands*enabled[None,:,None]
        self.sim.step(commands,active_mask=None if full_batch else active,
                      _active_nonempty=active_count>0)
        self._finish_raster_cursor(active,raster_mask)
        self.steps+=active.long()
        self._update_match(active)
        self._update_fuel(active,score_intent=(action==4))
        self._update_perception(active,active_count=active_count)
        done=torch.zeros_like(active)
        truncated=active&(self.steps>=self.horizon)
        # Shared zero-sum score signal: positive for scoring one's alliance,
        # negative when the other alliance scores.
        red=self.fuel_scored_event[:,:3].sum(-1).float()
        blue=self.fuel_scored_event[:,3:].sum(-1).float()
        rewards=torch.where(self.team_ids[None]==0,(red-blue)[:,None],(blue-red)[:,None]).expand(-1,6)
        rewards=torch.where(active[:,None],rewards,torch.zeros_like(rewards))
        if capture_observation:
            obs=self.observe(active)
            self._last_obs=torch.where(active[:,None,None],obs,self._last_obs)
        else:
            obs=self._last_obs
        if capture_info:
            info={"fuel_acquired_event":self.fuel_acquired_event.clone(),
                  "fuel_scored_event":self.fuel_scored_event.clone(),
                  "fuel_passed_event":self.fuel_passed_event.clone(),
                  "fuel_denied_event":self.fuel_denied_event.clone(),
                  "team_score":self.fuel_score_count.clone(),
                  "match_elapsed":self.match_elapsed.clone()}
        else:
            info={}
        return self._last_obs,rewards,done,truncated,info
