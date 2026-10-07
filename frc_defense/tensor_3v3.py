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
                 perception_interval=1, contact_iterations=3,
                 teammate_intent_knowledge=True,
                 sweeping_enabled=False,
                 perception_range=6., perception_fov=120.,
                 stereo_baseline=.0635, fuel_ball_diameter=.15,
                 perception_dropout=.02, position_noise=.015,
                 velocity_noise=.05, replan_interval=20,
                 normalize_observations=True, fused_sensor_rng=False,
                 field_sweep_spacing=.02,
                 behavior_probe=None, behavior_probe_robot=0,
                 behavior_probe_hub_active=True,
                 random_gamepiece_placement=False,
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
        self.stereo_baseline_m = float(stereo_baseline)
        self.perception_ball_diameter = float(fuel_ball_diameter)
        self.perception_fov_degrees = float(perception_fov)
        self.perception_fov = math.radians(self.perception_fov_degrees) * .5
        # Keep short-lived sensor history so a brief occlusion does not erase
        # a useful Fuel target from policy observations and pickup planning.
        self.perception_track_timeout = 2.0
        self.perception_dropout = float(perception_dropout)
        self.position_noise, self.velocity_noise = float(position_noise), float(velocity_noise)
        self.replan_interval = max(1, int(replan_interval))
        self.perception_interval = max(1, int(perception_interval))
        self._planner_tick_scalar=self.replan_interval-1
        self._planner_tick_aligned=True
        self._vectorized_scoring_enabled=True
        self.normalize_observations = bool(normalize_observations)
        modes = list(control_modes or ("deterministic",) * NUM_ROBOTS)
        if len(modes) != NUM_ROBOTS or any(mode not in CONTROL_MODES for mode in modes):
            raise ValueError("control_modes must assign none, nn, or deterministic to six robots")
        self.control_modes = tuple(modes)
        self.teammate_intent_knowledge=bool(teammate_intent_knowledge)
        self.sweeping_enabled=bool(sweeping_enabled)
        self._controlled_mode_mask=torch.tensor(
            [mode!="none" for mode in modes],device=self.device,dtype=torch.bool)
        self._controlled_robot_ids=torch.tensor(
            [i for i,mode in enumerate(modes) if mode!="none"],
            device=self.device,dtype=torch.long)
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
        if behavior_probe not in (None,"collect"):
            raise ValueError("behavior_probe must be None or 'collect'")
        if not 0 <= int(behavior_probe_robot) < NUM_ROBOTS:
            raise ValueError("behavior_probe_robot must be between 0 and 5")
        self.behavior_probe=behavior_probe
        self.behavior_probe_robot=int(behavior_probe_robot)
        self.behavior_probe_hub_active=bool(behavior_probe_hub_active)
        self.random_gamepiece_placement=bool(random_gamepiece_placement)
        self._probe_respawn_capacity=0
        if behavior_probe is not None:
            if (modes[self.behavior_probe_robot]!="deterministic" or
                    roles[self.behavior_probe_robot]!="offense"):
                raise ValueError("behavior probe robot must use deterministic offense")
            if fuel_count < 96 + 6 * preloaded_per_robot + self._probe_respawn_capacity:
                raise ValueError("behavior probe needs one reserved FUEL slot per hopper position")
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
        self._defense_robot_ids=torch.tensor(
            [robot for robot,role in enumerate(roles) if role=="defense"],
            device=self.device,dtype=torch.long)
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
            randomize=(randomize and behavior_probe is None),
            robot_length=(.6096 if behavior_probe else .9),
            robot_width=(.6096 if behavior_probe else .9),
            obstacles=obstacles, field_colliders=colliders,
            bump_regions=bumps, contact_iterations=contact_iterations,
            field_sweep_spacing=field_sweep_spacing)
        self.raster_sweep_points=torch.tensor(
            self._build_raster_spline_points(),device=self.device,dtype=torch.float32)
        self._raster_cursor=torch.zeros((self.n,NUM_ROBOTS),device=self.device,
                                        dtype=torch.long)
        self._raster_reanchor=torch.ones((self.n,NUM_ROBOTS),device=self.device,
                                         dtype=torch.bool)
        self._target_collect_heading=torch.zeros((self.n,NUM_ROBOTS),
                                                  device=self.device,dtype=torch.float32)
        self._target_fuel_index=torch.full((self.n,NUM_ROBOTS),-1,
                                            device=self.device,dtype=torch.long)
        self._target_turn_anchor=torch.zeros((self.n,NUM_ROBOTS,2),
                                             device=self.device,dtype=torch.float32)
        self._target_collecting=torch.zeros((self.n,NUM_ROBOTS),
                                            device=self.device,dtype=torch.bool)
        self._target_cluster_empty_ticks=torch.zeros((self.n,NUM_ROBOTS),
                                                      device=self.device,dtype=torch.long)
        self._avoidance_winner=torch.full(
            (self.n,self.sim._collision_pair_i.numel()),-1,
            device=self.device,dtype=torch.long)
        self._field_scales=torch.tensor((field_length,field_width),device=self.device)
        self._robot_occlusion_radius=.5*torch.sqrt(
            self.sim.length.square()+self.sim.width.square())
        self.randomize = bool(randomize and behavior_probe is None)
        self.drivetrain_config={"name":"built-in illustrative defaults",
            "randomize":self.randomize,"mass":55.0,"robot_length":.9,"robot_width":.9,
            "max_speed":4.8,"max_acceleration":8.0,"max_omega":8.0,"max_alpha":18.0,
            "swerve":self.sim.swerve.__dict__.copy()}
        self.field_feature_obstacles = torch.tensor(
            [(b.x, b.y, box_observation_radius(b)) for b in self.field_feature_boxes],
            device=self.device, dtype=torch.float32).reshape(-1, 3)

        by_name = {b.name: b for b in self.field_boxes}
        self.hub_centers = torch.tensor(
            [(by_name[n].x, by_name[n].y) for n in ("red_hub", "blue_hub")],
            device=self.device, dtype=torch.float32)
        self._score_x_fractions=torch.tensor(
            (.12,.30,.48,.66,.84),device=self.device,dtype=torch.float32)
        self._score_y_fractions=torch.tensor(
            (.12,.28,.44,.60,.76,.90),device=self.device,dtype=torch.float32)
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
        self._probe_respawn_cursor=torch.zeros(
            self.n,device=self.device,dtype=torch.long)
        self._probe_respawn_positions=torch.zeros(
            (self._probe_respawn_capacity,2),device=self.device,
            dtype=self.piece_pos.dtype)
        if self._probe_respawn_capacity:
            probe_indices=torch.arange(self._probe_respawn_capacity,
                                       device=self.device,dtype=torch.float32)
            x_fraction=torch.frac(probe_indices*.6180339887498949)
            if TEAM_IDS[self.behavior_probe_robot]==0:
                x_low=.75
                x_high=self.alliance_zone_depth-.45
            else:
                x_low=self.field_length-self.alliance_zone_depth+.45
                x_high=self.field_length-.75
            self._probe_respawn_positions[:,0]=x_low+(x_high-x_low)*x_fraction
            lower_lane=torch.full_like(probe_indices,.85)
            upper_lane=torch.full_like(probe_indices,self.sim.field_width-.85)
            self._probe_respawn_positions[:,1]=torch.where(
                (probe_indices.long()%2)==0,lower_lane,upper_lane)
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
        self.perception_quality_state = torch.zeros_like(self.track_age)
        self.track_confidence = torch.zeros_like(self.track_age)
        self.track_spatial_extent = torch.zeros(
            (self.n,NUM_ROBOTS,self.fuel_count,2),device=self.device)
        self.track_angular_extent = torch.zeros_like(self.track_age)
        self.track_merged = torch.zeros_like(self.track_mask)
        self._current_fuel_visibility = torch.zeros_like(self.track_mask)
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
        self._last_score_intent = torch.zeros(
            (self.n, NUM_ROBOTS),device=self.device,dtype=torch.bool)
        self.match_elapsed = torch.zeros(self.n, device=self.device)
        self.match_remaining = torch.full((self.n,), 160., device=self.device)
        self.steps = torch.zeros((self.n,), device=self.device, dtype=torch.long)
        self.planner_ticks = torch.zeros_like(self.steps)

        # AD* treats each robot slot as a batch row and shares the static field grid.
        from .tensor_adstar import TensorADStar
        proxy = SimpleNamespace(n=self.n * NUM_ROBOTS, device=self.device, sim=self.sim,
                                field_boxes=self.field_boxes)
        self.planner = TensorADStar(proxy, avoid_bumps=True)
        if behavior_probe == "collect":
            # Allow small tracking error around narrow hardware while keeping
            # exact square-footprint collision checks.
            self.planner.footprint_clearance=.04
        self._planner_controller_mode_key=tuple(
            i for i,mode in enumerate(modes) if mode!="none")
        self._planner_controller_modes_dirty=False
        self._route_segment_indices=torch.arange(
            self.planner.max_points-1,device=self.device)
        self._last_obs = torch.zeros((self.n, NUM_ROBOTS, OBS_DIM), device=self.device)
        self.reset(seed=seed)








    def _hub_line_spawn_poses(self):
        """Place each alliance's three robots against its HUB-side zone line."""
        pose=torch.zeros((self.n,NUM_ROBOTS,3),device=self.device,
                         dtype=self.sim.pose.dtype)
        lane_offsets=torch.tensor((-2.685,0.,2.665, -2.685,0.,2.665),
                                  device=self.device,dtype=pose.dtype)
        pose[:,:,1]=self.sim.field_width*.5+lane_offsets[None]
        pose[:,:,2]=torch.where(self.team_ids[None]==0,0.,math.pi)
        if self.randomize:
            pose[:,:,1]+=(self._random((self.n,NUM_ROBOTS))-.5)*.3
            pose[:,:,2]+=(self._random((self.n,NUM_ROBOTS))-.5)*.2
        # The front support point of the oriented chassis touches the vertical
        # alliance-side HUB face. Account for randomized size and yaw so every
        # robot remains on the same field line after reset.
        heading=pose[:,:,2]
        support_x=(.5*self.sim.length*heading.cos().abs()+
                   .5*self.sim.width*heading.sin().abs())
        red_face=torch.full_like(support_x,self.alliance_zone_depth)
        blue_face=torch.full_like(support_x,
                                  self.sim.field_length-self.alliance_zone_depth)
        face=torch.where(self.team_ids[None]==0,red_face,blue_face)
        pose[:,:,0]=torch.where(self.team_ids[None]==0,
                                face-support_x,face+support_x)
        return pose

    def reset(self, seed=None):
        if seed is not None:
            self.generator.manual_seed(int(seed))
            self._perception_rng_seed = int(seed) & 0xffffffff
        self._perception_rng_ticks.zero_()
        self.sim.reset(seed)
        self.sim.field_contact.zero_()
        self.sim.wall_contact.zero_()
        self.sim.robot_contact.zero_()
        pose=self._hub_line_spawn_poses()
        self.sim.pose.copy_(pose)
        self.sim._field_collision()
        self._prepare_score_grid()
        if self.dt > .1:
            # Playback batches physics coarsely to keep long matches responsive.
            # Bound physical yaw rate so a single coarse step cannot wrap the
            # chassis through a large angle before the controller can react.
            self.sim.omega_limit.clamp_(max=.3/self.dt)
        self._target_collect_heading.copy_(self.sim.pose[:,:,2])
        self._target_fuel_index.fill_(-1)
        self._target_turn_anchor.copy_(self.sim.pose[:,:,:2])
        self._target_collecting.zero_()
        self._target_cluster_empty_ticks.zero_()
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
        self.perception_quality_state.zero_()
        self.track_confidence.zero_(); self.track_spatial_extent.zero_()
        self.track_angular_extent.zero_(); self.track_merged.zero_()
        self._current_fuel_visibility.zero_()
        self.opponent_valid.zero_(); self.opponent_age.fill_(float("inf"))
        self.opponent_size.zero_()
        self._update_perception(torch.ones(self.n, device=self.device, dtype=torch.bool))
        self.last_actions.fill_(7)
        self._last_score_intent.zero_()
        self._last_obs = self.observe()
        return self._last_obs, {"field_layout": "2026_rebuilt", "robots": NUM_ROBOTS}

    def reset_done(self, mask):
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool).reshape(self.n)
        self.sim.reset_done(mask)
        pose=self._hub_line_spawn_poses()
        self.sim.pose.copy_(torch.where(mask[:,None,None],pose,self.sim.pose))
        self.sim._field_collision(mask)
        self._prepare_score_grid()
        # Keep the counter monotonic across matches so counter-based sensor
        # noise does not repeat from the same episode tick on every reset.
        self._target_collect_heading.copy_(torch.where(
            mask[:,None],self.sim.pose[:,:,2],self._target_collect_heading))
        self._target_fuel_index.masked_fill_(mask[:,None],-1)
        self._target_turn_anchor.copy_(torch.where(
            mask[:,None,None],self.sim.pose[:,:,:2],self._target_turn_anchor))
        self._target_collecting.masked_fill_(mask[:,None],False)
        self._target_cluster_empty_ticks.masked_fill_(mask[:,None],0)
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
        self.perception_quality_state[mask] = 0.
        self.track_confidence[mask] = 0.; self.track_spatial_extent[mask] = 0.
        self.track_angular_extent[mask] = 0.; self.track_merged[mask] = False
        self.opponent_valid[mask] = False; self.opponent_age[mask] = float("inf")
        self.opponent_size[mask] = 0.
        self._update_perception(mask)
        self.last_actions[mask] = 7
        self._last_score_intent[mask] = False
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
        if self.sweeping_enabled:
            self._prepare_raster_cursor(active)
        targets,action,carrying,fuel_delta=self._target_for_actions(actions,active)
        raster_mask=(self.sweeping_enabled & (self._deterministic_mode_mask &
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
            robot_radii=.5*torch.minimum(self.sim.length,self.sim.width)
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
            if self.sweeping_enabled:
                self.planner.plan_waypoints(
                    start,waypoint_chains,self._raster_cursor.reshape(-1),heading,lengths,widths,
                    lookahead=self._raster_waypoint_lookahead,
                    fallback_goal=goal,waypoint_mask=raster_mask.reshape(-1),
                    speed=speeds,lateral_friction=lateral,acceleration=accel,
                    active_mask=flat_active,robot_obstacles=robot_obstacles)
            else:
                self.planner.plan(
                    start,goal,heading,lengths,widths,speed=speeds,
                    lateral_friction=lateral,acceleration=accel,
                    active_mask=flat_active,robot_obstacles=robot_obstacles)
            # A depot approach can be a narrow field-edge route. If the other
            # chassis collectively seal its only entrance, retry that failed
            # route against static field geometry; near-term robot contention
            # is handled by the motion controller and physics contacts.
            depot_fuel=self._last_audit_fuel_targets
            depot_target=torch.zeros((self.n,NUM_ROBOTS),device=self.device,
                                     dtype=torch.bool)
            for box in self.field_boxes:
                if box.name.endswith("_depot"):
                    depot_target |= (
                        (depot_fuel[...,0]-box.x).abs()<=box.length/2) & (
                        (depot_fuel[...,1]-box.y).abs()<=box.width/2)
            depot_target &= (self.last_actions.reshape(self.n,NUM_ROBOTS)<4)
            failed=(self.planner.last_lengths.reshape(self.n,NUM_ROBOTS)<=2)
            far_goal=(goal.reshape(self.n,NUM_ROBOTS,2)-
                      self.sim.pose[:,:,:2]).norm(dim=-1)>1.25
            retry=(depot_target&failed&far_goal&planner_robot_mask[None])
            retry_flat=retry.reshape(-1)
            if getattr(self, "_capture_planner_retries", False):
                # Keep retry execution in the captured planner graph. The
                # planner uses a fixed six-row batch and masks persistent
                # writes, avoiding both .any() synchronization and nonzero
                # compaction while preserving the retry predicate on device.
                if self.sweeping_enabled:
                    self.planner.plan_waypoints(
                        start,waypoint_chains,self._raster_cursor.reshape(-1),heading,lengths,widths,
                        lookahead=self._raster_waypoint_lookahead,
                        fallback_goal=goal,waypoint_mask=raster_mask.reshape(-1),
                        speed=speeds,lateral_friction=lateral,acceleration=accel,
                        active_mask=retry_flat,robot_obstacles=None,
                        static_active_mask=True)
                else:
                    self.planner.plan(
                        start,goal,heading,lengths,widths,speed=speeds,
                        lateral_friction=lateral,acceleration=accel,
                        active_mask=retry_flat,robot_obstacles=None,
                        static_active_mask=True)
            elif retry_flat.any():
                if self.sweeping_enabled:
                    self.planner.plan_waypoints(
                        start,waypoint_chains,self._raster_cursor.reshape(-1),heading,lengths,widths,
                        lookahead=self._raster_waypoint_lookahead,
                        fallback_goal=goal,waypoint_mask=raster_mask.reshape(-1),
                        speed=speeds,lateral_friction=lateral,acceleration=accel,
                        active_mask=retry_flat,robot_obstacles=None)
                else:
                    self.planner.plan(
                        start,goal,heading,lengths,widths,speed=speeds,
                        lateral_friction=lateral,acceleration=accel,
                        active_mask=retry_flat,robot_obstacles=None)
        adstar_command=None
        adstar_tangent=None
        if not self.sweeping_enabled and self._has_deterministic_offense:
            adstar_command,adstar_tangent=self.planner.path_reference(
                start,self.sim.velocity[:,:,:2].reshape(-1,2),speeds,
                lookahead=(.25 if self.behavior_probe=="collect" else None))
            adstar_tangent=adstar_tangent.reshape(self.n,NUM_ROBOTS,2)
        hub=self.hub_centers[self.team_ids][None]
        hub_delta=hub-self.sim.pose[:,:,:2]
        hub_bearing=torch.atan2(hub_delta[...,1],hub_delta[...,0])
        angle_error=torch.atan2(torch.sin(hub_bearing-self.sim.pose[:,:,2]),
                                torch.cos(hub_bearing-self.sim.pose[:,:,2]))
        omega=swerve_heading_rate(angle_error,self.sim.omega_limit,self.sim.alpha,
                                  self.sim.velocity[:,:,2],self.dt)
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
        collect_braking_to_ferry=torch.zeros_like(ferrying)
        if (self.behavior_probe=="collect" and
                not self.behavior_probe_hub_active):
            collect_possession=(self.piece_owner[:,None,:]==
                torch.arange(NUM_ROBOTS,device=self.device)[None,:,None]).sum(-1)
            collect_braking_to_ferry=((action==7)&carrying&self._dumper_mask[None]&
                (collect_possession>=self.robot_fuel_capacities[None]))
        ferry_omega=swerve_heading_rate(ferry_error,self.sim.omega_limit,self.sim.alpha,
                                        self.sim.velocity[:,:,2],self.dt)
        omega=torch.where(ferrying|collect_braking_to_ferry,ferry_omega,omega)
        deterministic_robot=self._deterministic_mode_mask[None]
        deterministic_offense=(self._deterministic_mode_mask &
                                ~self._defense_role_mask)[None]
        pickup_active=(deterministic_offense &
                       ((action<4)|((action==6)&~carrying)))
        pickup_direction=fuel_delta
        if adstar_tangent is not None:
            target_based_picker=pickup_active & (not self.sweeping_enabled)
            tangent_norm=adstar_tangent.norm(dim=-1,keepdim=True)
            target_based_tangent=torch.where(tangent_norm>.5,adstar_tangent,
                targets-self.sim.pose[:,:,:2])
            pickup_direction=torch.where(target_based_picker[...,None],
                                         target_based_tangent,pickup_direction)
        if self.behavior_probe=="collect":
            pickup_direction=torch.where(
                (pickup_active & (action<4))[...,None],
                fuel_delta,pickup_direction)
        pickup_bearing=torch.atan2(pickup_direction[...,1],pickup_direction[...,0])
        pickup_error=torch.atan2(torch.sin(pickup_bearing-self.sim.pose[:,:,2]),
                                 torch.cos(pickup_bearing-self.sim.pose[:,:,2]))
        pickup_omega=swerve_heading_rate(pickup_error,self.sim.omega_limit,self.sim.alpha,
                                         self.sim.velocity[:,:,2],self.dt)
        omega=torch.where(pickup_active,pickup_omega,omega)
        trench_alignment=self.planner.last_trench_alignment.reshape(self.n,6)
        trench_heading=self.planner.last_heading.reshape(self.n,6)
        trench_error=torch.atan2(torch.sin(trench_heading-self.sim.pose[:,:,2]),
                                 torch.cos(trench_heading-self.sim.pose[:,:,2]))
        trench_omega=swerve_heading_rate(trench_error,self.sim.omega_limit,self.sim.alpha,
                                         self.sim.velocity[:,:,2],self.dt)
        omega=torch.where(trench_alignment,trench_omega,omega)
        # The physics kernel applies exact per-module speed desaturation after
        # combining translation and rotation. Keep planning at the chassis cap
        # instead of subtracting the worst-case rotational wheel speed twice.
        available_speed=self.sim.speed
        if adstar_command is None:
            command,_=self.planner.path_reference(
                start,self.sim.velocity[:,:,:2].reshape(-1,2),
                available_speed.reshape(-1),
                lookahead=(.25 if self.behavior_probe=="collect" else None))
        else:
            command=adstar_command
        command=command.reshape(self.n,6,2)
        command=torch.where((trench_alignment & (trench_error.abs()>.25))[...,None],
                            torch.zeros_like(command),command)
        distance=(targets-self.sim.pose[:,:,:2]).norm(dim=-1,keepdim=True)
        if self.behavior_probe=="collect":
            # A short staging route can leave AD*'s lookahead near its zero-
            # speed endpoint while the intake target is still distant. Bridge
            # that final approach instead of waiting for speed to recover,
            # but only when the direct chassis path is footprint-clear.
            direct=(targets-self.sim.pose[:,:,:2])
            direct=direct/direct.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            direct_speed=torch.minimum(self.sim.speed,
                                       (distance[...,0]*3.).clamp(max=4.8))
            direct_command=direct*direct_speed[...,None]
            direct_path=torch.stack((start,targets.reshape(-1,2)),dim=1)
            direct_clear=self.planner._footprint_path_clear(
                direct_path,heading,lengths,widths).reshape(self.n,NUM_ROBOTS)
            stalled_picker=(pickup_active & (action<4) & (distance[...,0]>.03) &
                ((distance[...,0]<1.6) | (command.norm(dim=-1)<.05)) & direct_clear)
            command=torch.where(stalled_picker[...,None],direct_command,command)
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
        stationary_pose=(self.sim.pose.clone() if "none" in self.control_modes else None)
        self.sim.step(commands,active_mask=None if full_batch else active,
                      _active_nonempty=active_count>0)
        if stationary_pose is not None:
            stationary=active[:,None]&~enabled[None,:]
            self.sim.pose.copy_(torch.where(
                stationary[...,None],stationary_pose,self.sim.pose))
            self.sim.velocity.copy_(torch.where(
                stationary[...,None],torch.zeros_like(self.sim.velocity),
                self.sim.velocity))
        if self.sweeping_enabled:
            self._finish_raster_cursor(active,raster_mask)
        self.steps+=active.long()
        self._update_match(active)
        self._update_fuel(active,score_intent=self._last_score_intent)
        if self.perception_interval == 1:
            self._update_perception(active,active_count=active_count)
        else:
            perception_tick = (self.steps % self.perception_interval) == 0
            update_active = active & perception_tick
            aging_active = active & ~perception_tick
            if aging_active.any():
                self.track_age = torch.where(
                    aging_active[:, None, None] & self.track_mask,
                    self.track_age + self.dt, self.track_age)
                self.track_mask &= ~((self.track_age > self.perception_track_timeout) &
                                     aging_active[:, None, None])
                self.opponent_age = torch.where(
                    aging_active[:, None] & self.opponent_valid,
                    self.opponent_age + self.dt, self.opponent_age)
                self.opponent_valid &= ~((self.opponent_age > 1.) & aging_active[:, None])
            update_count = int(update_active.sum().item())
            if update_count:
                self._update_perception(update_active, active_count=update_count)
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
