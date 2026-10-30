"""Six-robot, parameter-shared strategic environment for REBUILT 3v3 play.

The established 137-input/8-action ActorCritic is shared across robot slots.
Each row is a robot-centric observation; the reserved second local-track block
now carries the two nearest teammates.  The simulator returns one observation
and reward per robot while match termination and physics remain per world.
"""
from __future__ import annotations

import math
import os
from types import SimpleNamespace

import torch

from .tensor_sim import TensorVectorizedSimulator, swerve_heading_rate
from .fuel_physics_runtime import (initialize_fuel_physics, reset_fuel_physics,
                                  advance_with_fuel)
from .field import ALLIANCE_ZONE_DEPTH, midfield_respawn_points
from ._tensor_3v3_actions import TensorThreeVsThreeActionMixin
from ._tensor_3v3_constants import TEAM_IDS
from ._tensor_3v3_gamepieces import TensorThreeVsThreeGamepieceMixin
from ._tensor_3v3_navigation import TensorThreeVsThreeNavigationMixin
from ._tensor_3v3_observation import (
    ACTION_DIM, NUM_ROBOTS, OBS_DIM, TensorThreeVsThreeObservationMixin,
    _assert_device_condition,
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
                 robot_roles=None, robot_types=None, allow_nn_offense=False,
                 dt=.02, horizon=8000, randomize=True, fuel_count=504,
                 preloaded_per_robot=0, max_fuel_capacity=60,
                 max_scoring_bps=25.0, turret_fuel_capacity=40,
                 turret_scoring_bps=15.0,
                 perception_interval=2, contact_iterations=3,
                 teammate_intent_knowledge=False,
                 teammate_coordination_enabled=None,
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
                 field_width=8.07, obstacles=(), fuel_physics=True,
                 field_layout="2026_rebuilt", defer_initial_perception=False):
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
        # ROCm graph capture passed end-to-end parity on both WX9100 devices.
        self._target_action_graph_enabled = bool(
            torch.version.hip and self.n <= 8 and
            os.environ.get("AUTODRIVE_3V3_TARGET_GRAPH", "1") != "0")
        self._target_action_graph = None
        self._target_action_graph_signature = None
        self._target_action_graph_outputs = None
        self._target_action_graph_audit = None
        self._target_action_graph_actions = None
        self._target_action_graph_active = None
        self._target_action_graph_disabled = False
        # Audit goals are valid only after a policy action in this episode.
        # Keep validity separate from the CUDA graph's output buffers.
        self._last_action_audit_valid = torch.zeros(
            self.n, device=self.device, dtype=torch.bool)
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
        self._full_batch_step_scalar=0
        self._vectorized_scoring_enabled=True
        self.normalize_observations = bool(normalize_observations)
        modes = list(control_modes or ("deterministic",) * NUM_ROBOTS)
        if len(modes) != NUM_ROBOTS or any(mode not in CONTROL_MODES for mode in modes):
            raise ValueError("control_modes must assign none, nn, or deterministic to six robots")
        self.control_modes = tuple(modes)
        # This gates every behavior that uses teammate strategy or intent:
        # assigned search/ferry lanes, shared possession/training features,
        # target-based right-of-way, and predicted teammate traffic avoidance.
        # Physical teammate pose and velocity remain observable to model
        # onboard sensing and local collision response.
        if teammate_coordination_enabled is None:
            teammate_coordination_enabled=teammate_intent_knowledge
        self.teammate_coordination_enabled=bool(teammate_coordination_enabled)
        self.sweeping_enabled=bool(sweeping_enabled)
        self._controlled_mode_mask=torch.tensor(
            [mode!="none" for mode in modes],device=self.device,dtype=torch.bool)
        self._controlled_robot_ids=torch.tensor(
            [i for i,mode in enumerate(modes) if mode!="none"],
            device=self.device,dtype=torch.long)
        roles = list(robot_roles or ("offense",) * 3 + ("defense",) * 3)
        if len(roles) != NUM_ROBOTS or any(role not in ("offense", "defense") for role in roles):
            raise ValueError("robot_roles must assign offense or defense to six robots")
        if not allow_nn_offense and any(mode == "nn" and role != "defense" for mode, role in zip(modes, roles)):
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
        self._intake_collecting=torch.zeros((self.n,NUM_ROBOTS),device=self.device,
                                             dtype=torch.bool)
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
        self._deterministic_offense_robot_ids=tuple(
            i for i,(mode,role) in enumerate(zip(modes,roles))
            if mode=="deterministic" and role=="offense")
        self._deterministic_offense_robot_index=torch.tensor(
            self._deterministic_offense_robot_ids,device=self.device,dtype=torch.long)
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
        # Coordinated ferry/search uses a preassigned lane per robot slot.
        # Siloed fallback search instead projects each robot's own lateral
        # motion toward midfield without a teammate lane assignment.
        midfield_x=float(field_length)/2
        ferry_x=midfield_x+torch.where(self.team_ids==0,-.45,.45)
        ferry_lane=torch.tensor((1.35,4.035,6.72,6.72,4.035,1.35),
                                device=self.device,dtype=torch.float32)
        self.ferry_targets=torch.stack((ferry_x,ferry_lane),-1)
        self.field_search_lanes=self.ferry_targets[:,1]
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

        from .field import (box_observation_radius, bump_boxes, qualification_field,
                            static_collision_boxes)
        self.field_layout_id = str(field_layout)
        self.field_boxes = qualification_field(self.field_layout_id,
            field_length, field_width)
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
            max_speed=9.6, max_omega=16.0,
            obstacles=obstacles, field_colliders=colliders,
            bump_regions=bumps, contact_iterations=contact_iterations,
            field_sweep_spacing=field_sweep_spacing)
        # Policy localization is an explicit state boundary.  Physics owns
        # ``sim.pose``; observations, candidate construction, and controllers
        # consume this estimate.  The default estimator starts localized and
        # is advanced by wheel/gyro odometry below.  Scenario fixtures may
        # deliberately perturb an estimate without changing physical pose.
        self.estimated_pose = self.sim.pose.clone()
        # This is the estimator frame delivered to the policy. Phase 4
        # observation delay may temporarily restore an older captured value;
        # ``estimated_pose`` continues to advance in the background.
        self.policy_pose = self.estimated_pose.clone()
        self.raster_sweep_points=torch.tensor(
            self._build_raster_spline_points(),device=self.device,dtype=torch.float32)
        self._raster_cursor=torch.zeros((self.n,NUM_ROBOTS),device=self.device,
                                        dtype=torch.long)
        self._raster_reanchor=torch.ones((self.n,NUM_ROBOTS),device=self.device,
                                         dtype=torch.bool)
        self._no_track_search_active=torch.zeros(
            (self.n,NUM_ROBOTS),device=self.device,dtype=torch.bool)
        self._target_collect_heading=torch.zeros((self.n,NUM_ROBOTS),
                                                  device=self.device,dtype=torch.float32)
        self._target_fuel_index=torch.full((self.n,NUM_ROBOTS),-1,
                                            device=self.device,dtype=torch.long)
        self._target_fuel_selected_step=torch.full((self.n,NUM_ROBOTS),-1,
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
            "max_speed":9.6,"max_acceleration":8.0,"max_omega":16.0,"max_alpha":18.0,
            "swerve":self.sim.swerve.__dict__.copy()}
        self.field_feature_obstacles = torch.tensor(
            [(b.x, b.y, box_observation_radius(b)) for b in self.field_feature_boxes],
            device=self.device, dtype=torch.float32).reshape(-1, 3)
        self.field_occlusion_boxes = torch.tensor(
            [(b.x,b.y,b.length/2,b.width/2) for b in solids],
            device=self.device,dtype=torch.float32).reshape(-1,4)

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
        initialize_fuel_physics(self,fuel_physics)
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
        self._score_committed = torch.zeros_like(self._ferry_committed)
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
                                policy_pose=self.policy_pose,
                                field_boxes=self.field_boxes)
        self.planner = TensorADStar(proxy, avoid_bumps=True)
        self._planner_controller_mode_key=tuple(
            i for i,mode in enumerate(modes) if mode!="none")
        self._planner_controller_modes_dirty=False
        self._planner_traversal_request_signature=None
        self._last_applied_planner_traversal_signature=None
        self._planner_traversal_pending=torch.zeros((self.n,),device=self.device,
                                                     dtype=torch.bool)
        self._route_segment_indices=torch.arange(
            self.planner.max_points-1,device=self.device)
        self._last_obs = torch.zeros((self.n, NUM_ROBOTS, OBS_DIM), device=self.device)
        self.reset(seed=seed, defer_perception=defer_initial_perception)








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

    @torch.no_grad()
    def synchronize_policy_pose(self, mask=None):
        """Initialize estimated localization from physical pose at a reset.

        This is for episode/scenario setup only. Runtime decisions must read
        ``policy_pose`` and runtime movement advances it through odometry.
        """
        if mask is None:
            self.estimated_pose.copy_(self.sim.pose)
            self.policy_pose.copy_(self.estimated_pose)
            return
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool).reshape(self.n)
        reset = mask[:, None, None]
        self.estimated_pose.copy_(torch.where(reset, self.sim.pose,
                                              self.estimated_pose))
        self.policy_pose.copy_(torch.where(reset, self.estimated_pose,
                                           self.policy_pose))

    @torch.no_grad()
    def set_focal_pose_estimate(self, estimate, *, world=0, slot=3):
        """Set one robot's estimated ``(x, y, heading)`` without moving it.

        The helper is intentionally distinct from scenario placement, which
        edits physical truth.  It is the supported hook for localization
        errors and estimator recovery experiments.
        """
        if (type(world) is not int or not 0 <= world < self.n or
                type(slot) is not int or not 0 <= slot < NUM_ROBOTS):
            raise ValueError("world/slot is outside the policy-pose state")
        value = torch.as_tensor(estimate, device=self.device,
                                dtype=self.policy_pose.dtype).reshape(3)
        _assert_device_condition(self.device,torch.isfinite(value).all(),
                                 "pose estimate must be finite")
        self.estimated_pose[world, slot].copy_(value)
        self.policy_pose[world, slot].copy_(value)

    @torch.no_grad()
    def _advance_policy_pose_odometry(self, prior_velocity, active):
        """Integrate the simulator's idealized onboard wheel/gyro odometry feed.

        ``sim.velocity`` is used here as the modeled body-motion sensor input;
        it is noiseless and simulator-produced, but this estimator does not
        read ``sim.pose`` after reset. Hardware use requires replacing this
        feed with actual wheel/gyro measurements.
        """
        active = torch.as_tensor(active, device=self.device, dtype=torch.bool).reshape(self.n)
        estimate = self.estimated_pose.clone()
        linear_delta = .5 * (prior_velocity[..., :2] + self.sim.velocity[..., :2]) * self.dt
        angular_delta = .5 * (prior_velocity[..., 2] + self.sim.velocity[..., 2]) * self.dt
        estimate[..., :2] += linear_delta
        estimate[..., 2] += angular_delta
        estimate[..., 2] = torch.atan2(torch.sin(estimate[..., 2]),
                                       torch.cos(estimate[..., 2]))
        update = active[:, None] & self._controlled_mode_mask[None]
        self.estimated_pose.copy_(torch.where(update[..., None], estimate,
                                              self.estimated_pose))
        self.policy_pose.copy_(torch.where(update[..., None], self.estimated_pose,
                                           self.policy_pose))

    def _invalidate_last_action_audit(self, mask=None):
        """Invalidate action-derived audit data for reset worlds."""
        if mask is None:
            self._last_action_audit_valid.zero_()
            return
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool).reshape(self.n)
        self._last_action_audit_valid.masked_fill_(mask, False)

    def reset(self, seed=None, *, defer_perception=False):
        if type(defer_perception) is not bool:
            raise ValueError("defer_perception must be boolean")
        self._invalidate_last_action_audit()
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
        self.synchronize_policy_pose()
        self._prepare_score_grid()
        if self.dt > .1:
            # Playback batches physics coarsely to keep long matches responsive.
            # Bound physical yaw rate so a single coarse step cannot wrap the
            # chassis through a large angle before the controller can react.
            self.sim.omega_limit.clamp_(max=.3/self.dt)
        self._target_collect_heading.copy_(self.policy_pose[:,:,2])
        self._target_fuel_index.fill_(-1)
        self._target_fuel_selected_step.fill_(-1)
        self._target_turn_anchor.copy_(self.policy_pose[:,:,:2])
        self._target_collecting.zero_()
        self._intake_collecting.zero_()
        self._no_track_search_active.zero_()
        self._target_cluster_empty_ticks.zero_()
        self._raster_cursor.copy_(self._closest_raster_indices(self.policy_pose[:,:,:2]))
        self._raster_reanchor.fill_(True)
        self.match_elapsed.zero_(); self.match_remaining.fill_(160.)
        self.steps.zero_(); self.planner_ticks.fill_(self.replan_interval-1)
        self._planner_tick_scalar=self.replan_interval-1; self._planner_tick_aligned=True
        self._full_batch_step_scalar=0
        self._planner_traversal_request_signature=None
        self._last_applied_planner_traversal_signature=None
        self._planner_traversal_pending.zero_()
        self.hub_active.fill_(True); self.auto_fuel_scores.zero_(); self.fuel_score_count.zero_()
        for tensor in (self.fuel_acquisition_count, self.fuel_denied_count,
                       self.fuel_acquired_event, self.fuel_scored_event,
                       self.fuel_passed_event, self.fuel_denied_event,
                       self.next_intake, self.next_score, self.next_ferry,
                       self._ferry_committed,self._score_committed,
                       self._avoidance_winner,
                       self.last_hub_zone):
            tensor.zero_()
        self._avoidance_winner.fill_(-1)
        self._randomize_pass_zone_positions()
        self._initialize_fuel()
        reset_fuel_physics(self)
        self.track_mask.zero_(); self.track_age.fill_(float("inf"))
        self.perception_quality_state.zero_()
        self.track_confidence.zero_(); self.track_spatial_extent.zero_()
        self.track_angular_extent.zero_(); self.track_merged.zero_()
        self._current_fuel_visibility.zero_()
        self.opponent_valid.zero_(); self.opponent_age.fill_(float("inf"))
        self.opponent_size.zero_()
        self.last_actions.fill_(7)
        self._last_score_intent.zero_()
        if defer_perception:
            self._last_obs = None
            return None
        self._update_perception(torch.ones(self.n, device=self.device, dtype=torch.bool))
        self._last_obs = self.observe()
        return self._last_obs, {"field_layout": self.field_layout_id, "robots": NUM_ROBOTS}

    def reset_done(self, mask):
        if (self.device.type != "cpu" and
                (not isinstance(mask, torch.Tensor) or mask.device != self.device)):
            raise ValueError("GPU reset mask must already reside on the environment device")
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool).reshape(self.n)
        self._invalidate_last_action_audit(mask)
        self.sim.reset_done(mask)
        if self.n == 1:
            self._planner_traversal_request_signature=None
            self._last_applied_planner_traversal_signature=None
        self._planner_traversal_pending &= ~mask
        pose=self._hub_line_spawn_poses()
        self.sim.pose.copy_(torch.where(mask[:,None,None],pose,self.sim.pose))
        self.sim._field_collision(mask)
        self.synchronize_policy_pose(mask)
        self._prepare_score_grid()
        # Keep the counter monotonic across matches so counter-based sensor
        # noise does not repeat from the same episode tick on every reset.
        self._target_collect_heading.copy_(torch.where(
            mask[:,None],self.policy_pose[:,:,2],self._target_collect_heading))
        self._target_fuel_index.masked_fill_(mask[:,None],-1)
        self._target_fuel_selected_step.masked_fill_(mask[:,None],-1)
        self._target_turn_anchor.copy_(torch.where(
            mask[:,None,None],self.policy_pose[:,:,:2],self._target_turn_anchor))
        self._target_collecting.masked_fill_(mask[:,None],False)
        self._no_track_search_active.masked_fill_(mask[:,None],False)
        self._target_cluster_empty_ticks.masked_fill_(mask[:,None],0)
        reset_cursor=self._closest_raster_indices(self.policy_pose[:,:,:2])
        self._raster_cursor.copy_(torch.where(mask[:,None],reset_cursor,self._raster_cursor))
        self._raster_reanchor.copy_(torch.where(
            mask[:,None],torch.ones_like(self._raster_reanchor),self._raster_reanchor))
        self.sim.velocity.copy_(torch.where(mask[:, None, None], torch.zeros_like(self.sim.velocity),
                                            self.sim.velocity))
        for tensor in (self.match_elapsed, self.steps):
            tensor.masked_fill_(mask, 0)
        self.planner_ticks.copy_(torch.where(mask,
            torch.full_like(self.planner_ticks,self.replan_interval-1),self.planner_ticks))
        # Keep reset scheduling per world; do not inspect a device mask to
        # restore the host scalar fast path.
        self._planner_tick_aligned=False
        self._full_batch_step_scalar=None
        self.match_remaining.copy_(torch.where(mask, torch.full_like(self.match_remaining, 160.),
                                               self.match_remaining))
        for tensor in (self.auto_fuel_scores, self.fuel_score_count, self.fuel_acquisition_count,
                       self.fuel_denied_count, self.fuel_acquired_event, self.fuel_scored_event,
                       self.fuel_passed_event, self.fuel_denied_event,
                       self.next_intake, self.next_score, self.next_ferry,
                       self._ferry_committed,self._score_committed,
                       self._avoidance_winner,
                       self.last_hub_zone):
            tensor[mask] = 0
        self._avoidance_winner[mask] = -1
        self.hub_active[mask] = True
        self._randomize_pass_zone_positions(mask)
        self._initialize_fuel(mask)
        reset_fuel_physics(self,mask)
        self.track_mask[mask] = False; self.track_age[mask] = float("inf")
        self.perception_quality_state[mask] = 0.
        self.track_confidence[mask] = 0.; self.track_spatial_extent[mask] = 0.
        self.track_angular_extent[mask] = 0.; self.track_merged[mask] = False
        self.opponent_valid[mask] = False; self.opponent_age[mask] = float("inf")
        self.opponent_size[mask] = 0.
        self._update_perception(mask)
        self.last_actions[mask] = 7
        self._intake_collecting[mask] = False
        self._last_score_intent[mask] = False
        self._last_obs = torch.where(mask[:, None, None], self.observe(), self._last_obs)
        return self._last_obs

    def _target_for_actions_step(self, actions, active, full_batch,
                                 policy_action_override=None):
        """Replay the static target-selection tensor workload on HIP batches."""
        if (policy_action_override is not None or
                not self._target_action_graph_enabled or
                self._target_action_graph_disabled or not full_batch or
                self.device.type != "cuda" or
                not torch.is_tensor(actions) or not actions.is_cuda or
                actions.dtype != torch.long or
                torch.cuda.is_current_stream_capturing()):
            return self._target_for_actions(actions, active,
                policy_action_override=policy_action_override)

        signature = (self.n, self.control_modes, self.robot_roles,
                     self.robot_types, self.sweeping_enabled,
                     self.dt, self.field_length, self.sim.field_width,
                     self.fuel_physics is not None)
        if self._target_action_graph_signature != signature:
            self._target_action_graph = None
            self._target_action_graph_outputs = None
            self._target_action_graph_audit = None
            self._target_action_graph_signature = signature
        if self._target_action_graph_actions is None:
            self._target_action_graph_actions = torch.empty(
                (self.n, NUM_ROBOTS), device=self.device, dtype=torch.long)
            self._target_action_graph_active = torch.empty(
                (self.n,), device=self.device, dtype=torch.bool)

        self._target_action_graph_actions.copy_(actions.reshape(self.n, NUM_ROBOTS))
        self._target_action_graph_active.copy_(active)
        if self._target_action_graph is None:
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    outputs = self._target_for_actions(
                        self._target_action_graph_actions,
                        self._target_action_graph_active)
                audit = (self._last_audit_targets,
                         self._last_audit_fuel_targets,
                         self._last_audit_cluster_count)
                graph.replay()
            except Exception:
                self._target_action_graph_disabled = True
                return self._target_for_actions(actions, active)
            self._target_action_graph = graph
            self._target_action_graph_outputs = outputs
            self._target_action_graph_audit = audit
        else:
            self._target_action_graph.replay()
            (self._last_audit_targets,
             self._last_audit_fuel_targets,
             self._last_audit_cluster_count) = self._target_action_graph_audit
        return self._target_action_graph_outputs




    def _perceived_robot_obstacles(self):
        """Known same-team localization plus one anonymous observed opponent."""
        # Each observer receives three known team slots and one anonymous enemy
        # track. The final two slots remain disabled, preserving planner shape.
        known_ids = torch.stack(self._team_robot_ids)[self.team_ids]
        obstacles = self.policy_pose.new_zeros((self.n, NUM_ROBOTS, NUM_ROBOTS, 3))
        obstacles[:, :, :3, :2] = self.policy_pose[:, known_ids, :2]
        radii = .5 * torch.minimum(self.sim.length, self.sim.width)
        own = known_ids == torch.arange(NUM_ROBOTS, device=self.device)[:, None]
        obstacles[:, :, :3, 2] = radii[:, known_ids].masked_fill(own[None], 0.)
        observed_xy = self.opponent_pose[:, :, :2]
        observed_size = self.opponent_size[:, :, :2]
        valid = (self.opponent_valid & torch.isfinite(observed_xy).all(-1)
                 & torch.isfinite(observed_size).all(-1)
                 & (observed_size > 0.).all(-1))
        obstacles[:, :, 3, :2] = torch.where(valid[:, :, None], observed_xy, 0.)
        observed_radius = .5 * observed_size.amin(-1)
        obstacles[:, :, 3, 2] = torch.where(valid, observed_radius, 0.)
        return obstacles.reshape(-1, NUM_ROBOTS, 3)

    def step(self, actions, active_mask=None, *, capture_observation=True,
             capture_info=True, _active_count=None, _return_info=None,
             planner_goal_override=None, planner_goal_override_action=None,
             planner_goal_override_target_track=None,
             planner_goal_override_target_xy=None,
             planner_goal_override_traversal=False,
             planner_goal_heading_override=None,
             planner_traversal_cost_override=None,
             planner_traversal_cost_override_action=None,
             planner_traversal_cost_override_scale=1.0,
             planner_force_goal_replan=False,
             policy_action_override=None):
        full_batch=active_mask is None
        if full_batch and self._full_batch_step_scalar is not None:
            self._full_batch_step_scalar+=1
        elif not full_batch:
            self._full_batch_step_scalar=None
        if (not full_batch and self.device.type != "cpu" and
                (not isinstance(active_mask, torch.Tensor) or active_mask.device != self.device)):
            raise ValueError("GPU active mask must already reside on the environment device")
        active=(torch.ones(self.n,device=self.device,dtype=torch.bool) if full_batch else
                torch.as_tensor(active_mask,device=self.device,dtype=torch.bool).reshape(self.n))
        if _return_info is not None:
            capture_info=bool(_return_info)
        active_count=(self.n if full_batch else
                      int(_active_count) if _active_count is not None else None)
        if planner_goal_override_target_track is not None:
            if (type(planner_goal_override_target_track) is not int or
                    not 0 <= planner_goal_override_target_track < self.track_mask.shape[-1]):
                raise ValueError("planner collection target must be a valid sensor-track buffer index")
            if (self.n != 1 or self.control_modes[3] != "nn" or
                    self.robot_roles[3] != "offense"):
                raise ValueError("planner collection target override is restricted to neural offense slot 3")
            if (planner_goal_override is None or
                    type(planner_goal_override_action) is not int or
                    not 0 <= planner_goal_override_action < 4):
                raise ValueError("planner collection target must bind to a paired collect goal")
            if planner_goal_override_target_xy is None:
                raise ValueError("planner collection target requires its snapshot XY")
            planner_target_source_xy = torch.as_tensor(
                planner_goal_override_target_xy, device=self.device,
                dtype=self.policy_pose.dtype).reshape(2)
            _assert_device_condition(self.device,
                torch.isfinite(planner_target_source_xy).all(),
                "planner collection target snapshot XY must be finite")
        elif planner_goal_override_target_xy is not None:
            raise ValueError("planner collection target XY requires its sensor-track index")
        self._prepare_raster_cursor(active)
        targets,action,carrying,fuel_delta=self._target_for_actions_step(
            actions,active,full_batch,
            policy_action_override=policy_action_override)
        self._last_planner_collection_target_valid = None
        self._last_planner_collection_action_match = None
        self._last_planner_collection_goal_heading = None
        collection_target_valid = torch.zeros_like(active)
        collection_target_xy = None
        if planner_goal_override_target_track is not None:
            track_index = planner_goal_override_target_track
            collection_target_xy = self.track_pos[:, 3, track_index]
            source_gap = (collection_target_xy - planner_target_source_xy[None]).norm(dim=-1)
            target_valid = (
                self.track_mask[:, 3, track_index] &
                (self.track_age[:, 3, track_index] <= self.perception_track_timeout) &
                torch.isfinite(collection_target_xy).all(-1) & (source_gap <= 1.0))
            action_match = action[:, 3] == planner_goal_override_action
            collection_target_valid = active & target_valid & action_match
            self._last_planner_collection_target_valid = target_valid[0]
            self._last_planner_collection_action_match = action_match[0]
            selected_target_delta = collection_target_xy - self.policy_pose[:, 3, :2]
            selected_target_distance = selected_target_delta.norm(dim=-1, keepdim=True)
            current_forward = torch.stack((self.policy_pose[:, 3, 2].cos(),
                                           self.policy_pose[:, 3, 2].sin()), dim=-1)
            selected_target_delta = torch.where(selected_target_distance > 1e-8,
                selected_target_delta, current_forward)
            fuel_delta = fuel_delta.clone()
            fuel_delta[:, 3] = torch.where(collection_target_valid[:, None],
                selected_target_delta, fuel_delta[:, 3])
            selected_target_pickup_active = torch.zeros_like(action, dtype=torch.bool)
            selected_target_pickup_active[:, 3] = collection_target_valid
        else:
            selected_target_delta = None
            selected_target_pickup_active = torch.zeros_like(action, dtype=torch.bool)
        planner_goal_override_mask = torch.zeros_like(action, dtype=torch.bool)
        planner_goal_heading_mask = torch.zeros_like(action, dtype=torch.bool)
        planner_goal_heading = None
        if planner_goal_override is not None:
            if self.n != 1:
                raise ValueError("planner goal overrides currently require a single-world environment")
            if planner_goal_override_action is None:
                raise ValueError("planner goal override requires its paired action index")
            goal = torch.as_tensor(planner_goal_override, device=self.device,
                                   dtype=targets.dtype).reshape(2)
            if type(planner_goal_override_action) is not int:
                raise ValueError("planner goal override action must be an integer")
            expected_action = planner_goal_override_action
            if not 0 <= expected_action < 8:
                raise ValueError("planner goal override action must be in [0, 7]")
            if planner_goal_override_traversal and expected_action != 7:
                raise ValueError("planner goal traversal is only valid for action 7")
            _assert_device_condition(self.device,
                torch.isfinite(goal).all() & (goal[0] >= 0.) &
                (goal[0] < self.sim.field_length) & (goal[1] >= 0.) &
                (goal[1] < self.sim.field_width),
                "planner goal override must be a finite in-field XY endpoint")
            targets = targets.clone()
            override_permitted = expected_action != 7 or bool(planner_goal_override_traversal)
            planner_goal_override_mask[:, 3] = (
                (action[:, 3] == expected_action) & override_permitted)
            if planner_goal_override_target_track is not None:
                planner_goal_override_mask[:, 3] &= collection_target_valid
            targets[:, 3] = torch.where(planner_goal_override_mask[:, 3, None],
                                         goal[None], targets[:, 3])
            if planner_goal_override_target_track is not None:
                target_delta = collection_target_xy - goal[None]
                target_distance = target_delta.norm(dim=-1)
                selected_heading = torch.atan2(target_delta[:, 1], target_delta[:, 0])
                selected_heading = torch.where(target_distance > 1e-8,
                    selected_heading, self.policy_pose[:, 3, 2])
                planner_goal_heading = selected_heading[0]
                planner_goal_heading_mask[:, 3] = planner_goal_override_mask[:, 3]
                self._last_planner_collection_goal_heading = selected_heading[0]
            elif planner_goal_heading_override is not None:
                planner_goal_heading = torch.as_tensor(
                    planner_goal_heading_override, device=self.device,
                    dtype=targets.dtype).reshape(())
                _assert_device_condition(self.device,
                    torch.isfinite(planner_goal_heading),
                    "planner goal heading must be finite")
                planner_goal_heading_mask[:, 3] = planner_goal_override_mask[:, 3]
        elif planner_goal_heading_override is not None:
            raise ValueError("planner goal heading requires its paired XY goal")
        if (type(planner_force_goal_replan) is not bool or
                (planner_force_goal_replan and
                 (self.n != 1 or planner_goal_override is None or
                  planner_goal_override_action is None))):
            raise ValueError("forced focal planner refresh requires a selected focal goal")
        if planner_force_goal_replan:
            _assert_device_condition(self.device,planner_goal_override_mask[0,3],
                                     "forced focal planner refresh requires a selected focal goal")
        if planner_traversal_cost_override is None:
            if planner_traversal_cost_override_action is not None:
                raise ValueError("planner traversal cost action requires a cost map")
            planner_traversal_cost = None
            planner_traversal_cost_scale = 1.0
            traversal_signature=None
            traversal_request_changed=False
            self._planner_traversal_request_signature=None
        else:
            if (self.device.type != "cpu" and
                    (not isinstance(planner_traversal_cost_override,torch.Tensor) or
                     planner_traversal_cost_override.device != self.device)):
                raise ValueError("GPU traversal cost map must already reside on the environment device")
            if (self.device.type == "cuda" and
                    torch.cuda.is_current_stream_capturing()):
                raise ValueError(
                    "learned planner traversal overrides require eager decision-boundary validation")
            if (self.n != 1 or self.control_modes[3] != "nn" or
                    self.robot_roles[3] != "offense"):
                raise ValueError(
                    "planner traversal costs are restricted to the single-world neural offense in slot 3")
            if (planner_goal_override is None or
                    type(planner_goal_override_action) is not int or
                    type(planner_traversal_cost_override_action) is not int or
                    planner_traversal_cost_override_action != planner_goal_override_action):
                raise ValueError(
                    "planner traversal cost map must bind to its selected goal action")
            try:
                planner_traversal_cost_scale = float(planner_traversal_cost_override_scale)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "planner traversal cost scale must be finite and nonnegative") from exc
            if not math.isfinite(planner_traversal_cost_scale) or planner_traversal_cost_scale < 0.:
                raise ValueError(
                    "planner traversal cost scale must be finite and nonnegative")
            planner_traversal_cost = torch.as_tensor(
                planner_traversal_cost_override, device=self.device,
                dtype=self.policy_pose.dtype)
            expected_cost_shape = (self.planner.nx, self.planner.ny)
            if planner_traversal_cost.shape != expected_cost_shape:
                raise ValueError(
                    f"planner traversal cost map must have shape {expected_cost_shape}")
            cached = getattr(self, "_validated_planner_traversal_cost", None)
            try:
                version = planner_traversal_cost._version
            except RuntimeError:
                # Inference tensors have no version counter; validate each use.
                version = None
            if not (version is not None and cached is not None and
                    cached[0] is planner_traversal_cost and
                    cached[1] == version and cached[2] == planner_traversal_cost_scale):
                _assert_device_condition(self.device,
                    torch.isfinite(planner_traversal_cost).all() &
                    (planner_traversal_cost >= 0.).all(),
                    "planner traversal cost map must be finite and nonnegative")
                _assert_device_condition(self.device,torch.isfinite(
                    planner_traversal_cost * planner_traversal_cost_scale).all(),
                    "scaled planner traversal cost map must remain finite")
                if version is not None:
                    self._validated_planner_traversal_cost = (
                        planner_traversal_cost, version, planner_traversal_cost_scale)
            # A macro freezes one map tensor and reuses it for every physics
            # tick. Track both request and application separately so a changed
            # map/goal/action triggers once at the decision boundary and a
            # rejected override is not retried on every tick of that macro.
            try:
                traversal_version = planner_traversal_cost._version
            except RuntimeError:
                traversal_version = None
            if planner_goal_override is None:
                traversal_goal=None
            elif isinstance(planner_goal_override, (tuple, list)):
                traversal_goal = tuple(float(value) for value in planner_goal_override)
            else:
                try:
                    goal_version=planner_goal_override._version
                except RuntimeError:
                    goal_version=None
                traversal_goal=("tensor",id(planner_goal_override),goal_version)
            traversal_signature = {
                "map": planner_traversal_cost,
                "version": traversal_version,
                "scale": planner_traversal_cost_scale,
                "goal": traversal_goal,
                "action": planner_traversal_cost_override_action,
            }
            previous_request = self._planner_traversal_request_signature
            traversal_request_changed = (traversal_version is None or
                previous_request is None or
                previous_request["map"] is not planner_traversal_cost or
                previous_request["version"] != traversal_version or
                previous_request["scale"] != planner_traversal_cost_scale or
                previous_request["goal"] != traversal_goal or
                previous_request["action"] != planner_traversal_cost_override_action)
            self._planner_traversal_request_signature=traversal_signature
            if traversal_request_changed:
                self._planner_traversal_pending.fill_(True)
        raster_mask=((self._deterministic_mode_mask &
                      ~self._defense_role_mask)[None] &
                     ((self.sweeping_enabled &
                       ((action<4)|((action==6)&~carrying))) |
                      self._no_track_search_active))
        # AD* routes are consumed only by robots with an active controller.
        # Keep stationary robots' cached paths untouched; when a rollout
        # changes controller assignments, the trainer forces a fresh plan.
        planner_robot_mask=self._nn_mode_mask|self._deterministic_mode_mask
        planner_modes_dirty=self._planner_controller_modes_dirty
        self._planner_controller_modes_dirty=False
        # Replanning cadence is maintained per world. Host scheduler counts
        # remain metadata only; planner eligibility comes from device masks.
        self._planner_tick_aligned=False
        self.planner_ticks.copy_(torch.where(active,self.planner_ticks+1,self.planner_ticks))
        replan=active&((self.planner_ticks%self.replan_interval)==0)
        if planner_modes_dirty:
            replan=active
        flat_active=(replan[:,None]&planner_robot_mask[None]).reshape(-1)
        focal_override=(planner_goal_override_mask[:,3] & replan &
                        planner_robot_mask[3])
        forced_focal=(torch.full_like(active,planner_force_goal_replan) & active)
        traversal_pending=(self._planner_traversal_pending & active &
                           planner_goal_override_mask[:,3])
        flat_focal=(focal_override | forced_focal | traversal_pending)
        flat_active=flat_active.clone()
        flat_active[3::NUM_ROBOTS] |= flat_focal
        start=self.policy_pose[:,:,:2].reshape(-1,2); goal=targets.reshape(-1,2)
        heading=self.policy_pose[:,:,2].reshape(-1)
        # Plan with the intake extended whenever collection is requested,
        # including while the deployment guard keeps it retracted to clear a
        # nearby field element. This makes AD* route the escape safely.
        intake_extended=(self._turret_mask[None] | self._intake_collecting)
        # The intake projects 12in beyond the bumper, so the centered AD*
        # rectangle needs 24in extra total length to cover that forward reach.
        planner_lengths=(self.sim.length + intake_extended.to(self.sim.length.dtype)*.6096)
        lengths=planner_lengths.reshape(-1); widths=self.sim.width.reshape(-1)
        speeds=self.sim.speed.reshape(-1); accel=self.sim.accel.reshape(-1)
        lateral=self.sim.lateral_mu.reshape(-1)
        robot_obstacles=self._perceived_robot_obstacles()
        waypoint_chains=self.raster_sweep_points[None].expand(
            self.n,-1,-1,-1).reshape(-1,self.raster_sweep_points.shape[1],2)

        if self.sweeping_enabled:
            self.planner.plan_waypoints(
                start,waypoint_chains,self._raster_cursor.reshape(-1),heading,lengths,widths,
                lookahead=self._raster_waypoint_lookahead,
                fallback_goal=goal,waypoint_mask=raster_mask.reshape(-1),
                speed=speeds,lateral_friction=lateral,acceleration=accel,
                active_mask=flat_active,robot_obstacles=robot_obstacles,
                static_active_mask=True)
        else:
            self.planner.plan(
                start,goal,heading,lengths,widths,speed=speeds,
                lateral_friction=lateral,acceleration=accel,
                active_mask=flat_active,robot_obstacles=robot_obstacles,
                static_active_mask=True)

        if planner_traversal_cost is not None:
            rows=self.n*NUM_ROBOTS
            selected=flat_active.clone()
            focal=torch.zeros((rows,),device=self.device,dtype=torch.bool)
            focal[3::NUM_ROBOTS]=True
            permitted=(planner_goal_override_mask.reshape(rows) &
                       planner_robot_mask.reshape(rows) &
                       active[:,None].expand(-1,NUM_ROBOTS).reshape(rows))
            selected &= focal & permitted
            if self.sweeping_enabled:
                self.planner.plan_waypoints(
                    start,waypoint_chains,self._raster_cursor.reshape(-1),heading,lengths,widths,
                    lookahead=self._raster_waypoint_lookahead,
                    fallback_goal=goal,waypoint_mask=raster_mask.reshape(-1),
                    speed=speeds,lateral_friction=lateral,acceleration=accel,
                    active_mask=selected,robot_obstacles=robot_obstacles,
                    static_active_mask=True,
                    learned_traversal_cost=planner_traversal_cost,
                    learned_cost_scale=planner_traversal_cost_scale)
            else:
                self.planner.plan(
                    start,goal,heading,lengths,widths,speed=speeds,
                    lateral_friction=lateral,acceleration=accel,
                    active_mask=selected,robot_obstacles=robot_obstacles,
                    static_active_mask=True,
                    learned_traversal_cost=planner_traversal_cost,
                    learned_cost_scale=planner_traversal_cost_scale)
            self._last_applied_planner_traversal_signature=traversal_signature
            selected_world=selected.reshape(self.n,NUM_ROBOTS)[:,3]
            self._planner_traversal_pending &= ~selected_world
        else:
            self._planner_traversal_pending.zero_()
            self._last_applied_planner_traversal_signature=None

        # Retry failed depot routes with static geometry, preserving the retry
        # predicate on device and keeping the planner batch shape fixed.
        depot_fuel=self._last_audit_fuel_targets
        depot_target=torch.zeros((self.n,NUM_ROBOTS),device=self.device,dtype=torch.bool)
        for box in self.field_boxes:
            if box.name.endswith("_depot"):
                depot_target |= ((depot_fuel[...,0]-box.x).abs()<=box.length/2) & (
                    (depot_fuel[...,1]-box.y).abs()<=box.width/2)
        depot_target &= self.last_actions.reshape(self.n,NUM_ROBOTS)<4
        failed=self.planner.last_lengths.reshape(self.n,NUM_ROBOTS)<=2
        far_goal=(goal.reshape(self.n,NUM_ROBOTS,2)-self.policy_pose[:,:,:2]).norm(dim=-1)>1.25
        retry_flat=(depot_target & failed & far_goal &
                    planner_robot_mask[None] & active[:,None]).reshape(-1)
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
        if planner_traversal_cost is not None:
            retry_selected=retry_flat.clone()
            focal=torch.zeros_like(retry_selected)
            focal[3::NUM_ROBOTS]=True
            permitted=(planner_goal_override_mask.reshape(-1) &
                       planner_robot_mask.reshape(-1) &
                       active[:,None].expand(-1,NUM_ROBOTS).reshape(-1))
            retry_selected &= focal & permitted
            if self.sweeping_enabled:
                self.planner.plan_waypoints(
                    start,waypoint_chains,self._raster_cursor.reshape(-1),heading,lengths,widths,
                    lookahead=self._raster_waypoint_lookahead,
                    fallback_goal=goal,waypoint_mask=raster_mask.reshape(-1),
                    speed=speeds,lateral_friction=lateral,acceleration=accel,
                    active_mask=retry_selected,robot_obstacles=None,
                    static_active_mask=True,
                    learned_traversal_cost=planner_traversal_cost,
                    learned_cost_scale=planner_traversal_cost_scale)
            else:
                self.planner.plan(
                    start,goal,heading,lengths,widths,speed=speeds,
                    lateral_friction=lateral,acceleration=accel,
                    active_mask=retry_selected,robot_obstacles=None,
                    static_active_mask=True,
                    learned_traversal_cost=planner_traversal_cost,
                    learned_cost_scale=planner_traversal_cost_scale)
            self._planner_traversal_pending &= ~retry_selected.reshape(self.n,NUM_ROBOTS)[:,3]
        adstar_command=None
        adstar_tangent=None
        if not self.sweeping_enabled and self._has_deterministic_offense:
            adstar_command,adstar_tangent=self.planner.path_reference(
                start,self.sim.velocity[:,:,:2].reshape(-1,2),speeds)
            adstar_tangent=adstar_tangent.reshape(self.n,NUM_ROBOTS,2)
        hub=self.hub_centers[self.team_ids][None]
        hub_delta=hub-self.policy_pose[:,:,:2]
        hub_bearing=torch.atan2(hub_delta[...,1],hub_delta[...,0])
        angle_error=torch.atan2(torch.sin(hub_bearing-self.policy_pose[:,:,2]),
                                torch.cos(hub_bearing-self.policy_pose[:,:,2]))
        deterministic_offense=(self._deterministic_mode_mask &
                                ~self._defense_role_mask)[None]
        robot_hub_active=self.hub_active[:,self.team_ids]
        scoring_dumper=(action==4)&carrying&self._dumper_mask[None]
        score_centered=scoring_dumper&(angle_error.abs()<=math.radians(5.))
        aim_error=torch.where(score_centered,torch.zeros_like(angle_error),angle_error)
        omega=swerve_heading_rate(aim_error,self.sim.omega_limit,self.sim.alpha,
                                  self.sim.velocity[:,:,2],self.dt)
        score_fine_aim=scoring_dumper&(angle_error.abs()<=math.radians(15.))
        fine_error=torch.where(score_centered,torch.zeros_like(angle_error),angle_error)
        fine_omega=(3.*fine_error-1.5*self.sim.velocity[:,:,2]).clamp(
            -self.sim.omega_limit,self.sim.omega_limit)
        fine_omega=torch.where(score_centered,torch.zeros_like(fine_omega),fine_omega)
        omega=torch.where(score_fine_aim,fine_omega,omega)
        omega=torch.where(scoring_dumper,omega,torch.zeros_like(omega))
        from .fuel_physics_runtime import ferry_aim_targets
        home_zone_center=ferry_aim_targets(
            self,self.policy_pose[:,:,:2],self.team_ids[None,:].expand(self.n,-1))
        ferry_vector=home_zone_center-self.policy_pose[:,:,:2]
        ferry_bearing=torch.atan2(ferry_vector[...,1],ferry_vector[...,0])
        ferry_error=torch.atan2(torch.sin(ferry_bearing-self.policy_pose[:,:,2]),
                                torch.cos(ferry_bearing-self.policy_pose[:,:,2]))
        ferrying=(action==6)&carrying&self._dumper_mask[None]
        collect_braking_to_ferry=((action==7)&carrying&self._dumper_mask[None]&
                                  ~robot_hub_active)
        ferry_omega=swerve_heading_rate(ferry_error,self.sim.omega_limit,self.sim.alpha,
                                        self.sim.velocity[:,:,2],self.dt)
        omega=torch.where(ferrying|collect_braking_to_ferry,ferry_omega,omega)
        deterministic_robot=self._deterministic_mode_mask[None]
        deterministic_offense=(self._deterministic_mode_mask &
                                ~self._defense_role_mask)[None]
        pickup_active=(deterministic_offense &
                       ((action<4)|((action==6)&~carrying)))
        pickup_fuel_targets = self._last_audit_fuel_targets
        if planner_goal_override_target_track is not None:
            pickup_fuel_targets = pickup_fuel_targets.clone()
            pickup_fuel_targets[:, 3] = torch.where(
                collection_target_valid[:, None], collection_target_xy,
                pickup_fuel_targets[:, 3])
        pickup_fuel_delta=(pickup_fuel_targets-
                           self.policy_pose[:,:,:2])
        pickup_fuel_distance=pickup_fuel_delta.norm(dim=-1)
        pickup_direction=fuel_delta
        if adstar_tangent is not None:
            target_based_picker=pickup_active & (not self.sweeping_enabled)
            tangent_norm=adstar_tangent.norm(dim=-1,keepdim=True)
            fallback_direction=torch.where((action<4)[...,None],fuel_delta,
                targets-self.policy_pose[:,:,:2])
            target_based_tangent=torch.where(tangent_norm>.5,adstar_tangent,
                fallback_direction)
            # Follow AD* through the route, then aim the intake at its chosen
            # piece early enough to complete the required chassis turn during
            # the remaining approach. A route tangent can run along a field
            # boundary while fuel lies just inside it.
            fuel_bearing=torch.atan2(pickup_fuel_delta[...,1],
                                     pickup_fuel_delta[...,0])
            fuel_turn_error=torch.atan2(
                torch.sin(fuel_bearing-self.policy_pose[:,:,2]),
                torch.cos(fuel_bearing-self.policy_pose[:,:,2])).abs()
            turn_distance=(self.sim.speed*fuel_turn_error /
                           self.sim.omega_limit.clamp_min(.1))
            approach_distance=torch.maximum(torch.full_like(
                pickup_fuel_distance,.9),turn_distance)
            final_intake_approach=(~self._target_collecting &
                                   (pickup_fuel_distance<=approach_distance))
            target_based_tangent=torch.where(
                final_intake_approach[...,None],fuel_delta,target_based_tangent)
            pickup_direction=torch.where(target_based_picker[...,None],
                                         target_based_tangent,pickup_direction)
        if planner_goal_override_target_track is not None:
            pickup_direction = pickup_direction.clone()
            pickup_direction[:, 3] = torch.where(
                collection_target_valid[:, None], selected_target_delta,
                pickup_direction[:, 3])
        # While tracing a cluster, fuel_delta can point toward its inward
        # normal; keep following the route tangent until entering final pickup.
        pickup_bearing=torch.atan2(pickup_direction[...,1],pickup_direction[...,0])
        pickup_error=torch.atan2(torch.sin(pickup_bearing-self.policy_pose[:,:,2]),
                                 torch.cos(pickup_bearing-self.policy_pose[:,:,2]))
        pickup_omega=swerve_heading_rate(pickup_error,self.sim.omega_limit,self.sim.alpha,
                                         self.sim.velocity[:,:,2],self.dt)
        pickup_heading_active = pickup_active | selected_target_pickup_active
        omega=torch.where(pickup_heading_active,pickup_omega,omega)
        trench_alignment=self.planner.last_trench_alignment.reshape(self.n,6)
        trench_heading=self.planner.last_heading.reshape(self.n,6)
        trench_error=torch.atan2(torch.sin(trench_heading-self.policy_pose[:,:,2]),
                                 torch.cos(trench_heading-self.policy_pose[:,:,2]))
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
                available_speed.reshape(-1))
        else:
            command=adstar_command
        command=command.reshape(self.n,6,2)
        command=torch.where((trench_alignment & (trench_error.abs()>.25))[...,None],
                            torch.zeros_like(command),command)
        distance=(targets-self.policy_pose[:,:,:2]).norm(dim=-1,keepdim=True)
        # AD* can return a near-zero lookahead command at short intake staging
        # goals even when the straight chassis path is clear. Bridge that
        # final approach for every deterministic collector, not only audit
        # probes; retain planner routing whenever the direct path is blocked.
        offense_ids=self._deterministic_offense_robot_index
        offense_targets=targets.index_select(1,offense_ids)
        offense_pose=self.policy_pose.index_select(1,offense_ids)
        offense_distance=distance.index_select(1,offense_ids)
        offense_command=command.index_select(1,offense_ids)
        direct=(offense_targets-offense_pose[:,:,:2])
        direct=direct/direct.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        direct_speed=torch.minimum(self.sim.speed.index_select(1,offense_ids),
                                   offense_distance[:,:,0]*6.)
        direct_command=direct*direct_speed[...,None]
        direct_path=torch.stack((offense_pose[...,:2],offense_targets),dim=2)
        direct_clear=self.planner._footprint_path_clear(
            direct_path.reshape(-1,2,2).contiguous(),
            offense_pose[...,2].reshape(-1).contiguous(),
            planner_lengths.index_select(1,offense_ids).reshape(-1).contiguous(),
            self.sim.width.index_select(1,offense_ids).reshape(-1).contiguous()
        ).reshape(self.n,-1)
        stalled_collector=(pickup_active.index_select(1,offense_ids) &
            ~trench_alignment.index_select(1,offense_ids) &
            (offense_distance[:,:,0]>.03) &
            ((offense_distance[:,:,0]<1.6) |
             (offense_command.norm(dim=-1)<.05)) & direct_clear)
        command.index_copy_(1,offense_ids,torch.where(
            stalled_collector[...,None],direct_command,offense_command))
        # Rotate into the AD* travel tangent before taking the last step to a
        # fuel target; this keeps lateral chassis motion out of the intake
        # approach while preserving efficient swerve travel between targets.
        chassis_forward=torch.stack((torch.cos(self.policy_pose[:,:,2]),
                                     torch.sin(self.policy_pose[:,:,2])),-1)
        chassis_right=torch.stack((torch.sin(self.policy_pose[:,:,2]),
                                   -torch.cos(self.policy_pose[:,:,2])),-1)
        forward_speed=(command*chassis_forward).sum(-1,keepdim=True)
        nonholonomic_command=chassis_forward*forward_speed
        lateral_velocity=(self.sim.velocity[:,:,:2]*chassis_right).sum(-1)
        nonholonomic_pickup=(deterministic_offense & pickup_active &
            (self.sweeping_enabled | (pickup_fuel_distance<=.7)))
        command=torch.where(nonholonomic_pickup[...,None],
                            nonholonomic_command,command)
        pickup_turning=(nonholonomic_pickup &
            ((pickup_error.abs()>.25)|(lateral_velocity.abs()>.15)))
        command=torch.where(pickup_turning[...,None],torch.zeros_like(command),command)
        deterministic_scorer = ((self._deterministic_mode_mask &
                                 ~self._defense_role_mask)[None] &
                                (action == 4) & carrying)
        score_zone_x=self.policy_pose[:,:,0]
        in_score_zone=torch.where(self.team_ids[None,:]==0,
            score_zone_x<=self.alliance_zone_depth,
            score_zone_x>=self.field_length-self.alliance_zone_depth)
        score_staging=(deterministic_scorer & in_score_zone &
                       self._robots_clear_of_bumps(self.policy_pose[:,:,:2]))
        # The scoring gate accepts any stationary, bump-clear pose in the
        # alliance zone. Stop at the first such pose along the AD* route so a
        # detour around a hub bump does not continue back across the zone.
        command=torch.where(score_staging[...,None],torch.zeros_like(command),command)
        deterministic_picker = pickup_active
        # Raster spline targets stay ahead of the robot, so collectors keep
        # following through waypoint arrival instead of stopping.
        stop_radius = torch.where(deterministic_scorer, .02,
                                  torch.where(deterministic_picker, 0., .3))
        stop_radius = torch.where(planner_goal_heading_mask, .02, stop_radius)
        command=torch.where(distance<=stop_radius[...,None],torch.zeros_like(command),command)
        # Action 7 is native hold when no explicit strategic goal was supplied.
        # The MECO planner adapter may pair action 7 with a validated target to
        # traverse the existing guarded AD* route, without changing ordinary
        # native hold semantics.
        hold_action7=(action==7)&~planner_goal_override_mask
        command=torch.where(hold_action7[...,None],torch.zeros_like(command),command)
        trench_route_length=self.planner.last_lengths.reshape(self.n,6)
        trench_route_goal=self.planner.last_goal.reshape(self.n,6,2)
        trench_forward=torch.stack((trench_heading.cos(),trench_heading.sin()),-1)
        trench_remaining=((trench_route_goal-self.policy_pose[:,:,:2]) *
                          trench_forward).sum(-1).clamp_min(0.)
        trench_cross_speed=torch.minimum(self.sim.speed,
                                          trench_remaining*6.)
        trench_cross_command=trench_forward*trench_cross_speed[...,None]
        trench_crossing=(trench_alignment & (trench_error.abs()<=.25) &
                         (trench_route_length>1))
        command=torch.where(trench_crossing[...,None],trench_cross_command,command)
        # Keep following the route while the chassis turns to align the
        # intake; waiting for perfect alignment stalls starts and turnarounds.
        command=self._avoid_robot_contention(command,targets,active)
        # Contention avoidance may add a sidestep, but trench alignment is a
        # stationary rotation maneuver. Keep translation stopped until the
        # chassis is aligned so the sidestep cannot push it into a trench arm.
        command=torch.where((trench_alignment & (trench_error.abs()>.25))[...,None],
                            torch.zeros_like(command),command)
        if planner_goal_heading is not None:
            at_heading_goal = (planner_goal_heading_mask[:, 3] &
                               (distance[:, 3, 0] <= .04))
            # Before rotating, check the whole turn against static field
            # geometry at 22.5-degree samples and keep a conservative
            # circumscribed-radius clearance from observed robots.
            current_heading = self.policy_pose[:, 3, 2]
            heading_delta = torch.atan2(
                torch.sin(planner_goal_heading - current_heading),
                torch.cos(planner_goal_heading - current_heading))
            fractions = torch.linspace(0., 1., 37, device=self.device,
                                      dtype=self.policy_pose.dtype)
            sweep_headings = current_heading[0] + heading_delta[0] * fractions
            sweep_lengths = planner_lengths[0, 3].expand_as(sweep_headings)
            sweep_widths = self.sim.width[0, 3].expand_as(sweep_headings)
            blocked = self.planner._blocked(sweep_headings, sweep_lengths, sweep_widths)
            goal_ix, goal_iy = self.planner._indices(targets[0, 3:4])
            goal_cell = (goal_ix * self.planner.ny + goal_iy).reshape(1, 1)
            static_clear = ~blocked.flatten(1).gather(
                1, goal_cell.expand(blocked.shape[0], 1)).any()
            # Robot 3 is on the blue alliance; use its three same-team robot
            # slots (including itself, which is marked invalid below).
            known_ids = torch.stack(self._team_robot_ids)[self.team_ids[3]]
            known_xy = self.policy_pose[0, known_ids, :2]
            known_valid = known_ids != 3
            known_radius = .5 * torch.sqrt(
                self.sim.length[0, known_ids].square() +
                self.sim.width[0, known_ids].square())
            known_velocity = self.sim.velocity[0, known_ids, :2]
            opp_xy = self.opponent_pose[0, 3, :2].unsqueeze(0)
            opp_size = self.opponent_size[0, 3, :2]
            opp_valid = (self.opponent_valid[0, 3] &
                         torch.isfinite(opp_xy[0]).all() &
                         torch.isfinite(opp_size).all() & (opp_size > 0.).all())
            opp_radius = .5 * torch.sqrt(opp_size.square().sum())
            opp_velocity = self.opponent_velocity[0, 3, :2].unsqueeze(0)
            obstacle_xy = torch.cat((known_xy, opp_xy), dim=0)
            obstacle_valid = torch.cat((known_valid, opp_valid.reshape(1)))
            obstacle_radius = torch.cat((known_radius, opp_radius.reshape(1)))
            obstacle_velocity = torch.cat((known_velocity, opp_velocity), dim=0)
            # Wait for chassis translation to settle before turning. Then bound
            # the remaining stopping drift and each observed obstacle's motion
            # over the full commanded rotation duration.
            chassis_speed = self.sim.velocity[0, 3, :2].norm()
            chassis_accel = self.sim.accel[0, 3].clamp_min(.1)
            stopped = chassis_speed <= .05
            stopping_drift = chassis_speed.square() / (2. * chassis_accel) + chassis_speed * self.dt
            turn_angle = heading_delta[0].abs()
            turn_omega = self.sim.omega_limit[0, 3].clamp_min(.1)
            turn_alpha = self.sim.alpha[0, 3].clamp_min(.1)
            accel_turn_time = 2. * torch.sqrt(turn_angle / turn_alpha)
            cruise_turn_time = turn_angle / turn_omega + turn_omega / turn_alpha
            turn_seconds = torch.where(turn_angle <= turn_omega.square() / turn_alpha,
                                       accel_turn_time, cruise_turn_time) + .1
            relative_xy = obstacle_xy - targets[0, 3]
            relative_velocity = obstacle_velocity
            closest_time = (-(relative_xy * relative_velocity).sum(-1) /
                            relative_velocity.square().sum(-1).clamp_min(1e-8))
            closest_time = torch.minimum(closest_time.clamp_min(0.), turn_seconds)
            closest_xy = relative_xy + relative_velocity * closest_time[:, None]
            robot_radius = .5 * torch.sqrt(
                self.sim.length[0, 3].square() + self.sim.width[0, 3].square())
            clearance = (closest_xy.norm(dim=-1) >
                         robot_radius + obstacle_radius + stopping_drift + .08)
            dynamic_clear = clearance | ~obstacle_valid
            can_align = static_clear & dynamic_clear.all() & stopped
            # Pause at the endpoint if the turn is currently obstructed. The
            # native physics collision and route guards remain authoritative.
            command[:, 3] = torch.where(at_heading_goal[:, None],
                                        torch.zeros_like(command[:, 3]), command[:, 3])
            target_omega = swerve_heading_rate(
                heading_delta, self.sim.omega_limit[:, 3], self.sim.alpha[:, 3],
                self.sim.velocity[:, 3, 2], self.dt)
            omega[:, 3] = torch.where(
                at_heading_goal & can_align, target_omega,
                torch.where(at_heading_goal, torch.zeros_like(target_omega), omega[:, 3]))
        commands=torch.cat((command,omega[...,None]),-1)
        enabled=self._controlled_mode_mask
        commands=commands*enabled[None,:,None]
        prior_velocity = self.sim.velocity.clone()
        stationary_pose=(self.sim.pose.clone() if "none" in self.control_modes else None)
        if hasattr(self, "_curriculum_commands"):
            commands = self._curriculum_commands(commands, active)
        advance_with_fuel(self,commands,None if full_batch else active)
        self._advance_policy_pose_odometry(prior_velocity, active)
        if stationary_pose is not None:
            stationary=active[:,None]&~enabled[None,:]
            self.sim.pose.copy_(torch.where(
                stationary[...,None],stationary_pose,self.sim.pose))
            self.sim.velocity.copy_(torch.where(
                stationary[...,None],torch.zeros_like(self.sim.velocity),
                self.sim.velocity))
        self._finish_raster_cursor(active,raster_mask)
        self.steps+=active.long()
        self._update_match(active)
        self._update_fuel(active,score_intent=self._last_score_intent)
        if self.perception_interval == 1:
            self._update_perception(active,active_count=active_count)
        elif full_batch and self._full_batch_step_scalar is not None:
            if self._full_batch_step_scalar % self.perception_interval == 0:
                self._update_perception(active,active_count=self.n)
            else:
                from . import tensor_perception
                fused_aging=tensor_perception.age_tracks_3v3(
                    self.track_pos,self.track_vel,self.track_age,self.track_mask,
                    self.opponent_age,self.opponent_valid,active,self.dt,
                    self.perception_track_timeout)
                if not fused_aging:
                    self.track_age.copy_(torch.where(
                        active[:, None, None] & self.track_mask,
                        self.track_age + self.dt, self.track_age))
                    self.track_pos.add_(torch.where(
                        (active[:, None, None] & self.track_mask)[..., None],
                        self.track_vel * self.dt,
                        torch.zeros_like(self.track_vel)))
                    self.track_mask &= ~((self.track_age > self.perception_track_timeout) &
                                         active[:, None, None])
                    self.opponent_age.copy_(torch.where(
                        active[:, None] & self.opponent_valid,
                        self.opponent_age + self.dt, self.opponent_age))
                    self.opponent_valid &= ~((self.opponent_age > 1.) & active[:, None])
        else:
            perception_tick = (self.steps % self.perception_interval) == 0
            update_active = active & perception_tick
            aging_active = active & ~perception_tick
            self.track_age.copy_(torch.where(
                    aging_active[:, None, None] & self.track_mask,
                    self.track_age + self.dt, self.track_age))
            self.track_pos.add_(torch.where(
                    (aging_active[:, None, None] & self.track_mask)[..., None],
                    self.track_vel * self.dt, torch.zeros_like(self.track_vel)))
            self.track_mask &= ~((self.track_age > self.perception_track_timeout) &
                                     aging_active[:, None, None])
            self.opponent_age.copy_(torch.where(
                    aging_active[:, None] & self.opponent_valid,
                    self.opponent_age + self.dt, self.opponent_age))
            self.opponent_valid &= ~((self.opponent_age > 1.) & aging_active[:, None])
            self._update_perception(update_active)
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
