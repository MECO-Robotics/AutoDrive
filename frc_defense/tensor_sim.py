"""Fully tensorized, device-resident FRC defense simulator (optional PyTorch).

All per-world state and hot-path calculations stay on ``device``. This model
uses wheel-level swerve forces and motor/current limits. Bump traversal adds a
quasi-3D triangular height profile, chassis pitch/load transfer, and gravity on
the ramp grade. Suspension, chassis roll/bounce, wheel lift, CAN timing, and a
full electrical network are not resolved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

from .field import midfield_respawn_points
from .observation import normalize_tensor_observation_batch_in_place
from .tensor_physics import (TensorState, TensorSwerveParameters,
                             TensorVectorizedSimulator, swerve_heading_rate)
from .tensor_defense_gamepieces import TensorDefenseGamepieceMixin
from .tensor_defense_observation import TensorDefenseObservationMixin
from .tensor_defense_opponent import TensorDefenseOpponentMixin
from .tensor_defense_reset import TensorDefenseResetMixin
from .tensor_defense_step import TensorDefenseStepMixin

try:
    import torch
except ImportError:  # Keep package imports usable without the training extra.
    torch = None

if torch is not None:
    from . import tensor_fuel_candidate_rank as _fuel_candidate_rank_hip
    from .tensor_strategic_observation_proto import (
        fused_candidate_local_available as _fused_strategic_observation_available,
        pack_fused_candidate_local as _pack_fused_strategic_observation,
    )
    from .tensor_strategic_observation_full_proto import (
        integrated_available as _full_strategic_observation_available,
        observe as _full_strategic_observation,
    )
    from .tensor_perception import (FUSED_PERCEPTION_HIP_ENABLED,
                                    piece_occlusion_mask, visibility_mask)
    from . import tensor_perception_commit_proto as _perception_commit_proto
else:
    _fuel_candidate_rank_hip = None
    _fused_strategic_observation_available = None
    _pack_fused_strategic_observation = None
    _full_strategic_observation_available = None
    _full_strategic_observation = None
    piece_occlusion_mask = None
    visibility_mask = None
    _perception_commit_proto = None
    FUSED_PERCEPTION_HIP_ENABLED = False


def _require_torch():
    if torch is None:
        raise RuntimeError("Tensor simulation requires PyTorch; install torch first")


@dataclass(frozen=True)
class PerceptionState:
    own_pose: Any
    own_velocity: Any
    own_possession: Any
    capacity: int
    opponent_pose: Any
    opponent_velocity: Any
    opponent_valid: Any
    opponent_track_age: Any
    fuel_position: Any
    fuel_velocity: Any
    fuel_mask: Any
    fuel_track_age: Any
    hub_active: Any
    hub_centers: Any
    match_elapsed: Any


class TensorDefenseEnv(
        TensorDefenseObservationMixin, TensorDefenseResetMixin, TensorDefenseGamepieceMixin, TensorDefenseStepMixin,
        TensorDefenseOpponentMixin):
    """Vectorized FRC REBUILT offense/defense environment."""
    def __init__(self,num_envs=1,task="counter_defense",device="cuda",seed=0,opponent="guard",**kwargs):
        _require_torch()
        if task not in ("counter_defense","defense"): raise ValueError("task must be counter_defense or defense")
        # Training/evaluation can load the exact robot configuration without
        # changing the tensor hot path. A hardware config defaults to nominal
        # physics; explicitly set "randomize": true for sim-to-real variation.
        config_path = os.environ.get("FRC_DRIVETRAIN_CONFIG")
        drivetrain_config = None
        if config_path:
            config = json.loads(Path(config_path).read_text())
            drivetrain_config = config
            kwargs.setdefault("randomize", config.get("randomize", False))
            for name in ("robot_length", "robot_width", "mass", "max_speed",
                         "max_acceleration", "max_omega", "max_alpha",
                         "field_friction", "lateral_friction"):
                if name in config:
                    kwargs.setdefault(name, config[name])
            kwargs.setdefault("swerve", TensorSwerveParameters(**config.get("swerve", {})))
            if "gamepieces" in config:
                kwargs.setdefault("gamepiece_config", config["gamepieces"])
        self.n,self.device,self.task,self.opponent=int(num_envs),torch.device(device),task,opponent
        self.action_mode=str(kwargs.pop("action_mode","direct")).lower()
        mode_default=(opponent.get("mode","INTERCEPT") if isinstance(opponent,dict)
                      and task=="counter_defense" else
                      "INTERCEPT" if task=="counter_defense" else None)
        selected_mode=kwargs.pop("defense_mode",mode_default)
        self.defense_mode=None if selected_mode is None else str(selected_mode).upper()
        if self.defense_mode is not None and self.defense_mode not in (
                "INTERCEPT","LANE_BLOCK","FUEL_DENIAL","SHADOW","HUB_GUARD"):
            raise ValueError("defense_mode must be INTERCEPT, LANE_BLOCK, FUEL_DENIAL, SHADOW, or HUB_GUARD")
        if self.action_mode not in ("direct","tactical","strategic"):
            raise ValueError("action_mode must be direct, tactical, or strategic")
        self.action_dim={"direct":3,"tactical":2,"strategic":8}[self.action_mode]
        self.obs_dim={"direct":35,"tactical":38,"strategic":137}[self.action_mode]
        self.learned_opponent_fn=kwargs.pop("learned_opponent_fn",None)
        self.reuse_strategic_opponent_candidates=bool(
            kwargs.pop("reuse_strategic_opponent_candidates",True))
        self.reuse_strategic_own_candidates=bool(
            kwargs.pop("reuse_strategic_own_candidates",False))
        self._pending_strategic_own_candidates=None
        self.squared_fuel_candidate_distance=bool(
            kwargs.pop("squared_fuel_candidate_distance",False))
        self.skip_strategic_offense_metric_objective=bool(
            kwargs.pop("skip_strategic_offense_metric_objective",True))
        gamepiece_config=kwargs.pop("gamepiece_config",{})
        if not isinstance(gamepiece_config,dict):
            raise ValueError("gamepiece_config must be a mapping")
        # These are configurable robot assumptions, not REBUILT regulation limits.
        self.fuel_capacity=max(0,int(kwargs.pop("max_fuel_capacity",gamepiece_config.get("max_capacity",8))))
        self.intake_interval=float(kwargs.pop("intake_interval",gamepiece_config.get("intake_interval_s",.20)))
        self.score_interval=float(kwargs.pop("score_interval",gamepiece_config.get("score_interval_s",.10)))
        if self.intake_interval<0 or self.score_interval<0:
            raise ValueError("gamepiece intervals must be nonnegative")
        self.fuel_count=int(kwargs.pop("fuel_count",504))
        preload_setting=kwargs.pop("preloaded_per_robot",None)
        self.preloaded_per_robot=None if preload_setting is None else int(preload_setting)
        if not 96+6*(self.preloaded_per_robot or 0) <= self.fuel_count <= 600:
            raise ValueError("fuel_count must be 96 + 6*preloaded_per_robot through 600")
        if self.preloaded_per_robot is not None and not 0<=self.preloaded_per_robot<=8:
            raise ValueError("preloaded_per_robot must be between 0 and 8")
        self.dt=float(kwargs.pop("dt",.02)); self.horizon=int(kwargs.pop("horizon",8000)); self.base_goal_radius=float(kwargs.pop("goal_radius",.5))
        if self.action_mode=="strategic":
            self.horizon=max(self.horizon,math.ceil(160./self.dt))
        self.match_clock_step=self.dt
        self.adstar_replan_interval=max(1,int(kwargs.pop("adstar_replan_interval",20)))
        self._planner_tick=torch.zeros((self.n,),device=self.device,dtype=torch.long)
        self._planner_tick_scalar=self.adstar_replan_interval-1
        self._planner_tick_aligned=False
        self._planner_full_batch_active=False
        self.adstar_spawn_hint=bool(kwargs.pop("adstar_spawn_hint",True))
        self.randomize=bool(kwargs.pop("randomize",True))
        self.static_opponent_fraction=float(kwargs.pop("static_opponent_fraction",0.))
        if not 0. <= self.static_opponent_fraction <= 1.:
            raise ValueError("static_opponent_fraction must be between 0 and 1")
        self.normalize_observations=bool(kwargs.pop("normalize_observations",True))
        self.observation_noise=float(kwargs.pop("observation_noise",.01))
        self.observation_dropout=float(kwargs.pop("observation_dropout",.02))
        perception=kwargs.pop("perception_config",{}) or {}
        if not isinstance(perception,dict):
            raise ValueError("perception_config must be a mapping")
        self.perception_fov=float(perception.get("fov_degrees",360.))
        self.perception_range=float(perception.get("range_m",math.hypot(
            float(kwargs.get("field_length",16.54)),float(kwargs.get("field_width",8.07)))))
        self.perception_dropout=float(perception.get("detection_dropout",.02))
        self.perception_position_noise=float(perception.get("position_noise_m",.015))
        self.perception_velocity_noise=float(perception.get("velocity_noise_mps",.05))
        self.perception_track_timeout=float(perception.get("track_timeout_s",1.0))
        if not 0. < self.perception_fov <= 360. or self.perception_range <= 0.:
            raise ValueError("perception FOV and range must be positive (FOV <= 360)")
        if not 0. <= self.perception_dropout <= 1. or self.perception_track_timeout < 0.:
            raise ValueError("invalid perception dropout or track timeout")
        self.max_control_latency_steps=max(0,int(kwargs.pop("max_control_latency_steps",6)))
        self.max_observation_latency_steps=max(0,int(kwargs.pop("max_observation_latency_steps",8)))
        field_layout=kwargs.pop("field_layout","2026_rebuilt")
        field_colliders=kwargs.pop("field_colliders",None)
        self.field_boxes=()
        self.field_feature_boxes=()
        if field_colliders is None and field_layout=="2026_rebuilt":
            from .field import bump_boxes, rebuilt_field, static_collision_boxes
            self.field_boxes=rebuilt_field()
            solids=static_collision_boxes(self.field_boxes)
            self.field_feature_boxes=solids+bump_boxes(self.field_boxes)
            field_colliders=[box.as_tensor() for box in solids]
            kwargs.setdefault("bump_regions",[box.as_tensor() for box in bump_boxes(self.field_boxes)])
            kwargs.setdefault("field_length",16.54); kwargs.setdefault("field_width",8.07)
        elif field_colliders is not None:
            field_colliders=list(field_colliders)
        self.sim=TensorVectorizedSimulator(self.n,device,seed,dt=self.dt,obstacles=kwargs.pop("obstacles",()),
            field_colliders=field_colliders or (),**kwargs)
        self.sim.randomize=self.randomize
        if drivetrain_config is None:
            drivetrain_config = {"name": "built-in illustrative defaults",
                "randomize": self.randomize,
                "mass": 55.0, "robot_length": .9, "robot_width": .9,
                "max_speed": 4.5, "max_acceleration": 8.0,
                "max_omega": 8.0, "max_alpha": 18.0,
                "swerve": asdict(self.sim.swerve)}
        self.drivetrain_config = drivetrain_config
        self.device=self.sim.device
        self._full_observation_hip_available=False
        self._ppo_sparse_delayed_observation_capture=False
        self._ppo_observation_delay_groups=None
        self._adstar_planners=None
        self._adstar_defender_planners=None
        self._adstar_tactical_planner=None
        self._adstar_contact_latched=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        opponent_kind=opponent.get("value","") if isinstance(opponent,dict) else opponent
        if self.task=="defense" and (self.action_mode=="strategic" or
                isinstance(opponent_kind,str) and opponent_kind.lower() in ("adstar","offense","guard")):
            from .tensor_adstar import TensorADStar
            self._adstar_planners=TensorADStar(self)
        elif (self.task=="counter_defense" and (self.action_mode=="strategic" or
              isinstance(opponent_kind,str) and opponent_kind.lower() in ("guard", "adstar_defender",
                  "intercept","lane_block","fuel_denial","shadow","hub_guard","mirror",
                  "cutoff","velocity_intercept"))):
            from .tensor_adstar import TensorADStar
            self._adstar_defender_planners=TensorADStar(self,avoid_bumps=True)
        if self.action_mode in ("tactical","strategic") or (self.task=="defense" and self.defense_mode):
            from .tensor_adstar import TensorADStar
            self._adstar_tactical_planner=TensorADStar(self,avoid_bumps=True)
        if self.field_feature_boxes:
            from .field import box_observation_radius
            feature_values=[(b.x,b.y,box_observation_radius(b)) for b in self.field_feature_boxes]
            self.field_feature_obstacles=torch.tensor(feature_values,device=self.device,dtype=torch.float32)
        else:
            self.field_feature_obstacles=torch.empty((0,3),device=self.device)
        self.generator=torch.Generator(device=self.device).manual_seed(int(seed or 0)); self.steps=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.match_elapsed=torch.zeros(self.n,device=self.device)
        self.match_remaining=torch.full((self.n,),160.,device=self.device)
        self.hub_active=torch.ones((self.n,2),device=self.device,dtype=torch.bool)
        self.hub_centers=self._game_hub_centers()
        self.hub_inactive_first=torch.randint(2,(self.n,),device=self.device,generator=self.generator)
        self.auto_fuel_scores=torch.zeros((self.n,2),device=self.device,dtype=torch.long)
        self.piece_pos=torch.zeros((self.n,self.fuel_count,2),device=self.device)
        self.piece_vel=torch.zeros_like(self.piece_pos)
        self.piece_active=torch.zeros((self.n,self.fuel_count),device=self.device,dtype=torch.bool)
        self.piece_owner=torch.full((self.n,self.fuel_count),-1,device=self.device,dtype=torch.long)
        self.piece_zone=torch.full((self.n,self.fuel_count),-1,device=self.device,dtype=torch.long)
        self.piece_type=torch.full((self.n,self.fuel_count),-1,device=self.device,dtype=torch.long)
        self._midfield_respawn_positions=torch.tensor(
            midfield_respawn_points(self.fuel_count,self.sim.field_length,self.sim.field_width),
            device=self.device,dtype=self.piece_pos.dtype)
        # Per-robot tracks are the only FUEL representation exposed to a
        # controller. Simulator truth remains in piece_* for physics/scoring.
        self._track_pos=torch.zeros((self.n,2,self.fuel_count,2),device=self.device)
        self._track_vel=torch.zeros_like(self._track_pos)
        self._track_mask=torch.zeros((self.n,2,self.fuel_count),device=self.device,dtype=torch.bool)
        self._track_age=torch.full((self.n,2,self.fuel_count),float("inf"),device=self.device)
        self._opponent_track_pose=torch.zeros((self.n,2,3),device=self.device)
        self._opponent_track_velocity=torch.zeros_like(self._opponent_track_pose)
        self._opponent_track_valid=torch.zeros((self.n,2),device=self.device,dtype=torch.bool)
        self._opponent_track_age=torch.full((self.n,2),float("inf"),device=self.device)
        self.preloads_per_robot=torch.zeros((self.n,),device=self.device,dtype=torch.long)
        self.fuel_acquisition_count=torch.zeros((self.n,2),device=self.device,dtype=torch.long)
        self.fuel_score_count=torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_denied_count=torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_abandoned_count=torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_acquired_event=torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_scored_event=torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_denied_event=torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_abandoned_event=torch.zeros_like(self.fuel_acquisition_count)
        self.next_intake_time=torch.zeros((self.n,2),device=self.device)
        self.next_score_time=torch.zeros_like(self.next_intake_time)
        self._last_hub_zone=torch.zeros((self.n,2),device=self.device,dtype=torch.bool)
        self._last_strategic_action=torch.zeros((self.n,),device=self.device,dtype=torch.long)
        self._last_learned_opponent_action=torch.zeros((self.n,),device=self.device,dtype=torch.long)
        self.static_opponent_mask=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.goal=torch.zeros((self.n,2),device=self.device); self.goal_radius=torch.full((self.n,),self.base_goal_radius,device=self.device); self.previous=torch.zeros(self.n,device=self.device)
        self.last_contact=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.control_delay=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.observation_delay=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.action_history=torch.zeros((self.n,self.max_control_latency_steps+1,3),device=self.device)
        self.observation_history=torch.zeros((self.n,self.max_observation_latency_steps+1,self.obs_dim),device=self.device)
        # Per-world slot containing the newest raw observation. Keeping the
        # history circular avoids copying the full latency window at every
        # physics tick; inactive rows keep both their cursor and slots.
        self._observation_history_index=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self._observation_world_index=torch.arange(self.n,device=self.device)
        self._last_observation=torch.zeros((self.n,self.obs_dim),device=self.device)
        self.initial_distance=torch.zeros(self.n,device=self.device)
        self.path_length=torch.zeros(self.n,device=self.device)
        self.contact_steps=torch.zeros(self.n,device=self.device)
        self.contact_count=torch.zeros(self.n,device=self.device)
        self.last_static_contact=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.blocked_steps=torch.zeros(self.n,device=self.device)
        self.out_of_bounds_steps=torch.zeros(self.n,device=self.device)
        self.last_action=torch.zeros((self.n,3),device=self.device)

    def _game_hub_centers(self):
        """Official REBUILT HUB centers, sourced from field.py geometry anchors."""
        by_name={getattr(box,"name",""):box for box in self.field_boxes}
        if "red_hub" in by_name and "blue_hub" in by_name:
            points=[(by_name[name].x,by_name[name].y) for name in ("red_hub","blue_hub")]
        else:
            x=4.03+1.19/2
            points=[(x,self.sim.field_width/2),(self.sim.field_length-x,self.sim.field_width/2)]
        return torch.tensor(points,device=self.device,dtype=self.sim.pose.dtype)

    def _attacker_objective(self):
        robot=0 if self.task=="counter_defense" else 1
        hub=self.hub_centers[robot].expand(self.n,-1)
        position=self.sim.pose[:,robot,:2]
        direction=position-hub
        direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        radius=.595+.5*torch.sqrt(self.sim.length[:,robot].square()+
            self.sim.width[:,robot].square())+.02
        approach=hub+direction*radius[:,None]
        points,_,visible=self._perceived_fuel(robot)
        distance=(points-position[:,None,:]).norm(dim=-1).masked_fill(~visible,float("inf"))
        nearest,index=distance.min(-1)
        piece=points[torch.arange(self.n,device=self.device),index]
        carrying=self._own_possession(robot)>0
        score_target=carrying&self.hub_active[:,robot]
        return torch.where(score_target[:,None],approach,
            torch.where(torch.isfinite(nearest)[:,None],piece,approach))

    def _own_possession(self, robot):
        """Exact onboard possession estimate for the observing robot only."""
        return (self.piece_active & (self.piece_owner == int(robot))).sum(-1)
















    def _opponent_velocity_command(self,velocity,active_mask=None):
        command=torch.where(self.static_opponent_mask[:,None],torch.zeros_like(velocity),velocity)
        if active_mask is None:
            self._last_opponent_command=command
        else:
            old=getattr(self,"_last_opponent_command",torch.zeros_like(command))
            self._last_opponent_command=torch.where(active_mask[:,None],command,old)
        return command

    def _decode_learned_strategic_action(self, learned, action_mask):
        learned=torch.nan_to_num(torch.as_tensor(
            learned,device=self.device,dtype=self.sim.pose.dtype))
        if learned.ndim == 1 and learned.shape[0] == self.n:
            legacy_ids = int(getattr(self, "learned_opponent_action_dim", 8)) == 3
            if legacy_ids:
                legacy = learned.long().clamp(0, 2)
                action_class = (torch.where(legacy == 0, torch.zeros_like(legacy),
                    torch.where(legacy == 1, torch.full_like(legacy, 4), torch.full_like(legacy, 6)))
                    if self.task == "defense" else
                    torch.where(legacy == 0, torch.full_like(legacy, 5),
                        torch.where(legacy == 1, torch.ones_like(legacy), torch.zeros_like(legacy))))
            else:
                action_class = learned.long().clamp(0, 7)
        elif learned.ndim == 1 and self.n == 1 and learned.shape[0] in (3, 8):
            learned = learned[None, :]
        if learned.shape == (self.n, 8):
            action_class=learned.masked_fill(
                ~action_mask,torch.finfo(learned.dtype).min).argmax(-1)
        elif learned.shape == (self.n, 3):
            # Keep semantic compatibility for genuine three-action policies.
            legacy=learned.argmax(-1)
            if self.task=="defense":
                action_class=torch.where(legacy==0,torch.zeros_like(legacy),
                    torch.where(legacy==1,torch.full_like(legacy,4),torch.full_like(legacy,6)))
            else:
                action_class=torch.where(legacy==0,torch.full_like(legacy,5),
                    torch.where(legacy==1,torch.ones_like(legacy),torch.zeros_like(legacy)))
        elif not (learned.ndim == 1 and learned.shape[0] == self.n):
            raise ValueError(f"learned strategic opponent must return 3/8 scores or {(self.n,)} classes")
        return action_class
