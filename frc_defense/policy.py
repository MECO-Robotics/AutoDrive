"""Adapter from compact PPO observation/action vectors to runtime contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .types import ChassisCommand, Objective, Obstacle, RobotParameters, WorldState
from .observation import normalize_numpy_observation_in_place


def _obstacle_features(state: WorldState, x: float, y: float) -> np.ndarray:
    obstacles=list(state.obstacles)
    if abs(state.field_length-16.54)<.05 and abs(state.field_width-8.07)<.05:
        from .field import box_observation_radius, bump_boxes, rebuilt_field, static_collision_boxes
        boxes=rebuilt_field(state.field_length,state.field_width)
        boxes=static_collision_boxes(boxes)+bump_boxes(boxes)
        obstacles.extend(Obstacle(b.x,b.y,box_observation_radius(b))
                         for b in boxes)
    nearby=sorted(obstacles,key=lambda o:(o.x-x)**2+(o.y-y)**2)[:4]
    features=np.zeros((4,3),np.float32)
    for i,obstacle in enumerate(nearby): features[i]=(obstacle.x,obstacle.y,obstacle.radius)
    return features.ravel()


class PPOPolicy:
    """Wrap an SB3-compatible model without importing its training dependencies."""
    def __init__(self, model, normalize_observations: bool = True):
        self.model = model
        self.normalize_observations = normalize_observations

    def predict(self, state: WorldState, params: RobotParameters, goal: Objective) -> ChassisCommand:
        own, opp = state.own, state.opponent
        opponent_params = state.opponent_params or params
        obs = np.asarray([own.x, own.y, own.theta, own.vx, own.vy, own.omega,
                          opp.x, opp.y, opp.theta, opp.vx, opp.vy, opp.omega,
                          goal.x-own.x, goal.y-own.y, goal.radius, state.field_length, state.field_width,
                          params.length, opponent_params.length, params.width, opponent_params.width], dtype=np.float32)
        obs = np.concatenate((obs, np.asarray([params.max_acceleration,
                                               opponent_params.max_acceleration], dtype=np.float32)/10.))
        obs = np.concatenate([obs, _obstacle_features(state,own.x,own.y)])
        if self.normalize_observations:
            normalize_numpy_observation_in_place(
                obs,state.field_length,state.field_width,
                (params.max_speed,opponent_params.max_speed),
                (params.max_omega,opponent_params.max_omega))
        expected = getattr(getattr(self.model,"observation_space",None),"shape",None)
        if expected == (33,):
            obs = np.concatenate((obs[:21],obs[23:]))
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.nan_to_num(np.asarray(action, dtype=np.float32).reshape(-1)[:3])
        vx, vy = float(action[0])*params.max_speed, float(action[1])*params.max_speed
        speed = float(np.hypot(vx, vy))
        if speed > params.max_speed:
            vx, vy = vx*params.max_speed/speed, vy*params.max_speed/speed
        return ChassisCommand(vx, vy, float(action[2])*params.max_omega)


class TensorPPOPolicy:
    """Lazy Torch adapter for checkpoints produced by tensor_training.train."""

    def __init__(self, checkpoint, device: str = "cuda"):
        import torch
        from .tensor_training import ActorCritic

        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")
        self.device = requested
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.obs_dim = payload.get("obs_dim", 35)
        self.action_dim = payload.get("action_dim", 3)
        self.action_kind = payload.get("action_kind", "continuous")
        self.normalize_observations = payload.get("observation_normalization") == "fixed_physical_scale_v1"
        self.model = ActorCritic(self.obs_dim, self.action_dim, self.action_kind).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.torch = torch

    @staticmethod
    def _observation(state: WorldState, params: RobotParameters, goal: Objective,
                     normalize: bool = True) -> np.ndarray:
        own, opp = state.own, state.opponent
        opp_params = state.opponent_params or params
        obs = np.asarray([own.x, own.y, own.theta, own.vx, own.vy, own.omega,
                          opp.x, opp.y, opp.theta, opp.vx, opp.vy, opp.omega,
                          goal.x-own.x, goal.y-own.y, goal.radius, state.field_length,
                          state.field_width, params.length, opp_params.length,
                          params.width, opp_params.width], dtype=np.float32)
        limits=np.asarray([params.max_acceleration,opp_params.max_acceleration],dtype=np.float32)/10.
        vector=np.concatenate((obs,limits,_obstacle_features(state,own.x,own.y)))
        if normalize:
            normalize_numpy_observation_in_place(
                vector,state.field_length,state.field_width,
                (params.max_speed,opp_params.max_speed),
                (params.max_omega,opp_params.max_omega))
        return vector

    def predict(self, state: WorldState, params: RobotParameters, goal: Objective) -> ChassisCommand:
        vector=self._observation(state,params,goal,self.normalize_observations)
        if self.obs_dim == 33:
            vector=np.concatenate((vector[:21],vector[23:]))
        obs = self.torch.as_tensor(vector,device=self.device).unsqueeze(0)
        with self.torch.no_grad():
            action, _, _ = self.model.sample(obs,deterministic=True)
        action = action[0].detach().cpu().numpy()
        vx, vy = float(action[0])*params.max_speed, float(action[1])*params.max_speed
        speed = float(np.hypot(vx,vy))
        if speed > params.max_speed:
            vx, vy = vx*params.max_speed/speed, vy*params.max_speed/speed
        return ChassisCommand(vx,vy,float(action[2])*params.max_omega)


@dataclass(frozen=True)
class TacticalWaypoint:
    """A tactical target for deterministic AD* navigation."""
    objective: Objective


@dataclass(frozen=True)
class TacticalTarget:
    """One semantic option in a fixed tactical target catalog."""
    kind: str
    objective: Objective
    target_id: str = ""


@dataclass(frozen=True)
class TacticalDecision:
    """Selected game-state objective; downstream AD* owns all navigation."""
    target: TacticalTarget
    index: int
    scores: tuple[float, ...]


class TensorTacticalPolicy(TensorPPOPolicy):
    """Tensor PPO adapter for offense strategy, leaving navigation to AD*.

    Tactical checkpoints have a two-value normalized [x, y] subgoal action.
    Values in [-1, 1] map to field coordinates and are clamped to the robot's
    center-safe field region. The caller replans to the returned objective.
    An optional observation can be supplied by the caller when training uses
    additional route or history features; otherwise the standard world/goal/
    geometry observation is used.
    """

    def __init__(self, checkpoint, device: str = "cuda"):
        super().__init__(checkpoint, device)
        if self.action_dim != 2:
            raise ValueError("tactical checkpoints must have exactly 2 actions")

    def predict_waypoint(self, state: WorldState, params: RobotParameters,
                         goal: Objective,
                         *, observation: np.ndarray | None = None) -> TacticalWaypoint:
        """Map the learned normalized [x,y] action to a bounded field waypoint."""
        vector = (np.asarray(observation, dtype=np.float32).reshape(-1)
                  if observation is not None else
                  self._observation(state, params, goal, self.normalize_observations))
        if self.obs_dim == 33 and observation is None:
            vector = np.concatenate((vector[:21], vector[23:]))
        if vector.size != self.obs_dim:
            raise ValueError(f"tactical observation has {vector.size} features; expected {self.obs_dim}")
        obs = self.torch.as_tensor(vector, device=self.device).unsqueeze(0)
        with self.torch.no_grad():
            action, _, _ = self.model.sample(obs, deterministic=True)
        values = np.nan_to_num(action[0].detach().cpu().numpy(), nan=0., posinf=1., neginf=-1.)
        target = ((float(np.clip(values[0], -1., 1.)) + 1.) * .5 * state.field_length,
                  (float(np.clip(values[1], -1., 1.)) + 1.) * .5 * state.field_width)
        margin_x, margin_y = params.length / 2, params.width / 2
        x = float(np.clip(target[0], margin_x, max(margin_x, state.field_length - margin_x)))
        y = float(np.clip(target[1], margin_y, max(margin_y, state.field_width - margin_y)))
        return TacticalWaypoint(Objective(x, y, max(.1, goal.radius)))


class TensorTargetSelectionPolicy(TensorPPOPolicy):
    """Categorically choose pickup, score, or role-specific tactical strategy.

    The three output classes are fixed: index 0 selects pickup, index 1
    selects scoring, and index 2 selects the role-specific contest/intercept
    strategy. The environment resolves the selected class to the nearest
    currently valid gamepiece, scoring objective, or tactical region. The
    policy never outputs chassis velocity. `observation` can carry additional
    shared game history and AD* route/cost features from the training path.
    """

    def __init__(self, checkpoint, device: str = "cuda"):
        super().__init__(checkpoint, device)
        if self.action_dim != 3:
            raise ValueError("categorical tactical checkpoints must have exactly 3 classes")
        if getattr(self, "action_kind", None) != "categorical":
            raise ValueError("target selection requires an action_kind='categorical' checkpoint")

    def select_target(self, state: WorldState, params: RobotParameters,
                      goal: Objective, candidates: Sequence[TacticalTarget],
                      *, observation: np.ndarray | None = None) -> TacticalDecision:
        options = tuple(candidates)
        if len(options) != 3:
            raise ValueError("target selection requires exactly 3 semantic candidates")
        vector = (np.asarray(observation, dtype=np.float32).reshape(-1)
                  if observation is not None else
                  self._observation(state, params, goal, self.normalize_observations))
        if self.obs_dim == 33 and observation is None:
            vector = np.concatenate((vector[:21], vector[23:]))
        if vector.size != self.obs_dim:
            raise ValueError(f"tactical observation has {vector.size} features; expected {self.obs_dim}")
        obs = self.torch.as_tensor(vector, device=self.device).unsqueeze(0)
        with self.torch.no_grad():
            output = self.model(obs)
        logits = output[0] if isinstance(output, tuple) else output
        probabilities = self.torch.distributions.Categorical(logits=logits).probs[0]
        scores = np.nan_to_num(probabilities.detach().cpu().numpy(), nan=0.,
                               posinf=0., neginf=0.)
        index = int(np.argmax(scores))
        return TacticalDecision(options[index], index,
                               tuple(float(score) for score in scores))
