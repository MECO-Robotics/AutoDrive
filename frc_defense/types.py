"""Simulator-independent runtime contracts for learned chassis policies."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


@dataclass(frozen=True)
class RobotState:
    x: float
    y: float
    theta: float
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class WorldState:
    own: RobotState
    opponent: RobotState
    timestamp: float
    field_length: float
    field_width: float
    obstacles: tuple[Obstacle, ...] = ()
    opponent_params: RobotParameters | None = None


@dataclass(frozen=True)
class RobotParameters:
    length: float = 0.9
    width: float = 0.9
    max_speed: float = 4.8
    max_acceleration: float = 8.0
    max_omega: float = 8.0
    max_alpha: float = 18.0
    mass: float = 55.0

    def __post_init__(self):
        values = (self.length,self.width,self.max_speed,self.max_acceleration,
                  self.max_omega,self.max_alpha,self.mass)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("robot dimensions, mass, and motion limits must be finite and positive")


@dataclass(frozen=True)
class Objective:
    x: float
    y: float
    radius: float = 0.5

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x,self.y,self.radius)) or self.radius <= 0:
            raise ValueError("objective coordinates must be finite and radius positive")


@dataclass(frozen=True)
class ChassisCommand:
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0


class DefensePolicy(Protocol):
    def predict(self, state: WorldState, params: RobotParameters, goal: Objective) -> ChassisCommand: ...
