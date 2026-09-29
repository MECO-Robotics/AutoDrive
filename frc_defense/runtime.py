"""Safe, simulator-independent policy command boundary."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .types import ChassisCommand, Objective, RobotParameters, WorldState


class Policy(Protocol):
    def predict(self, state: WorldState, params: RobotParameters, goal: Objective) -> ChassisCommand: ...


@dataclass(frozen=True)
class CommandResult:
    raw: ChassisCommand
    constrained: ChassisCommand
    reason: str


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class SafePolicyRunner:
    """Constrain policy output and fall back to zero/manual on stale or invalid state."""

    def __init__(self, policy: Policy, params: RobotParameters, *, dt: float = 0.02,
                 stale_after: float = 0.15, clock: Callable[[], float] = time.time,
                 logger: Callable[[dict], None] | None = None):
        if dt <= 0 or stale_after <= 0:
            raise ValueError("dt and stale_after must be positive")
        self.policy, self.params = policy, params
        self.dt, self.stale_after, self.clock, self.logger = dt, stale_after, clock, logger
        self.previous = ChassisCommand()

    def _constrain(self, cmd: ChassisCommand, state: WorldState) -> ChassisCommand:
        p = self.params
        vx, vy = _clip(cmd.vx, p.max_speed), _clip(cmd.vy, p.max_speed)
        speed = math.hypot(vx, vy)
        if speed > p.max_speed:
            vx, vy = vx * p.max_speed / speed, vy * p.max_speed / speed
        dv = p.max_acceleration * self.dt
        vx = self.previous.vx + _clip(vx - self.previous.vx, dv)
        vy = self.previous.vy + _clip(vy - self.previous.vy, dv)
        omega = self.previous.omega + _clip(_clip(cmd.omega, p.max_omega) - self.previous.omega,
                                             p.max_alpha * self.dt)
        # Do not command farther into a field boundary; retain tangential motion.
        c, s = abs(math.cos(state.own.theta)), abs(math.sin(state.own.theta))
        margin_x = (p.length*c + p.width*s) / 2
        margin_y = (p.length*s + p.width*c) / 2
        if state.own.x <= margin_x and vx < 0 or state.own.x >= state.field_length - margin_x and vx > 0:
            vx = 0.0
        if state.own.y <= margin_y and vy < 0 or state.own.y >= state.field_width - margin_y and vy > 0:
            vy = 0.0
        return ChassisCommand(vx, vy, omega)

    def predict(self, state: WorldState, goal: Objective, *, manual: ChassisCommand | None = None) -> CommandResult:
        zero = ChassisCommand()
        now = self.clock()
        stale = not math.isfinite(state.timestamp) or now - state.timestamp > self.stale_after or state.timestamp > now + 0.05
        values = (state.own.x,state.own.y,state.own.theta,state.own.vx,state.own.vy,state.own.omega,
                  state.opponent.x,state.opponent.y,state.opponent.theta,state.opponent.vx,
                  state.opponent.vy,state.opponent.omega,state.field_length,state.field_width,goal.x,goal.y)
        obstacle_values = [v for obstacle in state.obstacles for v in (obstacle.x,obstacle.y,obstacle.radius)]
        valid = (all(math.isfinite(v) for v in values+tuple(obstacle_values)) and
                 state.field_length > 0 and state.field_width > 0 and
                 all(obstacle.radius > 0 for obstacle in state.obstacles))
        reason = "policy"
        try:
            raw = self.policy.predict(state, self.params, goal) if valid and not stale else zero
            if stale:
                reason = "stale-state"
            elif not valid or not all(math.isfinite(v) for v in (raw.vx, raw.vy, raw.omega)):
                reason, raw = "invalid-output", zero
        except Exception:
            reason, raw = "policy-error", zero
        if reason != "policy":
            raw = manual or zero
            if not all(math.isfinite(v) for v in (raw.vx, raw.vy, raw.omega)):
                raw = zero
        constrained = self._constrain(raw, state)
        self.previous = constrained
        result = CommandResult(raw, constrained, reason)
        if self.logger:
            self.logger({"timestamp": now, "raw": raw.__dict__, "constrained": constrained.__dict__, "reason": reason})
        return result
