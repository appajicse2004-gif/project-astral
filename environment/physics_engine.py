"""Deterministic 2D rocket physics with a fixed timestep.

Per step, with the rocket's mass m (dry mass + current fuel):

    thrust vector  F_t = T * (-sin(theta), cos(theta))     T = throttle * max_thrust
    acceleration   a   = (F_t + (0, -m*g)) / m
    ang. accel.    alpha = torque / I,   I = m * L^2 / 12
    mass flow      mdot = T / (isp * g0)  +  |rot_cmd| * rcs_max_flow

Integration: constant-acceleration update over the step,
    p(t+dt) = p(t) + v(t)*dt + 0.5*a*dt^2
    v(t+dt) = v(t) + a*dt
This reduces to the textbook Euler formulas as a special case, but is exact
for the constant forces used within one step (so free fall matches the
analytic solution), and needs no tuning for stability at dt = 1/60 s.
Mass is evaluated at the start of each step; the change within a single step
(< 0.2 kg at full burn) is neglected.

The engine knows nothing about rewards, learning or rendering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from config.environment_config import EnvironmentConfig, G0
from environment.collision import Outcome, TouchdownReport, classify_touchdown
from environment.landing_pad import LandingPad
from environment.rocket import (RocketState, lowest_point_height, make_state,
                                moment_of_inertia, total_mass)


@dataclass(frozen=True)
class StepResult:
    state: RocketState                      # snapshot after the step
    outcome: Outcome
    done: bool
    events: tuple[str, ...]                 # events raised during this step
    touchdown: TouchdownReport | None


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class PhysicsEngine:
    def __init__(self, config: EnvironmentConfig | None = None,
                 state: RocketState | None = None):
        self.config = config or EnvironmentConfig()
        self.pad = LandingPad(self.config.pad)
        self.reset(state)

    # ------------------------------------------------------------------ reset
    def reset(self, state: RocketState | None = None) -> RocketState:
        self.state = state.copy() if state is not None else make_state(self.config.rocket)
        self.outcome = Outcome.IN_FLIGHT
        self.touchdown: TouchdownReport | None = None
        self.events: list[str] = []         # all events since reset
        self._fuel_out_flagged = self.state.fuel <= 0.0
        return self.state.copy()

    @property
    def done(self) -> bool:
        return self.outcome.is_terminal

    # ------------------------------------------------------------------- step
    def step(self, throttle_cmd: float, rotation_cmd: float) -> StepResult:
        """Advance one fixed timestep.

        throttle_cmd in [0, 1]  : requested main-engine throttle
        rotation_cmd in [-1, 1] : requested attitude torque (+ = counter-clockwise)
        Commands are clipped to their valid ranges (actuator limits).
        """
        if self.done:
            raise RuntimeError(f"Episode already finished ({self.outcome.value}); call reset().")

        w, r, s = self.config.world, self.config.rocket, self.state
        dt = w.dt
        new_events: list[str] = []

        throttle_cmd = _clip(float(throttle_cmd), 0.0, 1.0)
        rotation_cmd = _clip(float(rotation_cmd), -1.0, 1.0)

        # Throttle actuator: rate-limited, and dead when out of propellant.
        if s.fuel <= 0.0:
            s.throttle = 0.0
            rotation_cmd = 0.0
        else:
            max_delta = r.throttle_rate * dt
            s.throttle += _clip(throttle_cmd - s.throttle, -max_delta, max_delta)
            s.throttle = _clip(s.throttle, 0.0, 1.0)

        # Propellant: if the step would need more than is left, scale thrust/torque down.
        mass = total_mass(s, r)
        mdot = (s.throttle * r.max_thrust / (r.isp * G0)) + abs(rotation_cmd) * r.rcs_max_flow
        needed = mdot * dt
        scale = 1.0
        if needed > s.fuel:
            scale = (s.fuel / needed) if needed > 0 else 0.0
        burned = needed * scale

        thrust = s.throttle * r.max_thrust * scale
        torque = rotation_cmd * r.max_torque * scale

        # Accelerations at the start-of-step mass.
        ax = -thrust * math.sin(s.theta) / mass
        ay = thrust * math.cos(s.theta) / mass - w.gravity
        alpha = torque / moment_of_inertia(mass, r)

        # Constant-acceleration integration.
        s.x += s.vx * dt + 0.5 * ax * dt * dt
        s.y += s.vy * dt + 0.5 * ay * dt * dt
        s.theta += s.omega * dt + 0.5 * alpha * dt * dt
        s.vx += ax * dt
        s.vy += ay * dt
        s.omega += alpha * dt
        s.theta = (s.theta + math.pi) % (2 * math.pi) - math.pi

        s.fuel = max(0.0, s.fuel - burned)
        s.time += dt
        s.ax, s.ay, s.alpha = ax, ay, alpha

        if s.fuel <= 1e-12 and not self._fuel_out_flagged:
            s.fuel = 0.0
            self._fuel_out_flagged = True
            new_events.append("FUEL_EXHAUSTED")

        # Termination checks: ground contact first, then leaving the world.
        height = lowest_point_height(s, r)
        if height <= 0.0:
            report = classify_touchdown(s, self.pad, self.config.criteria)
            self.touchdown = report
            self.outcome = report.outcome
            new_events.append(f"TOUCHDOWN:{report.outcome.value}")
            s.y -= height            # rest exactly on the ground
            s.vx = s.vy = s.omega = 0.0
            s.throttle = 0.0
        elif abs(s.x) > w.half_width or s.y > w.ceiling:
            self.outcome = Outcome.OUT_OF_BOUNDS
            new_events.append("OUT_OF_BOUNDS")

        self.events.extend(new_events)
        return StepResult(state=s.copy(), outcome=self.outcome, done=self.done,
                          events=tuple(new_events), touchdown=self.touchdown)
