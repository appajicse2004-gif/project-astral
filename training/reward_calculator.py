"""Configurable, inspectable reward function.

Every call returns a RewardBreakdown with ALL component names present (zeros
where inactive) so the dashboard can show why a reward or punishment occurred.

Loophole guards:
  * Progress terms are potential-based with shaping gamma = 1, so a static state
    earns exactly zero (gamma < 1 would pay (gamma-1)*Phi > 0 per step for hovering).
  * On a touchdown step the engine has already zeroed velocities (rest pose), so
    shaping potentials computed from it would pay the agent for crashing. Progress
    and per-step penalties are therefore skipped on that step; the terminal terms
    (computed from the recorded impact report) decide the outcome.
  * Time and fuel costs make hovering/loitering strictly unprofitable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from config.environment_config import EnvironmentConfig
from config.reward_config import RewardConfig
from environment.collision import Outcome
from environment.physics_engine import StepResult
from environment.rocket import RocketState, lowest_point_height

PROGRESS = ("progress_distance", "progress_horizontal", "progress_descent_speed",
            "progress_upright", "progress_angular")
STEP_PENALTIES = ("excess_horizontal_speed", "excess_tilt", "excess_rotation",
                  "fuel_usage", "time_penalty")
TERMINAL = ("terminal_success", "terminal_accuracy", "terminal_softness",
            "terminal_failure", "touchdown_severity", "landing_offset", "out_of_bounds", "timeout",
            "fuel_exhausted")
COMPONENT_NAMES = PROGRESS + STEP_PENALTIES + TERMINAL


@dataclass(frozen=True)
class RewardBreakdown:
    components: dict

    @property
    def total(self) -> float:
        return float(sum(self.components.values()))


class RewardCalculator:
    def __init__(self, cfg: RewardConfig, env_cfg: EnvironmentConfig):
        self.cfg = cfg
        self.env = env_cfg
        self._phi: dict | None = None
        self._fuel = 0.0

    # ----------------------------------------------------------------- helpers
    def _potentials(self, s: RocketState) -> dict:
        c, r, pad = self.cfg, self.env.rocket, self.env.pad
        h = max(0.0, lowest_point_height(s, r))
        dist = math.hypot(s.x - pad.x, s.y - r.length / 2)
        v_des = min(c.descent_cap, max(c.descent_base, c.descent_base + c.descent_slope * h))
        excess_descent = max(0.0, -s.vy - v_des)
        return {
            "progress_distance": -c.distance_weight * dist / c.distance_scale,
            "progress_horizontal": -c.horizontal_weight * abs(s.x - pad.x) / c.distance_scale,
            "progress_descent_speed": -c.descent_weight * excess_descent / c.descent_cap,
            "progress_upright": -c.upright_weight * abs(s.theta) / c.tilt_scale,
            "progress_angular": -c.angular_weight * abs(s.omega) / c.omega_scale,
        }

    def reset(self, state: RocketState) -> None:
        self._phi = self._potentials(state)
        self._fuel = state.fuel

    # -------------------------------------------------------------------- step
    def step(self, state: RocketState, result: StepResult, truncated: bool = False) -> RewardBreakdown:
        if self._phi is None:
            raise RuntimeError("RewardCalculator.reset() must be called first.")
        c, r = self.cfg, self.env.rocket
        comp = {name: 0.0 for name in COMPONENT_NAMES}
        touched = result.touchdown is not None

        if not touched:
            new_phi = self._potentials(state)
            for k in PROGRESS:
                comp[k] = c.shaping_gamma * new_phi[k] - self._phi[k]
            self._phi = new_phi

            h = max(0.0, lowest_point_height(state, r))
            vx_free = c.horizontal_speed_free + c.horizontal_speed_free_slope * h
            comp["excess_horizontal_speed"] = -c.horizontal_speed_penalty * max(0.0, abs(state.vx) - vx_free)
            comp["excess_tilt"] = -c.tilt_penalty * max(0.0, abs(state.theta) - c.tilt_free)
            comp["excess_rotation"] = -c.rotation_penalty * max(0.0, abs(state.omega) - c.rotation_free)
            comp["time_penalty"] = -c.time_penalty

        burned = max(0.0, self._fuel - state.fuel)
        self._fuel = state.fuel
        comp["fuel_usage"] = -c.fuel_usage_weight * burned / r.fuel_capacity if r.fuel_capacity > 0 else 0.0
        if "FUEL_EXHAUSTED" in result.events:
            comp["fuel_exhausted"] = -c.fuel_exhausted_penalty

        if touched:
            self._terminal_touchdown(comp, result)
        elif result.outcome is Outcome.OUT_OF_BOUNDS:
            comp["out_of_bounds"] = -c.out_of_bounds_penalty
        elif truncated:
            comp["timeout"] = -c.timeout_penalty
        return RewardBreakdown(components=comp)

    def _terminal_touchdown(self, comp: dict, result: StepResult) -> None:
        c, crit, pad = self.cfg, self.env.criteria, self.env.pad
        rep = result.touchdown
        out = rep.outcome
        if out is Outcome.SUCCESS:
            comp["terminal_success"] = c.success_reward
            comp["terminal_accuracy"] = c.accuracy_bonus * max(0.0, 1.0 - abs(rep.pad_offset) / pad.half_width)
            comp["terminal_softness"] = c.softness_bonus * max(
                0.0, 1.0 - rep.vertical_speed / crit.max_vertical_speed)
            return
        if out in (Outcome.CRASH, Outcome.SIDE_COLLISION):
            comp["terminal_failure"] = -c.crash_penalty
        elif out in (Outcome.HARD_LANDING, Outcome.EXCESSIVE_VELOCITY):
            comp["terminal_failure"] = -c.hard_landing_penalty
        elif out is Outcome.OFF_PAD:
            comp["terminal_failure"] = -c.off_pad_penalty
        impact = math.hypot(rep.horizontal_speed, rep.vertical_speed)
        comp["touchdown_severity"] = -c.touchdown_severity_weight * min(1.0, impact / c.touchdown_severity_scale)
        comp["landing_offset"] = -c.landing_offset_weight * min(
            1.0, abs(rep.pad_offset) / self.env.world.half_width)
