"""Gymnasium-independent core of the landing task.

Holds physics engine + reward calculator + observation builder + scenario
sampling, and implements reset/step with Gymnasium's (obs, reward, terminated,
truncated, info) contract. environment/rocket_env.py is a thin gym.Env wrapper
around this class. Keeping the logic here makes it testable without Gymnasium.

Action (2 floats in [-1, 1], symmetric as PPO expects):
    a[0] -> main throttle = (a[0] + 1) / 2   (so a[0] = 0 is half throttle)
    a[1] -> attitude torque command, + = counter-clockwise
Both are further limited by the physics actuator model (throttle slew rate).
"""
from __future__ import annotations

from dataclasses import asdict, fields, replace

import numpy as np

from config.environment_config import EnvironmentConfig, PadConfig
from config.reward_config import RewardConfig
from config.scenario_config import Scenario, ScenarioConfig
from environment.observation import OBS_DIM, build_observation
from environment.physics_engine import PhysicsEngine
from environment.rocket import make_state
from training.reward_calculator import COMPONENT_NAMES, RewardCalculator

_SCENARIO_KEYS = {f.name for f in fields(Scenario)}


def sample_scenario(cfg: ScenarioConfig, base: EnvironmentConfig, rng: np.random.Generator) -> Scenario:
    def u(rg):
        return float(rng.uniform(rg[0], rg[1]))

    pad_x = u(cfg.pad_x)
    limit = base.world.half_width - 2.0
    x = float(np.clip(pad_x + u(cfg.x_offset), -limit, limit))
    return Scenario(
        pad_x=pad_x, x=x, y=u(cfg.y), vx=u(cfg.vx), vy=u(cfg.vy), theta=u(cfg.theta),
        omega=u(cfg.omega), fuel_fraction=u(cfg.fuel_fraction),
        gravity=u(cfg.gravity) if cfg.gravity else base.world.gravity,
        dry_mass=u(cfg.dry_mass) if cfg.dry_mass else base.rocket.dry_mass,
    )


class LandingTask:
    def __init__(self, env_config: EnvironmentConfig | None = None,
                 scenario_config: ScenarioConfig | None = None,
                 reward_config: RewardConfig | None = None):
        self.base_config = env_config or EnvironmentConfig()
        self.scenario_config = scenario_config or ScenarioConfig()
        self.reward_config = reward_config or RewardConfig()
        self.engine: PhysicsEngine | None = None
        self.scenario: Scenario | None = None
        self.episode_config: EnvironmentConfig | None = None
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.steps = 0
        self.episode_return = 0.0
        self.component_totals = {k: 0.0 for k in COMPONENT_NAMES}

    @property
    def max_steps(self) -> int:
        return self.scenario_config.max_episode_steps

    # ------------------------------------------------------------------- reset
    def reset(self, rng: np.random.Generator, options: dict | None = None):
        scenario = sample_scenario(self.scenario_config, self.base_config, rng)
        if options:
            unknown = set(options) - _SCENARIO_KEYS
            if unknown:
                raise ValueError(f"Unknown scenario option(s): {sorted(unknown)}; valid: {sorted(_SCENARIO_KEYS)}")
            scenario = replace(scenario, **options)

        b = self.base_config
        cfg = replace(
            b,
            world=replace(b.world, gravity=scenario.gravity),
            rocket=replace(b.rocket, dry_mass=scenario.dry_mass),
            pad=PadConfig(x=scenario.pad_x, half_width=b.pad.half_width),
        )
        state = make_state(cfg.rocket, x=scenario.x, y=scenario.y, vx=scenario.vx, vy=scenario.vy,
                           theta=scenario.theta, omega=scenario.omega,
                           fuel_fraction=scenario.fuel_fraction)
        self.episode_config = cfg
        self.scenario = scenario
        self.engine = PhysicsEngine(cfg, state)
        self.reward = RewardCalculator(self.reward_config, cfg)
        self.reward.reset(self.engine.state)
        self._reset_counters()
        return build_observation(self.engine.state, cfg), {"scenario": asdict(scenario)}

    # -------------------------------------------------------------------- step
    def step(self, action):
        if self.engine is None:
            raise RuntimeError("Call reset() before step().")
        if self.engine.done:
            raise RuntimeError("Episode finished; call reset().")
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (2,):
            raise ValueError(f"Action must have 2 elements, got shape {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"Action contains non-finite values: {a}")
        a = np.clip(a, -1.0, 1.0)
        throttle_cmd, rotation_cmd = (a[0] + 1.0) / 2.0, a[1]

        result = self.engine.step(throttle_cmd, rotation_cmd)
        self.steps += 1
        terminated = result.done
        truncated = (not terminated) and self.steps >= self.max_steps

        breakdown = self.reward.step(result.state, result, truncated)
        reward = breakdown.total
        self.episode_return += reward
        for k, v in breakdown.components.items():
            self.component_totals[k] += v

        cfg = self.episode_config
        outcome = "timeout" if truncated else result.outcome.value
        info = {
            "outcome": outcome,
            "is_success": result.outcome.value == "success",
            "events": list(result.events),
            "reward_components": breakdown.components,
            "throttle_cmd": float(throttle_cmd),
            "rotation_cmd": float(rotation_cmd),
            "pad_offset": float(result.state.x - cfg.pad.x),
        }
        if result.touchdown is not None:
            td = asdict(result.touchdown)
            td["outcome"] = result.touchdown.outcome.value
            info["touchdown"] = td
        if terminated or truncated:
            info["episode_summary"] = {
                "return": self.episode_return,
                "length": self.steps,
                "outcome": outcome,
                "is_success": info["is_success"],
                "fuel_used": cfg.rocket.fuel_capacity * self.scenario.fuel_fraction - result.state.fuel,
                "final_x_offset": float(result.state.x - cfg.pad.x),
                "final_y": float(result.state.y),
                "reward_component_totals": dict(self.component_totals),
            }
        return build_observation(result.state, cfg), float(reward), terminated, truncated, info
