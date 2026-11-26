"""Gymnasium wrapper around LandingTask. All task logic lives in landing_task.py."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config.environment_config import EnvironmentConfig
from config.reward_config import RewardConfig
from config.scenario_config import ScenarioConfig
from environment.landing_task import LandingTask
from environment.observation import CLIP, OBS_DIM


class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, env_config: EnvironmentConfig | None = None,
                 scenario_config: ScenarioConfig | None = None,
                 reward_config: RewardConfig | None = None,
                 render_mode: str | None = None):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.task = LandingTask(env_config, scenario_config, reward_config)
        self.observation_space = spaces.Box(-CLIP, CLIP, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._renderer = None
        self._last_command = (0.0, 0.0)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self._renderer is not None:
            self._renderer.clear_trail()
        obs, info = self.task.reset(self.np_random, options)
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.task.step(action)
        self._last_command = (info["throttle_cmd"], info["rotation_cmd"])
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None
        if self._renderer is None:
            from visualization.simulation_renderer import SimulationRenderer
            self._renderer = SimulationRenderer(self.task.episode_config, caption="Project Astral - env")
        eng = self.task.engine
        self._renderer.render(eng.state, outcome=eng.outcome, report=eng.touchdown,
                              command=self._last_command, push_trail=not eng.done)
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
