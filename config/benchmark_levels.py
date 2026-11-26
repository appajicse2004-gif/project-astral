"""Fixed difficulty ladder for fair comparison of controllers and models.

L1 is the distribution the expert data / PPO fine-tuning used. L2 and L3 are
OUTSIDE it, so they test generalisation rather than memorisation.
"""
from __future__ import annotations

from config.scenario_config import ScenarioConfig

LEVELS = {
    "L0": ("default low start, centred",
           ScenarioConfig()),
    "L1": ("training distribution",
           ScenarioConfig(x_offset=(-10.0, 10.0), y=(20.0, 40.0), vx=(-1.0, 1.0), vy=(-0.5, 0.0),
                          theta=(-0.08, 0.08), omega=(-0.08, 0.08), max_episode_steps=900)),
    "L2": ("wider: bigger offset, speed, tilt",
           ScenarioConfig(x_offset=(-25.0, 25.0), y=(20.0, 50.0), vx=(-3.0, 3.0), vy=(-2.0, 0.0),
                          theta=(-0.2, 0.2), omega=(-0.2, 0.2), max_episode_steps=900)),
    "L3": ("hard: larger still + gravity and mass change",
           ScenarioConfig(x_offset=(-40.0, 40.0), y=(25.0, 60.0), vx=(-5.0, 5.0), vy=(-4.0, 0.0),
                          theta=(-0.35, 0.35), omega=(-0.4, 0.4), gravity=(8.0, 11.0),
                          dry_mass=(900.0, 1200.0), max_episode_steps=900)),
}
