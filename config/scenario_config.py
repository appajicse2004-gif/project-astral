"""Initial-condition ranges. The curriculum (Phase 7) will swap these per stage.

Every range is (low, high), sampled uniformly. None means "use the base config
value". The default is Stage-1-like: straight above the pad, at rest, upright.
"""
from __future__ import annotations

from dataclasses import dataclass

Range = tuple[float, float]


@dataclass(frozen=True)
class ScenarioConfig:
    pad_x: Range = (0.0, 0.0)
    x_offset: Range = (0.0, 0.0)          # start x relative to the pad centre
    y: Range = (20.0, 40.0)               # centre-of-mass height (low start: easier to find a first success)
    vx: Range = (0.0, 0.0)
    vy: Range = (0.0, 0.0)
    theta: Range = (0.0, 0.0)
    omega: Range = (0.0, 0.0)
    fuel_fraction: Range = (1.0, 1.0)
    gravity: Range | None = None
    dry_mass: Range | None = None
    max_episode_steps: int = 900          # 15 s of simulated time


@dataclass(frozen=True)
class Scenario:
    """One concrete episode start (fully resolved, recordable)."""
    pad_x: float
    x: float
    y: float
    vx: float
    vy: float
    theta: float
    omega: float
    fuel_fraction: float
    gravity: float
    dry_mass: float


# Named presets (CLI: --start low|tall). 'tall' is the original stage-1 setup (first run, "stage1").
PRESETS = {
    "low": ScenarioConfig(),
    "tall": ScenarioConfig(y=(60.0, 100.0), max_episode_steps=1800),
}
