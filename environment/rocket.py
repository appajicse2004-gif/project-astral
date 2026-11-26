"""Rocket state and the small geometric/mass helpers derived from it."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from config.environment_config import RocketConfig


@dataclass
class RocketState:
    """Full dynamic state of the rocket (centre of mass)."""
    x: float = 0.0
    y: float = 100.0
    vx: float = 0.0
    vy: float = 0.0
    theta: float = 0.0       # rad, 0 = upright, + = counter-clockwise
    omega: float = 0.0       # rad/s
    fuel: float = 0.0        # kg of propellant remaining
    throttle: float = 0.0    # actual (rate-limited) throttle in [0, 1]
    time: float = 0.0        # simulated seconds since reset
    # Last-step accelerations, stored for display and later recording.
    ax: float = 0.0
    ay: float = 0.0
    alpha: float = 0.0

    def copy(self) -> "RocketState":
        return replace(self)


def make_state(cfg: RocketConfig, *, x=0.0, y=100.0, vx=0.0, vy=0.0, theta=0.0,
               omega=0.0, fuel_fraction=1.0) -> RocketState:
    return RocketState(x=x, y=y, vx=vx, vy=vy, theta=theta, omega=omega,
                       fuel=cfg.fuel_capacity * fuel_fraction)


def total_mass(state: RocketState, cfg: RocketConfig) -> float:
    return cfg.dry_mass + state.fuel


def moment_of_inertia(mass: float, cfg: RocketConfig) -> float:
    """Slender-rod approximation about the centre of mass: I = m * L^2 / 12."""
    return mass * cfg.length ** 2 / 12.0


def thrust_direction(theta: float) -> tuple[float, float]:
    """Unit vector the engine pushes the rocket along (the rocket's 'up')."""
    return (-math.sin(theta), math.cos(theta))


def lowest_point_height(state: RocketState, cfg: RocketConfig) -> float:
    """Height above ground of the lowest point of the (rectangular) hull."""
    half_extent = (cfg.length / 2) * abs(math.cos(state.theta)) \
        + (cfg.width / 2) * abs(math.sin(state.theta))
    return state.y - half_extent


def hull_corners(state: RocketState, cfg: RocketConfig) -> list[tuple[float, float]]:
    """World-space hull corners, for rendering."""
    c, s = math.cos(state.theta), math.sin(state.theta)
    pts = []
    for lx, ly in ((-cfg.width / 2, -cfg.length / 2), (cfg.width / 2, -cfg.length / 2),
                   (cfg.width / 2, cfg.length / 2), (-cfg.width / 2, cfg.length / 2)):
        pts.append((state.x + lx * c - ly * s, state.y + lx * s + ly * c))
    return pts
