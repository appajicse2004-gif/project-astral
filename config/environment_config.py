"""Static configuration of the simulation: world, rocket, pad, landing criteria.

All physics parameters live here so they can be changed (or randomised by the
curriculum in Phase 7) without touching the physics engine. Units are SI:
metres, seconds, kilograms, newtons, radians.

Coordinate system: x to the right, y UP, ground at y = 0, x = 0 at world centre.
Angle theta = 0 means upright; positive theta is counter-clockwise (nose tilts
toward -x).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

G0 = 9.80665  # standard gravity, used only to convert specific impulse to mass flow


@dataclass(frozen=True)
class WorldConfig:
    gravity: float = 9.81          # m/s^2, acts downward
    dt: float = 1.0 / 60.0         # fixed simulation timestep (s)
    half_width: float = 100.0      # world spans x in [-half_width, +half_width]
    ceiling: float = 150.0         # leaving above this height ends the episode


@dataclass(frozen=True)
class RocketConfig:
    dry_mass: float = 1000.0       # kg, structure without propellant
    fuel_capacity: float = 400.0   # kg of propellant when full
    length: float = 10.0           # m, hull height (used for geometry + inertia)
    width: float = 2.0             # m, hull width
    max_thrust: float = 22000.0    # N, main engine at throttle 1.0
    isp: float = 250.0             # s, specific impulse -> mass flow = F / (isp*g0)
    throttle_rate: float = 1.5     # 1/s, max throttle change rate (actuator limit)
    max_torque: float = 10000.0    # N*m, attitude control at |command| = 1.0
    rcs_max_flow: float = 0.3      # kg/s propellant used by attitude control at |cmd| = 1


@dataclass(frozen=True)
class PadConfig:
    x: float = 0.0                 # pad centre
    half_width: float = 6.0        # pad spans x +/- half_width


@dataclass(frozen=True)
class LandingCriteria:
    """Thresholds evaluated at the moment of ground contact."""
    max_vertical_speed: float = 3.0     # m/s downward, safe limit
    max_horizontal_speed: float = 1.5   # m/s, safe limit
    max_angle: float = 0.15             # rad (~8.6 deg), safe limit
    max_angular_speed: float = 0.3      # rad/s, safe limit
    crash_vertical_speed: float = 10.0  # m/s downward: above this the rocket is destroyed
    crash_angle: float = 0.6            # rad (~34 deg): beyond this it tips over / hits sideways


@dataclass(frozen=True)
class EnvironmentConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    rocket: RocketConfig = field(default_factory=RocketConfig)
    pad: PadConfig = field(default_factory=PadConfig)
    criteria: LandingCriteria = field(default_factory=LandingCriteria)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
