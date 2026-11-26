"""Ground-contact classification.

Touching the ground is NOT a successful landing. At the instant of contact the
rocket's measured state is compared against LandingCriteria and a single primary
Outcome is assigned. The full list of exceeded safe thresholds is also kept so
later phases (reward breakdown, AI Learning Insights) can explain *why* a
touchdown failed from measured data.

Primary-outcome precedence (first match wins):
  SIDE_COLLISION      |angle| > crash_angle (tipped over / hit sideways)
  CRASH               downward speed > crash_vertical_speed (destroyed)
  HARD_LANDING        downward speed > max_vertical_speed OR |angle| > max_angle
  EXCESSIVE_VELOCITY  |horizontal speed| > max OR |angular speed| > max
  OFF_PAD             everything safe, but centre line is off the pad
  SUCCESS             all conditions met, on the pad
Fuel exhaustion is reported as an event during flight, not as a touchdown type.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from config.environment_config import LandingCriteria
from environment.landing_pad import LandingPad
from environment.rocket import RocketState


class Outcome(str, Enum):
    IN_FLIGHT = "in_flight"
    SUCCESS = "success"
    HARD_LANDING = "hard_landing"
    EXCESSIVE_VELOCITY = "excessive_velocity"
    OFF_PAD = "off_pad"
    SIDE_COLLISION = "side_collision"
    CRASH = "crash"
    OUT_OF_BOUNDS = "out_of_bounds"

    @property
    def is_terminal(self) -> bool:
        return self is not Outcome.IN_FLIGHT


@dataclass(frozen=True)
class TouchdownReport:
    outcome: Outcome
    time: float
    x: float
    pad_offset: float            # signed distance from pad centre
    vertical_speed: float        # m/s, positive = downward
    horizontal_speed: float      # m/s, signed
    angle: float                 # rad
    angular_speed: float         # rad/s
    fuel_remaining: float
    on_pad: bool
    failed_criteria: tuple[str, ...]  # which safe thresholds were exceeded


def classify_touchdown(state: RocketState, pad: LandingPad,
                       crit: LandingCriteria) -> TouchdownReport:
    vy_down = max(0.0, -state.vy)
    on_pad = pad.contains(state.x)

    failed = []
    if vy_down > crit.max_vertical_speed:
        failed.append("vertical_speed")
    if abs(state.vx) > crit.max_horizontal_speed:
        failed.append("horizontal_speed")
    if abs(state.theta) > crit.max_angle:
        failed.append("angle")
    if abs(state.omega) > crit.max_angular_speed:
        failed.append("angular_speed")
    if not on_pad:
        failed.append("off_pad")

    if abs(state.theta) > crit.crash_angle:
        outcome = Outcome.SIDE_COLLISION
    elif vy_down > crit.crash_vertical_speed:
        outcome = Outcome.CRASH
    elif vy_down > crit.max_vertical_speed or abs(state.theta) > crit.max_angle:
        outcome = Outcome.HARD_LANDING
    elif abs(state.vx) > crit.max_horizontal_speed or abs(state.omega) > crit.max_angular_speed:
        outcome = Outcome.EXCESSIVE_VELOCITY
    elif not on_pad:
        outcome = Outcome.OFF_PAD
    else:
        outcome = Outcome.SUCCESS

    return TouchdownReport(
        outcome=outcome, time=state.time, x=state.x, pad_offset=pad.offset(state.x),
        vertical_speed=vy_down, horizontal_speed=state.vx, angle=state.theta,
        angular_speed=state.omega, fuel_remaining=state.fuel, on_pad=on_pad,
        failed_criteria=tuple(failed),
    )
