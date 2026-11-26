"""Reward weights. Changing these never requires touching the physics engine.

Design principles (see training/reward_calculator.py):
  * Progress terms are POTENTIAL-BASED with shaping gamma = 1: reward = Phi(s') - Phi(s).
    The undiscounted episode sum telescopes to Phi(end) - Phi(start), and a rocket
    that sits still earns exactly zero shaping. (Using the PPO discount 0.99 here
    would pay (0.99-1)*Phi > 0 every step for hovering in a bad state: ~+0.11/step
    at 55 m, i.e. a hover loophole worth more than a landing. A regression test
    covers this.) Only improvement is paid.
  * Penalties are applied only to clearly bad states (above a free threshold).
  * Large terminal terms decide whether the landing was actually safe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RewardConfig:
    shaping_gamma: float = 1.0   # keep at 1.0; see note above. Independent of the PPO discount.

    # --- progress (potential-based) weights; total gain over a full approach ~ weight ---
    distance_weight: float = 20.0        # closing distance to the target point
    horizontal_weight: float = 10.0      # reducing horizontal error to pad centre
    descent_weight: float = 5.0          # reducing descent speed above the safe profile
    upright_weight: float = 3.0          # reducing tilt
    angular_weight: float = 2.0          # reducing spin rate
    distance_scale: float = 100.0        # m
    tilt_scale: float = 0.6              # rad
    omega_scale: float = 1.0             # rad/s

    # Desired descent-speed profile: v_des(h) = clip(base + slope*h, base, cap)
    descent_base: float = 1.5            # m/s near the ground
    descent_slope: float = 0.25          # 1/s
    descent_cap: float = 15.0            # m/s

    # --- per-step penalties (only above the free thresholds) ---
    horizontal_speed_free: float = 1.5   # m/s allowed at ground level ...
    horizontal_speed_free_slope: float = 0.2  # ... plus this much per metre of altitude
    horizontal_speed_penalty: float = 0.02
    tilt_free: float = 0.4               # rad
    tilt_penalty: float = 0.1
    rotation_free: float = 0.5           # rad/s
    rotation_penalty: float = 0.05
    fuel_usage_weight: float = 10.0      # burning the whole tank costs this much
    time_penalty: float = 0.02           # per step (was 0.005: hovering was cheaper than trying to land)

    # --- terminal ---
    success_reward: float = 100.0
    accuracy_bonus: float = 20.0         # scaled by closeness to pad centre (success only)
    softness_bonus: float = 10.0         # scaled by gentleness of touchdown (success only)
    crash_penalty: float = 100.0         # CRASH, SIDE_COLLISION
    hard_landing_penalty: float = 60.0   # HARD_LANDING, EXCESSIVE_VELOCITY
    off_pad_penalty: float = 40.0        # otherwise-safe touchdown outside the pad
    landing_offset_weight: float = 20.0  # any failed touchdown: scaled by miss distance
    out_of_bounds_penalty: float = 100.0
    timeout_penalty: float = 60.0        # was 20: hover-until-timeout must score worse than a hard landing
    touchdown_severity_weight: float = 20.0   # failed touchdowns: graded by impact speed, so slower is better
    touchdown_severity_scale: float = 15.0    # m/s at which the severity penalty is maxed out
    fuel_exhausted_penalty: float = 10.0

    def to_dict(self) -> dict:
        return asdict(self)
