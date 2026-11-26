"""Observation vector (12 floats), normalised and expressed relative to the pad.

Pad-relative coordinates let the agent generalise across target positions instead
of memorising an absolute x.
"""
from __future__ import annotations

import math

import numpy as np

from config.environment_config import EnvironmentConfig
from environment.rocket import RocketState, lowest_point_height, total_mass

VEL_SCALE = 20.0       # m/s
OMEGA_SCALE = 2.0      # rad/s
TWR_SCALE = 3.0
CLIP = 5.0

OBS_NAMES = (
    "pad_offset_x",        # (x - pad_x) / world half-width, signed horizontal distance to pad centre
    "altitude",            # height of rocket bottom above ground / ceiling
    "vx",                  # horizontal velocity / VEL_SCALE
    "vy",                  # vertical velocity / VEL_SCALE
    "angle",               # theta / pi
    "angular_velocity",    # omega / OMEGA_SCALE
    "fuel_fraction",       # remaining propellant / capacity
    "throttle",            # actual (rate-limited) throttle; makes the state Markov
    "target_distance",     # distance to target point above pad centre / ceiling
    "pad_x",               # absolute pad position / world half-width (target info)
    "pad_half_width",      # pad half-width / world half-width (target info)
    "thrust_to_weight",    # current max_thrust / (m*g) / TWR_SCALE (lets agent adapt to mass/gravity)
)
OBS_DIM = len(OBS_NAMES)


def build_observation(state: RocketState, cfg: EnvironmentConfig) -> np.ndarray:
    w, r, pad = cfg.world, cfg.rocket, cfg.pad
    target_dist = math.hypot(state.x - pad.x, state.y - r.length / 2)
    twr = r.max_thrust / (total_mass(state, r) * w.gravity)
    obs = np.array([
        (state.x - pad.x) / w.half_width,
        max(0.0, lowest_point_height(state, r)) / w.ceiling,
        state.vx / VEL_SCALE,
        state.vy / VEL_SCALE,
        state.theta / math.pi,
        state.omega / OMEGA_SCALE,
        state.fuel / r.fuel_capacity if r.fuel_capacity > 0 else 0.0,
        state.throttle,
        target_dist / w.ceiling,
        pad.x / w.half_width,
        pad.half_width / w.half_width,
        twr / TWR_SCALE,
    ], dtype=np.float64)
    return np.clip(obs, -CLIP, CLIP).astype(np.float32)
