"""Hand-written feedback controller (NOT an AI). Single shared copy.

Used as (a) a yardstick, (b) the teacher for behaviour cloning / DAgger.
Same logic as in collect_varied_expert_data.py; the other scripts can import
this instead of keeping their own copies.
"""
from __future__ import annotations

import math

import numpy as np

from environment.rocket import lowest_point_height, total_mass


def expert_controller(state, cfg):
    """Returns (throttle in [0,1], rotation in [-1,1]) for a RocketState and the episode EnvironmentConfig."""
    r, w, pad = cfg.rocket, cfg.world, cfg.pad
    h = max(0.0, lowest_point_height(state, r))
    ax_des = float(np.clip(-0.12 * (state.x - pad.x) - 0.9 * state.vx, -3.0, 3.0))
    mass = total_mass(state, r)
    max_acc = r.max_thrust / mass
    theta_des = float(np.clip(
        math.asin(np.clip(-ax_des / max(max_acc, 1e-6), -0.45, 0.45)), -0.45, 0.45))
    inertia = mass * r.length ** 2 / 12.0
    alpha_des = 12.0 * (theta_des - state.theta) - 7.0 * state.omega
    rotation = float(np.clip(alpha_des * inertia / r.max_torque, -1.0, 1.0))
    vy_des = -min(10.0, 1.5 + 0.25 * h)
    ay_des = float(np.clip(0.9 * (vy_des - state.vy), -8.0, 5.0))
    throttle = float(np.clip(
        mass * (w.gravity + ay_des) / (r.max_thrust * max(math.cos(state.theta), 0.3)), 0.0, 1.0))
    return throttle, rotation


def expert_action(task) -> np.ndarray:
    """Expert action in the environment's normalised action space [-1, 1]^2."""
    throttle, rotation = expert_controller(task.engine.state, task.episode_config)
    return np.array([2.0 * throttle - 1.0, rotation], dtype=np.float32)
