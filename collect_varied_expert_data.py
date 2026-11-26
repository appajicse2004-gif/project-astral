"""Project Astral: collect demonstrations from varied initial conditions.

Run from the project_astral root:
    python collect_varied_expert_data.py

Writes:
    data/expert_varied/expert_demos.npz
    data/expert_varied/expert_summary.txt

Does not modify project source files or PPO checkpoints.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from config.scenario_config import ScenarioConfig
from environment.landing_task import LandingTask
from environment.rocket import lowest_point_height, total_mass


def expert_controller(state, cfg):
    r, w, pad = cfg.rocket, cfg.world, cfg.pad
    h = max(0.0, lowest_point_height(state, r))
    ax_des = float(np.clip(-0.12 * (state.x - pad.x) - 0.9 * state.vx, -3.0, 3.0))
    mass = total_mass(state, r)
    max_acc = r.max_thrust / mass
    theta_des = float(np.clip(
        math.asin(np.clip(-ax_des / max(max_acc, 1e-6), -0.45, 0.45)),
        -0.45, 0.45
    ))
    inertia = mass * r.length**2 / 12.0
    alpha_des = 12.0 * (theta_des - state.theta) - 7.0 * state.omega
    rotation = float(np.clip(alpha_des * inertia / r.max_torque, -1.0, 1.0))
    vy_des = -min(10.0, 1.5 + 0.25 * h)
    ay_des = float(np.clip(0.9 * (vy_des - state.vy), -8.0, 5.0))
    throttle = float(np.clip(
        mass * (w.gravity + ay_des) /
        (r.max_thrust * max(math.cos(state.theta), 0.3)),
        0.0, 1.0
    ))
    return throttle, rotation


def main():
    # Moderate disturbances first; avoid claiming broad robustness from a
    # narrow or unverified distribution.
    scenario_cfg = ScenarioConfig(
        x_offset=(-10.0, 10.0),
        y=(20.0, 40.0),
        vx=(-1.0, 1.0),
        vy=(-0.5, 0.0),
        theta=(-0.08, 0.08),
        omega=(-0.08, 0.08),
        max_episode_steps=900,
    )
    task = LandingTask(scenario_config=scenario_cfg)
    rng = np.random.default_rng(20261124)

    obs_rows, action_rows, episode_ids = [], [], []
    outcomes, returns, lengths, scenarios = [], [], [], []
    n_episodes = 100

    for ep in range(n_episodes):
        obs, reset_info = task.reset(rng)
        scenarios.append(reset_info["scenario"])
        ep_return, length = 0.0, 0

        while True:
            state = task.engine.state
            throttle, rotation = expert_controller(state, task.episode_config)
            action = np.asarray([2.0 * throttle - 1.0, rotation], dtype=np.float32)

            obs_rows.append(np.asarray(obs, dtype=np.float32).copy())
            action_rows.append(action.copy())
            episode_ids.append(ep)

            obs, reward, terminated, truncated, info = task.step(action)
            ep_return += float(reward)
            length += 1

            if terminated or truncated:
                summary = info["episode_summary"]
                outcomes.append(str(summary["outcome"]))
                returns.append(ep_return)
                lengths.append(length)
                break

    out = Path("data/expert_varied")
    out.mkdir(parents=True, exist_ok=True)

    obs_array = np.asarray(obs_rows, dtype=np.float32)
    action_array = np.asarray(action_rows, dtype=np.float32)
    episode_id_array = np.asarray(episode_ids, dtype=np.int32)
    success_by_episode = np.asarray([o == "success" for o in outcomes], dtype=bool)
    success_mask = success_by_episode[episode_id_array]

    np.savez_compressed(
        out / "expert_demos.npz",
        observations=obs_array,
        actions=action_array,
        episode_ids=episode_id_array,
        success_mask=success_mask,
        episode_outcomes=np.asarray(outcomes, dtype="U32"),
        episode_returns=np.asarray(returns, dtype=np.float32),
        episode_lengths=np.asarray(lengths, dtype=np.int32),
    )

    counts = {}
    for outcome in outcomes:
        counts[outcome] = counts.get(outcome, 0) + 1

    lines = [
        "PROJECT ASTRAL — VARIED EXPERT DEMONSTRATION COLLECTION",
        f"Episodes: {n_episodes}",
        f"Transitions: {len(obs_array)}",
        f"Observation shape: {obs_array.shape}",
        f"Action shape: {action_array.shape}",
        f"Outcomes: {counts}",
        f"Successful episodes: {int(success_by_episode.sum())}/{n_episodes}",
        f"Transitions from successful episodes: {int(success_mask.sum())}",
        f"Mean episode length: {np.mean(lengths):.1f}",
        "Scenario ranges: x_offset +/-10m, vx +/-1m/s, vy -0.5..0m/s, "
        "theta +/-0.08rad, omega +/-0.08rad/s, altitude 20..40m",
        f"Dataset: {out / 'expert_demos.npz'}",
        "",
        "Only transitions with success_mask=True should be used as successful "
        "expert demonstrations for initial behaviour cloning.",
        "Inspect success rate and scenario coverage before PPO training.",
    ]
    report = "\n".join(lines)
    (out / "expert_summary.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
