"""Project Astral: collect expert demonstrations for PPO imitation pretraining.

Run from the project_astral root:
    python collect_expert_data.py

Writes:
    data/expert/expert_demos.npz
    data/expert/expert_summary.txt

This script does NOT modify source files or existing PPO checkpoints.
It uses the conventional controller only to generate labelled observations/actions.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from config.scenario_config import ScenarioConfig
from environment.landing_task import LandingTask
from environment.rocket import lowest_point_height, total_mass


def expert_controller(state, cfg):
    """PD lateral/attitude control plus conservative vertical-speed control."""
    r, w, pad = cfg.rocket, cfg.world, cfg.pad
    h = max(0.0, lowest_point_height(state, r))

    # Desired horizontal acceleration toward the pad, with velocity damping.
    ax_des = float(np.clip(
        -0.12 * (state.x - pad.x) - 0.9 * state.vx, -3.0, 3.0
    ))

    mass = total_mass(state, r)
    max_acc = r.max_thrust / mass
    theta_des = float(np.clip(
        math.asin(np.clip(-ax_des / max(max_acc, 1e-6), -0.45, 0.45)),
        -0.45, 0.45
    ))

    inertia = mass * r.length**2 / 12.0
    alpha_des = 12.0 * (theta_des - state.theta) - 7.0 * state.omega
    rotation = float(np.clip(alpha_des * inertia / r.max_torque, -1.0, 1.0))

    # Slow descent profile; reduce target descent speed near the pad.
    vy_des = -min(10.0, 1.5 + 0.25 * h)
    ay_des = float(np.clip(0.9 * (vy_des - state.vy), -8.0, 5.0))
    throttle = float(np.clip(
        mass * (w.gravity + ay_des) /
        (r.max_thrust * max(math.cos(state.theta), 0.3)),
        0.0, 1.0
    ))
    return throttle, rotation


def main():
    # Keep the first dataset aligned with the verified baseline's default
    # centred-start task. Broader randomized starts come only after this passes.
    scenario_cfg = ScenarioConfig()
    task = LandingTask(scenario_config=scenario_cfg)
    rng = np.random.default_rng(20261124)

    observations = []
    actions = []
    episode_ids = []
    episode_outcomes = []
    episode_returns = []
    episode_lengths = []

    n_episodes = 100
    for ep in range(n_episodes):
        obs, _ = task.reset(rng)
        ep_return = 0.0
        length = 0

        while True:
            state = task.engine.state
            throttle, rotation = expert_controller(state, task.episode_config)
            action = np.asarray([2.0 * throttle - 1.0, rotation], dtype=np.float32)

            # Store the observation BEFORE the action: this is the state the
            # policy must learn to map to the expert's action.
            observations.append(np.asarray(obs, dtype=np.float32).copy())
            actions.append(action.copy())
            episode_ids.append(ep)

            obs, reward, terminated, truncated, info = task.step(action)
            ep_return += float(reward)
            length += 1

            if terminated or truncated:
                summary = info["episode_summary"]
                episode_outcomes.append(str(summary["outcome"]))
                episode_returns.append(float(ep_return))
                episode_lengths.append(int(length))
                break

    out = Path("data/expert")
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "expert_demos.npz"

    obs_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32)
    episode_id_array = np.asarray(episode_ids, dtype=np.int32)
    success_mask = np.asarray(
        [episode_outcomes[i] == "success" for i in episode_id_array],
        dtype=np.bool_
    )

    np.savez_compressed(
        npz_path,
        observations=obs_array,
        actions=action_array,
        episode_ids=episode_id_array,
        success_mask=success_mask,
        episode_outcomes=np.asarray(episode_outcomes, dtype="U32"),
        episode_returns=np.asarray(episode_returns, dtype=np.float32),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int32),
    )

    counts = {}
    for outcome in episode_outcomes:
        counts[outcome] = counts.get(outcome, 0) + 1
    successes = counts.get("success", 0)

    report = [
        "PROJECT ASTRAL — EXPERT DEMONSTRATION COLLECTION",
        f"Episodes: {n_episodes}",
        f"Transitions: {len(obs_array)}",
        f"Observation shape: {obs_array.shape}",
        f"Action shape: {action_array.shape}",
        f"Outcomes: {counts}",
        f"Successful episodes: {successes}/{n_episodes}",
        f"Transitions from successful episodes: {int(success_mask.sum())}",
        f"Mean episode length: {np.mean(episode_lengths):.1f}",
        f"Dataset: {npz_path}",
        "",
        "Use successful-episode transitions for initial behaviour cloning.",
        "This dataset is an expert demonstration set, not PPO training.",
    ]
    report_text = "\n".join(report)
    (out / "expert_summary.txt").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
