"""`python main.py check-env`: validate the Gymnasium env and run random rollouts."""
from __future__ import annotations

import numpy as np


def run_env_check(episodes: int = 20, seed: int = 0) -> None:
    try:
        from gymnasium.utils.env_checker import check_env
    except ImportError as exc:
        raise SystemExit("gymnasium is required: pip install gymnasium") from exc
    from environment.rocket_env import RocketLandingEnv

    env = RocketLandingEnv()
    check_env(env, skip_render_check=True)
    print("gymnasium check_env: PASSED")

    outcomes: dict[str, int] = {}
    lengths, returns = [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(env.action_space.sample())
            done = term or trunc
        outcomes[info["outcome"]] = outcomes.get(info["outcome"], 0) + 1
        lengths.append(info["episode_summary"]["length"])
        returns.append(info["episode_summary"]["return"])
    print(f"random policy over {episodes} episodes (this is an UNTRAINED baseline):")
    print(f"  outcomes: {outcomes}")
    print(f"  mean length {np.mean(lengths):.0f} steps, mean return {np.mean(returns):.1f}")
