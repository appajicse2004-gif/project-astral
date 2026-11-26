"""Evaluation: run a policy on fresh episodes and report honest statistics.

Deliberately independent of Stable-Baselines3: it takes any predict(obs)->action
callable, so it is also testable without ML libraries. Evaluation never updates
a model. Success rates are always reported with the sample size and a 95%
Wilson confidence interval, because 'it landed 3 times' is not 'it can land'.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from config.environment_config import EnvironmentConfig
from config.reward_config import RewardConfig
from config.scenario_config import ScenarioConfig
from environment.landing_task import LandingTask


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# Outcomes whose touchdown was within the safe vertical speed and tilt limits
# (they failed only on sideways/spin speed or on position), i.e. a controlled descent.
SOFT_OUTCOMES = ("success", "off_pad", "excessive_velocity")


def ranking_key(summary: dict) -> tuple:
    """Order checkpoints: successes first, then controlled touchdowns, then return.

    Ranking by return alone let a model that hovers until timeout outrank one that
    is actually attempting landings (seen at 550k steps of the first run).
    """
    return (summary["success_rate"], summary.get("soft_touchdown_rate", 0.0), summary["mean_return"])


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


@dataclass
class EvalResult:
    label: str
    n_episodes: int
    seed: int
    n_success: int
    success_rate: float
    success_ci95: tuple[float, float]
    mean_return: float
    std_return: float
    mean_length: float
    outcomes: dict
    mean_pad_error: float | None        # mean |offset from pad centre| at touchdown (all touchdowns)
    mean_touchdown_speed: float | None  # mean downward speed at touchdown
    mean_fuel_used: float
    soft_touchdown_rate: float = 0.0   # touched down within the safe vertical speed AND tilt limits
    episodes: list = field(default_factory=list)   # compact per-episode records

    def to_dict(self, include_episodes: bool = True) -> dict:
        d = asdict(self)
        d["success_ci95"] = list(self.success_ci95)
        if not include_episodes:
            d.pop("episodes")
        return d

    def summary(self) -> dict:
        return {"label": self.label, "n_episodes": self.n_episodes, "success_rate": self.success_rate,
                "success_ci95": list(self.success_ci95), "mean_return": self.mean_return,
                "soft_touchdown_rate": self.soft_touchdown_rate}


def evaluate(predict: Callable[[np.ndarray], np.ndarray], env_config: EnvironmentConfig,
             scenario_config: ScenarioConfig, reward_config: RewardConfig,
             n_episodes: int, seed: int, label: str = "eval") -> EvalResult:
    task = LandingTask(env_config, scenario_config, reward_config)
    rng = np.random.default_rng(seed)
    records, outcomes = [], {}
    for _ in range(n_episodes):
        obs, info0 = task.reset(rng)
        while True:
            obs, _, terminated, truncated, info = task.step(predict(obs))
            if terminated or truncated:
                break
        s = info["episode_summary"]
        td = info.get("touchdown")
        outcomes[s["outcome"]] = outcomes.get(s["outcome"], 0) + 1
        records.append({
            "outcome": s["outcome"], "is_success": bool(s["is_success"]), "return": s["return"],
            "length": s["length"], "fuel_used": s["fuel_used"],
            "pad_error": abs(td["pad_offset"]) if td else None,
            "touchdown_speed": td["vertical_speed"] if td else None,
            "scenario": info0["scenario"],
        })
    n_success = sum(r["is_success"] for r in records)
    returns = [r["return"] for r in records]
    return EvalResult(
        label=label, n_episodes=n_episodes, seed=seed, n_success=n_success,
        success_rate=n_success / n_episodes if n_episodes else 0.0,
        success_ci95=wilson_interval(n_success, n_episodes),
        mean_return=float(np.mean(returns)) if returns else 0.0,
        std_return=float(np.std(returns)) if returns else 0.0,
        mean_length=float(np.mean([r["length"] for r in records])) if records else 0.0,
        outcomes=outcomes,
        mean_pad_error=_mean([r["pad_error"] for r in records if r["pad_error"] is not None]),
        mean_touchdown_speed=_mean([r["touchdown_speed"] for r in records if r["touchdown_speed"] is not None]),
        mean_fuel_used=float(np.mean([r["fuel_used"] for r in records])) if records else 0.0,
        soft_touchdown_rate=(sum(r["outcome"] in SOFT_OUTCOMES for r in records) / n_episodes) if n_episodes else 0.0,
        episodes=records,
    )


def sb3_predictor(model) -> Callable[[np.ndarray], np.ndarray]:
    """Deterministic (mean-action) policy, the standard way to evaluate PPO."""
    return lambda obs: model.predict(obs, deterministic=True)[0]


def format_result(r: EvalResult) -> str:
    lo, hi = r.success_ci95
    lines = [
        f"[{r.label}] {r.n_episodes} episodes (seed {r.seed})",
        f"  success rate : {r.success_rate:.1%}  ({r.n_success}/{r.n_episodes}, 95% CI {lo:.1%}-{hi:.1%})",
        f"  outcomes     : {r.outcomes}",
        f"  mean return  : {r.mean_return:.1f} +/- {r.std_return:.1f}",
    ]
    if r.mean_pad_error is not None:
        lines.append(f"  landing error: {r.mean_pad_error:.2f} m from pad centre (mean over touchdowns)")
        lines.append(f"  touchdown spd: {r.mean_touchdown_speed:.2f} m/s (mean downward)")
    lines.append(f"  soft touchdowns: {r.soft_touchdown_rate:.0%} (within safe speed and tilt limits)")
    lines.append(f"  fuel used    : {r.mean_fuel_used:.1f} kg   mean length {r.mean_length:.0f} steps")
    return "\n".join(lines)
