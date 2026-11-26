"""Fair comparison: hand-written controller vs cloned model vs cloned+PPO model.

Run from the project_astral folder:
    python compare_models.py
    python compare_models.py --episodes 50 --levels L1 L2
    python compare_models.py --model scratch=data/checkpoints/stage1c/final_step_xxx/model.zip

All policies fly IDENTICAL scenarios (same seed per level). Nothing is trained or modified.
Results are also saved to data/exports/compare_models_<timestamp>.csv (never overwrites).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from agent.expert_controller import expert_action
from config.benchmark_levels import LEVELS
from config.environment_config import EnvironmentConfig
from config.reward_config import RewardConfig
from environment.landing_task import LandingTask
from training.evaluation import SOFT_OUTCOMES, wilson_interval

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = [
    ("cloned only", ROOT / "data/models/expert_initialized_ppo.zip"),
    ("cloned + PPO", ROOT / "data/models/expert_initialized_ppo_final.zip"),
]


def run_policy(act, scenario, n_episodes: int, seed: int) -> dict:
    task = LandingTask(EnvironmentConfig(), scenario, RewardConfig())
    rng = np.random.default_rng(seed)
    outcomes, rets, fuel, pad_err, td_speed = {}, [], [], [], []
    for _ in range(n_episodes):
        obs, _ = task.reset(rng)
        while True:
            obs, _, terminated, truncated, info = task.step(act(task, obs))
            if terminated or truncated:
                break
        s, td = info["episode_summary"], info.get("touchdown")
        outcomes[s["outcome"]] = outcomes.get(s["outcome"], 0) + 1
        rets.append(s["return"]); fuel.append(s["fuel_used"])
        if td:
            pad_err.append(abs(td["pad_offset"])); td_speed.append(td["vertical_speed"])
    k = outcomes.get("success", 0)
    lo, hi = wilson_interval(k, n_episodes)
    return {"n": n_episodes, "successes": k, "rate": k / n_episodes, "ci_lo": lo, "ci_hi": hi,
            "soft": sum(outcomes.get(o, 0) for o in SOFT_OUTCOMES) / n_episodes,
            "outcomes": outcomes, "mean_return": float(np.mean(rets)), "mean_fuel": float(np.mean(fuel)),
            "pad_err": float(np.mean(pad_err)) if pad_err else float("nan"),
            "td_speed": float(np.mean(td_speed)) if td_speed else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=PATH",
                    help="add a model (repeatable). If none given, the two default expert-initialised models are used")
    args = ap.parse_args()

    policies = [("hand-written controller (not AI)", lambda task, obs: expert_action(task))]
    specs = [(m.split("=", 1)[0], Path(m.split("=", 1)[1])) for m in args.model] or DEFAULT_MODELS
    models = []
    for label, path in specs:
        if not path.exists():
            print(f"[skip] {label}: file not found: {path}")
            continue
        if not models:
            try:
                from stable_baselines3 import PPO
            except ImportError:
                raise SystemExit("stable-baselines3 is needed to load models (run inside the venv)")
        from stable_baselines3 import PPO
        m = PPO.load(str(path), device="cpu")
        policies.append((label, (lambda mm: (lambda task, obs: mm.predict(obs, deterministic=True)[0]))(m)))
        models.append(label)

    rows = []
    for li, level in enumerate(args.levels):
        desc, scenario = LEVELS[level]
        print(f"\n=== {level}: {desc}  ({args.episodes} identical flights per policy) ===")
        print(f"{'policy':<34}{'success':>16}{'95% CI':>15}{'soft TD':>9}{'pad err':>9}{'TD m/s':>8}  outcomes")
        for name, act in policies:
            r = run_policy(act, scenario, args.episodes, args.seed + li)
            print(f"{name:<34}{r['rate']:>9.0%} ({r['successes']:>3}/{r['n']:<3}){r['ci_lo']:>7.0%}-{r['ci_hi']:<6.0%}"
                  f"{r['soft']:>8.0%}{r['pad_err']:>9.1f}{r['td_speed']:>8.2f}  {r['outcomes']}")
            rows.append({"level": level, "policy": name, **{k: v for k, v in r.items() if k != "outcomes"},
                         "outcomes": str(r["outcomes"])})

    out = ROOT / "data" / "exports"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"compare_models_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {path}")
    print("Reading guide: 'soft TD' = touched down within safe speed and tilt limits. L2/L3 are outside the "
          "training distribution. Success rates come with sample size and 95% CI; small gaps are not real differences.")


if __name__ == "__main__":
    main()
