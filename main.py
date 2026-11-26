"""Project Astral entry point.

Phase 1 provides manual flight. Training / analysis / autonomous modes are added
in later phases.
"""
from __future__ import annotations

import argparse

from config.environment_config import EnvironmentConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Project Astral - RL rocket landing laboratory")
    sub = p.add_subparsers(dest="mode")
    m = sub.add_parser("manual", help="fly the rocket yourself with the keyboard")
    m.add_argument("--start-x", type=float, default=25.0)
    m.add_argument("--start-y", type=float, default=110.0)
    m.add_argument("--start-vx", type=float, default=0.0)
    m.add_argument("--smoke-frames", type=int, default=0,
                   help="render N frames with a dummy video driver and exit (testing)")
    c = sub.add_parser("check-env", help="validate the Gymnasium environment (needs gymnasium)")
    c.add_argument("--episodes", type=int, default=20)
    t = sub.add_parser("train", help="train the PPO agent (headless)")
    t.add_argument("--run-name", default=None)
    t.add_argument("--timesteps", type=int, default=200_000)
    t.add_argument("--envs", type=int, default=None)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--device", default=None, help="cpu (default) or cuda")
    t.add_argument("--eval-freq", type=int, default=None)
    t.add_argument("--eval-episodes", type=int, default=None)
    t.add_argument("--no-tensorboard", action="store_true")
    t.add_argument("--start", choices=["low", "tall"], default="low",
                   help="low: start 20-40 m up, 15 s limit (default) | tall: 60-100 m up, 30 s limit (first run)")
    t.add_argument("--resume", default=None, help="path to a checkpoint directory to continue from")
    e = sub.add_parser("evaluate", help="evaluate a saved model (no learning)")
    e.add_argument("--checkpoint", default=None)
    e.add_argument("--run", default=None)
    e.add_argument("--which", default="best", help="best | latest | <tag>")
    e.add_argument("--episodes", type=int, default=100)
    e.add_argument("--seed", type=int, default=12345)
    e.add_argument("--start", choices=["low", "tall"], default="low")
    return p


def main() -> None:
    args = build_parser().parse_args()
    config = EnvironmentConfig()
    if args.mode == "train":
        from modes.train_mode import run_train
        run_train(args)
    elif args.mode == "evaluate":
        from modes.train_mode import run_evaluate
        run_evaluate(args)
    elif args.mode == "check-env":
        from modes.env_check import run_env_check
        run_env_check(args.episodes)
    elif args.mode in (None, "manual"):
        from modes.manual_mode import run_manual
        run_manual(config, getattr(args, "start_x", 25.0), getattr(args, "start_y", 110.0),
                   getattr(args, "start_vx", 0.0), getattr(args, "smoke_frames", 0))


if __name__ == "__main__":
    main()
