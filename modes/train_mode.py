"""CLI handlers for `train` and `evaluate`."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent.model_manager import ModelManager
from config.environment_config import EnvironmentConfig
from config.reward_config import RewardConfig
from config.scenario_config import PRESETS, ScenarioConfig
from config.training_config import TrainingConfig


def run_train(args) -> None:
    from training.trainer import train
    cfg = TrainingConfig()
    overrides = {k: v for k, v in dict(seed=args.seed, n_envs=args.envs, device=args.device,
                                       eval_freq=args.eval_freq, eval_episodes=args.eval_episodes).items()
                 if v is not None}
    if args.no_tensorboard:
        overrides["use_tensorboard"] = False
    cfg = replace(cfg, **overrides)
    if args.resume:
        resume = Path(args.resume)
        run_name = resume.parent.name
    else:
        resume, run_name = None, args.run_name
        if not run_name:
            raise SystemExit("--run-name is required when not resuming")
    train(cfg, run_name, args.timesteps, resume_from=resume, train_scenario=PRESETS[args.start])


def run_evaluate(args) -> None:
    from training.evaluation import evaluate, format_result, sb3_predictor
    manager = ModelManager()
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    elif args.run:
        ckpt = manager.checkpoint_dir(args.run, args.which)
    else:
        raise SystemExit("give --checkpoint PATH or --run NAME")
    model, meta = manager.load(ckpt, device="cpu")
    print(f"Loaded {ckpt.name}: {meta['num_timesteps']:,} training timesteps. Evaluation does NOT update the model.")
    res = evaluate(sb3_predictor(model), EnvironmentConfig(), PRESETS[args.start], RewardConfig(),
                   args.episodes, args.seed, label="evaluation")
    print(format_result(res))
