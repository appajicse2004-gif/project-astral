"""Headless training driver (Phase 3). The live dashboard in Phase 5 will run this in a worker process."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from agent.model_manager import ModelManager
from agent.ppo_agent import build_ppo
from config.environment_config import EnvironmentConfig
from config.reward_config import RewardConfig
from config.scenario_config import ScenarioConfig
from config.training_config import DATA_DIR, TrainingConfig
from training.callbacks import CheckpointEvalCallback, EpisodeStatsCallback, ParameterUpdateCallback
from training.evaluation import evaluate, format_result, ranking_key, sb3_predictor


@dataclass
class TrainResult:
    run_name: str
    timesteps: int
    interrupted: bool
    final_checkpoint: Path
    final_eval: object
    duration_s: float


def build_vec_env(cfg: TrainingConfig, env_config, scenario, reward_config):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from environment.rocket_env import RocketLandingEnv

    def make():
        return Monitor(RocketLandingEnv(env_config, scenario, reward_config))

    return DummyVecEnv([make for _ in range(cfg.n_envs)])


def train(cfg: TrainingConfig, run_name: str, total_timesteps: int, *, env_config=None, reward_config=None,
          train_scenario=None, eval_scenario=None, resume_from: Path | str | None = None,
          manager: ModelManager | None = None, logs_root: Path | None = None) -> TrainResult:
    cfg.validate()
    env_config = env_config or EnvironmentConfig()
    reward_config = reward_config or RewardConfig()
    train_scenario = train_scenario or ScenarioConfig()
    eval_scenario = eval_scenario or train_scenario
    manager = manager or ModelManager()
    logs_dir = Path(logs_root or DATA_DIR / "logs") / run_name

    if resume_from is None and manager.run_dir(run_name).exists():
        raise FileExistsError(f"Run '{run_name}' already exists. Pick a new --run-name or use --resume.")

    vec_env = build_vec_env(cfg, env_config, train_scenario, reward_config)
    tb_log = None
    if cfg.use_tensorboard:
        try:
            import tensorboard  # noqa: F401
            tb_log = str(logs_dir / "tb")
        except ImportError:
            print("tensorboard not installed - continuing without it")

    if resume_from is not None:
        model, meta = manager.load(resume_from, env=vec_env, device=cfg.device, restore_rng=True)
        vec_env.seed(cfg.seed + 1000 + int(model.num_timesteps))   # don't replay the first run's scenarios
        print(f"Resuming '{run_name}' from {meta['num_timesteps']:,} timesteps "
              f"(PPO hyperparameters come from the checkpoint, not the current config)")
    else:
        model = build_ppo(vec_env, cfg, tensorboard_log=tb_log)

    param_cb = ParameterUpdateCallback(logs_dir / "param_updates.csv")
    episode_cb = EpisodeStatsCallback(logs_dir / "episodes.csv")
    ckpt_cb = CheckpointEvalCallback(manager, run_name, cfg, env_config, reward_config, train_scenario, eval_scenario)

    from stable_baselines3.common.callbacks import CallbackList
    t0, interrupted = time.time(), False
    try:
        model.learn(total_timesteps=total_timesteps, callback=CallbackList([param_cb, episode_cb, ckpt_cb]),
                    reset_num_timesteps=resume_from is None, tb_log_name=run_name)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted - saving a checkpoint before exit...")
    finally:
        param_cb.close()
        episode_cb.close()

    # Final evaluation on a DIFFERENT seed than the one used to pick the best checkpoint.
    res = evaluate(sb3_predictor(model), env_config, eval_scenario, reward_config,
                   cfg.final_eval_episodes, cfg.eval_seed + 1, label="final")
    score = ranking_key(res.summary())
    best = ckpt_cb.best_score
    tag = f"{'interrupted' if interrupted else 'final'}_step_{int(model.num_timesteps):09d}"
    path = manager.save(model, run_name, tag, training_config=cfg, env_config=env_config,
                        reward_config=reward_config, scenario_config=train_scenario, eval_result=res,
                        is_best=best is None or score > best)
    vec_env.close()
    print(format_result(res))
    print(f"Saved: {path}")
    return TrainResult(run_name, int(model.num_timesteps), interrupted, path, res, time.time() - t0)
