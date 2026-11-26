"""PPO construction (Stable-Baselines3). PPO itself is not re-implemented."""
from __future__ import annotations

from agent.policy_network import policy_kwargs
from config.training_config import TrainingConfig


def build_ppo(env, cfg: TrainingConfig, tensorboard_log: str | None = None):
    from stable_baselines3 import PPO
    cfg.validate()
    return PPO(
        "MlpPolicy", env,
        learning_rate=cfg.learning_rate, n_steps=cfg.n_steps, batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range, ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm, target_kl=cfg.target_kl,
        policy_kwargs=policy_kwargs(cfg), tensorboard_log=tensorboard_log,
        verbose=cfg.verbose, seed=cfg.seed, device=cfg.device,
    )
