"""Project Astral: behaviour-clone PPO from expert demonstrations, then PPO fine-tune.

Run from the project_astral root:
    python train_expert_initialized_ppo.py

Requires the previously generated:
    data/expert_varied/expert_demos.npz

Writes new, separate outputs under:
    data/models/expert_initialized_ppo.zip
    data/models/expert_initialized_ppo_final.zip

Does not overwrite existing training runs/checkpoints.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from config.environment_config import EnvironmentConfig
from config.reward_config import RewardConfig
from config.scenario_config import ScenarioConfig
from config.training_config import TrainingConfig
from training.trainer import build_vec_env
from agent.ppo_agent import build_ppo
from training.evaluation import evaluate, sb3_predictor, format_result


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "expert_varied" / "expert_demos.npz"
OUT = ROOT / "data" / "models"
BC_STEPS = 2500
PPO_STEPS = 150_000


def main():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Expert dataset not found: {DATASET}\n"
            "Run collect_varied_expert_data.py first."
        )

    data = np.load(DATASET, allow_pickle=False)
    obs = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    success_mask = data["success_mask"].astype(bool)

    if obs.ndim != 2 or obs.shape[1] != 12:
        raise ValueError(f"Expected observations shape (N, 12), got {obs.shape}")
    if actions.shape != (len(obs), 2):
        raise ValueError(f"Expected actions shape (N, 2), got {actions.shape}")
    if len(success_mask) != len(obs):
        raise ValueError("success_mask length does not match observations")
    if not success_mask.any():
        raise ValueError("No successful expert transitions in dataset")
    if not np.isfinite(obs).all() or not np.isfinite(actions).all():
        raise ValueError("Dataset contains NaN or infinite values")

    # Train only on transitions belonging to fully successful episodes.
    x = torch.as_tensor(obs[success_mask], dtype=torch.float32)
    y = torch.as_tensor(actions[success_mask], dtype=torch.float32)
    print(f"Loaded {len(x):,} successful transitions from {DATASET}")
    print(f"Observation shape: {tuple(x.shape)} | action shape: {tuple(y.shape)}")

    # Match the demonstration scenario distribution for initial fine-tuning.
    scenario = ScenarioConfig(
        x_offset=(-10.0, 10.0), y=(20.0, 40.0),
        vx=(-1.0, 1.0), vy=(-0.5, 0.0),
        theta=(-0.08, 0.08), omega=(-0.08, 0.08),
        max_episode_steps=900,
    )
    cfg = TrainingConfig(
        seed=7, n_envs=4, n_steps=256, batch_size=256,
        n_epochs=5, learning_rate=1e-4, ent_coef=0.001,
        eval_freq=50_000, eval_episodes=20,
        final_eval_episodes=50, use_tensorboard=False,
        verbose=1, device="cpu",
    )
    env_cfg, reward_cfg = EnvironmentConfig(), RewardConfig()
    vec_env = build_vec_env(cfg, env_cfg, scenario, reward_cfg)
    model = build_ppo(vec_env, cfg)

    # Behaviour cloning: regress the policy's deterministic Gaussian mean
    # toward the expert's normalized continuous actions.
    policy = model.policy
    actor_params = list(policy.mlp_extractor.policy_net.parameters())
    actor_params += list(policy.action_net.parameters())
    optimizer = torch.optim.Adam(actor_params, lr=3e-4)

    rng = np.random.default_rng(7)
    batch_size = 512
    policy.set_training_mode(True)
    for step in range(BC_STEPS):
        idx = rng.integers(0, len(x), size=min(batch_size, len(x)))
        xb, yb = x[idx], y[idx]
        features = policy.extract_features(xb)
        latent_pi, _ = policy.mlp_extractor(features)
        predicted = policy.action_net(latent_pi)
        loss = torch.nn.functional.mse_loss(predicted, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_params, 0.5)
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"[behaviour cloning] step {step+1}/{BC_STEPS} | action MSE={loss.item():.6f}")

    OUT.mkdir(parents=True, exist_ok=True)
    bc_path = OUT / "expert_initialized_ppo"
    model.save(str(bc_path))
    print(f"\nSaved behaviour-cloned PPO initialization: {bc_path}.zip")

    # PPO now updates this same policy using environment rewards.
    print(f"\nStarting PPO fine-tuning for {PPO_STEPS:,} timesteps...")
    model.learn(total_timesteps=PPO_STEPS, reset_num_timesteps=True)
    final_path = OUT / "expert_initialized_ppo_final"
    model.save(str(final_path))
    print(f"Saved PPO fine-tuned model: {final_path}.zip")

    result = evaluate(
        sb3_predictor(model), env_cfg, scenario, reward_cfg,
        n_episodes=50, seed=987654, label="PPO expert-initialized final"
    )
    print("\n" + format_result(result))
    print("\nNOTE: This is a first experimental run. Compare it with the expert "
          "controller on the same scenario distribution; do not assume it "
          "has learned robust landing unless evaluation supports that claim.")
    vec_env.close()


if __name__ == "__main__":
    main()
