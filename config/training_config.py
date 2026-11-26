"""PPO and training-loop configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    # --- PPO ---
    n_envs: int = 8                  # parallel copies of the env in one process
    n_steps: int = 512               # steps per env per rollout -> rollout = n_envs * n_steps
    batch_size: int = 256            # minibatch size; must divide the rollout size
    n_epochs: int = 10               # passes over each rollout
    learning_rate: float = 3e-4
    gamma: float = 0.995             # PPO discount (reward shaping has its own gamma = 1)
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005          # entropy regularisation
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # gradient clipping threshold
    target_kl: float | None = None
    # --- network ---
    net_arch: tuple[int, ...] = (128, 128)   # same sizes for actor and critic
    activation: str = "tanh"
    log_std_init: float = -0.5       # initial action std = exp(-0.5) ~ 0.61
    device: str = "cpu"
    # --- evaluation / checkpoints ---
    eval_freq: int = 50_000          # timesteps between evaluations (each saves a checkpoint)
    eval_episodes: int = 20
    eval_seed: int = 10_000          # fixed so checkpoints are compared on identical scenarios
    final_eval_episodes: int = 100   # final report uses a DIFFERENT seed (eval_seed + 1)
    use_tensorboard: bool = True
    verbose: int = 1

    @property
    def rollout_size(self) -> int:
        return self.n_envs * self.n_steps

    def validate(self) -> None:
        if self.rollout_size % self.batch_size != 0:
            raise ValueError(f"batch_size ({self.batch_size}) must divide n_envs*n_steps ({self.rollout_size})")
        if self.activation not in ("tanh", "relu"):
            raise ValueError("activation must be 'tanh' or 'relu'")
        if self.n_envs < 1 or self.n_steps < 2:
            raise ValueError("n_envs >= 1 and n_steps >= 2 required")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["net_arch"] = list(self.net_arch)
        return d
