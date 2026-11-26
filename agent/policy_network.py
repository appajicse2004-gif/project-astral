"""Network architecture settings and introspection.

Stable-Baselines3 builds the actor and critic inside ActorCriticPolicy
(shared nothing: separate 'pi' and 'vf' MLPs on the 12-d observation, plus a
learned log-std vector for the Gaussian action distribution). There is no
separate value_network module: that would just re-wrap the same object.
describe_network() reads the REAL parameter shapes so visualisations (Phase 5)
are drawn from the actual model rather than a hard-coded diagram.
"""
from __future__ import annotations

from config.training_config import TrainingConfig


def policy_kwargs(cfg: TrainingConfig) -> dict:
    import torch.nn as nn
    activations = {"tanh": nn.Tanh, "relu": nn.ReLU}
    return dict(
        net_arch=dict(pi=list(cfg.net_arch), vf=list(cfg.net_arch)),
        activation_fn=activations[cfg.activation],
        log_std_init=cfg.log_std_init,
    )


def describe_network(policy) -> list[dict]:
    """[{name, shape, numel}] for every learnable parameter tensor."""
    return [{"name": n, "shape": tuple(p.shape), "numel": int(p.numel())}
            for n, p in policy.named_parameters()]


def count_parameters(policy) -> int:
    return sum(p.numel() for p in policy.parameters())
