"""Training callbacks. Everything recorded here comes from the real training loop."""
from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import BaseCallback

from agent.model_manager import ModelManager
from training.evaluation import evaluate, format_result, ranking_key, sb3_predictor


class ParameterUpdateCallback(BaseCallback):
    """Measures how much the policy's parameters change across each PPO update.

    PPO collects a whole rollout with FIXED parameters, then updates them in
    n_epochs x minibatch gradient steps. A snapshot is taken each time a rollout
    starts (i.e. right after the previous update); the difference between
    consecutive snapshots is exactly one PPO update. Parameters never change
    between those points, so nothing is recorded per environment step.
    """

    def __init__(self, csv_path: Path | str | None = None, keep: int = 500):
        super().__init__()
        self.csv_path = Path(csv_path) if csv_path else None
        self.history: deque = deque(maxlen=keep)
        self._prev = None
        self._fh = None
        self._writer = None
        self.n_updates = 0

    def _snapshot(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.policy.state_dict().items()
                if v.is_floating_point()}

    def _on_training_start(self) -> None:
        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            self._fh = open(self.csv_path, "a", newline="")
            self._writer = csv.writer(self._fh)
            if new:
                self._writer.writerow(["update", "num_timesteps", "l2_change", "relative_change", "fraction_changed"])

    def _on_rollout_start(self) -> None:
        snap = self._snapshot()
        if self._prev is not None:
            sq = prev_sq = 0.0
            changed = total = 0
            layers = {}
            for name, cur in snap.items():
                d = cur - self._prev[name]
                l2 = float(torch.linalg.vector_norm(d))
                n_changed = int((d != 0).sum())
                sq += l2 * l2
                prev_sq += float(torch.linalg.vector_norm(self._prev[name])) ** 2
                changed += n_changed
                total += d.numel()
                layers[name] = {"l2": l2, "fraction_changed": n_changed / d.numel()}
            self.n_updates += 1
            rec = {"update": self.n_updates, "num_timesteps": int(self.model.num_timesteps),
                   "l2_change": sq ** 0.5, "relative_change": (sq ** 0.5) / max(prev_sq ** 0.5, 1e-12),
                   "fraction_changed": changed / max(total, 1), "layers": layers}
            self.history.append(rec)
            self.logger.record("astral/param_update_l2", rec["l2_change"])
            self.logger.record("astral/param_update_fraction_changed", rec["fraction_changed"])
            if self._writer is not None:
                self._writer.writerow([rec["update"], rec["num_timesteps"], rec["l2_change"],
                                       rec["relative_change"], rec["fraction_changed"]])
                self._fh.flush()
        self._prev = snap

    def _on_step(self) -> bool:
        return True

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class EpisodeStatsCallback(BaseCallback):
    """Logs every finished TRAINING episode to CSV and a rolling summary to the SB3 logger."""

    FIELDS = ["num_timesteps", "outcome", "is_success", "return", "length", "fuel_used",
              "pad_offset", "touchdown_vertical_speed", "touchdown_horizontal_speed",
              "touchdown_angle", "touchdown_angular_speed", "final_x_offset", "final_y"]

    def __init__(self, csv_path: Path | str | None = None, window: int = 100):
        super().__init__()
        self.csv_path = Path(csv_path) if csv_path else None
        self.recent: deque = deque(maxlen=window)
        self.episodes_completed = 0
        self._fh = None
        self._writer = None

    def _on_training_start(self) -> None:
        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            self._fh = open(self.csv_path, "a", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
            if new:
                self._writer.writeheader()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            s = info.get("episode_summary")
            if s is None:
                continue
            td = info.get("touchdown") or {}
            row = {"num_timesteps": int(self.num_timesteps), "outcome": s["outcome"],
                   "is_success": int(s["is_success"]), "return": s["return"], "length": s["length"],
                   "fuel_used": s["fuel_used"], "pad_offset": td.get("pad_offset"),
                   "touchdown_vertical_speed": td.get("vertical_speed"),
                   "touchdown_horizontal_speed": td.get("horizontal_speed"),
                   "touchdown_angle": td.get("angle"), "touchdown_angular_speed": td.get("angular_speed"),
                   "final_x_offset": s.get("final_x_offset"), "final_y": s.get("final_y")}
            self.recent.append(row)
            self.episodes_completed += 1
            if self._writer is not None:
                self._writer.writerow(row)
                self._fh.flush()
        return True

    def _on_rollout_end(self) -> None:
        if not self.recent:
            return
        n = len(self.recent)
        rec = self.logger.record
        rec("astral/episodes_completed", self.episodes_completed)
        rec("astral/train_success_rate", sum(r["is_success"] for r in self.recent) / n)
        rec("astral/train_crash_rate", sum(r["outcome"] in ("crash", "side_collision") for r in self.recent) / n)
        rec("astral/train_mean_return", sum(r["return"] for r in self.recent) / n)
        rec("astral/train_mean_fuel_used", sum(r["fuel_used"] for r in self.recent) / n)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class CheckpointEvalCallback(BaseCallback):
    """Every eval_freq timesteps: evaluate deterministically on fixed held-out seeds, save a checkpoint.

    The 'best' checkpoint is chosen by evaluation (success rate, then mean return),
    not by the latest training episode.
    """

    def __init__(self, manager: ModelManager, run_name: str, training_config, env_config, reward_config,
                 train_scenario, eval_scenario):
        super().__init__()
        self.manager, self.run_name = manager, run_name
        self.cfg, self.env_cfg, self.reward_cfg = training_config, env_config, reward_config
        self.train_scenario, self.eval_scenario = train_scenario, eval_scenario
        self._last_eval = None
        self.best_score = manager.best_score(run_name)
        self.last_result = None

    def _on_training_start(self) -> None:
        # BaseCallback.num_timesteps is only synced on the first step; read the model directly,
        # otherwise a resumed run (e.g. at 100k) would evaluate immediately.
        self._last_eval = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.cfg.eval_freq:
            self._last_eval = int(self.num_timesteps)
            self.evaluate_and_save()
        return True

    def evaluate_and_save(self) -> None:
        res = evaluate(sb3_predictor(self.model), self.env_cfg, self.eval_scenario, self.reward_cfg,
                       self.cfg.eval_episodes, self.cfg.eval_seed, label=f"eval@{self.num_timesteps}")
        self.last_result = res
        score = ranking_key(res.summary())
        is_best = self.best_score is None or score > self.best_score
        if is_best:
            self.best_score = score
        tag = f"step_{int(self.num_timesteps):09d}"
        self.manager.save(self.model, self.run_name, tag, training_config=self.cfg, env_config=self.env_cfg,
                          reward_config=self.reward_cfg, scenario_config=self.train_scenario,
                          eval_result=res, is_best=is_best)
        self.logger.record("astral/eval_success_rate", res.success_rate)
        self.logger.record("astral/eval_mean_return", res.mean_return)
        if self.cfg.verbose:
            print("\n" + format_result(res) + ("\n  ** new best checkpoint **" if is_best else "") + "\n")
