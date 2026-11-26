"""Checkpoint management. Never silently overwrites; every checkpoint is a new directory.

Layout:  data/checkpoints/<run>/run.json
         data/checkpoints/<run>/<tag>/model.zip       SB3 save: policy + OPTIMIZER state + hyperparameters
         data/checkpoints/<run>/<tag>/metadata.json   timesteps, all configs, eval results, library versions
         data/checkpoints/<run>/<tag>/rng_state.pkl   python/numpy/torch RNG states

Resume restores the model, optimizer state and the timestep counter. It does NOT
restore environment RNG streams or in-progress episodes, so a resumed run is a
faithful continuation of learning but not bit-for-bit identical to an
uninterrupted one.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import re
import shutil
import sys
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

from config.training_config import DATA_DIR

SCHEMA_VERSION = 1


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for pkg in ("torch", "stable-baselines3", "gymnasium", "numpy"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _as_dict(x):
    return asdict(x) if hasattr(x, "__dataclass_fields__") else x


class ModelManager:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else DATA_DIR / "checkpoints"

    # ---------------------------------------------------------------- paths
    def run_dir(self, run_name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
            raise ValueError("run name may only contain letters, digits, '_', '-', '.'")
        return self.root / run_name

    def read_run(self, run_name: str) -> dict:
        p = self.run_dir(run_name) / "run.json"
        if p.exists():
            return json.loads(p.read_text())
        return {"run_name": run_name, "created_utc": _utcnow(), "checkpoints": [], "best_checkpoint": None}

    def checkpoint_dir(self, run_name: str, which: str = "latest") -> Path:
        run = self.read_run(run_name)
        if not run["checkpoints"]:
            raise FileNotFoundError(f"No checkpoints recorded for run '{run_name}'")
        if which == "latest":
            tag = run["checkpoints"][-1]["tag"]
        elif which == "best":
            tag = run["best_checkpoint"] or run["checkpoints"][-1]["tag"]
        else:
            tag = which
        d = self.run_dir(run_name) / tag
        if not d.exists():
            raise FileNotFoundError(d)
        return d

    def best_score(self, run_name: str):
        """ranking_key of the current best checkpoint, or None."""
        run = self.read_run(run_name)
        for c in run["checkpoints"]:
            if c["tag"] == run["best_checkpoint"] and c.get("eval"):
                from training.evaluation import ranking_key
                return ranking_key(c["eval"])
        return None

    # ----------------------------------------------------------------- save
    def save(self, model, run_name: str, tag: str, *, training_config, env_config, reward_config,
             scenario_config, eval_result=None, is_best: bool = False, notes: str = "") -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
            raise ValueError("invalid checkpoint tag")
        run_dir = self.run_dir(run_name)
        final = run_dir / tag
        if final.exists():
            raise FileExistsError(f"Checkpoint already exists and will not be overwritten: {final}")
        run_dir.mkdir(parents=True, exist_ok=True)
        tmp = run_dir / (tag + ".partial")
        if tmp.exists():                 # leftover of an interrupted save, never a finished checkpoint
            shutil.rmtree(tmp)
        tmp.mkdir()

        model.save(str(tmp / "model"))   # -> model.zip
        import torch
        with open(tmp / "rng_state.pkl", "wb") as f:
            pickle.dump({"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state()}, f)
        meta = {
            "schema_version": SCHEMA_VERSION, "run_name": run_name, "tag": tag,
            "created_utc": _utcnow(), "num_timesteps": int(model.num_timesteps), "notes": notes,
            "training_config": _as_dict(training_config), "env_config": _as_dict(env_config),
            "reward_config": _as_dict(reward_config), "scenario_config": _as_dict(scenario_config),
            "evaluation": eval_result.to_dict() if eval_result is not None else None,
            "versions": _versions(),
        }
        _write_json_atomic(tmp / "metadata.json", meta)
        os.replace(tmp, final)

        run = self.read_run(run_name)
        run["checkpoints"].append({
            "tag": tag, "num_timesteps": int(model.num_timesteps), "created_utc": meta["created_utc"],
            "eval": eval_result.summary() if eval_result is not None else None, "is_best": bool(is_best)})
        if is_best:
            run["best_checkpoint"] = tag
        _write_json_atomic(run_dir / "run.json", run)
        return final

    # ----------------------------------------------------------------- load
    def load(self, checkpoint_dir: Path | str, env=None, device: str = "cpu", restore_rng: bool = False):
        from stable_baselines3 import PPO
        d = Path(checkpoint_dir)
        meta = json.loads((d / "metadata.json").read_text())
        saved, now = meta.get("versions", {}), _versions()
        for pkg in ("torch", "stable-baselines3"):
            if saved.get(pkg) and now.get(pkg) and saved[pkg] != now[pkg]:
                warnings.warn(f"Checkpoint saved with {pkg} {saved[pkg]}, loading with {now[pkg]}")
        model = PPO.load(str(d / "model.zip"), env=env, device=device)
        if restore_rng and (d / "rng_state.pkl").exists():
            import torch
            with open(d / "rng_state.pkl", "rb") as f:
                st = pickle.load(f)
            random.setstate(st["python"])
            np.random.set_state(st["numpy"])
            torch.set_rng_state(st["torch"])
        return model, meta
