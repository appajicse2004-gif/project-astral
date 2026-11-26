"""Visual comparison: how the rocket flies at each stage of training, on IDENTICAL starts.

Run from the project_astral folder (venv active):
    python compare_flights.py                     # all stages it can find, level L1
    python compare_flights.py --level L2          # harder starts
    python compare_flights.py --flights 12 --gif-flight 3
    python compare_flights.py --stage "my model=data/checkpoints/stage1c/final_step_000900000"

Writes (never overwrites, timestamped) into data/exports/:
    flight_comparison_<time>.png   trajectories, height and tilt for every stage
    flight_comparison_<time>.gif   one flight, all stages side by side, animated
Nothing is trained or modified. Open the GIF in a browser (Chrome/Edge) to see it move.

A picture of a handful of flights illustrates behaviour; it is not proof of performance.
Use compare_models.py for success rates with sample sizes and confidence intervals.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from agent.expert_controller import expert_action
from config.benchmark_levels import LEVELS
from config.environment_config import EnvironmentConfig
from config.reward_config import RewardConfig
from environment.landing_task import LandingTask

ROOT = Path(__file__).resolve().parent
DEFAULT_STAGES = [
    ("20k steps\n(first test run)", "data/checkpoints/smoke/final_step_000020480"),
    ("1M steps from scratch\n(stage1, old rewards)", "data/checkpoints/stage1/final_step_001003520"),
    ("cloned from\nhand-written controller", "data/models/expert_initialized_ppo.zip"),
    ("cloned + PPO\n(final)", "data/models/expert_initialized_ppo_final.zip"),
]
GREEN, RED, ORANGE, GREY = "#2ca25f", "#d7301f", "#e6850e", "#888888"
HULL_W, HULL_L = 2.0, 10.0


@dataclass
class Flight:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    throttle: np.ndarray
    outcome: str
    success: bool
    touchdown: dict | None


# ----------------------------------------------------------------------- flying
def fly(act, task: LandingTask, rng) -> Flight:
    obs, _ = task.reset(rng)
    s = task.engine.state
    rec = [(s.time, s.x, s.y, s.theta, s.throttle)]
    while True:
        obs, _, terminated, truncated, info = task.step(act(task, obs))
        s = task.engine.state
        rec.append((s.time, s.x, s.y, s.theta, s.throttle))
        if terminated or truncated:
            break
    a = np.asarray(rec)
    return Flight(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], info["outcome"], bool(info["is_success"]),
                  info.get("touchdown"))


def collect(stages, scenario, n_flights: int, seed: int) -> dict:
    """Every stage flies the same scenarios: same seed, and policies never touch the RNG."""
    out = {}
    for label, act in stages:
        task = LandingTask(EnvironmentConfig(), scenario, RewardConfig())
        rng = np.random.default_rng(seed)
        out[label] = [fly(act, task, rng) for _ in range(n_flights)]
    return out


def _color(f: Flight) -> str:
    if f.success:
        return GREEN
    return ORANGE if f.outcome in ("timeout", "out_of_bounds") else RED


# ---------------------------------------------------------------------- figures
def _limits(flights: dict, index: int | None = None, max_half: float = 100.0):
    pool = [f for fl in flights.values() for f in (fl if index is None else [fl[index]])]
    xmax = max(float(np.abs(f.x).max()) for f in pool)
    ymax = max(float(f.y.max()) for f in pool)
    half = max(20.0, min(max_half, math.ceil(xmax + 6)))
    return (-half, half), (0.0, max(30.0, ymax + 6))


def make_png(flights: dict, level_text: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(flights)
    n = len(labels)
    xlim, ylim = _limits(flights)
    fig, axes = plt.subplots(3, n, figsize=(3.4 * n, 11), squeeze=False,
                             gridspec_kw={"height_ratios": [1.5, 1, 1]})
    tmax = max(float(f.t[-1]) for fl in flights.values() for f in fl)
    for c, label in enumerate(labels):
        fl = flights[label]
        k = sum(f.success for f in fl)
        ax1, ax2, ax3 = axes[0][c], axes[1][c], axes[2][c]
        for f in fl:
            col = _color(f)
            ax1.plot(f.x, f.y, color=col, lw=1.2, alpha=0.8)
            ax1.plot(f.x[0], f.y[0], "o", color=GREY, ms=3)
            ax1.plot(f.x[-1], f.y[-1], "o" if f.success else "x", color=col, ms=6)
            ax2.plot(f.t, f.y, color=col, lw=1.1, alpha=0.8)
            ax3.plot(f.t, np.degrees(f.theta), color=col, lw=1.1, alpha=0.8)
        ax1.axhline(0, color="#444", lw=1)
        ax1.plot([-6, 6], [0, 0], color=GREEN, lw=6, solid_capstyle="butt", zorder=0)
        ax1.set_xlim(*xlim); ax1.set_ylim(*ylim)          # free aspect: keeps drift readable
        ax1.set_title(f"{label}\nlanded {k}/{len(fl)}", fontsize=10, color=GREEN if k else RED)
        ax1.set_xlabel("sideways position (m)  [green bar = pad]", fontsize=8)
        ax2.set_ylim(0, ylim[1]); ax2.set_xlim(0, tmax)
        ax2.set_xlabel("time (s)", fontsize=8)
        ax3.axhspan(-8.6, 8.6, color=GREEN, alpha=0.12)
        ax3.set_ylim(-45, 45); ax3.set_xlim(0, tmax)
        ax3.set_xlabel("time (s)", fontsize=8)
        if c == 0:
            ax1.set_ylabel("flight path: height (m)", fontsize=9)
            ax2.set_ylabel("height of rocket (m)", fontsize=9)
            ax3.set_ylabel("tilt (deg)  [shaded = safe]", fontsize=9)
        for ax in (ax2, ax3):
            ax.grid(alpha=0.25)
        if c > 0:
            ax2.set_yticklabels([]); ax3.set_yticklabels([])
    fig.suptitle(f"Project Astral: how the rocket flies at each stage   |   {level_text}\n"
                 f"{len(next(iter(flights.values())))} identical starting conditions per column.\n"
                 f"green = safe landing on the pad   red = failed landing   orange = timeout / left the map   "
                 f"(o = start or landing, x = failure)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_gif(flights: dict, index: int, path: Path, fps: int = 20, stride: int = 3) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon

    labels = list(flights)
    n = len(labels)
    xlim, ylim = _limits(flights, index, max_half=40.0)
    xr, yr = xlim[1] - xlim[0], ylim[1] + 1
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0 * yr / xr + 0.9), squeeze=False)
    axes = axes[0]
    artists = []
    for ax, label in zip(axes, labels):
        ax.set_xlim(*xlim); ax.set_ylim(-1, ylim[1]); ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#0b1020")
        ax.axhspan(-1, 0, color="#3a4660")
        ax.plot([-6, 6], [0, 0], color=GREEN, lw=5, solid_capstyle="butt")
        ax.set_title(label, fontsize=9)
        trail, = ax.plot([], [], color="#6f9bd8", lw=1)
        hull = Polygon(np.zeros((4, 2)), closed=True, color="#e8edf5", zorder=3)
        nose = Polygon(np.zeros((3, 2)), closed=True, color="#e6554a", zorder=4)
        flame, = ax.plot([], [], color="#ffae3c", lw=3, zorder=2)
        ax.add_patch(hull); ax.add_patch(nose)
        status = ax.text(0.03, 0.95, "", transform=ax.transAxes, ha="left", va="top", fontsize=9,
                         color="white", fontweight="bold")
        artists.append((trail, hull, nose, flame, status))

    lengths = [len(flights[l][index].t) for l in labels]
    n_frames = math.ceil(max(lengths) / stride) + fps * 2          # hold last frame ~2 s

    def update(k):
        for (label, (trail, hull, nose, flame, status)) in zip(labels, artists):
            f = flights[label][index]
            i = min(k * stride, len(f.t) - 1)
            x, y, th = f.x[i], f.y[i], f.theta[i]
            c, s = math.cos(th), math.sin(th)

            def world(lx, ly):
                return (x + lx * c - ly * s, y + lx * s + ly * c)
            hw, hl = HULL_W / 2, HULL_L / 2
            hull.set_xy([world(-hw, -hl), world(hw, -hl), world(hw, hl), world(-hw, hl)])
            nose.set_xy([world(-hw, hl), world(hw, hl), world(0, hl + 2.5)])
            if f.throttle[i] > 0.01:
                tip = world(0, -(hl + 1 + 7 * f.throttle[i]))
                base = world(0, -hl)
                flame.set_data([base[0], tip[0]], [base[1], tip[1]])
            else:
                flame.set_data([], [])
            trail.set_data(f.x[: i + 1], f.y[: i + 1])
            if i >= len(f.t) - 1:
                status.set_text(f.outcome.replace("_", " ").upper())
                status.set_color("#5be08f" if f.success else "#ff6b5e")
            else:
                status.set_text(f"t = {f.t[i]:.1f} s")
                status.set_color("white")
        return []

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return n_frames


# ----------------------------------------------------------------------- summary
def summarize(flights: dict) -> str:
    lines = [f"{'stage':<42}{'landed':>9}{'touchdown m/s':>15}{'pad error m':>13}  outcomes"]
    for label, fl in flights.items():
        outc: dict = {}
        for f in fl:
            outc[f.outcome] = outc.get(f.outcome, 0) + 1
        td = [f.touchdown for f in fl if f.touchdown]
        spd = np.mean([t["vertical_speed"] for t in td]) if td else float("nan")
        err = np.mean([abs(t["pad_offset"]) for t in td]) if td else float("nan")
        lines.append(f"{label.replace(chr(10), ' '):<42}{sum(f.success for f in fl):>5}/{len(fl):<3}"
                     f"{spd:>15.2f}{err:>13.1f}  {outc}")
    return "\n".join(lines)


# --------------------------------------------------------------------- stage setup
def _model_policy(model):
    return lambda task, obs: model.predict(obs, deterministic=True)[0]


def _resolve(path: str) -> Path:
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    return p / "model.zip" if p.is_dir() else p


def build_stages(args) -> list:
    stages = []
    try:
        from dataclasses import replace as _r
        from agent.ppo_agent import build_ppo
        from config.training_config import TrainingConfig
        from environment.rocket_env import RocketLandingEnv
        fresh = build_ppo(RocketLandingEnv(), _r(TrainingConfig(), verbose=0, use_tensorboard=False))
        stages.append(("untrained\n(random weights)", _model_policy(fresh)))
    except Exception as exc:                       # pragma: no cover
        print(f"[skip] untrained model could not be built: {exc}")

    specs = [(m.split("=", 1)[0], m.split("=", 1)[1]) for m in args.stage]
    if not args.no_defaults:
        specs = DEFAULT_STAGES + specs
    if specs:
        from stable_baselines3 import PPO
        for label, path in specs:
            p = _resolve(path)
            if not p.exists():
                print(f"[skip] {label.replace(chr(10), ' ')}: not found: {p}")
                continue
            stages.append((label, _model_policy(PPO.load(str(p), device="cpu"))))
    if not args.no_expert:
        stages.append(("hand-written controller\n(not AI, reference)", lambda task, obs: expert_action(task)))
    return stages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", choices=list(LEVELS), default="L1")
    ap.add_argument("--flights", type=int, default=8, help="flights per stage in the plot")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--gif-flight", type=int, default=0, help="which flight to animate")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--stage", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--no-defaults", action="store_true")
    ap.add_argument("--no-expert", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.gif_flight < args.flights:
        raise SystemExit("--gif-flight must be between 0 and flights-1")

    desc, scenario = LEVELS[args.level]
    stages = build_stages(args)
    if len(stages) < 2:
        raise SystemExit("Need at least two stages to compare (check the [skip] lines above).")
    print(f"Level {args.level}: {desc}. Flying {args.flights} identical starts per stage: "
          f"{', '.join(l.replace(chr(10), ' ') for l, _ in stages)}")
    flights = collect(stages, scenario, args.flights, args.seed)
    print("\n" + summarize(flights))

    out = ROOT / "data" / "exports"
    out.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    png = out / f"flight_comparison_{stamp}.png"
    make_png(flights, f"level {args.level}: {desc}", png)
    print(f"\nSaved picture: {png}")
    if not args.no_gif:
        gif = out / f"flight_comparison_{stamp}.gif"
        print("Rendering animation (about a minute)...")
        frames = make_gif(flights, args.gif_flight, gif)
        print(f"Saved animation ({frames} frames): {gif}")
    print("\nReminder: this is an illustration of a few flights. For success rates with confidence "
          "intervals run: python compare_models.py")


if __name__ == "__main__":
    main()
