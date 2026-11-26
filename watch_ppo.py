"""Watch a trained PPO model (or the hand-written controller) fly in the Pygame renderer.

Run from the project_astral folder:
    python watch_ppo.py                      # PPO model on level L1, press R for next flight
    python watch_ppo.py --level L2 --auto    # harder starts, automatically start the next flight
    python watch_ppo.py --controller expert  # hand-written controller (NOT AI), for side-by-side viewing
    python watch_ppo.py --checkpoint data/models/expert_initialized_ppo.zip   # the cloned-only model

Keys: ESC quit | R new flight | SPACE pause | +/- speed (0.25x to 8x, real slow-motion)
Inference only: nothing is trained or saved.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def steps_for_frame(speed: float, carry: float, cap: int = 32) -> tuple[int, float]:
    """How many physics steps to run this frame at the given speed, with fractional carry-over.

    speed 0.25 -> one step every 4th frame (real slow motion); speed 8 -> 8 steps per frame.
    """
    total = carry + speed
    n = int(total)
    return min(n, cap), total - n


def main() -> None:
    from config.benchmark_levels import LEVELS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="data/models/expert_initialized_ppo_final.zip")
    ap.add_argument("--controller", choices=["ppo", "expert"], default="ppo")
    ap.add_argument("--level", choices=list(LEVELS), default="L1")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions with the policy's noise (default: deterministic mean action)")
    ap.add_argument("--auto", action="store_true", help="start the next flight automatically after each one")
    args = ap.parse_args()

    import pygame
    from agent.expert_controller import expert_action
    from config.environment_config import EnvironmentConfig
    from config.reward_config import RewardConfig
    from environment.landing_task import LandingTask
    from visualization.simulation_renderer import SimulationRenderer

    model = None
    if args.controller == "ppo":
        from stable_baselines3 import PPO
        root = Path(__file__).resolve().parent
        ckpt = Path(args.checkpoint)
        ckpt = ckpt if ckpt.is_absolute() else root / ckpt
        if not ckpt.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt}")
        model = PPO.load(str(ckpt), device="cpu")
        who = f"PPO ({'sampled' if args.stochastic else 'deterministic'}) - {ckpt.name}"
    else:
        who = "hand-written controller (NOT AI)"

    desc, scenario = LEVELS[args.level]
    task = LandingTask(EnvironmentConfig(), scenario, RewardConfig())
    rng = np.random.default_rng(args.seed)
    obs, reset_info = task.reset(rng)
    renderer = SimulationRenderer(task.episode_config, caption=f"Project Astral - {who}")
    clock = pygame.time.Clock()

    speed, carry, paused, running = 1.0, 0.0, False, True
    episode, total_reward, last_action, last_info = 1, 0.0, np.zeros(2, np.float32), {}
    tally: dict[str, int] = {}
    done_frames = 0
    print(f"{who} | level {args.level}: {desc}")

    def new_flight():
        nonlocal obs, reset_info, episode, total_reward, last_action, last_info, done_frames
        obs, reset_info = task.reset(rng)
        renderer.clear_trail()
        episode += 1
        total_reward, last_action, last_info, done_frames = 0.0, np.zeros(2, np.float32), {}, 0

    try:
        while running:
            clock.tick(60)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_r:
                        new_flight()
                    elif ev.key == pygame.K_SPACE:
                        paused = not paused
                    elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        speed = min(8.0, speed * 2)
                    elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        speed = max(0.25, speed / 2)
            if not running:
                break

            if not paused and not task.engine.done:
                n, carry = steps_for_frame(speed, carry)
                for _ in range(n):
                    if model is not None:
                        a, _ = model.predict(obs, deterministic=not args.stochastic)
                    else:
                        a = expert_action(task)
                    last_action = np.asarray(a, dtype=np.float32).reshape(2)
                    obs, reward, terminated, truncated, last_info = task.step(last_action)
                    total_reward += float(reward)
                    if terminated or truncated:
                        outcome = last_info["outcome"]
                        tally[outcome] = tally.get(outcome, 0) + 1
                        break
            elif task.engine.done and args.auto:
                done_frames += 1
                if done_frames > 90:
                    new_flight()

            eng = task.engine
            n_done = sum(tally.values())
            extra = [
                f"CONTROLLER: {who}",
                f"level {args.level} | flight {episode} | reward {total_reward:8.1f}",
                f"action [{last_action[0]:+.2f}, {last_action[1]:+.2f}]",
                f"results: {tally.get('success', 0)}/{n_done} landed   {tally}",
                f"sim speed x{speed:g} {'PAUSED' if paused else ''}",
                "R new flight | SPACE pause | +/- speed | ESC quit",
            ]
            command = (float(last_info.get("throttle_cmd", 0.0)), float(last_info.get("rotation_cmd", 0.0)))
            renderer.render(eng.state, outcome=eng.outcome, report=eng.touchdown, command=command,
                            extra_lines=extra, push_trail=not eng.done)
    finally:
        renderer.close()
        n_done = sum(tally.values())
        if n_done:
            print(f"Session: {tally.get('success', 0)}/{n_done} landed. Outcomes: {tally}")


if __name__ == "__main__":
    main()
