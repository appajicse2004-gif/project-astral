"""Project Astral: fast conventional-controller baseline check.

Copy this file into the project_astral root and run:
    python astral_baseline_check.py
It does not modify project source files or train/overwrite PPO checkpoints.
"""
from __future__ import annotations
import csv, math
from pathlib import Path
import numpy as np
from environment.landing_task import LandingTask
from config.scenario_config import ScenarioConfig
from environment.rocket import total_mass, lowest_point_height


def controller(state, cfg):
    r, w, pad = cfg.rocket, cfg.world, cfg.pad
    h = max(0.0, lowest_point_height(state, r))
    # PD lateral correction; for the default centered-start benchmark this
    # should remain near zero, while still damping small drift/tilt.
    ax_des = float(np.clip(-0.12 * (state.x - pad.x) - 0.9 * state.vx, -3.0, 3.0))
    mass = total_mass(state, r)
    max_acc = r.max_thrust / mass
    theta_des = float(np.clip(math.asin(np.clip(-ax_des / max(max_acc, 1e-6), -0.45, 0.45)), -0.45, 0.45))
    inertia = mass * r.length**2 / 12.0
    alpha_des = 12.0 * (theta_des - state.theta) - 7.0 * state.omega
    rotation = float(np.clip(alpha_des * inertia / r.max_torque, -1.0, 1.0))
    # Follow a conservative descent-speed profile, slowing near the ground.
    vy_des = -min(10.0, 1.5 + 0.25 * h)
    ay_des = float(np.clip(0.9 * (vy_des - state.vy), -8.0, 5.0))
    throttle = float(np.clip(mass * (w.gravity + ay_des) /
                             (r.max_thrust * max(math.cos(state.theta), 0.3)), 0.0, 1.0))
    return throttle, rotation


def main():
    task = LandingTask(scenario_config=ScenarioConfig())
    rng = np.random.default_rng(42)
    rows, outcomes = [], {}
    for ep in range(100):
        _, reset_info = task.reset(rng)
        while True:
            s = task.engine.state
            throttle, rotation = controller(s, task.episode_config)
            _, _, terminated, truncated, info = task.step(np.array([2.0 * throttle - 1.0, rotation]))
            if terminated or truncated:
                summary = info['episode_summary']
                td = info.get('touchdown', {})
                outcome = summary['outcome']
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                rows.append({
                    'episode': ep + 1, 'outcome': outcome,
                    'steps': summary['length'], 'fuel_used_kg': round(summary['fuel_used'], 4),
                    'pad_offset_m': round(summary['final_x_offset'], 4),
                    'vertical_speed_mps': round(td.get('vertical_speed', float('nan')), 4),
                    'horizontal_speed_mps': round(td.get('horizontal_speed', float('nan')), 4),
                    'angle_rad': round(td.get('angle', float('nan')), 5),
                    'angular_speed_rps': round(td.get('angular_speed', float('nan')), 5),
                })
                break
    out = Path('data/logs/baseline_controller')
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'episodes.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    successes = outcomes.get('success', 0)
    print('\nPROJECT ASTRAL — BASELINE CONTROLLER')
    print(f'Episodes: {len(rows)} | Successes: {successes}/100 ({successes}%)')
    print(f'Outcomes: {outcomes}')
    landed = [r for r in rows if r['outcome'] == 'success']
    if landed:
        print(f"Mean successful touchdown vertical speed: {np.mean([r['vertical_speed_mps'] for r in landed]):.3f} m/s")
        print(f"Mean successful touchdown horizontal speed: {np.mean([r['horizontal_speed_mps'] for r in landed]):.3f} m/s")
        print(f"Mean successful pad offset: {np.mean([abs(r['pad_offset_m']) for r in landed]):.3f} m")
    print(f'Per-episode log: {csv_path}')
    print('NOTE: This is a conventional-controller benchmark, NOT PPO training.')

if __name__ == '__main__':
    main()
