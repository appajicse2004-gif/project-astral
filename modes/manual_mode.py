"""Human-controlled flight (no learning, no recording into any training data)."""
from __future__ import annotations

import os

from config.environment_config import EnvironmentConfig
from environment.physics_engine import PhysicsEngine
from environment.rocket import make_state


def run_manual(config: EnvironmentConfig, start_x: float, start_y: float,
               start_vx: float, headless_frames: int = 0) -> None:
    """Fly the rocket with the keyboard. headless_frames > 0 runs that many
    frames with no input and exits (used to smoke-test rendering without a window)."""
    if headless_frames:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("pygame is required for manual mode: pip install pygame") from exc
    from visualization.simulation_renderer import SimulationRenderer

    def fresh():
        return make_state(config.rocket, x=start_x, y=start_y, vx=start_vx)

    engine = PhysicsEngine(config, fresh())
    renderer = SimulationRenderer(config, caption="Project Astral - manual flight")
    clock = pygame.time.Clock()
    dt = config.world.dt
    accumulator, speed, frames = 0.0, 1.0, 0
    command = (0.0, 0.0)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    engine.reset(fresh())
                    renderer.clear_trail()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(4.0, speed * 2)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(0.25, speed / 2)

        keys = pygame.key.get_pressed()
        throttle = 1.0 if (keys[pygame.K_UP] or keys[pygame.K_w]) else 0.0
        rot = float(keys[pygame.K_LEFT] or keys[pygame.K_a]) - float(keys[pygame.K_RIGHT] or keys[pygame.K_d])
        command = (throttle, rot)

        accumulator += min(clock.tick(60) / 1000.0, 0.1) * speed
        while accumulator >= dt and not engine.done:
            engine.step(*command)
            accumulator -= dt

        renderer.render(engine.state, outcome=engine.outcome, report=engine.touchdown,
                        command=command, extra_lines=[f"sim speed x{speed:g}"],
                        push_trail=not engine.done)
        frames += 1
        if headless_frames and frames >= headless_frames:
            running = False

    renderer.close()
