"""Pygame renderer for the rocket simulation.

Rendering only: it reads a RocketState and draws it. It never modifies physics
state. pygame is imported lazily so the rest of the project (physics, tests,
later headless training) works without it.
"""
from __future__ import annotations

import math
from collections import deque

from config.environment_config import EnvironmentConfig
from environment.collision import Outcome, TouchdownReport
from environment.rocket import RocketState, hull_corners

BG = (8, 12, 22)
GRID = (18, 26, 42)
GROUND = (40, 52, 74)
PAD = (60, 200, 140)
HULL = (210, 220, 235)
NOSE = (230, 90, 80)
FLAME = (255, 170, 60)
VEL = (80, 200, 255)
ACC = (255, 220, 80)
TRAIL = (70, 110, 170)
TEXT = (190, 205, 225)
DIM = (110, 125, 150)
GOOD = (80, 220, 140)
BAD = (240, 90, 90)


class SimulationRenderer:
    def __init__(self, config: EnvironmentConfig, width: int = 920, height: int = 720,
                 caption: str = "Project Astral"):
        import pygame  # lazy import
        self.pg = pygame
        self.cfg = config
        pygame.init()
        pygame.display.set_caption(caption)
        self.screen = pygame.display.set_mode((width, height))
        self.font = pygame.font.SysFont("consolas,menlo,dejavusansmono", 15)
        self.big = pygame.font.SysFont("consolas,menlo,dejavusansmono", 26, bold=True)
        self.w, self.h = width, height
        self.margin_bottom = 45
        self.scale = (width - 20) / (2 * config.world.half_width)   # px per metre
        self.trail: deque[tuple[float, float]] = deque(maxlen=900)

    # --------------------------------------------------------------- helpers
    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        return (int(self.w / 2 + x * self.scale),
                int(self.h - self.margin_bottom - y * self.scale))

    def clear_trail(self) -> None:
        self.trail.clear()

    # ------------------------------------------------------------------ draw
    def render(self, state: RocketState, *, outcome: Outcome = Outcome.IN_FLIGHT,
               report: TouchdownReport | None = None, command: tuple[float, float] = (0.0, 0.0),
               extra_lines: list[str] | None = None, push_trail: bool = True) -> None:
        pg, scr, cfg = self.pg, self.screen, self.cfg
        scr.fill(BG)

        # Reference grid every 20 m.
        step = 20
        for gx in range(-int(cfg.world.half_width), int(cfg.world.half_width) + 1, step):
            pg.draw.line(scr, GRID, self.to_screen(gx, 0), self.to_screen(gx, cfg.world.ceiling))
        for gy in range(0, int(cfg.world.ceiling) + 1, step):
            pg.draw.line(scr, GRID, self.to_screen(-cfg.world.half_width, gy),
                         self.to_screen(cfg.world.half_width, gy))

        # Ground and pad.
        gx0, gy0 = self.to_screen(-cfg.world.half_width, 0)
        pg.draw.rect(scr, GROUND, (0, gy0, self.w, self.h - gy0))
        px0, py0 = self.to_screen(cfg.pad.x - cfg.pad.half_width, 0)
        px1, _ = self.to_screen(cfg.pad.x + cfg.pad.half_width, 0)
        pg.draw.rect(scr, PAD, (px0, py0 - 3, px1 - px0, 6))
        cx, cy = self.to_screen(cfg.pad.x, 0)
        pg.draw.line(scr, PAD, (cx, cy - 10), (cx, cy + 10), 2)      # target centre

        # Trail.
        if push_trail:
            self.trail.append((state.x, state.y))
        if len(self.trail) > 1:
            pg.draw.lines(scr, TRAIL, False, [self.to_screen(*p) for p in self.trail], 1)

        # Flame (length proportional to actual throttle).
        c, s = math.cos(state.theta), math.sin(state.theta)
        half_l = cfg.rocket.length / 2
        if state.throttle > 0.01:
            flame_len = 1.0 + 7.0 * state.throttle
            base = (state.x + half_l * s, state.y - half_l * c)
            tip = (state.x + (half_l + flame_len) * s, state.y - (half_l + flame_len) * c)
            pg.draw.line(scr, FLAME, self.to_screen(*base), self.to_screen(*tip), 5)

        # Hull + nose cone.
        corners = hull_corners(state, cfg.rocket)
        pg.draw.polygon(scr, HULL, [self.to_screen(*p) for p in corners])
        nose_tip = (state.x - (half_l + 2.5) * s, state.y + (half_l + 2.5) * c)
        pg.draw.polygon(scr, NOSE, [self.to_screen(*corners[2]), self.to_screen(*corners[3]),
                                    self.to_screen(*nose_tip)])

        # Velocity (blue) and acceleration (yellow) vectors.
        origin = self.to_screen(state.x, state.y)
        self._arrow(origin, (state.vx * 1.5, state.vy * 1.5), VEL)
        self._arrow(origin, (state.ax * 1.5, state.ay * 1.5), ACC)

        self._hud(state, outcome, report, command, extra_lines or [])
        pg.display.flip()

    def _arrow(self, origin, vec, color) -> None:
        end = (origin[0] + vec[0] * self.scale, origin[1] - vec[1] * self.scale)
        if abs(vec[0]) + abs(vec[1]) > 0.05:
            self.pg.draw.line(self.screen, color, origin, end, 2)
            self.pg.draw.circle(self.screen, color, (int(end[0]), int(end[1])), 3)

    def _hud(self, st, outcome, report, command, extra) -> None:
        pg, scr, r = self.pg, self.screen, self.cfg.rocket
        lines = [
            f"t        {st.time:7.2f} s",
            f"pos      ({st.x:7.2f}, {st.y:7.2f}) m",
            f"vel      ({st.vx:7.2f}, {st.vy:7.2f}) m/s",
            f"accel    ({st.ax:7.2f}, {st.ay:7.2f}) m/s2",
            f"angle    {math.degrees(st.theta):7.1f} deg",
            f"ang.rate {math.degrees(st.omega):7.1f} deg/s",
            f"throttle {st.throttle:7.2f}  (cmd {command[0]:.2f})",
            f"rotation cmd {command[1]:+.2f}",
            f"fuel     {st.fuel:7.1f} kg",
        ] + extra
        for i, text in enumerate(lines):
            scr.blit(self.font.render(text, True, TEXT), (12, 10 + 18 * i))

        # Fuel bar.
        bx, by, bw, bh = 12, 10 + 18 * len(lines) + 6, 160, 10
        frac = st.fuel / r.fuel_capacity if r.fuel_capacity else 0.0
        pg.draw.rect(scr, GRID, (bx, by, bw, bh))
        pg.draw.rect(scr, FLAME if frac > 0.2 else BAD, (bx, by, int(bw * frac), bh))

        legend = "blue: velocity   yellow: acceleration"
        scr.blit(self.font.render(legend, True, DIM), (self.w - 330, 10))
        controls = "W/UP thrust   A/LEFT rotate CCW   D/RIGHT rotate CW   R reset   +/- speed   ESC quit"
        scr.blit(self.font.render(controls, True, DIM), (12, self.h - 24))

        if outcome is not Outcome.IN_FLIGHT:
            color = GOOD if outcome is Outcome.SUCCESS else BAD
            label = outcome.value.replace("_", " ").upper()
            scr.blit(self.big.render(label, True, color), (self.w // 2 - 120, 60))
            if report is not None:
                detail = (f"impact vy {report.vertical_speed:.2f} m/s, vx {report.horizontal_speed:.2f} m/s, "
                          f"angle {math.degrees(report.angle):.1f} deg, "
                          f"pad offset {report.pad_offset:.1f} m")
                scr.blit(self.font.render(detail, True, TEXT), (self.w // 2 - 230, 96))
                if report.failed_criteria:
                    scr.blit(self.font.render("exceeded: " + ", ".join(report.failed_criteria),
                                              True, BAD), (self.w // 2 - 230, 116))

    def close(self) -> None:
        self.pg.quit()
