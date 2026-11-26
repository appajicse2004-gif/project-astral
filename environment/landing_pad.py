"""Landing pad geometry."""
from __future__ import annotations

from config.environment_config import PadConfig


class LandingPad:
    def __init__(self, cfg: PadConfig):
        self.cfg = cfg

    @property
    def x(self) -> float:
        return self.cfg.x

    @property
    def half_width(self) -> float:
        return self.cfg.half_width

    def offset(self, x: float) -> float:
        """Signed horizontal distance from pad centre (+ = right of centre)."""
        return x - self.cfg.x

    def contains(self, x: float) -> bool:
        """True if the rocket's centre line is over the pad."""
        return abs(self.offset(x)) <= self.cfg.half_width
