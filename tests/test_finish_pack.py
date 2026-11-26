import unittest

import numpy as np

from agent.expert_controller import expert_action
from config.benchmark_levels import LEVELS
from environment.landing_task import LandingTask
from watch_ppo import steps_for_frame


class TestViewerSpeed(unittest.TestCase):
    def test_slow_motion_is_real(self):
        carry, steps = 0.0, 0
        for _ in range(40):                       # 40 frames at 0.25x -> 10 steps
            n, carry = steps_for_frame(0.25, carry)
            steps += n
        self.assertEqual(steps, 10)

    def test_normal_and_fast(self):
        self.assertEqual(steps_for_frame(1.0, 0.0)[0], 1)
        self.assertEqual(steps_for_frame(8.0, 0.0)[0], 8)

    def test_cap(self):
        self.assertEqual(steps_for_frame(100.0, 0.0, cap=32)[0], 32)


class TestExpertController(unittest.TestCase):
    def fly(self, level, n, seed=5):
        task = LandingTask(scenario_config=LEVELS[level][1])
        rng = np.random.default_rng(seed)
        ok = 0
        for _ in range(n):
            task.reset(rng)
            while True:
                *_, term, trunc, info = task.step(expert_action(task))
                if term or trunc:
                    break
            ok += info["is_success"]
        return ok

    def test_lands_on_default_and_training_distribution(self):
        self.assertEqual(self.fly("L0", 25), 25)
        self.assertGreaterEqual(self.fly("L1", 25), 24)

    def test_levels_get_harder(self):
        self.assertGreater(self.fly("L1", 30), self.fly("L3", 30))


try:
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class TestFlightComparison(unittest.TestCase):
    def test_same_scenarios_for_every_stage_and_files_written(self):
        import tempfile
        from pathlib import Path
        from compare_flights import collect, make_gif, make_png, summarize
        stages = [("free fall", lambda t, o: np.array([-1.0, 0.0])),
                  ("controller", lambda t, o: expert_action(t))]
        fl = collect(stages, LEVELS["L1"][1], 3, 11)
        # identical scenarios: every stage starts from the same height and sideways position
        for a, b in zip(fl["free fall"], fl["controller"]):
            self.assertEqual((a.x[0], a.y[0]), (b.x[0], b.y[0]))
        self.assertEqual(sum(f.success for f in fl["free fall"]), 0)
        self.assertEqual(sum(f.success for f in fl["controller"]), 3)
        self.assertIn("3/3", summarize(fl))
        with tempfile.TemporaryDirectory() as d:
            png, gif = Path(d) / "a.png", Path(d) / "a.gif"
            make_png(fl, "test", png)
            make_gif(fl, 0, gif, stride=6)
            self.assertGreater(png.stat().st_size, 5000)
            self.assertGreater(gif.stat().st_size, 5000)


if __name__ == "__main__":
    unittest.main()
