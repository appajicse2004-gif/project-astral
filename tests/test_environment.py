import math
import unittest
from dataclasses import replace

import numpy as np

from config.environment_config import EnvironmentConfig
from config.scenario_config import ScenarioConfig
from environment.landing_task import LandingTask, sample_scenario
from environment.observation import CLIP, OBS_DIM, OBS_NAMES, build_observation
from environment.rocket import RocketState
from training.reward_calculator import COMPONENT_NAMES

ENV = EnvironmentConfig()


def rng(seed=0):
    return np.random.default_rng(seed)


class TestObservation(unittest.TestCase):
    def test_shape_dtype_and_names(self):
        obs = build_observation(RocketState(y=50.0, fuel=400.0), ENV)
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertEqual(len(OBS_NAMES), OBS_DIM)
        self.assertEqual(obs.dtype, np.float32)

    def test_finite_and_bounded_even_for_extreme_state(self):
        obs = build_observation(RocketState(x=1e6, y=1e6, vx=1e6, vy=-1e6, omega=1e6, fuel=400.0), ENV)
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertTrue(np.all(np.abs(obs) <= CLIP))

    def test_pad_relative_offset(self):
        a = build_observation(RocketState(x=10.0, y=50.0, fuel=400.0), ENV)
        shifted = replace(ENV, pad=replace(ENV.pad, x=30.0))
        b = build_observation(RocketState(x=40.0, y=50.0, fuel=400.0), shifted)
        self.assertAlmostEqual(float(a[0]), float(b[0]), places=6)
        self.assertNotAlmostEqual(float(a[OBS_NAMES.index("pad_x")]), float(b[OBS_NAMES.index("pad_x")]))

    def test_values_match_state(self):
        obs = build_observation(RocketState(y=50.0, vx=10.0, vy=-20.0, theta=math.pi / 2, fuel=200.0), ENV)
        d = dict(zip(OBS_NAMES, obs))
        self.assertAlmostEqual(d["vx"], 0.5, places=5)
        self.assertAlmostEqual(d["vy"], -1.0, places=5)
        self.assertAlmostEqual(d["angle"], 0.5, places=5)
        self.assertAlmostEqual(d["fuel_fraction"], 0.5, places=5)


class TestScenario(unittest.TestCase):
    def test_default_is_vertical_start_above_pad(self):
        s = sample_scenario(ScenarioConfig(), ENV, rng())
        self.assertEqual((s.x, s.vx, s.vy, s.theta, s.omega), (0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertTrue(20.0 <= s.y <= 40.0)          # low start (new default)
        self.assertEqual(s.gravity, ENV.world.gravity)

    def test_seeded_reproducible_and_varied(self):
        cfg = ScenarioConfig(x_offset=(-30, 30), vx=(-3, 3), theta=(-0.3, 0.3), gravity=(8.0, 11.0))
        a, b, c = (sample_scenario(cfg, ENV, rng(s)) for s in (1, 1, 2))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertTrue(8.0 <= a.gravity <= 11.0)

    def test_start_x_stays_inside_world(self):
        cfg = ScenarioConfig(pad_x=(90, 90), x_offset=(50, 50))
        self.assertLess(abs(sample_scenario(cfg, ENV, rng()).x), ENV.world.half_width)


class TestPresets(unittest.TestCase):
    def test_low_and_tall_presets(self):
        from config.scenario_config import PRESETS
        self.assertEqual(PRESETS["low"].y, (20.0, 40.0))
        self.assertEqual(PRESETS["low"].max_episode_steps, 900)
        self.assertEqual(PRESETS["tall"].y, (60.0, 100.0))       # the original first-run setup
        self.assertEqual(PRESETS["tall"].max_episode_steps, 1800)


class TestLandingTask(unittest.TestCase):
    def make(self, **sc):
        t = LandingTask(scenario_config=ScenarioConfig(**sc) if sc else None)
        obs, info = t.reset(rng())
        return t, obs, info

    def test_reset_outputs(self):
        t, obs, info = self.make()
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertIn("scenario", info)

    def test_action_mapping_and_physics(self):
        t, *_ = self.make()
        t.step([-1.0, 0.0])           # zero throttle -> pure free fall
        self.assertAlmostEqual(t.engine.state.ay, -ENV.world.gravity, places=9)
        self.assertEqual(t.engine.state.throttle, 0.0)

    def test_actions_clipped(self):
        t, *_ = self.make()
        *_, info = t.step([50.0, -50.0])
        self.assertEqual(info["throttle_cmd"], 1.0)
        self.assertEqual(info["rotation_cmd"], -1.0)

    def test_bad_actions_rejected(self):
        t, *_ = self.make()
        with self.assertRaises(ValueError):
            t.step([0.0])
        with self.assertRaises(ValueError):
            t.step([float("nan"), 0.0])

    def test_free_fall_terminates_with_crash(self):
        t, *_ = self.make()
        terminated = False
        for _ in range(3000):
            obs, r, terminated, truncated, info = t.step([-1.0, 0.0])
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["outcome"], "crash")
        self.assertFalse(info["is_success"])
        self.assertLess(info["episode_summary"]["return"], -50)
        self.assertEqual(info["touchdown"]["outcome"], "crash")

    def test_truncation_at_max_steps(self):
        t, *_ = self.make(max_episode_steps=5, y=(100.0, 100.0))
        for i in range(5):
            _, _, terminated, truncated, info = t.step([0.0, 0.0])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["outcome"], "timeout")
        self.assertEqual(info["episode_summary"]["length"], 5)

    def test_step_after_end_raises(self):
        t, *_ = self.make(max_episode_steps=1)
        t.step([0.0, 0.0])
        t.engine.reset()  # engine not done, but task semantics: caller must reset after truncation
        # A finished (terminated) episode must raise:
        t2, *_ = self.make(y=(6.0, 6.0))
        for _ in range(600):
            _, _, term, trunc, _ = t2.step([-1.0, 0.0])
            if term:
                break
        with self.assertRaises(RuntimeError):
            t2.step([0.0, 0.0])

    def test_reward_equals_sum_of_components(self):
        t, *_ = self.make()
        _, r, _, _, info = t.step([0.3, 0.2])
        self.assertEqual(set(info["reward_components"]), set(COMPONENT_NAMES))
        self.assertAlmostEqual(r, sum(info["reward_components"].values()), places=9)

    def test_episode_component_totals_sum_to_return(self):
        t, *_ = self.make()
        for _ in range(3000):
            *_, term, trunc, info = t.step([-1.0, 0.0])
            if term or trunc:
                break
        ep = info["episode_summary"]
        self.assertAlmostEqual(ep["return"], sum(ep["reward_component_totals"].values()), places=6)

    def test_episode_summary_has_diagnostics(self):
        t, *_ = self.make()
        for _ in range(3000):
            *_, term, trunc, info = t.step([-1.0, 0.0])
            if term or trunc:
                break
        ep = info["episode_summary"]
        self.assertIn("final_x_offset", ep)
        self.assertIn("final_y", ep)
        self.assertAlmostEqual(ep["final_x_offset"], 0.0, places=6)    # straight fall stays above the pad
        self.assertEqual(info["touchdown"]["horizontal_speed"], info["touchdown"]["horizontal_speed"])
        for k in ("angle", "angular_speed", "vertical_speed"):
            self.assertIn(k, info["touchdown"])

    def test_options_override_scenario_and_physics(self):
        t = LandingTask()
        t.reset(rng(), options={"gravity": 1.62, "y": 50.0, "x": 10.0, "pad_x": 5.0})
        t.step([-1.0, 0.0])
        self.assertAlmostEqual(t.engine.state.vy, -1.62 * ENV.world.dt, places=9)
        self.assertEqual(t.episode_config.pad.x, 5.0)

    def test_unknown_option_rejected(self):
        with self.assertRaises(ValueError):
            LandingTask().reset(rng(), options={"warp_speed": 9})

    def test_mass_override_changes_dynamics(self):
        t = LandingTask()
        t.reset(rng(), options={"dry_mass": 2000.0, "y": 50.0})
        t.step([1.0, 0.0])
        t2 = LandingTask()
        t2.reset(rng(), options={"y": 50.0})
        t2.step([1.0, 0.0])
        self.assertLess(t.engine.state.ay, t2.engine.state.ay)

    def test_deterministic_given_seed(self):
        def run(seed):
            cfg = ScenarioConfig(x_offset=(-20, 20), vx=(-2, 2), theta=(-0.2, 0.2))
            t = LandingTask(scenario_config=cfg)
            obs, _ = t.reset(rng(seed))
            out = [obs.copy()]
            for i in range(50):
                obs, r, *_ = t.step([math.sin(i / 5), math.cos(i / 3)])
                out.append((obs.copy(), r))
            return out
        a, b = run(7), run(7)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual([x[1] for x in a[1:]], [x[1] for x in b[1:]])

    def test_random_policy_rollouts_stay_valid(self):
        gen = rng(3)
        t = LandingTask(scenario_config=ScenarioConfig(x_offset=(-40, 40), vx=(-3, 3), theta=(-0.4, 0.4)))
        for ep in range(15):
            obs, _ = t.reset(gen)
            for _ in range(2000):
                obs, r, term, trunc, info = t.step(gen.uniform(-1, 1, 2))
                self.assertTrue(np.all(np.isfinite(obs)))
                self.assertTrue(math.isfinite(r))
                if term or trunc:
                    break
            self.assertTrue(term or trunc)


try:
    import gymnasium  # noqa: F401
    HAVE_GYM = True
except ImportError:
    HAVE_GYM = False


@unittest.skipUnless(HAVE_GYM, "gymnasium not installed - run these on your machine")
class TestGymWrapper(unittest.TestCase):
    def test_check_env(self):
        from gymnasium.utils.env_checker import check_env
        from environment.rocket_env import RocketLandingEnv
        check_env(RocketLandingEnv(), skip_render_check=True)

    def test_spaces_and_seeding(self):
        from environment.rocket_env import RocketLandingEnv
        env = RocketLandingEnv(scenario_config=ScenarioConfig(x_offset=(-30, 30)))
        o1, _ = env.reset(seed=5)
        o2, _ = env.reset(seed=5)
        np.testing.assert_array_equal(o1, o2)
        self.assertTrue(env.observation_space.contains(o1))
        self.assertTrue(env.action_space.contains(env.action_space.sample()))
        self.assertEqual(env.action_space.shape, (2,))

    def test_random_episode_terminates(self):
        from environment.rocket_env import RocketLandingEnv
        env = RocketLandingEnv()
        env.reset(seed=0)
        for _ in range(5000):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            self.assertTrue(env.observation_space.contains(obs))
            if term or trunc:
                break
        self.assertTrue(term or trunc)
        self.assertIn("episode_summary", info)


if __name__ == "__main__":
    unittest.main()
