import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from config.environment_config import EnvironmentConfig
from config.reward_config import RewardConfig
from config.scenario_config import ScenarioConfig
from training.evaluation import evaluate, format_result, wilson_interval

ENV, RW, SC = EnvironmentConfig(), RewardConfig(), ScenarioConfig()


def free_fall(obs):
    return np.array([-1.0, 0.0])


def scripted_lander(obs):
    """Test-only hand-written controller, used to prove the evaluator counts successes correctly."""
    import math
    vy, h, th, om = obs[3] * 20, obs[1] * 150, obs[4] * math.pi, obs[5] * 2
    tv = -max(1.0, min(15.0, 0.5 * h))
    thr = 0.62 + 0.3 * (tv - vy)
    return np.array([2 * min(1.0, max(0.0, thr)) - 1, min(1.0, max(-1.0, -2 * th - 3 * om))])


class TestWilson(unittest.TestCase):
    def test_zero_successes(self):
        lo, hi = wilson_interval(0, 20)
        self.assertEqual(lo, 0.0)
        self.assertTrue(0.10 < hi < 0.20)

    def test_all_successes(self):
        lo, hi = wilson_interval(20, 20)
        self.assertEqual(hi, 1.0)
        self.assertTrue(0.80 < lo < 0.90)

    def test_empty_sample_is_uninformative(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_more_data_narrows_interval(self):
        a = wilson_interval(5, 10)
        b = wilson_interval(50, 100)
        self.assertLess(b[1] - b[0], a[1] - a[0])


class TestRanking(unittest.TestCase):
    def test_attempting_landing_outranks_hovering_even_with_worse_return(self):
        from training.evaluation import ranking_key
        hover = {"success_rate": 0.0, "soft_touchdown_rate": 0.0, "mean_return": -22.0}
        trying = {"success_rate": 0.0, "soft_touchdown_rate": 1.0, "mean_return": -95.0}
        self.assertGreater(ranking_key(trying), ranking_key(hover))

    def test_success_rate_dominates(self):
        from training.evaluation import ranking_key
        a = {"success_rate": 0.1, "soft_touchdown_rate": 0.0, "mean_return": -200.0}
        b = {"success_rate": 0.0, "soft_touchdown_rate": 1.0, "mean_return": 50.0}
        self.assertGreater(ranking_key(a), ranking_key(b))

    def test_old_summaries_without_soft_rate_still_rank(self):
        from training.evaluation import ranking_key
        self.assertEqual(ranking_key({"success_rate": 0.0, "mean_return": -5.0}), (0.0, 0.0, -5.0))


class TestEvaluate(unittest.TestCase):
    def test_free_fall_never_succeeds(self):
        r = evaluate(free_fall, ENV, SC, RW, 10, seed=1)
        self.assertEqual(r.n_episodes, 10)
        self.assertEqual(r.n_success, 0)
        self.assertEqual(r.success_rate, 0.0)
        self.assertEqual(sum(r.outcomes.values()), 10)
        self.assertEqual(r.outcomes, {"crash": 10})
        self.assertEqual(len(r.episodes), 10)
        self.assertGreater(r.mean_touchdown_speed, ENV.criteria.crash_vertical_speed)

    def test_soft_touchdown_rate_counts_controlled_descents(self):
        r = evaluate(scripted_lander, ENV, SC, RW, 5, seed=1)
        self.assertEqual(r.soft_touchdown_rate, 1.0)
        self.assertIn("soft_touchdown_rate", r.summary())
        self.assertEqual(evaluate(free_fall, ENV, SC, RW, 5, seed=1).soft_touchdown_rate, 0.0)

    def test_scripted_controller_succeeds_and_is_counted(self):
        r = evaluate(scripted_lander, ENV, SC, RW, 10, seed=1)
        self.assertEqual(r.n_success, 10)
        self.assertEqual(r.success_rate, 1.0)
        self.assertLess(r.mean_touchdown_speed, ENV.criteria.max_vertical_speed)
        self.assertLess(r.mean_pad_error, ENV.pad.half_width)
        self.assertGreater(r.mean_return, 100)

    def test_same_seed_same_result_different_seed_different_scenarios(self):
        sc = ScenarioConfig(x_offset=(-30, 30), vx=(-2, 2))
        a = evaluate(free_fall, ENV, sc, RW, 5, seed=3)
        b = evaluate(free_fall, ENV, sc, RW, 5, seed=3)
        c = evaluate(free_fall, ENV, sc, RW, 5, seed=4)
        self.assertEqual([e["scenario"] for e in a.episodes], [e["scenario"] for e in b.episodes])
        self.assertNotEqual([e["scenario"] for e in a.episodes], [e["scenario"] for e in c.episodes])
        self.assertEqual(a.mean_return, b.mean_return)

    def test_ci_and_sample_size_reported(self):
        r = evaluate(scripted_lander, ENV, SC, RW, 8, seed=2)
        text = format_result(r)
        self.assertIn("8 episodes", text)
        self.assertIn("95% CI", text)
        self.assertEqual(len(r.success_ci95), 2)
        json.dumps(r.to_dict())            # must be serialisable for checkpoint metadata

    def test_evaluation_is_read_only_for_policy(self):
        calls = []
        def spy(obs):
            calls.append(obs.shape)
            return free_fall(obs)
        evaluate(spy, ENV, SC, RW, 2, seed=0)
        self.assertTrue(all(s == (12,) for s in calls))


try:
    import torch  # noqa: F401
    import stable_baselines3  # noqa: F401
    import gymnasium  # noqa: F401
    HAVE_ML = True
except ImportError:
    HAVE_ML = False

from tests.test_agent import TINY  # noqa: E402


@unittest.skipUnless(HAVE_ML, "torch / stable-baselines3 / gymnasium not installed")
class TestRealTraining(unittest.TestCase):
    """Real PPO on the real environment, tiny budget. Verifies plumbing, NOT that the agent learned to land."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _model(self):
        from agent.ppo_agent import build_ppo
        from training.trainer import build_vec_env
        env = build_vec_env(TINY, ENV, SC, RW)
        return build_ppo(env, TINY), env

    def test_ppo_updates_parameters_only_at_rollout_boundaries(self):
        from training.callbacks import ParameterUpdateCallback
        model, env = self._model()
        before = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
        cb = ParameterUpdateCallback()
        model.learn(total_timesteps=TINY.rollout_size * 4, callback=cb)
        after = model.policy.state_dict()
        changed = sum(float((after[k] - before[k]).abs().max()) > 0 for k in before)
        self.assertGreater(changed, 0, "no parameter changed: training did nothing")
        self.assertGreaterEqual(len(cb.history), 1)
        self.assertLessEqual(len(cb.history), 4)            # at most one record per rollout, never per step
        for rec in cb.history:
            self.assertGreater(rec["l2_change"], 0.0)
            self.assertTrue(0.0 < rec["fraction_changed"] <= 1.0)
        env.close()

    def test_save_load_roundtrip_identical_predictions_and_optimizer_state(self):
        from agent.model_manager import ModelManager
        model, env = self._model()
        model.learn(total_timesteps=TINY.rollout_size * 2)
        mgr = ModelManager(self.root)
        d = mgr.save(model, "t", "step_a", training_config=TINY, env_config=ENV, reward_config=RW, scenario_config=SC)
        self.assertTrue((d / "model.zip").exists())
        meta = json.loads((d / "metadata.json").read_text())
        self.assertEqual(meta["num_timesteps"], model.num_timesteps)
        self.assertIn("reward_config", meta)
        self.assertTrue((d / "rng_state.pkl").exists())

        loaded, _ = mgr.load(d, device="cpu")
        obs = np.random.default_rng(0).uniform(-1, 1, (5, 12)).astype(np.float32)
        a1, _ = model.predict(obs, deterministic=True)
        a2, _ = loaded.predict(obs, deterministic=True)
        np.testing.assert_allclose(a1, a2, atol=1e-6)
        s1, s2 = model.policy.optimizer.state_dict(), loaded.policy.optimizer.state_dict()
        self.assertGreater(len(s2["state"]), 0, "optimizer state was not restored")
        self.assertEqual(s1["param_groups"][0]["lr"], s2["param_groups"][0]["lr"])
        k = next(iter(s1["state"]))
        np.testing.assert_allclose(s1["state"][k]["exp_avg"].cpu().numpy(), s2["state"][k]["exp_avg"].cpu().numpy(), atol=1e-6)
        env.close()

    def test_never_overwrites_existing_checkpoint(self):
        from agent.model_manager import ModelManager
        model, env = self._model()
        mgr = ModelManager(self.root)
        kw = dict(training_config=TINY, env_config=ENV, reward_config=RW, scenario_config=SC)
        mgr.save(model, "t", "same", **kw)
        with self.assertRaises(FileExistsError):
            mgr.save(model, "t", "same", **kw)
        env.close()

    def test_resume_continues_timesteps_instead_of_resetting(self):
        from agent.model_manager import ModelManager
        from training.trainer import build_vec_env
        model, env = self._model()
        model.learn(total_timesteps=TINY.rollout_size * 2)
        t_saved = model.num_timesteps
        mgr = ModelManager(self.root)
        d = mgr.save(model, "t", "first", training_config=TINY, env_config=ENV, reward_config=RW, scenario_config=SC)
        env2 = build_vec_env(TINY, ENV, SC, RW)
        resumed, meta = mgr.load(d, env=env2, device="cpu", restore_rng=True)
        self.assertEqual(resumed.num_timesteps, t_saved)
        resumed.learn(total_timesteps=TINY.rollout_size * 2, reset_num_timesteps=False)
        self.assertEqual(resumed.num_timesteps, t_saved + TINY.rollout_size * 2)
        env.close(); env2.close()

    def test_full_train_run_creates_checkpoints_logs_and_final_eval(self):
        from agent.model_manager import ModelManager
        from training.trainer import train
        mgr = ModelManager(self.root / "ckpt")
        res = train(TINY, "smoke", TINY.rollout_size * 4, manager=mgr, logs_root=self.root / "logs")
        run = mgr.read_run("smoke")
        tags = [c["tag"] for c in run["checkpoints"]]
        self.assertTrue(any(t.startswith("final_step_") for t in tags), tags)
        self.assertIsNotNone(run["best_checkpoint"])
        self.assertEqual(res.final_eval.n_episodes, TINY.final_eval_episodes)
        self.assertTrue((self.root / "logs" / "smoke" / "param_updates.csv").exists())
        self.assertTrue((self.root / "logs" / "smoke" / "episodes.csv").exists())
        with self.assertRaises(FileExistsError):               # existing run name is refused, not overwritten
            train(TINY, "smoke", TINY.rollout_size, manager=mgr, logs_root=self.root / "logs")


if __name__ == "__main__":
    unittest.main()
