import unittest

try:
    import torch  # noqa: F401
    import stable_baselines3  # noqa: F401
    import gymnasium  # noqa: F401
    HAVE_ML = True
except ImportError:
    HAVE_ML = False

from config.training_config import TrainingConfig

TINY = TrainingConfig(n_envs=2, n_steps=64, batch_size=64, n_epochs=2, net_arch=(16, 16),
                      eval_freq=256, eval_episodes=3, final_eval_episodes=3,
                      use_tensorboard=False, verbose=0)


class TestTrainingConfig(unittest.TestCase):
    def test_default_is_valid(self):
        TrainingConfig().validate()

    def test_batch_must_divide_rollout(self):
        with self.assertRaises(ValueError):
            TrainingConfig(n_envs=3, n_steps=100, batch_size=64).validate()

    def test_to_dict_is_json_friendly(self):
        import json
        json.dumps(TrainingConfig().to_dict())


@unittest.skipUnless(HAVE_ML, "torch / stable-baselines3 / gymnasium not installed")
class TestAgent(unittest.TestCase):
    def setUp(self):
        from training.trainer import build_vec_env
        from config.environment_config import EnvironmentConfig
        from config.reward_config import RewardConfig
        from config.scenario_config import ScenarioConfig
        self.env = build_vec_env(TINY, EnvironmentConfig(), ScenarioConfig(), RewardConfig())

    def tearDown(self):
        self.env.close()

    def test_builds_actor_critic_with_expected_shapes(self):
        from agent.ppo_agent import build_ppo
        from agent.policy_network import count_parameters, describe_network
        from environment.observation import OBS_DIM
        model = build_ppo(self.env, TINY)
        shapes = {d["name"]: d["shape"] for d in describe_network(model.policy)}
        self.assertEqual(shapes["mlp_extractor.policy_net.0.weight"], (16, OBS_DIM))
        self.assertEqual(shapes["mlp_extractor.value_net.0.weight"], (16, OBS_DIM))
        self.assertEqual(shapes["action_net.weight"], (2, 16))
        self.assertEqual(shapes["value_net.weight"], (1, 16))
        self.assertEqual(shapes["log_std"], (2,))
        self.assertGreater(count_parameters(model.policy), 0)

    def test_predict_returns_valid_action(self):
        from agent.ppo_agent import build_ppo
        import numpy as np
        model = build_ppo(self.env, TINY)
        o = self.env.reset()
        action, _ = model.predict(o[0], deterministic=True)
        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.all(np.abs(action) <= 1.0))


if __name__ == "__main__":
    unittest.main()
