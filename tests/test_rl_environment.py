"""RL environment tests."""

import numpy as np


class TestInventoryEnv:
    """Test the Gymnasium InventoryEnv."""

    def test_env_creation(self):
        """Verify environment can be created with defaults."""
        from app.rl.environment import InventoryEnv
        env = InventoryEnv()
        assert env.observation_space.shape == (5,)
        assert env.action_space.n == 5

    def test_env_reset(self):
        """Verify reset returns valid observation and info."""
        from app.rl.environment import InventoryEnv
        env = InventoryEnv(seed=42)
        obs, info = env.reset(seed=42)
        assert obs.shape == (5,)
        assert all(0.0 <= o <= 1.0 for o in obs)
        assert "day" in info

    def test_env_step(self):
        """Verify step returns valid outputs."""
        from app.rl.environment import InventoryEnv
        env = InventoryEnv(seed=42)
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (5,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_env_episode_completes(self):
        """Verify episode terminates after episode_length steps."""
        from app.rl.environment import InventoryEnv
        env = InventoryEnv(episode_length=10, seed=42)
        env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            steps += 1
        assert steps == 10

    def test_env_summary(self):
        """Verify episode summary contains expected keys."""
        from app.rl.environment import InventoryEnv
        env = InventoryEnv(episode_length=5, seed=42)
        env.reset(seed=42)
        for _ in range(5):
            env.step(2)
        summary = env.get_episode_summary()
        assert "total_reward" in summary
        assert "service_level" in summary
        assert "total_cost" in summary

    def test_env_seed_reproducibility(self):
        """Verify same seed produces same results."""
        from app.rl.environment import InventoryEnv
        env1 = InventoryEnv(seed=42, episode_length=10)
        env2 = InventoryEnv(seed=42, episode_length=10)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

        for _ in range(10):
            o1, r1, _, _, _ = env1.step(2)
            o2, r2, _, _, _ = env2.step(2)
            np.testing.assert_array_almost_equal(o1, o2)
            assert r1 == r2

    def test_env_check(self):
        """Run gymnasium.utils.env_checker.check_env()."""
        from gymnasium.utils.env_checker import check_env

        from app.rl.environment import InventoryEnv
        env = InventoryEnv(seed=42)
        # check_env will raise if there are issues
        check_env(env, skip_render_check=True)
