"""RL Trainer — trains all RL agents on InventoryEnv and collects metrics."""

import logging
import time
from typing import Optional

import numpy as np

from app.rl.agents.hybrid_ga_rl import HybridGARLAgent
from app.rl.agents.q_learning import QLearningAgent, SARSAAgent
from app.rl.environment import InventoryEnv

logger = logging.getLogger(__name__)

# Optional deep RL agents
try:
    import torch  # noqa: F401

    from app.rl.agents.dqn import DQNAgent
    from app.rl.agents.ppo import A2CAgent, PPOAgent
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("PyTorch not available — DQN/PPO/A2C agents will be skipped")


def _run_tabular_training(
    agent,
    env: InventoryEnv,
    n_episodes: int,
    agent_name: str,
    is_sarsa: bool = False,
) -> dict:
    """Train a tabular agent (Q-Learning or SARSA) and return metrics."""
    rewards_history = []
    service_levels = []

    for _ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        if is_sarsa:
            action = agent.select_action(obs)

        while not done:
            if is_sarsa:
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_action = agent.select_action(next_obs)
                agent.update(obs, action, reward, next_obs, done, next_action=next_action)
                obs = next_obs
                action = next_action
            else:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.update(obs, action, reward, next_obs, done)
                obs = next_obs
            total_reward += reward

        if hasattr(agent, "decay_epsilon"):
            agent.decay_epsilon()
        rewards_history.append(total_reward)
        summary = env.get_episode_summary()
        service_levels.append(summary["service_level"])

    result = {
        "agent": agent_name,
        "rewards": rewards_history,
        "service_levels": service_levels,
        "final_epsilon": getattr(agent, "epsilon", None),
    }
    if hasattr(agent, "get_q_table_stats"):
        result["q_table_stats"] = agent.get_q_table_stats()
    return result


def _run_dqn_training(
    agent: "DQNAgent",
    env: InventoryEnv,
    n_episodes: int,
) -> dict:
    """Train DQN agent."""
    rewards_history = []
    service_levels = []

    for _ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

        rewards_history.append(total_reward)
        summary = env.get_episode_summary()
        service_levels.append(summary["service_level"])

    return {
        "agent": "DQN",
        "rewards": rewards_history,
        "service_levels": service_levels,
        "final_epsilon": agent.epsilon,
    }


def _run_ppo_training(
    agent: "PPOAgent",
    env: InventoryEnv,
    n_episodes: int,
    rollout_steps: int = 256,
) -> dict:
    """Train PPO agent."""
    rewards_history = []
    service_levels = []

    for _ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        steps_in_rollout = 0

        while not done:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, done, log_prob, value)
            obs = next_obs
            total_reward += reward
            steps_in_rollout += 1

            if steps_in_rollout >= rollout_steps and not done:
                _, _, last_value = agent.select_action(obs)
                agent.update(last_value)
                steps_in_rollout = 0

        # End-of-episode update
        agent.update(last_value=0.0)
        rewards_history.append(total_reward)
        summary = env.get_episode_summary()
        service_levels.append(summary["service_level"])

    return {
        "agent": "PPO",
        "rewards": rewards_history,
        "service_levels": service_levels,
    }


def _run_a2c_training(
    agent: "A2CAgent",
    env: InventoryEnv,
    n_episodes: int,
) -> dict:
    """Train A2C agent."""
    rewards_history = []
    service_levels = []

    for _ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, done, log_prob, value)
            obs = next_obs
            total_reward += reward
            steps += 1

            if steps % agent.n_steps == 0 and not done:
                _, _, last_value = agent.select_action(obs)
                agent.update(last_value)

        agent.update(last_value=0.0)
        rewards_history.append(total_reward)
        summary = env.get_episode_summary()
        service_levels.append(summary["service_level"])

    return {
        "agent": "A2C",
        "rewards": rewards_history,
        "service_levels": service_levels,
    }


class RLTrainer:
    """Coordinates training of all RL agents on InventoryEnv."""

    def __init__(
        self,
        n_episodes: int = 300,
        episode_length: int = 90,
        env_kwargs: Optional[dict] = None,
        seed: int = 42,
        convergence_window: int = 20,
        convergence_threshold: float = 0.05,
    ):
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.env_kwargs = env_kwargs or {}
        self.results: dict[str, dict] = {}
        self.seed = seed
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold

        # Set global seeds for reproducibility
        np.random.seed(seed)
        if HAS_TORCH:
            torch.manual_seed(seed)

    def _make_env(self, offset: int = 0) -> InventoryEnv:
        return InventoryEnv(
            episode_length=self.episode_length,
            seed=self.seed + offset,
            **self.env_kwargs,
        )

    def train_all(self, on_progress=None) -> dict[str, dict]:
        """Train all available agents and return results dict."""
        total_agents = 6 if HAS_TORCH else 3
        trained = 0

        def _report(name, status):
            nonlocal trained
            trained += 1
            msg = f"RL Training [{trained}/{total_agents}]: {name} — {status}"
            logger.info(msg)
            if on_progress:
                on_progress("rl", msg)

        # --- Tabular agents ---
        t0 = time.time()
        env = self._make_env(offset=0)
        q_agent = QLearningAgent()
        self.results["Q-Learning"] = _run_tabular_training(
            q_agent, env, self.n_episodes, "Q-Learning"
        )
        _report("Q-Learning", f"done in {time.time()-t0:.1f}s")

        t0 = time.time()
        env = self._make_env(offset=1)
        sarsa_agent = SARSAAgent()
        self.results["SARSA"] = _run_tabular_training(
            sarsa_agent, env, self.n_episodes, "SARSA", is_sarsa=True
        )
        _report("SARSA", f"done in {time.time()-t0:.1f}s")

        # --- GA-RL Hybrid ---
        t0 = time.time()
        env = self._make_env(offset=2)
        hybrid_agent = HybridGARLAgent()
        hybrid_agent.initialize_from_ga(env)
        hybrid_result = _run_tabular_training(
            hybrid_agent, env, self.n_episodes, "GA-RL Hybrid"
        )
        hybrid_result["ga_stats"] = hybrid_agent.get_stats().get("ga", {})
        self.results["GA-RL Hybrid"] = hybrid_result
        _report("GA-RL Hybrid", f"done in {time.time()-t0:.1f}s")

        # --- Deep RL agents (require PyTorch) ---
        if HAS_TORCH:
            t0 = time.time()
            env = self._make_env(offset=3)
            dqn_agent = DQNAgent()
            self.results["DQN"] = _run_dqn_training(dqn_agent, env, self.n_episodes)
            _report("DQN", f"done in {time.time()-t0:.1f}s")

            t0 = time.time()
            env = self._make_env(offset=4)
            ppo_agent = PPOAgent()
            self.results["PPO"] = _run_ppo_training(ppo_agent, env, self.n_episodes)
            _report("PPO", f"done in {time.time()-t0:.1f}s")

            t0 = time.time()
            env = self._make_env(offset=5)
            a2c_agent = A2CAgent()
            self.results["A2C"] = _run_a2c_training(a2c_agent, env, self.n_episodes)
            _report("A2C", f"done in {time.time()-t0:.1f}s")
        else:
            logger.warning("Skipping DQN/PPO/A2C (PyTorch not installed)")

        return self.results

    def get_comparison_data(self) -> dict:
        """Return structured data for visualization."""
        comparison = {}
        for name, result in self.results.items():
            rewards = result["rewards"]
            svc = result["service_levels"]
            comparison[name] = {
                "final_reward": float(np.mean(rewards[-20:])),
                "best_reward": float(np.max(rewards)),
                "mean_reward": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "final_service_level": float(np.mean(svc[-20:])) if svc else 0.0,
                "mean_service_level": float(np.mean(svc)) if svc else 0.0,
                "convergence_episode": int(self._find_convergence(rewards)),
                "rewards_history": [float(r) for r in rewards],
                "service_level_history": [float(s) for s in svc],
            }
        return comparison

    @staticmethod
    def _find_convergence(rewards: list, window: int = 20, threshold: float = 0.05) -> int:
        """Find the episode where reward stabilizes (std/|mean| < threshold)."""
        if len(rewards) < window * 2:
            return len(rewards)
        for i in range(window, len(rewards)):
            segment = rewards[i - window:i]
            mean = np.mean(segment)
            if mean != 0 and np.std(segment) / abs(mean) < threshold:
                return i - window
        return len(rewards)
