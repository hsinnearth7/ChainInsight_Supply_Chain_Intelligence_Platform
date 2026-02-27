"""Q-Learning and SARSA agents for inventory management RL.

Tabular agents that discretize the continuous 5-dim observation space
into bins and maintain a Q-table for action selection.
"""

import logging

import numpy as np

from app.rl.agents.base import BaseTabularAgent

logger = logging.getLogger(__name__)


class QLearningAgent(BaseTabularAgent):
    """Off-policy TD control using Q-Learning (max over next actions)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info(
            "QLearningAgent initialized: n_bins=%d, alpha=%.3f, gamma=%.2f, "
            "epsilon=%.2f→%.2f (decay=%.4f)",
            self.n_bins, self.alpha, self.gamma, self.epsilon, self.epsilon_min, self.epsilon_decay,
        )

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        **kwargs,
    ) -> float:
        """Q-Learning update: Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]."""
        state = self._discretize(obs)
        next_state = self._discretize(next_obs)
        current_q = self.q_table[state][action]
        next_max_q = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * next_max_q
        td_error = td_target - current_q
        self.q_table[state][action] += self.alpha * td_error
        self.steps += 1
        return abs(td_error)

    def __repr__(self) -> str:
        return (
            f"QLearningAgent(n_actions={self.n_actions}, n_bins={self.n_bins}, "
            f"alpha={self.alpha}, gamma={self.gamma}, "
            f"epsilon={self.epsilon:.4f}, steps={self.steps})"
        )


class SARSAAgent(BaseTabularAgent):
    """On-policy TD control using SARSA.

    Identical interface to QLearningAgent except the update step uses the
    actual next action (on-policy) rather than the greedy maximum.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info(
            "SARSAAgent initialized: n_bins=%d, alpha=%.3f, gamma=%.2f, "
            "epsilon=%.2f→%.2f (decay=%.4f)",
            self.n_bins, self.alpha, self.gamma, self.epsilon, self.epsilon_min, self.epsilon_decay,
        )

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        next_action: int = 0,
        **kwargs,
    ) -> float:
        """SARSA update: Q(s,a) += alpha * [r + gamma * Q(s',a') - Q(s,a)].

        Uses the actual next action (on-policy) instead of max.
        """
        state = self._discretize(obs)
        next_state = self._discretize(next_obs)
        current_q = self.q_table[state][action]
        next_q = 0.0 if done else self.q_table[next_state][next_action]
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        self.q_table[state][action] += self.alpha * td_error
        self.steps += 1
        return abs(td_error)

    def __repr__(self) -> str:
        return (
            f"SARSAAgent(n_actions={self.n_actions}, n_bins={self.n_bins}, "
            f"alpha={self.alpha}, gamma={self.gamma}, "
            f"epsilon={self.epsilon:.4f}, steps={self.steps})"
        )
