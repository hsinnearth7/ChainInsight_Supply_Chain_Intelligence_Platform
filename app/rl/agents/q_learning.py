"""Q-Learning and SARSA agents for inventory management RL.

Tabular agents that discretize the continuous 5-dim observation space
into bins and maintain a Q-table for action selection.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class QLearningAgent:
    """Off-policy TD control using Q-Learning (max over next actions).

    Discretizes the continuous [0,1]^5 observation space into bins per
    dimension, producing a finite state space suitable for tabular methods.
    """

    def __init__(
        self,
        n_actions: int = 5,
        n_bins: int = 10,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Bin edges for each of the 5 observation dimensions (all [0, 1])
        self.bin_edges = [
            np.linspace(0.0, 1.0, n_bins + 1)[1:-1] for _ in range(5)
        ]

        # Q-table: state (tuple of ints) → array of shape (n_actions,)
        self.q_table: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

        self.steps = 0
        self.episodes = 0
        logger.info(
            "QLearningAgent initialized: n_bins=%d, alpha=%.3f, gamma=%.2f, "
            "epsilon=%.2f→%.2f (decay=%.4f)",
            n_bins, alpha, gamma, epsilon, epsilon_min, epsilon_decay,
        )

    def _discretize(self, obs: np.ndarray) -> tuple[int, ...]:
        """Convert continuous observation to a tuple of bin indices."""
        return tuple(
            int(np.digitize(obs[i], self.bin_edges[i]))
            for i in range(len(self.bin_edges))
        )

    def select_action(self, obs: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        state = self._discretize(obs)
        q_values = self.q_table[state]
        # Break ties randomly
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)
        return int(np.random.choice(best_actions))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> float:
        """Q-Learning update: Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)].

        Returns the TD error magnitude.
        """
        state = self._discretize(obs)
        next_state = self._discretize(next_obs)

        current_q = self.q_table[state][action]
        next_max_q = 0.0 if done else np.max(self.q_table[next_state])

        td_target = reward + self.gamma * next_max_q
        td_error = td_target - current_q

        self.q_table[state][action] += self.alpha * td_error
        self.steps += 1

        return abs(td_error)

    def decay_epsilon(self) -> None:
        """Decay exploration rate (call once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes += 1

    def get_q_table_stats(self) -> dict:
        """Return diagnostic statistics about the Q-table."""
        if not self.q_table:
            return {
                "table_size": 0,
                "non_zero_entries": 0,
                "total_cells": 0,
                "sparsity": 1.0,
                "max_q": 0.0,
                "min_q": 0.0,
                "mean_q": 0.0,
            }

        all_values = np.array([v for v in self.q_table.values()])
        non_zero = int(np.count_nonzero(all_values))
        total_cells = all_values.size

        return {
            "table_size": len(self.q_table),
            "non_zero_entries": non_zero,
            "total_cells": total_cells,
            "sparsity": 1.0 - (non_zero / total_cells) if total_cells else 1.0,
            "max_q": float(np.max(all_values)),
            "min_q": float(np.min(all_values)),
            "mean_q": float(np.mean(all_values)),
            "steps": self.steps,
            "episodes": self.episodes,
            "epsilon": self.epsilon,
        }

    def save(self, path: str | Path) -> None:
        """Persist Q-table and hyperparameters to an .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        states = []
        values = []
        for state, q_vals in self.q_table.items():
            states.append(state)
            values.append(q_vals)

        np.savez_compressed(
            path,
            states=np.array(states, dtype=np.int32) if states else np.empty((0, 5), dtype=np.int32),
            values=np.array(values, dtype=np.float64) if values else np.empty((0, self.n_actions), dtype=np.float64),
            params=np.array([
                self.n_actions, self.n_bins, self.alpha, self.gamma,
                self.epsilon, self.epsilon_min, self.epsilon_decay,
                self.steps, self.episodes,
            ]),
        )
        logger.info("Saved Q-table (%d states) to %s", len(self.q_table), path)

    def load(self, path: str | Path) -> None:
        """Load Q-table and hyperparameters from an .npz file."""
        path = Path(path)
        data = np.load(path, allow_pickle=False)

        params = data["params"]
        self.n_actions = int(params[0])
        self.n_bins = int(params[1])
        self.alpha = float(params[2])
        self.gamma = float(params[3])
        self.epsilon = float(params[4])
        self.epsilon_min = float(params[5])
        self.epsilon_decay = float(params[6])
        self.steps = int(params[7])
        self.episodes = int(params[8])

        # Rebuild bin edges for potentially changed n_bins
        self.bin_edges = [
            np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1] for _ in range(5)
        ]

        # Rebuild Q-table
        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )
        states = data["states"]
        values = data["values"]
        for i in range(len(states)):
            key = tuple(int(x) for x in states[i])
            self.q_table[key] = values[i].copy()

        logger.info(
            "Loaded Q-table (%d states, %d steps) from %s",
            len(self.q_table), self.steps, path,
        )

    def __repr__(self) -> str:
        return (
            f"QLearningAgent(n_actions={self.n_actions}, n_bins={self.n_bins}, "
            f"alpha={self.alpha}, gamma={self.gamma}, "
            f"epsilon={self.epsilon:.4f}, steps={self.steps})"
        )


class SARSAAgent:
    """On-policy TD control using SARSA.

    Identical interface to QLearningAgent except the update step uses the
    actual next action (on-policy) rather than the greedy maximum.
    """

    def __init__(
        self,
        n_actions: int = 5,
        n_bins: int = 10,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.bin_edges = [
            np.linspace(0.0, 1.0, n_bins + 1)[1:-1] for _ in range(5)
        ]

        self.q_table: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

        self.steps = 0
        self.episodes = 0
        logger.info(
            "SARSAAgent initialized: n_bins=%d, alpha=%.3f, gamma=%.2f, "
            "epsilon=%.2f→%.2f (decay=%.4f)",
            n_bins, alpha, gamma, epsilon, epsilon_min, epsilon_decay,
        )

    def _discretize(self, obs: np.ndarray) -> tuple[int, ...]:
        """Convert continuous observation to a tuple of bin indices."""
        return tuple(
            int(np.digitize(obs[i], self.bin_edges[i]))
            for i in range(len(self.bin_edges))
        )

    def select_action(self, obs: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        state = self._discretize(obs)
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)
        return int(np.random.choice(best_actions))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        next_action: int,
        done: bool,
    ) -> float:
        """SARSA update: Q(s,a) += alpha * [r + gamma * Q(s',a') - Q(s,a)].

        Uses the actual next action (on-policy) instead of max.
        Returns the TD error magnitude.
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

    def decay_epsilon(self) -> None:
        """Decay exploration rate (call once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes += 1

    def get_q_table_stats(self) -> dict:
        """Return diagnostic statistics about the Q-table."""
        if not self.q_table:
            return {
                "table_size": 0,
                "non_zero_entries": 0,
                "total_cells": 0,
                "sparsity": 1.0,
                "max_q": 0.0,
                "min_q": 0.0,
                "mean_q": 0.0,
            }

        all_values = np.array([v for v in self.q_table.values()])
        non_zero = int(np.count_nonzero(all_values))
        total_cells = all_values.size

        return {
            "table_size": len(self.q_table),
            "non_zero_entries": non_zero,
            "total_cells": total_cells,
            "sparsity": 1.0 - (non_zero / total_cells) if total_cells else 1.0,
            "max_q": float(np.max(all_values)),
            "min_q": float(np.min(all_values)),
            "mean_q": float(np.mean(all_values)),
            "steps": self.steps,
            "episodes": self.episodes,
            "epsilon": self.epsilon,
        }

    def save(self, path: str | Path) -> None:
        """Persist Q-table and hyperparameters to an .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        states = []
        values = []
        for state, q_vals in self.q_table.items():
            states.append(state)
            values.append(q_vals)

        np.savez_compressed(
            path,
            states=np.array(states, dtype=np.int32) if states else np.empty((0, 5), dtype=np.int32),
            values=np.array(values, dtype=np.float64) if values else np.empty((0, self.n_actions), dtype=np.float64),
            params=np.array([
                self.n_actions, self.n_bins, self.alpha, self.gamma,
                self.epsilon, self.epsilon_min, self.epsilon_decay,
                self.steps, self.episodes,
            ]),
        )
        logger.info("Saved SARSA Q-table (%d states) to %s", len(self.q_table), path)

    def load(self, path: str | Path) -> None:
        """Load Q-table and hyperparameters from an .npz file."""
        path = Path(path)
        data = np.load(path, allow_pickle=False)

        params = data["params"]
        self.n_actions = int(params[0])
        self.n_bins = int(params[1])
        self.alpha = float(params[2])
        self.gamma = float(params[3])
        self.epsilon = float(params[4])
        self.epsilon_min = float(params[5])
        self.epsilon_decay = float(params[6])
        self.steps = int(params[7])
        self.episodes = int(params[8])

        self.bin_edges = [
            np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1] for _ in range(5)
        ]

        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )
        states = data["states"]
        values = data["values"]
        for i in range(len(states)):
            key = tuple(int(x) for x in states[i])
            self.q_table[key] = values[i].copy()

        logger.info(
            "Loaded SARSA Q-table (%d states, %d steps) from %s",
            len(self.q_table), self.steps, path,
        )

    def __repr__(self) -> str:
        return (
            f"SARSAAgent(n_actions={self.n_actions}, n_bins={self.n_bins}, "
            f"alpha={self.alpha}, gamma={self.gamma}, "
            f"epsilon={self.epsilon:.4f}, steps={self.steps})"
        )
