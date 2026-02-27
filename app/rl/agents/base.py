"""Base classes for RL agents."""

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np


class BaseTabularAgent(ABC):
    """Base for Q-Learning and SARSA agents.

    Provides shared functionality: discretization, epsilon-greedy action
    selection, epsilon decay, Q-table statistics, persistence.
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

    @abstractmethod
    def update(self, obs, action, reward, next_obs, done, **kwargs) -> float:
        """Perform the TD update. Returns TD error magnitude."""
        ...

    def decay_epsilon(self) -> None:
        """Decay exploration rate (call once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes += 1

    def get_q_table_stats(self) -> dict:
        """Return diagnostic statistics about the Q-table."""
        if not self.q_table:
            return {
                "table_size": 0, "non_zero_entries": 0, "total_cells": 0,
                "sparsity": 1.0, "max_q": 0.0, "min_q": 0.0, "mean_q": 0.0,
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
        states, values = [], []
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
        self.bin_edges = [np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1] for _ in range(5)]
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float64))
        states = data["states"]
        values = data["values"]
        for i in range(len(states)):
            self.q_table[tuple(int(x) for x in states[i])] = values[i].copy()


class BasePolicyAgent(ABC):
    """Base for PPO and A2C agents.

    Defines the shared interface for neural-network-based policy agents.
    """

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> tuple[int, float, float]:
        """Sample action, return (action, log_prob, value)."""
        ...

    @abstractmethod
    def store_transition(self, obs, action, reward, done, log_prob, value) -> None:
        """Store a single transition."""
        ...

    @abstractmethod
    def update(self, last_value: float) -> dict:
        """Run policy update, return metrics dict."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...
