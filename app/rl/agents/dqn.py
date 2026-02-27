"""Deep Q-Network (DQN) agent for inventory management RL.

Implements a DQN agent with experience replay and target network
for training on the InventoryEnv environment. The agent learns
an optimal reorder policy by interacting with the environment
and minimizing temporal-difference error on Q-value estimates.
"""

import logging
import random
from collections import deque
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for the DQN agent but was not found. "
        "Install it with:  pip install torch  "
        "or visit https://pytorch.org/get-started/locally/ for platform-specific instructions."
    ) from exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size circular buffer for storing experience tuples.

    Each experience is a (state, action, reward, next_state, done) tuple.
    When the buffer is full, the oldest experience is overwritten.
    """

    def __init__(self, capacity: int) -> None:
        self._buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity,
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition in the buffer."""
        self._buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch of transitions.

        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
            each with leading dimension equal to *batch_size*.
        """
        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch, strict=False)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Multi-layer perceptron that estimates Q-values for each action.

    Architecture: input_dim -> 128 -> 128 -> n_actions
    Uses ReLU activations between hidden layers.
    """

    def __init__(self, input_dim: int = 5, n_actions: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for each action.

        Args:
            x: Batch of state observations, shape (batch, input_dim).

        Returns:
            Q-value estimates, shape (batch, n_actions).
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """Deep Q-Network agent with experience replay and target network.

    The agent follows an epsilon-greedy exploration strategy that decays
    over time, shifting from exploration to exploitation.

    Args:
        state_dim: Dimensionality of the observation space.
        n_actions: Number of discrete actions available.
        lr: Learning rate for the Adam optimizer.
        gamma: Discount factor for future rewards.
        epsilon: Initial exploration rate.
        epsilon_min: Floor for the exploration rate after decay.
        epsilon_decay: Multiplicative decay applied to epsilon after each update.
        buffer_size: Maximum capacity of the replay buffer.
        batch_size: Number of transitions sampled per training step.
        target_update_freq: How many training steps between target network syncs.
    """

    def __init__(
        self,
        state_dim: int = 5,
        n_actions: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device selection (CUDA if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("DQNAgent using device: %s", self.device)

        # Networks
        self.policy_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target net is never trained directly

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Step counter used for target network synchronisation
        self.step_count: int = 0

        logger.info(
            "DQNAgent initialised — state_dim=%d, n_actions=%d, buffer=%d, "
            "batch=%d, gamma=%.3f, epsilon=%.2f->%.2f (decay=%.4f), "
            "target_update_freq=%d",
            state_dim, n_actions, buffer_size, batch_size,
            gamma, epsilon, epsilon_min, epsilon_decay, target_update_freq,
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: Union[np.ndarray, list]) -> int:
        """Choose an action using epsilon-greedy exploration.

        With probability *epsilon* a random action is taken; otherwise the
        action with the highest Q-value from the policy network is selected.

        Args:
            obs: Current environment observation (numpy array or list).

        Returns:
            Integer action index in [0, n_actions).
        """
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        # Convert observation to tensor
        state_tensor = torch.as_tensor(
            np.asarray(obs, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Experience storage and training trigger
    # ------------------------------------------------------------------

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Store a transition and train if the buffer has enough samples.

        Args:
            obs: State before the action.
            action: Action taken.
            reward: Reward received.
            next_obs: State after the action.
            done: Whether the episode terminated.

        Returns:
            Training loss value if a training step was performed, else ``None``.
        """
        self.replay_buffer.push(obs, action, reward, next_obs, done)

        if len(self.replay_buffer) < self.batch_size:
            return None

        loss = self._train_step()
        return loss

    # ------------------------------------------------------------------
    # Core training logic
    # ------------------------------------------------------------------

    def _train_step(self) -> float:
        """Sample a batch and perform one gradient-descent step on MSE loss.

        The loss is computed between the Q-value of the action taken and the
        bootstrapped target:  r + gamma * max_a' Q_target(s', a')  (zero if done).

        Returns:
            Scalar loss value for logging.
        """
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size,
        )

        # Numpy -> Torch tensors
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q-values for chosen actions: Q(s, a)
        q_values = self.policy_net(states_t).gather(1, actions_t)

        # Target Q-values: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1, keepdim=True).values
            target_q = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        # MSE loss
        loss = nn.functional.mse_loss(q_values, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Periodically synchronise target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.debug(
                "Target network synced at step %d (epsilon=%.4f)",
                self.step_count, self.epsilon,
            )

        return loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save agent state (policy network weights) to disk.

        Args:
            path: File path for the checkpoint (e.g. ``"models/dqn.pt"``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        logger.info("DQN policy network saved to %s", path)

    def load(self, path: Union[str, Path]) -> None:
        """Load policy network weights and sync target network.

        Args:
            path: File path to a previously saved checkpoint.
        """
        path = Path(path)
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        logger.info("DQN policy network loaded from %s", path)
