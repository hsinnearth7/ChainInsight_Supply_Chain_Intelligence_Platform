"""PPO and A2C agents for inventory management reinforcement learning.

Implements Proximal Policy Optimization (PPO) and Advantage Actor-Critic (A2C)
using a shared ActorCritic neural network architecture.  Designed to work with
:class:`app.rl.environment.InventoryEnv` (obs: Box(5,), action: Discrete(5)).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for the RL agents but was not found. "
        "Install it with:  pip install torch  "
        "(or visit https://pytorch.org/get-started/locally/ for platform-specific instructions)"
    ) from exc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic network.

    Architecture
    ------------
    Shared backbone : Linear(state_dim, 128) -> ReLU -> Linear(128, 128) -> ReLU
    Actor head      : Linear(128, n_actions) -> Softmax   (policy distribution)
    Critic head     : Linear(128, 1)                      (state-value estimate)
    """

    def __init__(self, state_dim: int = 5, n_actions: int = 5) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(128, n_actions)
        self.critic_head = nn.Linear(128, 1)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Orthogonal initialization (standard for PPO/A2C)."""
        for module in self.backbone.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)

        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Observation batch of shape ``(batch, state_dim)``.

        Returns
        -------
        action_probs : torch.Tensor
            Softmax probability distribution over actions ``(batch, n_actions)``.
        state_value : torch.Tensor
            Estimated state value ``(batch, 1)``.
        """
        features = self.backbone(x)
        action_probs = F.softmax(self.actor_head(features), dim=-1)
        state_value = self.critic_head(features)
        return action_probs, state_value


# ---------------------------------------------------------------------------
# Rollout buffer (used by PPO, and optionally by A2C for multi-step)
# ---------------------------------------------------------------------------


@dataclass
class RolloutBuffer:
    """Stores trajectory data and computes GAE advantages / discounted returns.

    All data is kept as Python lists during collection, then converted to
    tensors when :meth:`get` is called.
    """

    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)

    # Computed by compute_returns
    advantages: Optional[np.ndarray] = field(default=None, repr=False)
    returns: Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.rewards)

    # ------------------------------------------------------------------
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    # ------------------------------------------------------------------
    def compute_returns(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
    ) -> None:
        """Compute GAE advantages and discounted returns.

        Parameters
        ----------
        gamma : float
            Discount factor.
        gae_lambda : float
            GAE lambda for bias-variance trade-off.
        last_value : float
            Bootstrap value for the final state (0 if terminal).
        """
        n = self.size
        advantages = np.zeros(n, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        gae = 0.0
        next_value = last_value
        for t in reversed(range(n)):
            delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
            next_value = values[t]

        self.advantages = advantages
        self.returns = advantages + values

    # ------------------------------------------------------------------
    def get(
        self,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
    ) -> Generator[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        None,
        None,
    ]:
        """Yield mini-batches of experience as tensors.

        Yields
        ------
        batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError(
                "Call compute_returns() before iterating over the buffer."
            )

        if device is None:
            device = torch.device("cpu")

        n = self.size
        indices = np.arange(n)
        np.random.shuffle(indices)

        states_arr = np.array(self.states, dtype=np.float32)
        actions_arr = np.array(self.actions, dtype=np.int64)
        log_probs_arr = np.array(self.log_probs, dtype=np.float32)

        for start in range(0, n, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            yield (
                torch.tensor(states_arr[idx], dtype=torch.float32, device=device),
                torch.tensor(actions_arr[idx], dtype=torch.long, device=device),
                torch.tensor(log_probs_arr[idx], dtype=torch.float32, device=device),
                torch.tensor(self.advantages[idx], dtype=torch.float32, device=device),
                torch.tensor(self.returns[idx], dtype=torch.float32, device=device),
            )

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset the buffer for a new rollout."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------


from app.rl.agents.base import BasePolicyAgent


class PPOAgent(BasePolicyAgent):
    """Proximal Policy Optimization (clip variant).

    Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the observation space.
    n_actions : int
        Number of discrete actions.
    lr : float
        Learning rate for the Adam optimizer.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda.
    clip_epsilon : float
        PPO clipping parameter.
    epochs : int
        Number of optimization epochs per update.
    batch_size : int
        Mini-batch size for SGD.
    entropy_coef : float
        Entropy bonus coefficient (encourages exploration).
    value_coef : float
        Value-loss coefficient.
    """

    def __init__(
        self,
        state_dim: int = 5,
        n_actions: int = 5,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("PPOAgent using device: %s", self.device)

        self.policy = ActorCritic(state_dim, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

        # Running statistics for logging
        self._update_count = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self, obs: np.ndarray
    ) -> tuple[int, float, float]:
        """Sample an action from the current policy.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape ``(state_dim,)``.

        Returns
        -------
        action : int
            Sampled discrete action.
        log_prob : float
            Log-probability of the sampled action.
        value : float
            Estimated state value.
        """
        state_t = torch.tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action_probs, state_value = self.policy(state_t)

        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return (
            action.item(),
            log_prob.item(),
            state_value.item(),
        )

    # ------------------------------------------------------------------
    # Transition storage
    # ------------------------------------------------------------------

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Append a single transition to the rollout buffer."""
        self.buffer.add(obs, action, reward, done, log_prob, value)

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update(self, last_value: float = 0.0) -> dict:
        """Run PPO clipped-objective update over multiple epochs.

        Parameters
        ----------
        last_value : float
            Bootstrap value for the final state (pass 0 when the episode
            ended in a terminal state).

        Returns
        -------
        dict
            Training metrics: ``policy_loss``, ``value_loss``,
            ``entropy``, ``approx_kl``.
        """
        if self.buffer.size == 0:
            logger.warning("PPOAgent.update() called with empty buffer.")
            return {}

        self.buffer.compute_returns(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            last_value=last_value,
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        num_batches = 0

        for _epoch in range(self.epochs):
            for (
                batch_states,
                batch_actions,
                batch_old_log_probs,
                batch_advantages,
                batch_returns,
            ) in self.buffer.get(self.batch_size, self.device):

                # Normalize advantages within the batch
                adv = batch_advantages
                if adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Forward pass
                action_probs, state_values = self.policy(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO clipped surrogate objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (simple MSE)
                value_loss = F.mse_loss(
                    state_values.squeeze(-1), batch_returns
                )

                # Combined loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - new_log_probs).mean().item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_approx_kl += approx_kl
                num_batches += 1

        self.buffer.clear()
        self._update_count += 1

        metrics = {
            "policy_loss": total_policy_loss / max(num_batches, 1),
            "value_loss": total_value_loss / max(num_batches, 1),
            "entropy": total_entropy / max(num_batches, 1),
            "approx_kl": total_approx_kl / max(num_batches, 1),
        }
        logger.info(
            "PPO update #%d  |  policy_loss=%.4f  value_loss=%.4f  entropy=%.4f  approx_kl=%.4f",
            self._update_count,
            metrics["policy_loss"],
            metrics["value_loss"],
            metrics["entropy"],
            metrics["approx_kl"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and optimizer state to *path*."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "update_count": self._update_count,
            },
            path,
        )
        logger.info("PPOAgent saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights and optimizer state from *path*."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._update_count = checkpoint.get("update_count", 0)
        logger.info("PPOAgent loaded from %s (update_count=%d)", path, self._update_count)


# ---------------------------------------------------------------------------
# A2C Agent
# ---------------------------------------------------------------------------


class A2CAgent(BasePolicyAgent):
    """Advantage Actor-Critic (A2C) — synchronous, single-step updates.

    Unlike PPO, A2C performs a single gradient step per update without
    importance-sampling clipping or experience replay.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the observation space.
    n_actions : int
        Number of discrete actions.
    lr : float
        Learning rate for the Adam optimizer.
    gamma : float
        Discount factor.
    entropy_coef : float
        Entropy bonus coefficient.
    value_coef : float
        Value-loss coefficient.
    n_steps : int
        Number of environment steps collected before each update.
    """

    def __init__(
        self,
        state_dim: int = 5,
        n_actions: int = 5,
        lr: float = 7e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        n_steps: int = 5,
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_steps = n_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("A2CAgent using device: %s", self.device)

        self.policy = ActorCritic(state_dim, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Transition accumulators (simple lists — no replay)
        self._states: list[np.ndarray] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []
        self._log_probs: list[float] = []
        self._values: list[float] = []

        self._update_count = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self, obs: np.ndarray
    ) -> tuple[int, float, float]:
        """Sample an action from the current policy.

        Returns
        -------
        action : int
        log_prob : float
        value : float
        """
        state_t = torch.tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action_probs, state_value = self.policy(state_t)

        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return (
            action.item(),
            log_prob.item(),
            state_value.item(),
        )

    # ------------------------------------------------------------------
    # Transition storage
    # ------------------------------------------------------------------

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Store a single transition for the next update."""
        self._states.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)
        self._dones.append(done)
        self._log_probs.append(log_prob)
        self._values.append(value)

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update(self, last_value: float = 0.0) -> dict:
        """Single-step advantage actor-critic update (no clipping, no replay).

        Parameters
        ----------
        last_value : float
            Bootstrap value for the final state.

        Returns
        -------
        dict
            Training metrics: ``policy_loss``, ``value_loss``, ``entropy``.
        """
        n = len(self._rewards)
        if n == 0:
            logger.warning("A2CAgent.update() called with no stored transitions.")
            return {}

        # ------- Compute n-step returns -------
        returns = np.zeros(n, dtype=np.float32)
        R = last_value
        for t in reversed(range(n)):
            R = self._rewards[t] + self.gamma * R * (1.0 - float(self._dones[t]))
            returns[t] = R

        # ------- Convert to tensors -------
        states_t = torch.tensor(
            np.array(self._states, dtype=np.float32), device=self.device
        )
        actions_t = torch.tensor(
            np.array(self._actions, dtype=np.int64), device=self.device
        )
        returns_t = torch.tensor(returns, device=self.device)

        # ------- Forward pass (with gradient) -------
        action_probs, state_values = self.policy(states_t)
        state_values = state_values.squeeze(-1)

        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        # ------- Advantage (no GAE — simple TD residual) -------
        advantages = returns_t - state_values.detach()

        # ------- Losses -------
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(state_values, returns_t)

        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        # ------- Clear stored transitions -------
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()
        self._log_probs.clear()
        self._values.clear()

        self._update_count += 1

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
        logger.info(
            "A2C update #%d  |  policy_loss=%.4f  value_loss=%.4f  entropy=%.4f",
            self._update_count,
            metrics["policy_loss"],
            metrics["value_loss"],
            metrics["entropy"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and optimizer state to *path*."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "update_count": self._update_count,
            },
            path,
        )
        logger.info("A2CAgent saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights and optimizer state from *path*."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._update_count = checkpoint.get("update_count", 0)
        logger.info("A2CAgent loaded from %s (update_count=%d)", path, self._update_count)
