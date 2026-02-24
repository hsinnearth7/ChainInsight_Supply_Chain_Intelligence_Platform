"""Gymnasium-compatible Inventory Management Environment.

Simulates daily inventory decisions for a single product/category.
An agent observes the current inventory state and decides how much to reorder.
"""

import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


class InventoryEnv(gym.Env):
    """Single-product inventory management environment.

    State (5-dim continuous):
        [0] current_stock       — units on hand (normalized by max_stock)
        [1] pending_orders      — units in transit (normalized by max_stock)
        [2] days_since_order    — days since last reorder (normalized by lead_time)
        [3] demand_trend        — rolling 7-day demand average (normalized)
        [4] stockout_days       — consecutive days of zero stock (normalized)

    Action (discrete, 5 choices):
        0 = do nothing
        1 = order 0.5 × EOQ
        2 = order 1.0 × EOQ
        3 = order 1.5 × EOQ
        4 = order 2.0 × EOQ

    Reward:
        - holding_cost   = holding_rate_daily × unit_cost × current_stock
        - stockout_cost  = stockout_penalty × unmet_demand
        - ordering_cost  = fixed ordering cost if an order is placed
        reward = -(holding_cost + stockout_cost + ordering_cost)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        unit_cost: float = 50.0,
        daily_demand_mean: float = 20.0,
        daily_demand_std: float = 5.0,
        lead_time: int = 5,
        eoq: float = 200.0,
        safety_stock: float = 50.0,
        ordering_cost: float = 50.0,
        holding_rate: float = 0.25,
        stockout_penalty: float = 10.0,
        max_stock: float = 2000.0,
        episode_length: int = 90,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # Product parameters
        self.unit_cost = unit_cost
        self.daily_demand_mean = daily_demand_mean
        self.daily_demand_std = daily_demand_std
        self.lead_time = lead_time
        self.eoq = eoq
        self.safety_stock = safety_stock
        self.ordering_cost = ordering_cost
        self.holding_rate_daily = holding_rate / 365.0
        self.stockout_penalty = stockout_penalty
        self.max_stock = max_stock
        self.episode_length = episode_length

        # Action: 5 discrete choices
        self.action_space = spaces.Discrete(5)
        self.action_to_qty = {
            0: 0.0,
            1: 0.5 * eoq,
            2: 1.0 * eoq,
            3: 1.5 * eoq,
            4: 2.0 * eoq,
        }

        # Observation: 5-dim normalized continuous
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

        # Internal state
        self.current_stock = 0.0
        self.pending_orders: list[tuple[int, float]] = []  # (arrival_day, qty)
        self.days_since_order = 0
        self.demand_history: list[float] = []
        self.stockout_days = 0
        self.day = 0

        # Episode tracking
        self.episode_rewards: list[float] = []
        self.episode_costs: dict[str, float] = {
            "holding": 0.0,
            "stockout": 0.0,
            "ordering": 0.0,
        }
        self.episode_service_level: list[bool] = []

    @classmethod
    def from_product_data(
        cls,
        unit_cost: float,
        daily_demand: float,
        lead_time: int,
        safety_stock: float,
        current_stock: float,
        ordering_cost: float = 50.0,
        holding_rate: float = 0.25,
        **kwargs,
    ) -> "InventoryEnv":
        """Create environment from real product data."""
        annual_demand = daily_demand * 365
        h = holding_rate * unit_cost
        eoq = np.sqrt(2 * annual_demand * ordering_cost / max(h, 0.01))
        demand_std = daily_demand * 0.25  # assume 25% CV

        env = cls(
            unit_cost=unit_cost,
            daily_demand_mean=daily_demand,
            daily_demand_std=demand_std,
            lead_time=max(lead_time, 1),
            eoq=max(eoq, 1.0),
            safety_stock=safety_stock,
            ordering_cost=ordering_cost,
            holding_rate=holding_rate,
            max_stock=max(current_stock * 3, eoq * 3, 500.0),
            **kwargs,
        )
        return env

    def _get_obs(self) -> np.ndarray:
        """Return normalized observation."""
        demand_trend = (
            np.mean(self.demand_history[-7:]) / max(self.daily_demand_mean, 1.0)
            if self.demand_history
            else 1.0
        )
        pending_total = sum(qty for _, qty in self.pending_orders)

        obs = np.array([
            np.clip(self.current_stock / self.max_stock, 0, 1),
            np.clip(pending_total / self.max_stock, 0, 1),
            np.clip(self.days_since_order / max(self.lead_time, 1), 0, 1),
            np.clip(demand_trend, 0, 1),
            np.clip(self.stockout_days / 10.0, 0, 1),
        ], dtype=np.float32)
        return obs

    def _get_info(self) -> dict:
        return {
            "day": self.day,
            "current_stock": self.current_stock,
            "pending_orders": len(self.pending_orders),
            "stockout_days": self.stockout_days,
            "episode_costs": self.episode_costs.copy(),
            "service_level": (
                np.mean(self.episode_service_level)
                if self.episode_service_level else 0.0
            ),
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Initialize stock between safety_stock and 2×safety_stock
        self.current_stock = self.np_random.uniform(
            self.safety_stock, self.safety_stock * 2
        )
        self.pending_orders = []
        self.days_since_order = 0
        self.demand_history = []
        self.stockout_days = 0
        self.day = 0
        self.episode_rewards = []
        self.episode_costs = {"holding": 0.0, "stockout": 0.0, "ordering": 0.0}
        self.episode_service_level = []

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 1. Place order if action > 0
        order_cost = 0.0
        order_qty = self.action_to_qty[action]
        if order_qty > 0:
            arrival_day = self.day + self.lead_time
            self.pending_orders.append((arrival_day, order_qty))
            order_cost = self.ordering_cost
            self.days_since_order = 0
        else:
            self.days_since_order += 1

        # 2. Receive pending orders that arrive today
        arrived = [qty for (arr, qty) in self.pending_orders if arr <= self.day]
        self.pending_orders = [
            (arr, qty) for (arr, qty) in self.pending_orders if arr > self.day
        ]
        self.current_stock += sum(arrived)

        # 3. Generate daily demand (stochastic)
        demand = max(0, self.np_random.normal(self.daily_demand_mean, self.daily_demand_std))
        self.demand_history.append(demand)

        # 4. Fulfill demand
        fulfilled = min(demand, self.current_stock)
        unmet = demand - fulfilled
        self.current_stock -= fulfilled
        self.current_stock = min(self.current_stock, self.max_stock)  # cap

        # Track service level
        self.episode_service_level.append(unmet == 0)

        # 5. Track stockout streak
        if self.current_stock <= 0:
            self.stockout_days += 1
        else:
            self.stockout_days = 0

        # 6. Compute costs
        holding_cost = self.holding_rate_daily * self.unit_cost * self.current_stock
        stockout_cost = self.stockout_penalty * unmet

        self.episode_costs["holding"] += holding_cost
        self.episode_costs["stockout"] += stockout_cost
        self.episode_costs["ordering"] += order_cost

        # 7. Reward
        reward = -(holding_cost + stockout_cost + order_cost)
        self.episode_rewards.append(reward)

        # 8. Advance day
        self.day += 1
        terminated = self.day >= self.episode_length
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def get_episode_summary(self) -> dict:
        """Return summary metrics for the completed episode."""
        return {
            "total_reward": sum(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "total_holding_cost": self.episode_costs["holding"],
            "total_stockout_cost": self.episode_costs["stockout"],
            "total_ordering_cost": self.episode_costs["ordering"],
            "total_cost": sum(self.episode_costs.values()),
            "service_level": (
                np.mean(self.episode_service_level)
                if self.episode_service_level else 0.0
            ),
            "days": self.day,
        }
