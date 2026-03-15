"""S&OP (Sales & Operations Planning) simulator.

Combines demand forecasts, capacity constraints, and supply constraints
to simulate planning scenarios and compare outcomes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.settings import get_sop_config

logger = logging.getLogger(__name__)


@dataclass
class SupplyConstraint:
    """Supply-side constraint for a material/supplier."""
    supplier_id: str
    material: str
    max_qty_per_period: float
    lead_time_days: int = 14
    reliability_pct: float = 0.95


@dataclass
class SOPScenario:
    """A single S&OP planning scenario."""
    name: str
    demand_multiplier: float = 1.0
    capacity_multiplier: float = 1.0
    supply_reliability: float = 0.95
    description: str = ""


@dataclass
class SOPResult:
    """Result of a single S&OP simulation run."""
    scenario_name: str
    fill_rate: float
    avg_utilization: float
    total_inventory_cost: float
    stockout_events: int
    periods_simulated: int
    period_details: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Side-by-side comparison of multiple S&OP scenarios."""
    results: list[SOPResult]
    best_scenario: str
    summary: dict[str, Any] = field(default_factory=dict)


class SOPSimulator:
    """Sales & Operations Planning simulator with scenario analysis."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or get_sop_config()
        self.horizon_days = self.config.get("planning_horizon_days", 90)
        self.time_bucket = self.config.get("time_bucket", "weekly")
        self.target_fill_rate = self.config.get("target_fill_rate", 0.95)
        self.target_turns = self.config.get("target_inventory_turns", 12)
        self.stockout_penalty = self.config.get("stockout_penalty", 50.0)
        self.holding_rate = self.config.get("holding_cost_rate", 0.25)

    def create_default_scenarios(self) -> list[SOPScenario]:
        """Create baseline, optimistic, and conservative scenarios."""
        return [
            SOPScenario(
                name="baseline",
                demand_multiplier=1.0,
                capacity_multiplier=1.0,
                supply_reliability=0.95,
                description="Expected demand with current capacity",
            ),
            SOPScenario(
                name="optimistic",
                demand_multiplier=1.2,
                capacity_multiplier=1.1,
                supply_reliability=0.98,
                description="20% demand increase with expanded capacity",
            ),
            SOPScenario(
                name="conservative",
                demand_multiplier=0.85,
                capacity_multiplier=0.95,
                supply_reliability=0.90,
                description="15% demand decrease with reduced supply reliability",
            ),
        ]

    def simulate(
        self,
        base_demand: pd.DataFrame,
        daily_capacity: float,
        scenario: SOPScenario,
        seed: int = 42,
    ) -> SOPResult:
        """Run S&OP simulation for a single scenario.

        Args:
            base_demand: DataFrame with columns [period, demand].
            daily_capacity: Base daily production capacity.
            scenario: Scenario parameters.
            seed: Random seed for stochastic elements.

        Returns:
            SOPResult with KPIs and period details.
        """
        rng = np.random.RandomState(seed)

        # Determine periods
        bucket_days = 7 if self.time_bucket == "weekly" else 30
        n_periods = max(1, self.horizon_days // bucket_days)

        # Generate or use demand
        if base_demand.empty:
            period_demands = rng.normal(1000, 200, size=n_periods).clip(min=0)
        else:
            if len(base_demand) >= n_periods:
                period_demands = base_demand["demand"].values[:n_periods]
            else:
                period_demands = np.resize(base_demand["demand"].values, n_periods)

        # Apply scenario multipliers
        adjusted_demand = period_demands * scenario.demand_multiplier
        period_capacity = daily_capacity * bucket_days * scenario.capacity_multiplier

        # Simulate
        fulfilled = 0
        total_demand = 0
        inventory = period_capacity * 0.3  # start with 30% capacity as initial inventory
        stockout_events = 0
        total_holding_cost = 0
        total_stockout_cost = 0
        period_details = []

        for i in range(n_periods):
            demand = float(adjusted_demand[i])
            total_demand += demand

            # Supply arrives (with reliability)
            supply_arrived = period_capacity * (
                1.0 if rng.random() < scenario.supply_reliability else 0.7
            )
            inventory += supply_arrived

            # Fulfill demand
            fulfilled_qty = min(demand, inventory)
            fulfilled += fulfilled_qty
            unfulfilled = demand - fulfilled_qty
            inventory -= fulfilled_qty

            if unfulfilled > 0:
                stockout_events += 1
                total_stockout_cost += unfulfilled * self.stockout_penalty

            holding_cost = inventory * self.holding_rate * (bucket_days / 365)
            total_holding_cost += holding_cost

            utilization = demand / period_capacity if period_capacity > 0 else 0

            period_details.append({
                "period": i + 1,
                "demand": round(demand, 1),
                "capacity": round(period_capacity, 1),
                "fulfilled": round(fulfilled_qty, 1),
                "inventory": round(inventory, 1),
                "utilization": round(min(utilization, 2.0), 3),
                "stockout": unfulfilled > 0,
            })

        fill_rate = fulfilled / total_demand if total_demand > 0 else 1.0
        avg_util = float(np.mean([d["utilization"] for d in period_details]))

        return SOPResult(
            scenario_name=scenario.name,
            fill_rate=float(fill_rate),
            avg_utilization=float(avg_util),
            total_inventory_cost=round(total_holding_cost + total_stockout_cost, 2),
            stockout_events=stockout_events,
            periods_simulated=n_periods,
            period_details=period_details,
        )

    def compare_scenarios(
        self,
        base_demand: pd.DataFrame,
        daily_capacity: float,
        scenarios: list[SOPScenario] | None = None,
    ) -> ComparisonReport:
        """Run and compare multiple S&OP scenarios.

        Args:
            base_demand: Base demand forecast.
            daily_capacity: Base daily production capacity.
            scenarios: Scenarios to compare. Uses defaults if None.

        Returns:
            ComparisonReport with all results and best scenario.
        """
        scenarios = scenarios or self.create_default_scenarios()
        results = []

        for i, scenario in enumerate(scenarios):
            result = self.simulate(base_demand, daily_capacity, scenario, seed=42 + i)
            results.append(result)
            logger.info(
                "S&OP scenario '%s': fill_rate=%.1f%%, utilization=%.1f%%",
                scenario.name, result.fill_rate * 100, result.avg_utilization * 100,
            )

        # Best scenario = highest fill rate, then lowest cost
        best = max(results, key=lambda r: (r.fill_rate, -r.total_inventory_cost))

        summary = {
            "scenarios_compared": len(results),
            "best_scenario": best.scenario_name,
            "best_fill_rate": best.fill_rate,
            "target_fill_rate": self.target_fill_rate,
            "target_met": best.fill_rate >= self.target_fill_rate,
        }

        return ComparisonReport(results=results, best_scenario=best.scenario_name, summary=summary)

    def calculate_kpis(self, result: SOPResult) -> dict[str, Any]:
        """Extract KPIs from simulation result."""
        return {
            "scenario": result.scenario_name,
            "fill_rate": result.fill_rate,
            "avg_utilization": result.avg_utilization,
            "inventory_cost": result.total_inventory_cost,
            "stockout_events": result.stockout_events,
            "periods": result.periods_simulated,
            "meets_target": result.fill_rate >= self.target_fill_rate,
        }
