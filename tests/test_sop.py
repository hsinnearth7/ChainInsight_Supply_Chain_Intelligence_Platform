"""Tests for S&OP simulation module."""

import pandas as pd
import pytest

from app.sop.simulator import ComparisonReport, SOPResult, SOPScenario, SOPSimulator


@pytest.fixture
def simulator():
    return SOPSimulator(config={
        "planning_horizon_days": 28,
        "time_bucket": "weekly",
        "target_fill_rate": 0.95,
        "target_inventory_turns": 12,
        "stockout_penalty": 50.0,
        "holding_cost_rate": 0.25,
    })


@pytest.fixture
def sample_demand():
    return pd.DataFrame({
        "period": [1, 2, 3, 4],
        "demand": [1000, 1200, 800, 1100],
    })


class TestSOPSimulator:
    def test_create_default_scenarios(self, simulator):
        scenarios = simulator.create_default_scenarios()
        assert len(scenarios) == 3
        names = {s.name for s in scenarios}
        assert names == {"baseline", "optimistic", "conservative"}

    def test_simulate_baseline(self, simulator, sample_demand):
        scenario = SOPScenario(name="test", demand_multiplier=1.0, capacity_multiplier=1.0)
        result = simulator.simulate(sample_demand, daily_capacity=200, scenario=scenario)
        assert isinstance(result, SOPResult)
        assert 0 <= result.fill_rate <= 1.0
        assert result.periods_simulated > 0
        assert len(result.period_details) > 0

    def test_simulate_empty_demand(self, simulator):
        scenario = SOPScenario(name="empty")
        result = simulator.simulate(pd.DataFrame(columns=["period", "demand"]), daily_capacity=200, scenario=scenario)
        assert result.periods_simulated > 0

    def test_simulate_high_demand_causes_stockouts(self, simulator):
        df = pd.DataFrame({"period": [1, 2, 3, 4], "demand": [99999, 99999, 99999, 99999]})
        scenario = SOPScenario(name="overload", demand_multiplier=1.0)
        result = simulator.simulate(df, daily_capacity=100, scenario=scenario)
        assert result.fill_rate < 1.0
        assert result.stockout_events > 0

    def test_compare_scenarios(self, simulator, sample_demand):
        report = simulator.compare_scenarios(sample_demand, daily_capacity=200)
        assert isinstance(report, ComparisonReport)
        assert len(report.results) == 3
        assert report.best_scenario in {"baseline", "optimistic", "conservative"}

    def test_calculate_kpis(self, simulator, sample_demand):
        scenario = SOPScenario(name="test")
        result = simulator.simulate(sample_demand, daily_capacity=200, scenario=scenario)
        kpis = simulator.calculate_kpis(result)
        assert "fill_rate" in kpis
        assert "avg_utilization" in kpis
        assert "inventory_cost" in kpis
        assert "stockout_events" in kpis

    def test_deterministic_simulation(self, simulator, sample_demand):
        scenario = SOPScenario(name="test")
        r1 = simulator.simulate(sample_demand, daily_capacity=200, scenario=scenario, seed=42)
        r2 = simulator.simulate(sample_demand, daily_capacity=200, scenario=scenario, seed=42)
        assert r1.fill_rate == r2.fill_rate
        assert r1.total_inventory_cost == r2.total_inventory_cost

    def test_scenario_multipliers_affect_results(self, simulator, sample_demand):
        base = SOPScenario(name="base", demand_multiplier=1.0)
        high = SOPScenario(name="high", demand_multiplier=2.0)
        r_base = simulator.simulate(sample_demand, daily_capacity=200, scenario=base)
        r_high = simulator.simulate(sample_demand, daily_capacity=200, scenario=high)
        # Higher demand should lead to lower fill rate or higher cost
        assert r_high.fill_rate <= r_base.fill_rate or r_high.total_inventory_cost >= r_base.total_inventory_cost

    def test_period_details_structure(self, simulator, sample_demand):
        scenario = SOPScenario(name="test")
        result = simulator.simulate(sample_demand, daily_capacity=200, scenario=scenario)
        for detail in result.period_details:
            assert "period" in detail
            assert "demand" in detail
            assert "capacity" in detail
            assert "fulfilled" in detail
            assert "utilization" in detail
