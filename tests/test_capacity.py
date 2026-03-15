"""Tests for capacity planning module."""

import pandas as pd
import pytest

from app.capacity.models import CapacityPlanner, CapacityProfile


@pytest.fixture
def planner():
    return CapacityPlanner(config={
        "production_lines": 2,
        "shift_hours": 8,
        "max_overtime_hours": 4,
        "changeover_cost_per_hour": 150,
        "utilization_target": 0.85,
        "planning_horizon_days": 28,
    })


@pytest.fixture
def sample_demand():
    return pd.DataFrame({
        "period": [f"W{i}" for i in range(1, 5)],
        "demand": [5000, 6000, 4500, 7000],
    })


class TestCapacityPlanner:
    def test_build_default_profiles(self, planner):
        profiles = planner.build_default_profiles()
        assert len(profiles) == 2
        assert all(isinstance(p, CapacityProfile) for p in profiles)
        assert all(p.line_id.startswith("LINE-") for p in profiles)

    def test_check_feasibility_empty_demand(self, planner):
        report = planner.check_feasibility(pd.DataFrame(columns=["period", "demand"]))
        assert report.feasible
        assert report.avg_utilization == 0.0
        assert report.bottleneck_count == 0

    def test_check_feasibility_normal_demand(self, planner, sample_demand):
        report = planner.check_feasibility(sample_demand)
        assert isinstance(report.avg_utilization, float)
        assert 0 <= report.avg_utilization <= 2.0
        assert isinstance(report.bottleneck_count, int)

    def test_check_feasibility_excess_demand(self, planner):
        df = pd.DataFrame({
            "period": ["W1"],
            "demand": [999999],
        })
        report = planner.check_feasibility(df)
        assert not report.feasible
        assert report.bottleneck_count > 0
        assert len(report.bottlenecks) > 0

    def test_suggest_adjustments_overtime(self, planner):
        from app.capacity.models import Bottleneck
        bns = [Bottleneck(period="W1", line_id="ALL", demand=10000, capacity=8000, deficit=2000, utilization=1.25)]
        adjustments = planner.suggest_adjustments(bns)
        assert len(adjustments) == 1
        assert adjustments[0].action in ("overtime", "outsource")

    def test_suggest_adjustments_empty(self, planner):
        adjustments = planner.suggest_adjustments([])
        assert adjustments == []

    def test_feasibility_report_has_utilizations(self, planner, sample_demand):
        report = planner.check_feasibility(sample_demand)
        assert len(report.period_utilizations) > 0
        assert report.demand_coverage > 0

    def test_custom_profiles(self, planner, sample_demand):
        profiles = [
            CapacityProfile(line_id="MEGA", product_group="all", max_throughput_per_day=50000, efficiency_factor=0.9),
        ]
        report = planner.check_feasibility(sample_demand, profiles=profiles)
        assert report.feasible  # Mega line can handle anything
