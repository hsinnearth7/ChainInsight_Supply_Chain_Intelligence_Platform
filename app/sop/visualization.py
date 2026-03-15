"""S&OP simulation charts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.config import CHART_BG_COLOR, CHART_DPI, CHART_TEXT_COLOR, COLOR_PALETTE

logger = logging.getLogger(__name__)


def plot_demand_supply_balance(
    period_details: list[dict[str, Any]],
    output_dir: str,
) -> str:
    """Plot demand vs fulfilled demand over time."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=CHART_BG_COLOR)
    ax.set_facecolor(CHART_BG_COLOR)

    if not period_details:
        ax.text(0.5, 0.5, "No simulation data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=COLOR_PALETTE["gray"])
    else:
        periods = [d["period"] for d in period_details]
        demand = [d["demand"] for d in period_details]
        fulfilled = [d["fulfilled"] for d in period_details]
        capacity = [d["capacity"] for d in period_details]

        ax.plot(periods, demand, color=COLOR_PALETTE["danger"], linewidth=2, label="Demand", marker="o", markersize=4)
        ax.plot(periods, fulfilled, color=COLOR_PALETTE["success"], linewidth=2, label="Fulfilled", marker="s", markersize=4)
        ax.plot(periods, capacity, color=COLOR_PALETTE["gray"], linewidth=1, linestyle="--", label="Capacity")
        ax.fill_between(periods, fulfilled, demand, alpha=0.15, color=COLOR_PALETTE["danger"])

    ax.set_ylabel("Units", color=CHART_TEXT_COLOR)
    ax.set_xlabel("Period", color=CHART_TEXT_COLOR)
    ax.set_title("Demand-Supply Balance", color=CHART_TEXT_COLOR, fontsize=13, fontweight="bold")
    ax.legend()
    ax.tick_params(colors=CHART_TEXT_COLOR)

    plt.tight_layout()
    path = str(Path(output_dir) / "chart_sop_balance.png")
    fig.savefig(path, dpi=CHART_DPI, facecolor=CHART_BG_COLOR)
    plt.close(fig)
    logger.info("Saved S&OP balance chart: %s", path)
    return path


def plot_scenario_comparison(
    results: list[dict[str, Any]],
    output_dir: str,
) -> str:
    """Plot scenario comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=CHART_BG_COLOR)

    if not results:
        axes[0].text(0.5, 0.5, "No scenarios", transform=axes[0].transAxes,
                     ha="center", va="center", fontsize=14, color=COLOR_PALETTE["gray"])
    else:
        names = [r["scenario_name"] for r in results]
        colors = [COLOR_PALETTE["primary"], COLOR_PALETTE["success"], COLOR_PALETTE["warning"]]
        colors = colors[:len(names)]

        # Fill Rate
        fill_rates = [r["fill_rate"] * 100 for r in results]
        axes[0].bar(names, fill_rates, color=colors, alpha=0.8, edgecolor="white")
        axes[0].set_title("Fill Rate (%)", color=CHART_TEXT_COLOR, fontsize=11)
        axes[0].set_ylim(0, 110)

        # Utilization
        utils = [r["avg_utilization"] * 100 for r in results]
        axes[1].bar(names, utils, color=colors, alpha=0.8, edgecolor="white")
        axes[1].set_title("Avg Utilization (%)", color=CHART_TEXT_COLOR, fontsize=11)
        axes[1].set_ylim(0, 150)

        # Cost
        costs = [r["total_inventory_cost"] for r in results]
        axes[2].bar(names, costs, color=colors, alpha=0.8, edgecolor="white")
        axes[2].set_title("Inventory Cost ($)", color=CHART_TEXT_COLOR, fontsize=11)

    for ax in axes:
        ax.set_facecolor(CHART_BG_COLOR)
        ax.tick_params(colors=CHART_TEXT_COLOR)

    fig.suptitle("S&OP Scenario Comparison", color=CHART_TEXT_COLOR, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = str(Path(output_dir) / "chart_sop_scenarios.png")
    fig.savefig(path, dpi=CHART_DPI, facecolor=CHART_BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved S&OP scenario comparison chart: %s", path)
    return path
