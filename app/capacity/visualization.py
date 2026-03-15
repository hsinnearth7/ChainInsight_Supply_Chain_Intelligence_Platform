"""Capacity planning charts."""

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


def plot_utilization_timeline(
    utilizations: dict[str, float],
    output_dir: str,
    target: float = 0.85,
) -> str:
    """Plot capacity utilization over time periods."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=CHART_BG_COLOR)
    ax.set_facecolor(CHART_BG_COLOR)

    periods = list(utilizations.keys())
    values = [v * 100 for v in utilizations.values()]

    colors = [
        COLOR_PALETTE["danger"] if v > 100
        else COLOR_PALETTE["warning"] if v > target * 100
        else COLOR_PALETTE["success"]
        for v in values
    ]

    ax.bar(periods, values, color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(y=target * 100, color=COLOR_PALETTE["primary"], linestyle="--", linewidth=1.5, label=f"Target ({target*100:.0f}%)")
    ax.axhline(y=100, color=COLOR_PALETTE["danger"], linestyle="-", linewidth=1, label="Max Capacity")

    ax.set_ylabel("Utilization (%)", color=CHART_TEXT_COLOR)
    ax.set_xlabel("Period", color=CHART_TEXT_COLOR)
    ax.set_title("Capacity Utilization Timeline", color=CHART_TEXT_COLOR, fontsize=13, fontweight="bold")
    ax.legend()
    ax.tick_params(colors=CHART_TEXT_COLOR)

    if len(periods) > 10:
        ax.set_xticks(ax.get_xticks()[::2])

    plt.tight_layout()
    path = str(Path(output_dir) / "chart_capacity_utilization.png")
    fig.savefig(path, dpi=CHART_DPI, facecolor=CHART_BG_COLOR)
    plt.close(fig)
    logger.info("Saved capacity utilization chart: %s", path)
    return path


def plot_bottleneck_timeline(
    bottlenecks: list[dict[str, Any]],
    output_dir: str,
) -> str:
    """Plot bottleneck periods with demand vs capacity."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=CHART_BG_COLOR)
    ax.set_facecolor(CHART_BG_COLOR)

    if not bottlenecks:
        ax.text(0.5, 0.5, "No bottlenecks detected", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=COLOR_PALETTE["success"])
    else:
        periods = [b["period"] for b in bottlenecks]
        demands = [b["demand"] for b in bottlenecks]
        capacities = [b["capacity"] for b in bottlenecks]

        x = np.arange(len(periods))
        width = 0.35
        ax.bar(x - width/2, demands, width, label="Demand", color=COLOR_PALETTE["danger"], alpha=0.8)
        ax.bar(x + width/2, capacities, width, label="Capacity", color=COLOR_PALETTE["success"], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(periods)
        ax.legend()

    ax.set_ylabel("Units", color=CHART_TEXT_COLOR)
    ax.set_title("Bottleneck Periods — Demand vs Capacity", color=CHART_TEXT_COLOR, fontsize=13, fontweight="bold")
    ax.tick_params(colors=CHART_TEXT_COLOR)

    plt.tight_layout()
    path = str(Path(output_dir) / "chart_capacity_bottleneck.png")
    fig.savefig(path, dpi=CHART_DPI, facecolor=CHART_BG_COLOR)
    plt.close(fig)
    logger.info("Saved bottleneck chart: %s", path)
    return path
