"""Demand sensing charts."""

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


def plot_signal_timeline(
    signals: list[dict[str, Any]],
    output_dir: str,
) -> str:
    """Plot demand signals over time by source."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=CHART_BG_COLOR)
    ax.set_facecolor(CHART_BG_COLOR)

    source_colors = {
        "pos": COLOR_PALETTE["primary"],
        "social": COLOR_PALETTE["purple"],
        "weather": COLOR_PALETTE["teal"],
    }

    if not signals:
        ax.text(0.5, 0.5, "No signals available", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=COLOR_PALETTE["gray"])
    else:
        for source, color in source_colors.items():
            source_signals = [s for s in signals if s.get("source") == source]
            if source_signals:
                x = range(len(source_signals))
                y = [s["signal_value"] for s in source_signals]
                ax.scatter(x, y, c=color, label=source.upper(), alpha=0.6, s=20)

    ax.set_ylabel("Signal Value", color=CHART_TEXT_COLOR)
    ax.set_xlabel("Signal Index", color=CHART_TEXT_COLOR)
    ax.set_title("Demand Signals by Source", color=CHART_TEXT_COLOR, fontsize=13, fontweight="bold")
    ax.legend()
    ax.tick_params(colors=CHART_TEXT_COLOR)

    plt.tight_layout()
    path = str(Path(output_dir) / "chart_sensing_signals.png")
    fig.savefig(path, dpi=CHART_DPI, facecolor=CHART_BG_COLOR)
    plt.close(fig)
    logger.info("Saved sensing signals chart: %s", path)
    return path


def plot_forecast_adjustment(
    adjustments: list[dict[str, Any]],
    output_dir: str,
) -> str:
    """Plot base vs adjusted forecast comparison."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=CHART_BG_COLOR)
    ax.set_facecolor(CHART_BG_COLOR)

    if not adjustments:
        ax.text(0.5, 0.5, "No adjustments", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=COLOR_PALETTE["gray"])
    else:
        x = range(len(adjustments))
        base = [a.get("forecast", 0) for a in adjustments]
        adjusted = [a.get("adjusted_forecast", 0) for a in adjustments]

        ax.plot(x, base, color=COLOR_PALETTE["gray"], linewidth=2, label="Base Forecast", linestyle="--")
        ax.plot(x, adjusted, color=COLOR_PALETTE["primary"], linewidth=2, label="Adjusted Forecast")
        ax.fill_between(x, base, adjusted, alpha=0.2, color=COLOR_PALETTE["primary"])

    ax.set_ylabel("Demand", color=CHART_TEXT_COLOR)
    ax.set_xlabel("Product", color=CHART_TEXT_COLOR)
    ax.set_title("Forecast Adjustment — Base vs Sensing-Adjusted", color=CHART_TEXT_COLOR, fontsize=13, fontweight="bold")
    ax.legend()
    ax.tick_params(colors=CHART_TEXT_COLOR)

    plt.tight_layout()
    path = str(Path(output_dir) / "chart_sensing_adjustment.png")
    fig.savefig(path, dpi=CHART_DPI, facecolor=CHART_BG_COLOR)
    plt.close(fig)
    logger.info("Saved sensing adjustment chart: %s", path)
    return path
