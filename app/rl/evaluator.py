"""RL Evaluator — generates comparison charts and KPI summaries."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.config import CHART_DPI, CHART_BG_COLOR, CHART_TEXT_COLOR, COLOR_PALETTE

logger = logging.getLogger(__name__)

# Color cycle for agents
AGENT_COLORS = [
    COLOR_PALETTE["primary"],    # Q-Learning
    COLOR_PALETTE["success"],    # SARSA
    COLOR_PALETTE["danger"],     # DQN
    COLOR_PALETTE["warning"],    # PPO
    COLOR_PALETTE["purple"],     # A2C
    COLOR_PALETTE["teal"],       # GA-RL Hybrid
]


def _style_ax(ax, title: str, xlabel: str = "", ylabel: str = ""):
    """Apply consistent ChainInsight styling."""
    ax.set_facecolor(CHART_BG_COLOR)
    ax.set_title(title, fontsize=13, fontweight="bold", color=CHART_TEXT_COLOR, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=CHART_TEXT_COLOR)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=CHART_TEXT_COLOR)
    ax.tick_params(colors=CHART_TEXT_COLOR, labelsize=9)
    ax.grid(True, alpha=0.3)


class RLEvaluator:
    """Generates RL comparison charts from trainer results."""

    def __init__(self, comparison_data: dict, output_dir: str | Path):
        self.data = comparison_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chart_paths: list[str] = []

    def generate_all_charts(self) -> list[str]:
        """Generate all RL evaluation charts. Returns list of file paths."""
        self.chart_paths = []
        self._plot_reward_curves()
        self._plot_service_levels()
        self._plot_agent_comparison_bar()
        self._plot_convergence_analysis()
        self._plot_reward_distribution()
        self._plot_summary_table()
        logger.info("Generated %d RL charts in %s", len(self.chart_paths), self.output_dir)
        return self.chart_paths

    def _save(self, fig, name: str):
        path = self.output_dir / name
        fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", facecolor=CHART_BG_COLOR)
        plt.close(fig)
        self.chart_paths.append(str(path))

    def _plot_reward_curves(self):
        """Chart 23: Reward learning curves for all agents."""
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.set_facecolor(CHART_BG_COLOR)

        for i, (name, metrics) in enumerate(self.data.items()):
            rewards = metrics["rewards_history"]
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            # Smoothed curve
            window = min(20, len(rewards) // 5) if len(rewards) > 20 else 1
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
                ax.plot(smoothed, label=f"{name} (smoothed)", color=color, linewidth=2)
                ax.plot(rewards, color=color, alpha=0.15, linewidth=0.5)
            else:
                ax.plot(rewards, label=name, color=color, linewidth=2)

        _style_ax(ax, "RL Agent Reward Learning Curves", "Episode", "Total Reward")
        ax.legend(fontsize=9, loc="lower right")
        self._save(fig, "chart_23_rl_reward_curves.png")

    def _plot_service_levels(self):
        """Chart 24: Service level (fill rate) over training."""
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.set_facecolor(CHART_BG_COLOR)

        for i, (name, metrics) in enumerate(self.data.items()):
            svc = metrics["service_level_history"]
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            window = min(20, len(svc) // 5) if len(svc) > 20 else 1
            if window > 1:
                smoothed = np.convolve(svc, np.ones(window) / window, mode="valid")
                ax.plot(smoothed, label=name, color=color, linewidth=2)
            else:
                ax.plot(svc, label=name, color=color, linewidth=2)

        ax.axhline(y=0.95, color=COLOR_PALETTE["danger"], linestyle="--",
                    alpha=0.7, label="95% Target")
        _style_ax(ax, "Service Level (Fill Rate) During Training", "Episode", "Service Level")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="lower right")
        self._save(fig, "chart_24_rl_service_levels.png")

    def _plot_agent_comparison_bar(self):
        """Chart 25: Bar chart comparing final performance metrics."""
        agents = list(self.data.keys())
        n = len(agents)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.set_facecolor(CHART_BG_COLOR)

        colors = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(n)]

        # Final reward
        vals = [self.data[a]["final_reward"] for a in agents]
        axes[0].barh(agents, vals, color=colors, edgecolor="white", linewidth=0.5)
        _style_ax(axes[0], "Final Avg Reward (last 20 ep)", "Reward")
        for j, v in enumerate(vals):
            axes[0].text(v, j, f" {v:.0f}", va="center", fontsize=9, color=CHART_TEXT_COLOR)

        # Service level
        vals = [self.data[a]["final_service_level"] * 100 for a in agents]
        axes[1].barh(agents, vals, color=colors, edgecolor="white", linewidth=0.5)
        _style_ax(axes[1], "Final Service Level (%)", "Service Level %")
        axes[1].axvline(x=95, color=COLOR_PALETTE["danger"], linestyle="--", alpha=0.7)
        for j, v in enumerate(vals):
            axes[1].text(v, j, f" {v:.1f}%", va="center", fontsize=9, color=CHART_TEXT_COLOR)

        # Convergence
        vals = [self.data[a]["convergence_episode"] for a in agents]
        axes[2].barh(agents, vals, color=colors, edgecolor="white", linewidth=0.5)
        _style_ax(axes[2], "Convergence Episode", "Episode")
        for j, v in enumerate(vals):
            axes[2].text(v, j, f" {v}", va="center", fontsize=9, color=CHART_TEXT_COLOR)

        fig.suptitle("RL Agent Performance Comparison", fontsize=15,
                     fontweight="bold", color=CHART_TEXT_COLOR, y=1.02)
        fig.tight_layout()
        self._save(fig, "chart_25_rl_comparison.png")

    def _plot_convergence_analysis(self):
        """Chart 26: Rolling reward variance to show convergence stability."""
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.set_facecolor(CHART_BG_COLOR)

        window = 20
        for i, (name, metrics) in enumerate(self.data.items()):
            rewards = metrics["rewards_history"]
            if len(rewards) < window:
                continue
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            rolling_std = [
                np.std(rewards[max(0, j - window):j])
                for j in range(window, len(rewards))
            ]
            ax.plot(range(window, len(rewards)), rolling_std,
                    label=name, color=color, linewidth=2)

        _style_ax(ax, "Reward Stability (Rolling Std Dev, window=20)",
                  "Episode", "Reward Std Dev")
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=9)
        self._save(fig, "chart_26_rl_convergence.png")

    def _plot_reward_distribution(self):
        """Chart 27: Violin/box plot of reward distributions."""
        agents = list(self.data.keys())
        all_rewards = [self.data[a]["rewards_history"] for a in agents]

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.set_facecolor(CHART_BG_COLOR)

        bp = ax.boxplot(all_rewards, labels=agents, patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "white", "markersize": 6})
        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(AGENT_COLORS[j % len(AGENT_COLORS)])
            patch.set_alpha(0.7)

        _style_ax(ax, "Reward Distribution Across Training", ylabel="Total Reward")
        ax.tick_params(axis="x", rotation=15)
        self._save(fig, "chart_27_rl_reward_distribution.png")

    def _plot_summary_table(self):
        """Chart 28: Summary table of all agents."""
        agents = list(self.data.keys())
        columns = ["Agent", "Mean Reward", "Best Reward", "Reward Std",
                    "Service Level", "Convergence Ep"]

        cell_data = []
        for name in agents:
            m = self.data[name]
            cell_data.append([
                name,
                f"{m['mean_reward']:.1f}",
                f"{m['best_reward']:.1f}",
                f"{m['reward_std']:.1f}",
                f"{m['mean_service_level']*100:.1f}%",
                str(m["convergence_episode"]),
            ])

        fig, ax = plt.subplots(figsize=(14, max(3, len(agents) * 0.6 + 2)))
        fig.set_facecolor(CHART_BG_COLOR)
        ax.set_facecolor(CHART_BG_COLOR)
        ax.axis("off")

        table = ax.table(
            cellText=cell_data,
            colLabels=columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.6)

        # Style header
        for j in range(len(columns)):
            cell = table[0, j]
            cell.set_facecolor(COLOR_PALETTE["primary"])
            cell.set_text_props(color="white", fontweight="bold")

        # Color rows
        for i in range(len(agents)):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            for j in range(len(columns)):
                cell = table[i + 1, j]
                cell.set_facecolor(color + "20")  # 12% opacity

        ax.set_title("RL Agent Summary Comparison", fontsize=14,
                     fontweight="bold", color=CHART_TEXT_COLOR, pad=20)
        self._save(fig, "chart_28_rl_summary.png")

    def get_kpis(self) -> dict:
        """Return KPI dict for storage in DB."""
        kpis = {}
        for name, m in self.data.items():
            key = name.lower().replace("-", "_").replace(" ", "_")
            kpis[f"rl_{key}_mean_reward"] = m["mean_reward"]
            kpis[f"rl_{key}_best_reward"] = m["best_reward"]
            kpis[f"rl_{key}_service_level"] = m["mean_service_level"]
            kpis[f"rl_{key}_convergence"] = m["convergence_episode"]

        # Best agent
        best = max(self.data.items(), key=lambda x: x[1]["final_reward"])
        kpis["rl_best_agent"] = best[0]
        kpis["rl_best_reward"] = best[1]["final_reward"]
        kpis["rl_best_service_level"] = best[1]["final_service_level"]

        return kpis
