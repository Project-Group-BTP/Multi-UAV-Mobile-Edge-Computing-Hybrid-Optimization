import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, List, Dict, Optional, Tuple
from pathlib import Path


class ComparativePlotter:
    """Generate comparative plots across multiple algorithm runs."""

    def __init__(self, smoothing_window: int = 5, dpi: int = 300):
        """Initialize the plotter with smoothing settings.

        Args:
            smoothing_window: Window size for moving average smoothing
            dpi: DPI for saved plots
        """
        self.smoothing_window = smoothing_window
        self.dpi = dpi
        self.colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ]
        self.linestyles = ["-", "--", "-.", ":"]
        self.data: Dict[str, Dict] = {}

    def load_run(self, log_dir: str, algorithm_name: str) -> bool:
        """Load log data from a training run directory.

        Args:
            log_dir: Path to the log directory containing log_data_*.json
            algorithm_name: Name of the algorithm (for labeling)

        Returns:
            True if load successful, False otherwise
        """
        log_files = list(Path(log_dir).glob("log_data_*.json"))
        if not log_files:
            print(f"❌ No log_data_*.json found in {log_dir}")
            return False

        log_file = log_files[0]
        try:
            with open(log_file, "r") as f:
                log_data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load {log_file}: {e}")
            return False

        if not log_data:
            print(f"❌ Log file is empty: {log_file}")
            return False

        self.data[algorithm_name] = self._process_data(log_data)
        print(f"✅ Loaded {algorithm_name} from {log_file}")
        return True

    def _process_data(self, log_data: List[Dict]) -> Dict[str, Any]:
        """Process raw log data and extract metrics."""
        processed: Dict[str, Any] = {}

        # Determine x-axis key
        x_key = "update" if "update" in log_data[0] else "episode"
        x_label = "Update" if x_key == "update" else "Episode"

        # Extract all metrics
        metrics = [
            "reward",
            "latency",
            "energy",
            "fairness",
            "offline_rate",
            "actor_loss",
            "critic_loss",
            "entropy_loss",
            "alpha_loss",
        ]

        processed[x_key] = [entry.get(x_key) for entry in log_data]
        processed["x_label"] = x_label

        for metric in metrics:
            values = [entry.get(metric) for entry in log_data]
            # Filter out None values but keep track of valid indices
            processed[metric] = values

        return processed

    def _smooth_data(self, y_data: List[float]) -> Tuple[List[float], List[int]]:
        """Apply moving average smoothing to data.

        Returns:
            Tuple of (smoothed_data, valid_indices)
        """
        valid_indices = [i for i, v in enumerate(y_data) if v is not None]
        if not valid_indices:
            return [], []

        valid_data = [y_data[i] for i in valid_indices]

        if len(valid_data) > self.smoothing_window:
            smoothed = np.convolve(
                valid_data,
                np.ones(self.smoothing_window) / self.smoothing_window,
                mode="valid",
            )
            return smoothed.tolist(), valid_indices[: len(smoothed)]
        else:
            return valid_data, valid_indices

    def plot_comparison(self, metric: str, output_path: str, ylabel: Optional[str] = None) -> None:
        """Plot a single metric comparison across all loaded algorithms.

        Args:
            metric: Metric name to plot (e.g., 'reward', 'actor_loss')
            output_path: Path where to save the plot
            ylabel: Optional custom y-axis label
        """
        plt.figure(figsize=(14, 7))

        if not self.data:
            print("❌ No data loaded. Call load_run() first.")
            return

        has_data = False
        for idx, (algo_name, data) in enumerate(self.data.items()):
            if metric not in data:
                continue

            y_data = data[metric]
            if not any(v is not None for v in y_data):
                continue

            has_data = True
            x_key = "update" if "update" in data else "episode"
            x_data = data[x_key]

            # Smooth the data
            y_smooth, valid_indices = self._smooth_data(y_data)
            if not valid_indices:
                continue

            x_filtered = [x_data[i] for i in valid_indices[: len(y_smooth)]]

            # Plot smoothed and raw data
            color = self.colors[idx % len(self.colors)]
            linestyle = self.linestyles[idx % len(self.linestyles)]

            plt.plot(
                x_filtered,
                y_smooth,
                linewidth=2.5,
                label=algo_name,
                color=color,
                linestyle=linestyle,
            )

            # Plot raw data as light background
            x_raw = [x_data[i] for i in valid_indices]
            y_raw = [y_data[i] for i in valid_indices]
            plt.scatter(x_raw, y_raw, alpha=0.1, s=20, color=color)

        if not has_data:
            print(f"⚠️  No valid data for metric: {metric}")
            plt.close()
            return

        x_label = list(self.data.values())[0]["x_label"]
        y_label = ylabel or metric.replace("_", " ").title()
        title = f"{y_label} Comparison"

        plt.xlabel(x_label, fontsize=12, fontweight="bold")
        plt.ylabel(y_label, fontsize=12, fontweight="bold")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(fontsize=11, loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
        print(f"✅ Saved {output_path}")

    def plot_all_comparisons(self, output_dir: str) -> None:
        """Generate all comparison plots.

        Args:
            output_dir: Directory to save all plots
        """
        os.makedirs(output_dir, exist_ok=True)

        # Define which metrics to plot
        env_metrics = ["reward", "latency", "energy", "fairness", "offline_rate"]
        loss_metrics = ["actor_loss", "critic_loss", "entropy_loss", "alpha_loss"]

        all_metrics = env_metrics + loss_metrics

        for metric in all_metrics:
            # Check if any algorithm has this metric
            if any(metric in algo_data for algo_data in self.data.values()):
                output_path = os.path.join(output_dir, f"comparison_{metric}.png")
                self.plot_comparison(metric, output_path)

        # Create a summary figure with subplots
        self._plot_summary(output_dir)

    def _plot_summary(self, output_dir: str) -> None:
        """Create a 2x4 summary figure with key metrics."""
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        axes = axes.flatten()

        summary_metrics = [
            "reward",
            "latency",
            "energy",
            "fairness",
            "offline_rate",
            "actor_loss",
            "critic_loss",
            "alpha_loss",
        ]

        for ax_idx, metric in enumerate(summary_metrics):
            ax = axes[ax_idx]

            has_data = False
            for idx, (algo_name, data) in enumerate(self.data.items()):
                if metric not in data:
                    continue

                y_data = data[metric]
                if not any(v is not None for v in y_data):
                    continue

                has_data = True
                x_key = "update" if "update" in data else "episode"
                x_data = data[x_key]

                # Smooth the data
                y_smooth, valid_indices = self._smooth_data(y_data)
                if not valid_indices:
                    continue

                x_filtered = [x_data[i] for i in valid_indices[: len(y_smooth)]]

                color = self.colors[idx % len(self.colors)]
                linestyle = self.linestyles[idx % len(self.linestyles)]

                ax.plot(
                    x_filtered,
                    y_smooth,
                    linewidth=2,
                    label=algo_name,
                    color=color,
                    linestyle=linestyle,
                )

            if has_data:
                x_label = list(self.data.values())[0]["x_label"]
                y_label = metric.replace("_", " ").title()
                ax.set_xlabel(x_label, fontsize=10, fontweight="bold")
                ax.set_ylabel(y_label, fontsize=10, fontweight="bold")
                ax.set_title(y_label, fontsize=11, fontweight="bold")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for\n{metric}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        output_path = os.path.join(output_dir, "comparison_summary.png")
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
        print(f"✅ Saved {output_path}")


def compare_algorithms(log_dirs: List[str], algorithm_names: List[str], output_dir: str, smoothing_window: int = 5) -> None:
    """Convenience function to compare multiple algorithm runs.

    Args:
        log_dirs: List of directories containing log files
        algorithm_names: List of algorithm names (must match log_dirs length)
        output_dir: Directory to save comparison plots
        smoothing_window: Window size for moving average smoothing
    """
    if len(log_dirs) != len(algorithm_names):
        print("❌ Number of log directories must match number of algorithm names")
        return

    plotter = ComparativePlotter(smoothing_window=smoothing_window)

    for log_dir, algo_name in zip(log_dirs, algorithm_names):
        if not os.path.isdir(log_dir):
            print(f"❌ Directory not found: {log_dir}")
            continue

        plotter.load_run(log_dir, algo_name)

    if plotter.data:
        plotter.plot_all_comparisons(output_dir)
        print(f"\n✅ All comparative plots saved to {output_dir}")
    else:
        print("❌ No data loaded successfully")
