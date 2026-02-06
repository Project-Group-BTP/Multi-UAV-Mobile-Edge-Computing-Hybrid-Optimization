import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_metric(
    x: list,
    y: list,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
    smoothing_window: int = 5,
) -> None:
    """Plot a single metric with optional smoothing and save it."""
    plt.figure(figsize=(12, 6))

    # Apply smoothing using moving average if window > 1
    if len(y) > smoothing_window and smoothing_window > 1:
        y_smooth = np.convolve(y, np.ones(smoothing_window) / smoothing_window, mode="valid")
        x_smooth = x[: len(y_smooth)]
        plt.plot(x_smooth, y_smooth, linewidth=2, label="Smoothed", color="#1f77b4")
        plt.plot(x, y, alpha=0.3, linestyle="--", label="Raw", color="#1f77b4")
    else:
        plt.plot(x, y, linewidth=2, color="#1f77b4")

    plt.xlabel(xlabel, fontsize=12, fontweight="bold")
    plt.ylabel(ylabel, fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend() if len(y) > smoothing_window and smoothing_window > 1 else None
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_plots(log_file: str, output_dir: str, output_file_prefix: str, timestamp: str, smoothing_window: int = 5) -> None:
    """Generate all required plots from JSON log file."""

    with open(log_file, "r") as file:
        log_data: list[dict] = json.load(file)

    os.makedirs(output_dir, exist_ok=True)

    if not log_data:
        print(f"❌ Log file is empty: {log_file}")
        return

    # Determine x-axis (episode or update)
    if "update" in log_data[0]:
        x_axis_key = "update"
        x_label = "Update"
    elif "episode" in log_data[0]:
        x_axis_key = "episode"
        x_label = "Episode"
    else:
        print("❌ Log file does not contain 'episode' or 'update' keys.")
        return

    # Extract data
    parameters: dict = {
        x_axis_key: [entry[x_axis_key] for entry in log_data],
        "reward": [entry.get("reward") for entry in log_data],
        "latency": [entry.get("latency") for entry in log_data],
        "energy": [entry.get("energy") for entry in log_data],
        "fairness": [entry.get("fairness") for entry in log_data],
        "offline_rate": [entry.get("offline_rate") for entry in log_data],
        "actor_loss": [entry.get("actor_loss") for entry in log_data],
        "critic_loss": [entry.get("critic_loss") for entry in log_data],
        "entropy_loss": [entry.get("entropy_loss") for entry in log_data],
        "alpha_loss": [entry.get("alpha_loss") for entry in log_data],
    }

    x_data = parameters[x_axis_key]

    # Plot environment metrics
    metrics_to_plot = ["reward", "latency", "energy", "fairness", "offline_rate"]
    for metric in metrics_to_plot:
        if any(v is not None for v in parameters[metric]):
            y_data = [v if v is not None else np.nan for v in parameters[metric]]
            # Remove NaN values for cleaner plots
            valid_indices = [i for i, v in enumerate(y_data) if not np.isnan(v)]
            x_filtered = [x_data[i] for i in valid_indices]
            y_filtered = [y_data[i] for i in valid_indices]

            title = f"{metric.replace('_', ' ').title()} vs {x_label}"
            output_path = os.path.join(output_dir, f"{output_file_prefix}_{metric}_{timestamp}.png")
            plot_metric(x_filtered, y_filtered, x_label, metric.title(), title, output_path, smoothing_window)

    # Plot loss curves (if available)
    loss_metrics = ["actor_loss", "critic_loss", "entropy_loss", "alpha_loss"]
    for metric in loss_metrics:
        if any(v is not None for v in parameters[metric]):
            y_data = [v if v is not None else np.nan for v in parameters[metric]]
            valid_indices = [i for i, v in enumerate(y_data) if not np.isnan(v)]
            x_filtered = [x_data[i] for i in valid_indices]
            y_filtered = [y_data[i] for i in valid_indices]

            title = f"{metric.replace('_', ' ').title()} vs {x_label}"
            output_path = os.path.join(output_dir, f"{output_file_prefix}_{metric}_{timestamp}.png")
            plot_metric(x_filtered, y_filtered, x_label, metric.title(), title, output_path, smoothing_window)

    print(f"✅ All plots saved to {output_dir}\n")
