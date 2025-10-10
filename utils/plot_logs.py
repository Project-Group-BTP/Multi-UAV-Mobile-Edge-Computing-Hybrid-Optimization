import os
import json
import matplotlib.pyplot as plt


def plot_metric(x: list, y: list, xlabel: str, ylabel: str, title: str, output_path: str) -> None:
    """Helper function to plot a single metric as a scatter plot and save it separately"""
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_plots(log_file: str, output_dir: str, output_file_prefix: str, timestamp: str) -> None:
    """Generate plots from the logs stored in 'log_file'"""

    with open(log_file, "r") as file:
        log_data: list[dict] = json.load(file)
    os.makedirs(output_dir, exist_ok=True)

    if "update" in log_data[0]:
        x_axis_key: str = "update"
        x_label: str = "Update"
    elif "episode" in log_data[0]:
        x_axis_key = "episode"
        x_label = "Episode"
    else:
        print("❌ Log file does not contain 'episode' or 'update' keys.")
        return
    parameters: dict = {x_axis_key: [entry[x_axis_key] for entry in log_data], "reward": [entry["reward"] for entry in log_data], "latency": [entry["latency"] for entry in log_data], "energy": [entry["energy"] for entry in log_data], "fairness": [entry["fairness"] for entry in log_data]}
    metrics: list[str] = ["reward", "latency", "energy", "fairness"]
    for metric in metrics:
        title: str = f"{metric.replace('_', ' ').title()} vs {x_label}"
        output_path: str = os.path.join(output_dir, f"{output_file_prefix}_{metric}_{timestamp}.png")
        plot_metric(parameters[x_axis_key], parameters[metric], x_label, metric.title(), title, output_path)
    plot_metric(parameters["latency"], parameters["fairness"], "Latency", "Fairness", "Fairness vs Latency", os.path.join(output_dir, f"{output_file_prefix}_fairness_vs_latency_{timestamp}.png"))
    plot_metric(parameters["energy"], parameters["fairness"], "Energy", "Fairness", "Fairness vs Energy", os.path.join(output_dir, f"{output_file_prefix}_fairness_vs_energy_{timestamp}.png"))
    plot_metric(parameters["latency"], parameters["energy"], "Latency", "Energy", "Energy vs Latency", os.path.join(output_dir, f"{output_file_prefix}_energy_vs_latency_{timestamp}.png"))
    print(f"✅ All scatter plots saved to {output_dir}\n")
