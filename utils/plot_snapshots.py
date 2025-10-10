from environment.env import Env
import config
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import numpy as np


def plot_snapshot(env: Env, progress_step: int, step: int, save_dir: str, name: str, timestamp: str, initial: bool = False) -> None:
    """Generates and saves a plot of the current environment state."""
    save_path = f"{save_dir}/state_images_{timestamp}/{name}_{progress_step:04d}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, config.AREA_WIDTH)
    ax.set_ylim(0, config.AREA_HEIGHT)
    ax.set_aspect("equal")
    ax.set_title(f"Simulation Snapshot at {name.title()}: {progress_step}, Step: {step}")
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")

    # Plot UEs as blue dots
    ue_positions: np.ndarray = np.array([ue.pos for ue in env.ues])
    ax.scatter(ue_positions[:, 0], ue_positions[:, 1], c="blue", marker=".", label="UEs")

    # Plot UAVs and their connections
    for uav in env.uavs:
        # UAV position (red square)
        ax.scatter(uav.pos[0], uav.pos[1], c="red", marker="s", s=100, label=f"UAV" if uav.id == 0 else "")

        # UAV coverage radius
        coverage_circle: Circle = Circle((uav.pos[0], uav.pos[1]), config.UAV_COVERAGE_RADIUS, color="red", alpha=0.1)
        ax.add_patch(coverage_circle)

        # Lines to covered UEs (green)
        for ue in uav.current_covered_ues:
            ax.plot([uav.pos[0], ue.pos[0]], [uav.pos[1], ue.pos[1]], "g-", lw=0.5, label="UE Association" if "ue_assoc" not in plt.gca().get_legend_handles_labels()[1] else "")

        # Line to collaborator (dashed magenta)
        if uav.current_collaborator:
            ax.plot([uav.pos[0], uav.current_collaborator.pos[0]], [uav.pos[1], uav.current_collaborator.pos[1]], "m--", lw=1.0, label="UAV Collaboration")

    # Create a clean legend
    handles, labels = ax.get_legend_handles_labels()
    by_label: dict = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    # Save the figure
    if initial:
        plt.savefig(f"{save_path}/initial.png")
    else:
        plt.savefig(f"{save_path}/step_{step:04d}.png")

    plt.close(fig)
