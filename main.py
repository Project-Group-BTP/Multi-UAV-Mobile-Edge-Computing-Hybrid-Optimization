# Temporary script to run the environment with random actions and visualize the state

from environment.env import Env
import config
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os


def plot_snapshot(env: Env, time_step: int, save_path: str | None = None) -> None:
    """Generates and saves a plot of the current environment state."""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, config.AREA_WIDTH)
    ax.set_ylim(0, config.AREA_HEIGHT)
    ax.set_aspect("equal")
    ax.set_title(f"Simulation Snapshot at Time Step: {time_step}")
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
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    # Save the figure
    if save_path:
        plt.savefig(f"{save_path}/step_{time_step:04d}.png")

    plt.close(fig)  # Close the figure to free up memory


def generate_random_actions(env: Env, max_retries: int = 20) -> list[np.ndarray]:
    """
    Generates a valid random action for each UAV, now with collision avoidance.
    It sequentially generates an action for each UAV, ensuring the new position
    respects the minimum separation distance from other UAVs' new positions.
    """
    actions: list[np.ndarray] = []
    new_positions: list[np.ndarray] = []  # Store the intended new (x, y) positions of UAVs
    max_dist: float = config.UAV_SPEED * config.TIME_SLOT_DURATION

    for uav in env.uavs:
        current_pos: np.ndarray = uav.pos[:2]

        for _ in range(max_retries):
            # Step 1: Generate a candidate random move
            angle: float = np.random.uniform(0, 2 * np.pi)
            dist: float = np.random.uniform(0, max_dist)

            candidate_x: float = current_pos[0] + dist * np.cos(angle)
            candidate_y: float = current_pos[1] + dist * np.sin(angle)

            # Ensure the UAV stays within the simulation area boundaries
            candidate_x = np.clip(candidate_x, 0, config.AREA_WIDTH)
            candidate_y = np.clip(candidate_y, 0, config.AREA_HEIGHT)
            candidate_pos: np.ndarray = np.array([candidate_x, candidate_y])

            # Step 2: Check for collision with previously decided moves for other UAVs
            is_valid: bool = True
            for other_new_pos in new_positions:
                if np.linalg.norm(candidate_pos - other_new_pos) < config.MIN_UAV_SEPARATION:
                    is_valid = False
                    break  # Collision detected, try a new random move

            if is_valid:
                # Step 3: If the move is valid, accept it and break the retry loop
                new_positions.append(candidate_pos)
                actions.append(candidate_pos)
                break

        else:  # This 'else' triggers if the 'for' loop completes without a 'break'
            # Step 4: If no valid move was found, the UAV stays in its current position
            new_positions.append(current_pos)
            actions.append(current_pos)

    return actions


def main() -> None:
    # Initialize the environment
    env: Env = Env()

    # Create a directory to save visualization frames
    vis_dir: str = "simulation_frames"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        print(f"Created directory: {vis_dir}")

    print("Starting simulation with random actions (with collision avoidance)...")

    # Run the simulation for a set number of time slots
    for t in range(config.STEPS_PER_EPISODE):
        # 1. Generate random actions for each UAV
        actions_raw: list[np.ndarray] = generate_random_actions(env)
        actions: np.ndarray = np.array(actions_raw)
        # 2. Step the environment forward
        env.step(actions)

        # 3. Visualize and save the current state periodically
        if t % 5 == 0:
            plot_snapshot(env, t, save_path=vis_dir)
            print(f"Saved frame for time step {t}")

    print(f"\nSimulation finished. Visualization frames are saved in the '{vis_dir}' directory.")


if __name__ == "__main__":
    main()
