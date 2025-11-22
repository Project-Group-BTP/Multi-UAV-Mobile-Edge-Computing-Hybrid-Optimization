from marl_models.base_model import MARLModel
from environment.env import Env
from utils.logger import Logger, Log
from utils.plot_snapshots import plot_snapshot

# from utils.plot_snapshots import update_trajectories, reset_trajectories  # trajectory tracking, comment if not needed
import config
import numpy as np
import time


def test_model(env: Env, model: MARLModel, logger: Logger, num_episodes: int) -> None:
    start_time: float = time.time()
    episode_log: Log = Log()

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        model.reset()
        episode_reward: float = 0.0
        episode_latency: float = 0.0
        episode_energy: float = 0.0
        episode_fairness: float = 0.0
        # reset_trajectories(env)  # tracking code, comment if not needed
        plot_snapshot(env, episode, 0, logger.log_dir, "episode", logger.timestamp, True)

        for step in range(1, config.STEPS_PER_EPISODE + 1):
            if step % config.TEST_IMG_FREQ == 0:
                plot_snapshot(env, episode, step, logger.log_dir, "episode", logger.timestamp)

            actions: np.ndarray = model.select_actions(obs, exploration=False)
            next_obs, rewards, (total_latency, total_energy, jfi) = env.step(actions)
            # update_trajectories(env)  # tracking code, comment if not needed
            done: bool = step >= config.STEPS_PER_EPISODE
            obs = next_obs

            episode_reward += np.sum(rewards)
            episode_latency += total_latency
            episode_energy += total_energy
            episode_fairness = jfi
            if done:
                break

        episode_log.append(episode_reward, episode_latency, episode_energy, episode_fairness)
        if episode % config.TEST_LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            logger.log_metrics(episode, episode_log, config.TEST_LOG_FREQ, elapsed_time, "episode")
