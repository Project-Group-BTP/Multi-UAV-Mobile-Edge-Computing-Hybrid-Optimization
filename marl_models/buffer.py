import random
import numpy as np
from collections import deque
from typing import Deque, List, Tuple


class ReplayBuffer:
    def __init__(self, max_size: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=max_size)

    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray, dones: np.ndarray) -> None:
        """Store one experience tuple: (joint_obs, joint_actions, joint_rewards, joint_next_obs, joint_dones)"""
        self.buffer.append((obs, actions, rewards, next_obs, dones))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of experiences."""
        batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = random.sample(self.buffer, batch_size)
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = map(np.array, zip(*batch))
        return obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch

    def __len__(self) -> int:
        return len(self.buffer)
