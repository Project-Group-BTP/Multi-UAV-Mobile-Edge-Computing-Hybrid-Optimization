from abc import ABC, abstractmethod
from typing import List
import numpy as np


class MARLModel(ABC):
    """
    Abstract Base Class for Multi-Agent Reinforcement Learning models.
    This class defines the essential methods that any MARL algorithm implementation
    must have to be compatible with the training framework.
    """

    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, device: str):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

    @abstractmethod
    def select_actions(self, observations: List[np.ndarray], exploration: bool) -> np.ndarray:
        """
        Selects actions for all agents based on their observations.

        Args:
            observations (List[np.ndarray]): A list of observations, one for each agent.
            exploration (bool): Whether to apply exploration noise/strategy.

        Returns:
            np.ndarray: A numpy array of shape (num_agents, action_dim).
        """
        pass

    @abstractmethod
    def update(self, batch_size: int):
        """
        Performs a learning update on the model's networks using a batch of experiences.

        Args:
            batch_size (int): The number of experiences to sample from the replay buffer.
        """
        pass

    @abstractmethod
    def save(self, directory: str):
        pass

    @abstractmethod
    def load(self, directory: str):
        pass

    @abstractmethod
    def add_to_buffer(self, obs, actions, rewards, next_obs, dones):
        pass
