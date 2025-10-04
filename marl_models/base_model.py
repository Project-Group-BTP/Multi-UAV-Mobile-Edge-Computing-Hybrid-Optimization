from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union
import numpy as np
import torch

OffPolicyExperienceBatch = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
OnPolicyExperienceBatch = Dict[str, torch.Tensor]
ExperienceBatch = Union[OffPolicyExperienceBatch, OnPolicyExperienceBatch]


class MARLModel(ABC):
    """
    Abstract Base Class for Multi-Agent Reinforcement Learning models.
    This class defines the essential methods that any MARL algorithm implementation
    must have to be compatible with the training framework.
    """

    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str):
        self.model_name = model_name
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

    @abstractmethod
    def select_actions(self, observations: List[np.ndarray], exploration: bool) -> List[np.ndarray]:
        """
        Selects actions for all agents based on their observations.
        """
        pass

    @abstractmethod
    def update(self, batch: ExperienceBatch) -> None:
        """
        Performs a learning update on the model's networks using a batch of experiences.

        Args:
            batch (ExperienceBatch): A dictionary (for on-policy) or a tuple (for off-policy).

        Returns:
            dict: A dictionary containing loss information for logging.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the model's internal state (if any) for a new episode.
        """
        pass

    @abstractmethod
    def save(self, directory: str):
        pass

    @abstractmethod
    def load(self, directory: str):
        pass
