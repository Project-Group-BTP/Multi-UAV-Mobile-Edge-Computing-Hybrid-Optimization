import numpy as np
from typing import List
from marl_models.base_model import MARLModel, ExperienceBatch


class RandomModel(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)

    def select_actions(self, observations: List[np.ndarray], exploration: bool = True) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, (self.num_agents, self.action_dim))

    def update(self, batch: ExperienceBatch) -> None:
        pass

    def reset(self) -> None:
        pass

    def save(self, directory: str) -> None:
        pass

    def load(self, directory: str) -> None:
        pass
