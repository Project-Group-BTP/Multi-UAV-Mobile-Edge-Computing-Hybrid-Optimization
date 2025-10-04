import torch
import numpy as np
from typing import Dict, Generator, Union


class RolloutBuffer:  # It stores transitions, computes advantages, and provides mini-batches.
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, state_dim: int, buffer_size: int, device: str) -> None:
        self.num_agents: int = num_agents
        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.state_dim: int = state_dim
        self.buffer_size: int = buffer_size
        self.device: str = device

        # Initialize storage
        self.states: np.ndarray = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.observations: np.ndarray = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.actions: np.ndarray = np.zeros((buffer_size, num_agents, action_dim), dtype=np.float32)
        self.log_probs: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.rewards: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.dones: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.values: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)

        # For GAE calculation
        self.advantages: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.returns: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)

        self.step: int = 0

    def add(self, state: np.ndarray, obs: np.ndarray, actions: np.ndarray, log_probs: np.ndarray, rewards: np.ndarray, dones: np.ndarray, values: np.ndarray) -> None:
        if self.step >= self.buffer_size:
            raise ValueError("Rollout buffer overflow")

        self.states[self.step] = state
        self.observations[self.step] = obs
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values

        self.step += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        """
        Computes the advantages and returns for the collected trajectories using GAE.
        This should be called at the end of a rollout.
        """
        last_gae_lam: float = 0.0
        for t in reversed(range(self.buffer_size)):
            is_terminal: float = self.dones[t, 0]

            if t == self.buffer_size - 1:
                next_values: np.ndarray = last_values
            else:
                next_values = self.values[t + 1]

            delta: np.ndarray = self.rewards[t] + gamma * next_values * (1.0 - self.dones[t]) - self.values[t]
            self.advantages[t] = last_gae_lam = delta + gamma * gae_lambda * (1.0 - self.dones[t]) * last_gae_lam

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        A generator that yields mini-batches from the buffer.
        """
        num_samples: int = self.buffer_size * self.num_agents

        states: np.ndarray = np.repeat(self.states, self.num_agents, axis=0)
        obs: np.ndarray = self.observations.reshape(-1, self.obs_dim)
        actions: np.ndarray = self.actions.reshape(-1, self.action_dim)  # Reshape to (N, action_dim)
        log_probs: np.ndarray = self.log_probs.reshape(-1)
        advantages: np.ndarray = self.advantages.reshape(-1)
        returns: np.ndarray = self.returns.reshape(-1)
        values: np.ndarray = self.values.reshape(-1)

        indices: np.ndarray = np.random.permutation(num_samples)

        for start in range(0, num_samples, batch_size):
            end: int = start + batch_size
            batch_indices: np.ndarray = indices[start:end]

            yield {
                "states": torch.as_tensor(states[batch_indices], device=self.device),
                "obs": torch.as_tensor(obs[batch_indices], device=self.device),
                "actions": torch.as_tensor(actions[batch_indices], device=self.device),
                "old_log_probs": torch.as_tensor(log_probs[batch_indices], device=self.device),
                "advantages": torch.as_tensor(advantages[batch_indices], device=self.device),
                "returns": torch.as_tensor(returns[batch_indices], device=self.device),
                "old_values": torch.as_tensor(values[batch_indices], device=self.device),
            }

    def clear(self) -> None:
        self.step = 0
