from marl_models.base_model import MARLModel
from marl_models.buffer import ReplayBuffer
from marl_models.utils import soft_update, GaussianNoise
from marl_models.maddpg.agents import ActorNetwork, CriticNetwork
import config
import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Tuple


class MADDPG(MARLModel):
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(num_agents, obs_dim, action_dim, device)
        self.total_obs_dim: int = num_agents * obs_dim
        self.total_action_dim: int = num_agents * action_dim

        # Create networks for each agent
        self.actors: List[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.critics: List[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]
        self.target_actors: List[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.target_critics: List[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]
        self._init_target_networks()

        # Create optimizers
        self.actor_optimizers: List[torch.optim.Adam] = [torch.optim.Adam(actor.parameters(), lr=config.LEARNING_RATE) for actor in self.actors]
        self.critic_optimizers: List[torch.optim.Adam] = [torch.optim.Adam(critic.parameters(), lr=config.LEARNING_RATE) for critic in self.critics]

        # Replay Buffer
        self.buffer: ReplayBuffer = ReplayBuffer(config.BUFFER_SIZE)

        # Exploration Noise
        self.noise: List[GaussianNoise] = [GaussianNoise() for _ in range(num_agents)]

    def select_actions(self, observations: List[np.ndarray], exploration: bool) -> np.ndarray:
        actions: List[np.ndarray] = []
        with torch.no_grad():
            for i, obs in enumerate(observations):
                obs_tensor: torch.Tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                action: np.ndarray = self.actors[i](obs_tensor).squeeze(0).cpu().numpy()

                if exploration:
                    action += self.noise[i].sample()

                actions.append(np.clip(action, -1.0, 1.0))

        return np.array(actions)

    def add_to_buffer(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray, dones: np.ndarray) -> None:
        self.buffer.add(obs, actions, rewards, next_obs, dones)

    def update(self, batch_size: int) -> None:
        if len(self.buffer) < batch_size:
            return

        batch_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] = self.buffer.sample(batch_size)
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = batch_data

        obs_tensor: torch.Tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        actions_tensor: torch.Tensor = torch.as_tensor(actions_batch, dtype=torch.float32, device=self.device)
        rewards_tensor: torch.Tensor = torch.as_tensor(rewards_batch, dtype=torch.float32, device=self.device)
        next_obs_tensor: torch.Tensor = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)
        dones_tensor: torch.Tensor = torch.as_tensor(dones_batch, dtype=torch.float32, device=self.device)

        obs_flat: torch.Tensor = obs_tensor.reshape(batch_size, -1)
        next_obs_flat: torch.Tensor = next_obs_tensor.reshape(batch_size, -1)
        actions_flat: torch.Tensor = actions_tensor.reshape(batch_size, -1)

        for agent_idx in range(self.num_agents):
            # ----- Update Critic -----
            with torch.no_grad():
                next_actions: List[torch.Tensor] = [self.target_actors[i](next_obs_tensor[:, i, :]) for i in range(self.num_agents)]
                next_actions_tensor: torch.Tensor = torch.cat(next_actions, dim=1)
                target_q_value: torch.Tensor = self.target_critics[agent_idx](next_obs_flat, next_actions_tensor)
                agent_reward: torch.Tensor = rewards_tensor[:, agent_idx].unsqueeze(1)
                agent_done: torch.Tensor = dones_tensor[:, agent_idx].unsqueeze(1)
                y: torch.Tensor = agent_reward + config.DISCOUNT_FACTOR * target_q_value * (1 - agent_done)

            current_q_value: torch.Tensor = self.critics[agent_idx](obs_flat, actions_flat)

            critic_loss: torch.Tensor = F.mse_loss(current_q_value, y)
            self.critic_optimizers[agent_idx].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.critic_optimizers[agent_idx].step()

            # ----- Update Actor -----
            pred_actions_tensor: torch.Tensor = actions_tensor.detach().clone()
            pred_actions_tensor[:, agent_idx, :] = self.actors[agent_idx](obs_tensor[:, agent_idx, :])
            pred_actions_flat: torch.Tensor = pred_actions_tensor.reshape(batch_size, -1)

            actor_loss: torch.Tensor = -self.critics[agent_idx](obs_flat, pred_actions_flat).mean()
            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.actor_optimizers[agent_idx].step()

            # ----- Soft update target networks -----
            soft_update(self.target_actors[agent_idx], self.actors[agent_idx], config.UPDATE_FACTOR)
            soft_update(self.target_critics[agent_idx], self.critics[agent_idx], config.UPDATE_FACTOR)

        for n in self.noise:
            n.decay()

    def _init_target_networks(self) -> None:
        # Copy initial weights
        for actor, target_actor in zip(self.actors, self.target_actors):
            target_actor.load_state_dict(actor.state_dict())
        for critic, target_critic in zip(self.critics, self.target_critics):
            target_critic.load_state_dict(critic.state_dict())

    def save(self, directory: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i in range(self.num_agents):
            torch.save(self.actors[i].state_dict(), os.path.join(directory, f"actor_{i}.pth"))
            torch.save(self.critics[i].state_dict(), os.path.join(directory, f"critic_{i}.pth"))

    def load(self, directory: str) -> None:
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(torch.load(os.path.join(directory, f"actor_{i}.pth")))
            self.critics[i].load_state_dict(torch.load(os.path.join(directory, f"critic_{i}.pth")))
            # Also load target networks for stability
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
