from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.utils import soft_update, GaussianNoise
from marl_models.matd3.agents import ActorNetwork, CriticNetwork
import config
import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Tuple


class MATD3(MARLModel):
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(num_agents, obs_dim, action_dim, device)
        self.total_obs_dim: int = num_agents * obs_dim
        self.total_action_dim: int = num_agents * action_dim

        # Create networks for each agent
        self.actors: List[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.critics_1: List[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]
        self.critics_2: List[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]
        self.target_actors: List[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.target_critics_1: List[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]
        self.target_critics_2: List[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]

        self._init_target_networks()

        # Create optimizers
        self.actor_optimizers: List[torch.optim.Adam] = [torch.optim.Adam(actor.parameters(), lr=config.LEARNING_RATE) for actor in self.actors]
        self.critic_1_optimizers: List[torch.optim.Adam] = [torch.optim.Adam(critic.parameters(), lr=config.LEARNING_RATE) for critic in self.critics_1]
        self.critic_2_optimizers: List[torch.optim.Adam] = [torch.optim.Adam(critic.parameters(), lr=config.LEARNING_RATE) for critic in self.critics_2]

        # Exploration Noise
        self.noise: List[GaussianNoise] = [GaussianNoise() for _ in range(num_agents)]

        # Delayed Updates Counter
        self.update_counter: int = 0

    def select_actions(self, observations: List[np.ndarray], exploration: bool) -> np.ndarray:
        """Selects actions for all agents based on their observations (decentralized execution)."""
        actions: List[np.ndarray] = []
        with torch.no_grad():
            for i, obs in enumerate(observations):
                obs_tensor: torch.Tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                action: np.ndarray = self.actors[i](obs_tensor).squeeze(0).cpu().numpy()

                if exploration:
                    action += self.noise[i].sample()

                actions.append(np.clip(action, -1.0, 1.0))

        return np.array(actions)

    def update(self, batch: ExperienceBatch) -> None:
        assert isinstance(batch, tuple) and len(batch) == 5, "MATD3 expects OffPolicyExperienceBatch (tuple of 5 elements)"
        self.update_counter += 1
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = batch
        obs_tensor: torch.Tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        actions_tensor: torch.Tensor = torch.as_tensor(actions_batch, dtype=torch.float32, device=self.device)
        rewards_tensor: torch.Tensor = torch.as_tensor(rewards_batch, dtype=torch.float32, device=self.device)
        next_obs_tensor: torch.Tensor = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)
        dones_tensor: torch.Tensor = torch.as_tensor(dones_batch, dtype=torch.float32, device=self.device)

        batch_size: int = obs_tensor.shape[0]
        obs_flat: torch.Tensor = obs_tensor.reshape(batch_size, -1)
        next_obs_flat: torch.Tensor = next_obs_tensor.reshape(batch_size, -1)
        actions_flat: torch.Tensor = actions_tensor.reshape(batch_size, -1)

        for agent_idx in range(self.num_agents):
            # ----- Update Critic -----
            with torch.no_grad():
                # Get next actions from target actors and add clipped noise
                next_actions: List[torch.Tensor] = []
                for i in range(self.num_agents):
                    next_action_i: torch.Tensor = self.target_actors[i](next_obs_tensor[:, i, :])
                    noise: torch.Tensor = torch.randn_like(next_action_i) * config.TARGET_POLICY_NOISE
                    clipped_noise: torch.Tensor = torch.clamp(noise, -config.NOISE_CLIP, config.NOISE_CLIP)
                    next_actions.append(torch.clamp(next_action_i + clipped_noise, -1.0, 1.0))

                next_actions_tensor: torch.Tensor = torch.cat(next_actions, dim=1)

                # Compute target Q-value using the minimum of the two target critics
                target_q1: torch.Tensor = self.target_critics_1[agent_idx](next_obs_flat, next_actions_tensor)
                target_q2: torch.Tensor = self.target_critics_2[agent_idx](next_obs_flat, next_actions_tensor)
                target_q_min: torch.Tensor = torch.min(target_q1, target_q2)

                agent_reward: torch.Tensor = rewards_tensor[:, agent_idx].unsqueeze(1)
                agent_done: torch.Tensor = dones_tensor[:, agent_idx].unsqueeze(1)
                y: torch.Tensor = agent_reward + config.DISCOUNT_FACTOR * target_q_min * (1 - agent_done)

            # Update both critic networks
            current_q1: torch.Tensor = self.critics_1[agent_idx](obs_flat, actions_flat)
            critic_loss_1: torch.Tensor = F.mse_loss(current_q1, y)
            self.critic_1_optimizers[agent_idx].zero_grad()
            critic_loss_1.backward()
            torch.nn.utils.clip_grad_norm_(self.critics_1[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.critic_1_optimizers[agent_idx].step()

            current_q2: torch.Tensor = self.critics_2[agent_idx](obs_flat, actions_flat)
            critic_loss_2: torch.Tensor = F.mse_loss(current_q2, y)
            self.critic_2_optimizers[agent_idx].zero_grad()
            critic_loss_2.backward()
            torch.nn.utils.clip_grad_norm_(self.critics_2[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.critic_2_optimizers[agent_idx].step()

        # Delayed Policy and Target Network Updates
        if self.update_counter % config.POLICY_UPDATE_FREQ == 0:
            for agent_idx in range(self.num_agents):
                # ----- Update Actor -----
                # The actor loss is calculated using only the first critic
                pred_actions_tensor: torch.Tensor = actions_tensor.detach().clone()
                pred_actions_tensor[:, agent_idx, :] = self.actors[agent_idx](obs_tensor[:, agent_idx, :])
                pred_actions_flat: torch.Tensor = pred_actions_tensor.reshape(batch_size, -1)

                actor_loss: torch.Tensor = -self.critics_1[agent_idx](obs_flat, pred_actions_flat).mean()
                self.actor_optimizers[agent_idx].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), config.MAX_GRAD_NORM)
                self.actor_optimizers[agent_idx].step()

                # ----- Soft update all target networks -----
                soft_update(self.target_actors[agent_idx], self.actors[agent_idx], config.UPDATE_FACTOR)
                soft_update(self.target_critics_1[agent_idx], self.critics_1[agent_idx], config.UPDATE_FACTOR)
                soft_update(self.target_critics_2[agent_idx], self.critics_2[agent_idx], config.UPDATE_FACTOR)

            for n in self.noise:
                n.decay()

    def _init_target_networks(self) -> None:
        for actor, target_actor in zip(self.actors, self.target_actors):
            target_actor.load_state_dict(actor.state_dict())
        for critic1, target_critic1 in zip(self.critics_1, self.target_critics_1):
            target_critic1.load_state_dict(critic1.state_dict())
        for critic2, target_critic2 in zip(self.critics_2, self.target_critics_2):
            target_critic2.load_state_dict(critic2.state_dict())

    def save(self, directory: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i in range(self.num_agents):
            torch.save(self.actors[i].state_dict(), os.path.join(directory, f"actor_{i}.pth"))
            torch.save(self.critics_1[i].state_dict(), os.path.join(directory, f"critic_1_{i}.pth"))
            torch.save(self.critics_2[i].state_dict(), os.path.join(directory, f"critic_2_{i}.pth"))

    def load(self, directory: str) -> None:
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(torch.load(os.path.join(directory, f"actor_{i}.pth")))
            self.critics_1[i].load_state_dict(torch.load(os.path.join(directory, f"critic_1_{i}.pth")))
            self.critics_2[i].load_state_dict(torch.load(os.path.join(directory, f"critic_2_{i}.pth")))
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics_1[i].load_state_dict(self.critics_1[i].state_dict())
            self.target_critics_2[i].load_state_dict(self.critics_2[i].state_dict())
