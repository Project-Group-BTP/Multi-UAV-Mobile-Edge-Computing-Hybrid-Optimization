from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.attention_mappo.agents import ActorNetwork, CriticNetwork
import config
import numpy as np
import os
import torch
from torch.distributions import Normal


class AttentionMAPPO(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.actor: ActorNetwork = ActorNetwork(obs_dim, action_dim).to(device)
        self.critic: CriticNetwork = CriticNetwork(obs_dim).to(device)
        self.actor_optimizer: torch.optim.Adam = torch.optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer: torch.optim.Adam = torch.optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        obs_tensor: torch.Tensor = torch.as_tensor(np.array(observations), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist: Normal = self.actor(obs_tensor)
            actions: torch.Tensor = dist.sample() if exploration else dist.mean
        return np.clip(actions.cpu().numpy(), -1.0, 1.0)

    def get_action_and_value(self, obs: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # The actor treats each agent as an independent sample in the batch: Needs (Num_Agents, Obs_Dim)
        obs_tensor_actor: torch.Tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        # The critic needs to see the "whole swarm" at once to compute attention between agents : Needs (1, Num_Agents, Obs_Dim)
        obs_tensor_critic: torch.Tensor = obs_tensor_actor.unsqueeze(0)

        with torch.no_grad():
            dist: Normal = self.actor(obs_tensor_actor)
            actions: torch.Tensor = dist.sample()
            log_probs: torch.Tensor = dist.log_prob(actions).sum(dim=-1)

            values: torch.Tensor = self.critic(obs_tensor_critic).view(-1)  # (1, Num_Agents) -> (Num_Agents,)

        clipped_actions: np.ndarray = np.clip(actions.cpu().numpy(), -1.0, 1.0)
        return (clipped_actions, log_probs.cpu().numpy(), values.cpu().numpy())

    def update(self, batch: ExperienceBatch) -> dict:
        """Expects a batch from AttentionRolloutBuffer: (Batch_Size, Num_Agents, Dim)"""
        assert isinstance(batch, dict), "MAPPO expects OnPolicyExperienceBatch (dict)"
        obs_batch: torch.Tensor = batch["obs"]
        actions_batch: torch.Tensor = batch["actions"]
        old_log_probs_batch: torch.Tensor = batch["old_log_probs"]
        advantages_batch: torch.Tensor = batch["advantages"]
        returns_batch: torch.Tensor = batch["returns"]
        old_values_batch: torch.Tensor = batch["old_values"]

        # Normalize advantages
        advantages_batch: torch.Tensor = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

        # Critic Update
        values: torch.Tensor = self.critic(obs_batch)  # (Batch, Num_Agents)

        # Value Clipping
        values_clipped: torch.Tensor = old_values_batch + torch.clamp(values - old_values_batch, -config.PPO_CLIP_EPS, config.PPO_CLIP_EPS)
        vf_loss1: torch.Tensor = (values - returns_batch).pow(2)
        vf_loss2: torch.Tensor = (values_clipped - returns_batch).pow(2)
        critic_loss: torch.Tensor = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

        # Actor Update
        # Flatten for Actor: The Actor processes agents independently (local attention).
        # (Batch, Num_Agents) -> (Batch * Num_Agents)
        flat_obs: torch.Tensor = obs_batch.view(-1, self.obs_dim)
        flat_actions: torch.Tensor = actions_batch.view(-1, config.ACTION_DIM)
        flat_old_log_probs: torch.Tensor = old_log_probs_batch.view(-1)
        flat_advantages: torch.Tensor = advantages_batch.view(-1)

        dist: Normal = self.actor(flat_obs)
        new_log_probs: torch.Tensor = dist.log_prob(flat_actions).sum(dim=-1)
        ratio: torch.Tensor = torch.exp(new_log_probs - flat_old_log_probs)
        surr1: torch.Tensor = ratio * flat_advantages
        surr2: torch.Tensor = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPS, 1.0 + config.PPO_CLIP_EPS) * flat_advantages
        actor_loss: torch.Tensor = -torch.min(surr1, surr2).mean()

        entropy: torch.Tensor = dist.entropy().mean()
        actor_loss -= config.PPO_ENTROPY_COEF * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), config.MAX_GRAD_NORM)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.MAX_GRAD_NORM)
        self.critic_optimizer.step()

        # Return losses for logging (same format as standard MAPPO)
        return {
            "actor": float(actor_loss.detach().item()),
            "critic": float(critic_loss.detach().item()),
            "entropy": float(entropy.detach().item()),
        }

    def reset(self) -> None:
        pass

    def save(self, directory: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            os.path.join(directory, "attention_mappo.pth"),
        )

    def load(self, directory: str) -> None:
        path = os.path.join(directory, "attention_mappo.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model file not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
