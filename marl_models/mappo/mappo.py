from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.mappo.agents import ActorNetwork, CriticNetwork
import config
import numpy as np
import os
import torch
from torch.distributions import Normal


class MAPPO(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, state_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.state_dim: int = state_dim

        # Create networks
        self.actors: ActorNetwork = ActorNetwork(obs_dim, action_dim).to(device)
        self.critics: CriticNetwork = CriticNetwork(state_dim).to(device)

        # Create optimizers
        self.actor_optimizer: torch.optim.Adam = torch.optim.Adam(self.actors.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer: torch.optim.Adam = torch.optim.Adam(self.critics.parameters(), lr=config.CRITIC_LR)

    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        obs_tensor: torch.Tensor = torch.as_tensor(np.array(observations), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist: Normal = self.actors(obs_tensor)
            if exploration:
                actions: torch.Tensor = dist.sample()  # Stochastic actions for exploration
            else:
                actions = dist.mean  # Deterministic actions for evaluation

        # Clip actions to be within the valid range [-1, 1]
        return np.clip(actions.cpu().numpy(), -1.0, 1.0)

    def get_action_and_value(self, obs: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_tensor: torch.Tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        state_tensor: torch.Tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            dist: Normal = self.actors(obs_tensor)
            actions: torch.Tensor = dist.sample()

            # Get the log probability of the sampled actions
            log_probs: torch.Tensor = dist.log_prob(actions).sum(dim=-1)

            # Get the value of the current state from the critic network
            values: torch.Tensor = self.critics(state_tensor).squeeze(-1)

        clipped_actions: np.ndarray = np.clip(actions.cpu().numpy(), -1.0, 1.0)
        return clipped_actions, log_probs.cpu().numpy(), values.cpu().numpy()

    def update(self, batch: ExperienceBatch) -> None:
        assert isinstance(batch, dict), "MAPPO expects OnPolicyExperienceBatch (dict)"
        obs_batch: torch.Tensor = batch["obs"]
        actions_batch: torch.Tensor = batch["actions"]
        old_log_probs_batch: torch.Tensor = batch["old_log_probs"]
        advantages_batch: torch.Tensor = batch["advantages"]
        returns_batch: torch.Tensor = batch["returns"]
        states_batch: torch.Tensor = batch["states"]
        old_values_batch: torch.Tensor = batch["old_values"]

        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

        # Critic Loss
        values: torch.Tensor = self.critics(states_batch).squeeze(-1)
        # Value clipping
        values_clipped: torch.Tensor = old_values_batch + torch.clamp(values - old_values_batch, -config.PPO_CLIP_EPS, config.PPO_CLIP_EPS)
        vf_loss1: torch.Tensor = (values - returns_batch).pow(2)
        vf_loss2: torch.Tensor = (values_clipped - returns_batch).pow(2)
        critic_loss: torch.Tensor = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

        # Actor Loss
        dist: Normal = self.actors(obs_batch)
        new_log_probs: torch.Tensor = dist.log_prob(actions_batch).sum(dim=-1)
        ratio: torch.Tensor = torch.exp(new_log_probs - old_log_probs_batch)

        # PPO surrogate loss
        surr1: torch.Tensor = ratio * advantages_batch
        surr2: torch.Tensor = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPS, 1.0 + config.PPO_CLIP_EPS) * advantages_batch
        actor_loss: torch.Tensor = -torch.min(surr1, surr2).mean()

        # Adding entropy bonus for exploration
        entropy_loss: torch.Tensor = dist.entropy().mean()
        actor_loss -= config.PPO_ENTROPY_COEF * entropy_loss

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors.parameters(), config.MAX_GRAD_NORM)
        self.actor_optimizer.step()

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), config.MAX_GRAD_NORM)
        self.critic_optimizer.step()

    def reset(self) -> None:
        pass  # Nothing to reset

    def save(self, directory: str) -> None:
        torch.save(
            {
                "actor": self.actors.state_dict(),
                "critic": self.critics.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            os.path.join(directory, "mappo.pth"),
        )

    def load(self, directory: str) -> None:
        path: str = os.path.join(directory, "mappo.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model file not found: {path}")
        checkpoint: dict = torch.load(path, map_location=self.device)
        self.actors.load_state_dict(checkpoint["actor"])
        self.critics.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
