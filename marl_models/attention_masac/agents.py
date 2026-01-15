import config
from marl_models.buffer_and_helpers import layer_init
from marl_models.attention import AttentionActorBase, AttentionCriticBase
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorNetwork(AttentionActorBase):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__(obs_dim)
        self.mean: nn.Linear = layer_init(nn.Linear(self.hidden_dim, action_dim))
        self.log_std: nn.Linear = layer_init(nn.Linear(self.hidden_dim, action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self.get_feature_embedding(obs)
        mean: torch.Tensor = self.mean(x)
        log_std: torch.Tensor = torch.clamp(self.log_std(x), min=config.LOG_STD_MIN, max=config.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std: torch.Tensor = log_std.exp()
        dist: Normal = Normal(mean, std)

        # Reparameterization Trick
        x_t: torch.Tensor = dist.rsample()
        y_t: torch.Tensor = torch.tanh(x_t)  # Squash to [-1, 1]
        action: torch.Tensor = y_t

        # Log prob correction for tanh
        log_prob: torch.Tensor = dist.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + config.EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class CriticNetwork(AttentionCriticBase):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__(obs_dim, action_dim)

        # Final Head: Output 1 value
        self.q_head: nn.Sequential = nn.Sequential(layer_init(nn.Linear(self.fusion_dim, self.mlp_dim)), nn.LayerNorm(self.mlp_dim), nn.ReLU(), layer_init(nn.Linear(self.mlp_dim, 1)))

    def forward(self, obs_tensor: torch.Tensor, action_tensor: torch.Tensor, agent_index: int) -> torch.Tensor:
        embedding: torch.Tensor = self.get_eval_embedding(obs_tensor, action_tensor, agent_index)
        return self.q_head(embedding)
