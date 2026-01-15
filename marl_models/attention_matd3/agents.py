from marl_models.buffer_and_helpers import layer_init
from marl_models.attention import AttentionActorBase, AttentionCriticBase
import torch
import torch.nn as nn


class ActorNetwork(AttentionActorBase):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__(obs_dim)
        self.out: nn.Linear = layer_init(nn.Linear(self.hidden_dim, action_dim), std=0.01)  # Small std for output

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.get_feature_embedding(obs)
        return torch.tanh(self.out(x))


class CriticNetwork(AttentionCriticBase):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__(obs_dim, action_dim)

        # Final Head: Output 1 value
        self.q_head: nn.Sequential = nn.Sequential(layer_init(nn.Linear(self.fusion_dim, self.mlp_dim)), nn.LayerNorm(self.mlp_dim), nn.ReLU(), layer_init(nn.Linear(self.mlp_dim, 1)))

    def forward(self, obs_tensor: torch.Tensor, action_tensor: torch.Tensor, agent_index: int) -> torch.Tensor:
        embedding: torch.Tensor = self.get_eval_embedding(obs_tensor, action_tensor, agent_index)
        return self.q_head(embedding)
