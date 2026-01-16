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
        self.log_std: nn.Parameter = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs: torch.Tensor) -> Normal:
        x: torch.Tensor = self.get_feature_embedding(obs)
        mean: torch.Tensor = torch.tanh(self.mean(x))
        log_std: torch.Tensor = torch.clamp(self.log_std, config.LOG_STD_MIN, config.LOG_STD_MAX)
        std: torch.Tensor = torch.exp(log_std)
        return Normal(mean, std)


class CriticNetwork(AttentionCriticBase):
    def __init__(self, obs_dim: int) -> None:
        # action_dim=0 because this is V(s), not Q(s,a)
        super().__init__(obs_dim, action_dim=0)
        self.v_head: nn.Sequential = nn.Sequential(layer_init(nn.Linear(self.fusion_dim, self.mlp_dim)), nn.LayerNorm(self.mlp_dim), nn.ReLU(), layer_init(nn.Linear(self.mlp_dim, 1)))

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encodes observations once for the whole batch, then runs attention heads Num_Agents times.
        Args:
            obs_tensor: (Batch, Num_Agents, Obs_Dim)
        Returns:
            values: (Batch, Num_Agents)
        """
        num_agents: int = obs_tensor.shape[1]

        # Run the heavy encoder once for all agents
        # (Batch, Num_Agents, Obs) -> (Batch, Num_Agents, Hidden)
        all_embeddings: torch.Tensor = self.get_all_embeddings(obs_tensor)

        values_list: list[torch.Tensor] = [self.v_head(self.attend_to_others(all_embeddings, num_agents, agent_index=i)) for i in range(num_agents)]

        return torch.cat(values_list, dim=1)  # (Batch, Num_Agents)
