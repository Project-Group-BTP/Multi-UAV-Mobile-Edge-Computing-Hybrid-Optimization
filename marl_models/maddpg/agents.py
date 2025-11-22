import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Added layer normalization and orthogonal initialization for better training stability
# Can be adapted for other models as needed


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(ActorNetwork, self).__init__()
        self.fc1: nn.Linear = layer_init(nn.Linear(obs_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.out: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, action_dim), std=0.01)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = F.relu(self.ln1(self.fc1(input)))
        x = F.relu(self.ln2(self.fc2(x)))
        return torch.tanh(self.out(x))


class CriticNetwork(nn.Module):
    def __init__(self, total_obs_dim: int, total_action_dim: int) -> None:
        super(CriticNetwork, self).__init__()
        self.fc1: nn.Linear = layer_init(nn.Linear(total_obs_dim + total_action_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.out: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, 1))

    def forward(self, joint_obs: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = torch.cat([joint_obs, joint_action], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.out(x)
