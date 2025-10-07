import config
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(ActorNetwork, self).__init__()
        self.fc1: nn.Linear = nn.Linear(obs_dim, config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM)
        self.out: nn.Linear = nn.Linear(config.MLP_HIDDEN_DIM, action_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.out(x))


class CriticNetwork(nn.Module):
    def __init__(self, total_obs_dim: int, total_action_dim: int) -> None:
        super(CriticNetwork, self).__init__()
        self.fc1: nn.Linear = nn.Linear(total_obs_dim + total_action_dim, config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM)
        self.out: nn.Linear = nn.Linear(config.MLP_HIDDEN_DIM, 1)

    def forward(self, joint_obs: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = torch.cat([joint_obs, joint_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
