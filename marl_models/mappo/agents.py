import config
from marl_models.buffer_and_helpers import layer_init
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1: nn.Linear = layer_init(nn.Linear(obs_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.mean: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, action_dim))
        self.log_std: nn.Parameter = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs: torch.Tensor) -> Normal:
        x: torch.Tensor = torch.relu(self.ln1(self.fc1(obs)))
        x = torch.relu(self.ln2(self.fc2(x)))
        # Output the mean of the distribution. Tanh activation scales it to [-1, 1].
        mean: torch.Tensor = torch.tanh(self.mean(x))
        log_std: torch.Tensor = torch.clamp(self.log_std, config.LOG_STD_MIN, config.LOG_STD_MAX)
        std: torch.Tensor = torch.exp(log_std)
        return Normal(mean, std)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.fc1: nn.Linear = layer_init(nn.Linear(state_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.out: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return self.out(x)
