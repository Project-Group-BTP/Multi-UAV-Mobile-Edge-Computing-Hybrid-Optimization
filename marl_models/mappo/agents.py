import config
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(ActorNetwork, self).__init__()
        self.fc1: nn.Linear = nn.Linear(obs_dim, config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM)
        self.mean: nn.Linear = nn.Linear(config.MLP_HIDDEN_DIM, action_dim)

        # Making the log of the standard deviation a learnable parameter.
        # This is often more stable than learning the std deviation directly.
        self.log_std: nn.Parameter = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs: torch.Tensor) -> Normal:
        x: torch.Tensor = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        # Output the mean of the distribution. Tanh activation scales it to [-1, 1].
        mean: torch.Tensor = torch.tanh(self.mean(x))

        # Clamp the log_std to prevent it from becoming too large or small.
        log_std: torch.Tensor = torch.clamp(self.log_std, config.LOG_STD_MIN, config.LOG_STD_MAX)

        # The standard deviation is the exponent of the log_std.
        std: torch.Tensor = torch.exp(log_std)
        return Normal(mean, std)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super(CriticNetwork, self).__init__()
        self.fc1: nn.Linear = nn.Linear(state_dim, config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM)
        self.out: nn.Linear = nn.Linear(config.MLP_HIDDEN_DIM, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.out(x)
