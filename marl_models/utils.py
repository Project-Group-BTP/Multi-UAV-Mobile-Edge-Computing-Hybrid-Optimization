import config
import torch
import torch.nn as nn
import numpy as np


def get_device() -> str:
    """Check if GPU is available and set device accordingly."""
    if torch.cuda.is_available():
        print("\n🤖 Found GPU, using CUDA.")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("\n🤖 Using MPS (Apple Silicon GPU).")
        return "mps"
    else:
        print("\n⚙️ No GPU available, using CPU.")
        return "cpu"


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float):
    """Performs a soft update of the target network's parameters."""
    with torch.no_grad():
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.copy_(tau * param + (1.0 - tau) * target_param)


class GaussianNoise:
    """Gaussian noise with decay for exploration."""

    def __init__(self) -> None:
        self.scale: float = config.INITIAL_NOISE_SCALE

    def sample(self) -> np.ndarray:
        return np.random.normal(0, self.scale, config.ACTION_DIM)

    def decay(self) -> None:
        self.scale = max(config.MIN_NOISE_SCALE, self.scale * config.NOISE_DECAY_RATE)

    def reset(self) -> None:
        self.scale = config.INITIAL_NOISE_SCALE
