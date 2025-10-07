from marl_models.base_model import MARLModel
from marl_models.maddpg.maddpg import MADDPG
from marl_models.matd3.matd3 import MATD3
from marl_models.mappo.mappo import MAPPO
from marl_models.masac.masac import MASAC
from marl_models.random_baseline.random_model import RandomModel
import config
import torch
import os
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


def get_model(model_name: str) -> MARLModel:
    device = get_device()
    if model_name == "maddpg":
        return MADDPG(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    elif model_name == "matd3":
        return MATD3(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    elif model_name == "mappo":
        return MAPPO(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, state_dim=config.STATE_DIM, device=device)
    elif model_name == "masac":
        return MASAC(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    elif model_name == "random":
        return RandomModel(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_name}. Supported types: maddpg, matd3, mappo, masac, random")


def save_models(model: MARLModel, progress_step: int, name: str, final: bool = False):
    save_dir: str = f"saved_models/{model.model_name}"
    if final:
        save_dir = f"{save_dir}/final"
    else:
        save_dir = f"{save_dir}/{name.lower()}_{progress_step:04d}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.save(save_dir)
    if final:
        print(f"📁 Final models saved in: {save_dir}\n")
    else:
        print(f"📁 Models saved for {name.lower()} {progress_step} in: {save_dir}\n")


def soft_update(target_net: torch.nn.Module, source_net: torch.nn.Module, tau: float):
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
