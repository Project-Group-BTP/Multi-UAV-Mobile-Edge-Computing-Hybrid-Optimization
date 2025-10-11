from marl_models.base_model import MARLModel
from marl_models.maddpg.maddpg import MADDPG
from marl_models.matd3.matd3 import MATD3
from marl_models.mappo.mappo import MAPPO
from marl_models.masac.masac import MASAC
from marl_models.random_baseline.random_model import RandomModel
import config
import torch
import os


def get_device() -> str:
    """Check if GPU is available and set device accordingly."""
    if torch.cuda.is_available():
        print("\nFound GPU, using CUDA.\n")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("\nUsing MPS (Apple Silicon GPU).\n")
        return "mps"
    else:
        print("\nNo GPU available, using CPU.\n")
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


def save_models(model: MARLModel, progress_step: int, name: str, timestamp: str, final: bool = False, total_steps: int = 0):
    save_dir: str = f"saved_models/{model.model_name}_{timestamp}"
    if final:
        save_dir = f"{save_dir}/final"
    else:
        save_dir = f"{save_dir}/{name.lower()}_{progress_step:04d}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.save(save_dir)

    if total_steps > 0:
        step_count_path: str = os.path.join(save_dir, "total_steps.txt")
        with open(step_count_path, "w") as f:
            f.write(str(total_steps))

    if final:
        print(f"📁 Final models saved in: {save_dir}\n")
    else:
        print(f"📁 Models saved for {name.lower()} {progress_step} in: {save_dir}\n")


def load_step_count(directory: str) -> int:
    step_count_path: str = os.path.join(directory, "total_steps.txt")
    if os.path.exists(step_count_path):
        with open(step_count_path, "r") as f:
            return int(f.read().strip())
    return 0
