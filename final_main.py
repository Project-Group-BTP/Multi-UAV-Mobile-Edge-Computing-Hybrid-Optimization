from marl_models.utils import get_device
from marl_models.base_model import MARLModel
from marl_models.maddpg.maddpg import MADDPG
from marl_models.matd3.matd3 import MATD3
from marl_models.mappo.mappo import MAPPO
from marl_models.random_baseline.random_model import RandomModel
from marl_models.greedy_baseline.greedy_model import GreedyModel
from environment.env import Env
from train import train_on_policy, train_off_policy
from utils.logger import Logger
import config
import argparse
import time
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--resume", type=bool, help="Resume training from saved model", default=False)
    args = parser.parse_args()

    model_name = config.MODEL.lower()
    env: Env = Env()
    model: MARLModel
    device = get_device()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n🚀 Training started at {timestamp} for {config.NUM_EPISODES} episodes\n")
    logger = Logger(
        log_dir="./train_logs",
        log_file_name=f"logs_{timestamp}.txt",
        log_data_file_name=f"log_data_{timestamp}.json",
    )

    if model_name == "maddpg":
        model = MADDPG(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    elif model_name == "matd3":
        model = MATD3(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    elif model_name == "mappo":
        model = MAPPO(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, state_dim=config.NUM_UAVS * config.OBS_DIM_SINGLE, device=device)
    elif model_name == "random":
        model = RandomModel(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    elif model_name == "greedy":
        model = GreedyModel(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
        model.set_environment(env)
    else:
        raise ValueError(f"Unknown model type: {model_name}. " f"Supported types: maddpg, matd3, mappo, random, greedy")

    if args.resume:
        model.load(config.RESUME_DIRECTORY)
        print(f"📂 Resumed training from: {config.RESUME_DIRECTORY}\n")

    start_time = time.time()
    if model_name in ["maddpg", "matd3"]:
        train_off_policy(env, model, model_name, args.resume)
    elif model_name == "mappo":
        train_on_policy(env, model, model_name, args.resume)


if __name__ == "__main__":
    main()
