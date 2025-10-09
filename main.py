from marl_models.utils import get_model
from marl_models.base_model import MARLModel
from environment.env import Env
from train import train_on_policy, train_off_policy, train_random
from test import test_model
from utils.logger import Logger
from utils.plot_logs import generate_plots
import config
import torch
import numpy as np
import argparse
from datetime import datetime


def start_training(args: argparse.Namespace):
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n🚀 Training started at {timestamp} for {args.num_episodes} episodes\n")
    logger: Logger = Logger(
        log_dir="./train_logs",
        log_file_name=f"logs_{timestamp}.txt",
        log_data_file_name=f"log_data_{timestamp}.json",
        config_file_name=f"config_{timestamp}.json",
    )
    logger.log_configs()

    if args.resume:
        logger.load_configs(f"{config.RESUME_DIRECTORY}/config.json")

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    env: Env = Env()
    model_name: str = config.MODEL.lower()
    model: MARLModel = get_model(model_name)

    if args.resume:
        model.load(config.RESUME_DIRECTORY)
        print(f"📥 Models loaded successfully from {config.RESUME_DIRECTORY}")
        print(f"📂 Resumed training from: {config.RESUME_DIRECTORY}\n")

    if model_name in ["maddpg", "matd3", "masac"]:
        train_off_policy(env, model, logger, args.num_episodes)
    elif model_name == "mappo":
        train_on_policy(env, model, logger, args.num_episodes)
    else:  # "random"
        train_random(env, model, logger, args.num_episodes)  # Training = Testing for random model

    print("✅ Training Completed!\n")
    print("📊 Generating plots...\n")
    generate_plots(log_file=f"./train_logs/log_data_{timestamp}.json", output_dir="./train_plots/", output_file_prefix="train")


def start_testing(args: argparse.Namespace):
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n🚀 Testing started at {timestamp} for {args.num_episodes} episodes\n")
    logger: Logger = Logger(
        log_dir="./test_logs",
        log_file_name=f"logs_{timestamp}.txt",
        log_data_file_name=f"log_data_{timestamp}.json",
        config_file_name=f"config_{timestamp}.json",
    )
    logger.load_configs(args.config_path)

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    env: Env = Env()
    model_name: str = config.MODEL.lower()
    model: MARLModel = get_model(model_name)

    model.load(args.model_path)
    print(f"📥 Models loaded successfully from {args.model_path}")

    test_model(env, model, logger, args.num_episodes)

    print("✅ Testing Completed!\n")
    print("📊 Generating plots...\n")
    generate_plots(log_file=f"./test_logs/log_data_{timestamp}.json", output_dir="./test_plots/", output_file_prefix="test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--num_episodes", type=int, required=True)
    train_parser = subparsers.add_parser("train", parents=[parent_parser])
    train_parser.add_argument("--resume", action="store_true", default=False)

    test_parser = subparsers.add_parser("test", parents=[parent_parser])
    test_parser.add_argument("--model_path", type=str, required=True)
    test_parser.add_argument("--config_path", type=str, required=True)

    args = parser.parse_args()
    if args.mode == "train":
        start_training(args)
    elif args.mode == "test":
        start_testing(args)
    print("🎉 All done!")
