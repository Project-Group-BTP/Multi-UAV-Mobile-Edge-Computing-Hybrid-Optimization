from marl_models.base_model import MARLModel
from environment.env import Env
from marl_models.utils import get_model, load_step_count
from train import train_on_policy, train_off_policy, train_random
from test import test_model
from utils.logger import Logger
from utils.plot_logs import generate_plots
import config
import torch
import numpy as np
import argparse
import warnings
from datetime import datetime


def start_training(args: argparse.Namespace):
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n🚀 Training started at {timestamp} for {args.num_episodes} episodes\n")
    logger: Logger = Logger("train_logs", timestamp)
    resume_training: bool = args.resume_path is not None
    if resume_training:
        if args.config_path is None:
            raise ValueError("If --resume_path is provided, --config_path must also be provided.")
        else:
            logger.load_configs(args.config_path)  # Resume training with old config
    else:  # Fresh training
        if args.config_path is not None:
            warnings.warn("--config_path is ignored during training unless --resume_path is also provided.")
        logger.log_configs()

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    env: Env = Env()
    model_name: str = config.MODEL.lower()
    model: MARLModel = get_model(model_name)

    total_step_count: int = 0  # for off policy models
    if resume_training:
        model.load(args.resume_path)
        total_step_count = load_step_count(args.resume_path)
        print(f"📥 Models loaded successfully from {args.resume_path}")
        print(f"📂 Resumed training from: {args.resume_path}\n")

    if model_name in ["maddpg", "matd3", "masac"]:
        train_off_policy(env, model, logger, args.num_episodes, total_step_count)
    elif model_name == "mappo":
        train_on_policy(env, model, logger, args.num_episodes)
    else:  # "random"
        train_random(env, model, logger, args.num_episodes)

    print("✅ Training Completed!\n")
    print("📊 Generating plots...")
    generate_plots(f"train_logs/log_data_{timestamp}.json", "train_plots/", "train", timestamp)


def start_testing(args: argparse.Namespace):
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n🚀 Testing started at {timestamp} for {args.num_episodes} episodes\n")
    logger: Logger = Logger("test_logs", timestamp)
    logger.load_configs(args.config_path)

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    env: Env = Env()
    model: MARLModel = get_model(config.MODEL.lower())

    model.load(args.model_path)
    print(f"📥 Models loaded successfully from {args.model_path}")

    test_model(env, model, logger, args.num_episodes)

    print("✅ Testing Completed!\n")
    print("📊 Generating plots...")
    generate_plots(f"test_logs/log_data_{timestamp}.json", "test_plots/", "test", timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--num_episodes", type=int, required=True)
    train_parser = subparsers.add_parser("train", parents=[parent_parser])
    train_parser.add_argument("--resume_path", type=str, default=None)
    train_parser.add_argument("--config_path", type=str, default=None)

    test_parser = subparsers.add_parser("test", parents=[parent_parser])
    test_parser.add_argument("--model_path", type=str, required=True)
    test_parser.add_argument("--config_path", type=str, required=True)

    args = parser.parse_args()
    if args.mode == "train":
        start_training(args)
    elif args.mode == "test":
        start_testing(args)
    print("🎉 All done!")
