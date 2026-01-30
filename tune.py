import optuna
import argparse
import os
import numpy as np
import torch
import json
from datetime import datetime
import warnings
import optuna.visualization as vis

import config
from environment.env import Env
from marl_models.utils import get_model
from utils.logger import Logger
from train import train_on_policy, train_off_policy

# Suppress warnings for cleaner output during tuning
warnings.filterwarnings("ignore")


def objective(trial: optuna.Trial, stage: int, model_name: str, num_episodes: int) -> float:
    """
    Optuna Objective Function.
    Adjusts config based on 'stage' and runs a training session.
    """

    # --- STAGE 1: Objective Tuning (Reward Weights & Caching) ---
    if stage == 1:
        # We tune the definition of "Success" first
        config.ALPHA_1 = trial.suggest_float("alpha_1", 1.0, 15.0, step=0.5)  # Latency
        config.ALPHA_2 = trial.suggest_float("alpha_2", 0.1, 5.0, step=0.1)  # Energy
        config.ALPHA_3 = trial.suggest_float("alpha_3", 1.0, 10.0, step=0.1)  # Fairness
        config.GDSF_SMOOTHING_FACTOR = trial.suggest_float("gdsf_beta", 0.1, 0.9, step=0.05)

    # --- STAGE 2: Agent Tuning (Hyperparameters) ---
    elif stage == 2:
        # We tune the solver to reach the success defined in Stage 1
        # (Assuming Stage 1 params are hardcoded/loaded in config.py now)
        config.ACTOR_LR = trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True)
        config.CRITIC_LR = trial.suggest_float("critic_lr", 1e-5, 1e-3, log=True)
        config.PPO_BATCH_SIZE = trial.suggest_categorical("batch_size", [32, 64, 128])
        config.MLP_HIDDEN_DIM = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        config.DISCOUNT_FACTOR = trial.suggest_float("gamma", 0.90, 0.99, step=0.01)

    # --- Setup Environment & Model ---
    np.random.seed(config.SEED + trial.number)  # Change seed per trial
    torch.manual_seed(config.SEED + trial.number)

    env = Env()
    model = get_model(model_name)

    # Minimal Logger for Tuning (Prevent cluttering disk with 100s of logs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_log_dir = f"tuning_logs/{model_name}/trial_{trial.number}"
    if not os.path.exists(tuning_log_dir):
        os.makedirs(tuning_log_dir)
    logger = Logger(tuning_log_dir, timestamp)

    # --- Execution ---
    try:
        final_score: float = 0.0
        if model_name in ["maddpg", "matd3", "masac"]:
            final_score = train_off_policy(env, model, logger, num_episodes, 0, trial)
        elif model_name == "mappo":
            final_score = train_on_policy(env, model, logger, num_episodes, trial)

        return final_score

    except optuna.TrialPruned:
        raise  # Let Optuna handle the pruning exception
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float("-inf")  # Return lowest possible score on failure


def run_tuning(args):
    print(f"\n🎯 Starting Stage {args.stage} Tuning for {config.MODEL}...")
    print(f"📝 Episodes per trial: {args.episodes}")
    print(f"🔍 Trials: {args.trials}")

    # Use MedianPruner for Early Stopping
    # It stops a trial if its intermediate result is worse than the median of previous trials
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Don't prune the first 5 trials (let them complete)
        n_warmup_steps=10,  # Don't prune early steps of any trial
        interval_steps=1,  # Check for pruning at every report
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{config.MODEL}_stage_{args.stage}",
        pruner=pruner,
    )

    func = lambda trial: objective(trial, args.stage, config.MODEL.lower(), args.episodes)
    study.optimize(func, n_trials=args.trials)

    print("\n🏆 Tuning Completed!")
    print(f"Best Trial Score: {study.best_value}")
    print(f"Best Trial Number: {study.best_trial.number}")
    print("Best Parameters:")
    print(json.dumps(study.best_params, indent=4))

    # Save best params and study summary
    save_path = f"tuning_logs/{config.MODEL}/stage_{args.stage}.json"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "n_trials": len(study.trials),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }
    
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"💾 Saved best parameters to {save_path}")
    
    # Generate plots
    try:
        plot_tuning_results(study, config.MODEL, args.stage)
    except Exception as e:
        print(f"⚠️ Could not generate plots: {e}")


def plot_tuning_results(study: optuna.Study, model_name: str, stage: int) -> None:
    """Generates and saves tuning result plots."""

    plot_dir = f"tuning_logs/{model_name}/plots_stage_{stage}"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Parameter Importance
    fig = vis.plot_param_importances(study)
    fig.write_image(f"{plot_dir}/param_importance.png")

    # Optimization History
    fig = vis.plot_optimization_history(study)
    fig.write_image(f"{plot_dir}/optimization_history.png")

    # Slice Plot - Shows individual parameter effects
    fig = vis.plot_slice(study)
    fig.write_image(f"{plot_dir}/slice_plot.png")

    # Intermediate Values - Shows learning curves across trials
    fig = vis.plot_intermediate_values(study)
    fig.write_image(f"{plot_dir}/intermediate_values.png")

    print(f"📊 Saved tuning plots to {plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning Module")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        required=True,
        help="1: Tune Rewards/Env, 2: Tune Agent",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Episodes per trial (Lower than full training)",
    )
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of trials to run"
    )

    args = parser.parse_args()
    run_tuning(args)
