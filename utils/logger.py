import config as default_config
import json
import os
import numpy as np


class Log:
    def __init__(self) -> None:
        self.rewards: list[float] = []
        self.latencies: list[float] = []
        self.energies: list[float] = []
        self.fairness_scores: list[float] = []
        self.offline_rates: list[float] = []
        # Training losses (optional, may be empty for random baseline)
        self.actor_losses: list[float | None] = []
        self.critic_losses: list[float | None] = []
        self.entropy_losses: list[float | None] = []
        self.alpha_losses: list[float | None] = []

    def append(
        self,
        reward: float,
        latency: float,
        energy: float,
        fairness: float,
        offline_rate: float,
        *,
        actor_loss: float | None = None,
        critic_loss: float | None = None,
        entropy_loss: float | None = None,
        alpha_loss: float | None = None,
    ) -> None:
        """Append metrics for one logging interval. Loss fields are optional and can be None.

        Parameters are averaged/aggregated externally by training loop before being passed here.
        """
        self.rewards.append(reward)
        self.latencies.append(latency)
        self.energies.append(energy)
        self.fairness_scores.append(fairness)
        self.offline_rates.append(offline_rate)

        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.entropy_losses.append(entropy_loss)
        self.alpha_losses.append(alpha_loss)


class Logger:
    def __init__(self, log_dir: str, timestamp: str) -> None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.timestamp: str = timestamp
        self.log_dir: str = log_dir
        self.log_file_path: str = os.path.join(self.log_dir, f"logs_{timestamp}.txt")
        self.json_file_path: str = os.path.join(self.log_dir, f"log_data_{timestamp}.json")
        self.config_file_path: str = os.path.join(self.log_dir, f"config_{timestamp}.json")

    def log_configs(self) -> None:
        config_dict: dict = {
            key: getattr(default_config, key)
            for key in dir(default_config)
            if key.isupper()
            and not key.startswith("__")
            and not callable(getattr(default_config, key))
        }

        # Custom serializer for numpy arrays
        def numpy_encoder(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(self.config_file_path, "w") as f:
            json.dump(config_dict, f, indent=4, default=numpy_encoder)
        print(f"📝 Configs saved to {self.config_file_path}")

    def load_configs(self, config_path: str) -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Config file not found: {config_path}")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            # Convert lists back to numpy arrays where appropriate
            if isinstance(getattr(default_config, key, None), np.ndarray):
                setattr(default_config, key, np.array(value))
            else:
                setattr(default_config, key, value)

        print(f"✅ Configs loaded from {config_path}")

    def log_metrics(
        self,
        progress_step: int,
        log: Log,
        log_freq: int,
        elapsed_time: float,
        name: str,
        losses: dict | None = None,
    ) -> None:
        """Log aggregated metrics to text and JSON files.

        `losses` is optional and may contain keys like 'actor', 'critic', 'entropy', 'alpha'. If present,
        they will be included in the saved logs. The training loops should pass averaged loss values for the
        logging interval (matching `log_freq`) when available.
        """
        rewards_slice: np.ndarray = np.array(log.rewards[-log_freq:])
        latencies_slice: np.ndarray = np.array(log.latencies[-log_freq:])
        energies_slice: np.ndarray = np.array(log.energies[-log_freq:])
        fairness_slice: np.ndarray = np.array(log.fairness_scores[-log_freq:])
        offline_slice: np.ndarray = np.array(log.offline_rates[-log_freq:])

        reward_avg: float = float(np.mean(rewards_slice))
        latency_avg: float = float(np.mean(latencies_slice))
        energy_avg: float = float(np.mean(energies_slice))
        fairness_avg: float = float(np.mean(fairness_slice))
        offline_avg: float = float(np.mean(offline_slice))
        
        # Prepare loss averages from the Log object if available; prefer explicit `losses` dict when provided
        def _safe_mean(lst: list) -> float | None:
            if not lst:
                return None
            vals = [x for x in lst[-log_freq:] if x is not None]
            if not vals:
                return None
            return float(np.mean(np.array(vals)))

        actor_avg = None
        critic_avg = None
        entropy_avg = None
        alpha_avg = None

        if losses is not None:
            actor_loss_val = losses.get("actor")
            actor_avg = float(actor_loss_val) if actor_loss_val is not None else None
            critic_loss_val = losses.get("critic")
            critic_avg = float(critic_loss_val) if critic_loss_val is not None else None
            entropy_loss_val = losses.get("entropy")
            entropy_avg = (float(entropy_loss_val) if entropy_loss_val is not None else None)
            alpha_loss_val = losses.get("alpha")
            alpha_avg = float(alpha_loss_val) if alpha_loss_val is not None else None
        else:
            actor_avg = _safe_mean(log.actor_losses)
            critic_avg = _safe_mean(log.critic_losses)
            entropy_avg = _safe_mean(log.entropy_losses)
            alpha_avg = _safe_mean(log.alpha_losses)

        # Build human readable log message
        loss_parts: list[str] = []
        if actor_avg is not None:
            loss_parts.append(f"Actor Loss: {actor_avg:.6f}")
        if critic_avg is not None:
            loss_parts.append(f"Critic Loss: {critic_avg:.6f}")
        if entropy_avg is not None:
            loss_parts.append(f"Entropy Loss: {entropy_avg:.6f}")
        if alpha_avg is not None:
            loss_parts.append(f"Alpha Loss: {alpha_avg:.6f}")

        loss_str = " | ".join(loss_parts) + " | " if loss_parts else ""

        log_msg: str = (
            f"🔄 {name.title()} {progress_step} | "
            f"Total Reward: {reward_avg:.3f} | "
            f"Total Latency: {latency_avg:.3f} | "
            f"Total Energy: {energy_avg:.3f} | "
            f"Final Fairness: {fairness_avg:.3f} | "
            f"Offline Rate: {offline_avg:.3f} | "
            + loss_str
            + f"Elapsed Time: {elapsed_time:.2f}s\n"
        )

        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(log_msg)

        # Prepare JSON entry, only include keys that are not None
        data_entry: dict = {
            name.lower(): progress_step,
            "reward": reward_avg,
            "latency": latency_avg,
            "energy": energy_avg,
            "fairness": fairness_avg,
            "offline_rate": offline_avg,
            "time": elapsed_time,
        }
        if actor_avg is not None:
            data_entry["actor_loss"] = actor_avg
        if critic_avg is not None:
            data_entry["critic_loss"] = critic_avg
        if entropy_avg is not None:
            data_entry["entropy_loss"] = entropy_avg
        if alpha_avg is not None:
            data_entry["alpha_loss"] = alpha_avg

        json_data: list[dict] = []
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, "r") as jf:
                try:
                    json_data = json.load(jf)
                except json.JSONDecodeError:
                    json_data = []

        json_data.append(data_entry)
        with open(self.json_file_path, "w") as f:
            json.dump(json_data, f, indent=4)
