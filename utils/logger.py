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

    def append(self, reward: float, latency: float, energy: float, fairness: float) -> None:
        self.rewards.append(reward)
        self.latencies.append(latency)
        self.energies.append(energy)
        self.fairness_scores.append(fairness)


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
        config_dict: dict = {key: getattr(default_config, key) for key in dir(default_config) if key.isupper() and not key.startswith("__") and not callable(getattr(default_config, key))}

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

    def log_metrics(self, progress_step: int, log: Log, log_freq: int, elapsed_time: float, name: str) -> None:
        rewards_slice: np.ndarray = np.array(log.rewards[-log_freq:])
        latencies_slice: np.ndarray = np.array(log.latencies[-log_freq:])
        energies_slice: np.ndarray = np.array(log.energies[-log_freq:])
        fairness_slice: np.ndarray = np.array(log.fairness_scores[-log_freq:])

        reward_avg: float = float(np.mean(rewards_slice))
        latency_avg: float = float(np.mean(latencies_slice))
        energy_avg: float = float(np.mean(energies_slice))
        fairness_avg: float = float(np.mean(fairness_slice))
        log_msg: str = f"🔄 {name.title()} {progress_step} | " f"Total Reward: {reward_avg:.3f} | " f"Total Latency: {latency_avg:.3f} | " f"Total Energy: {energy_avg:.3f} | " f"Final Fairness: {fairness_avg:.3f} | " f"Elapsed Time: {elapsed_time:.2f}s\n"

        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(log_msg)

        data_entry: dict = {name.lower(): progress_step, "reward": reward_avg, "latency": latency_avg, "energy": energy_avg, "fairness": fairness_avg, "time": elapsed_time}
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
