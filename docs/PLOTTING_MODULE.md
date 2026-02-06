# Quick Reference: Plotting System

## Generate Single-Run Plots

After training, generate plots for one algorithm:

```python
from utils.plot_logs import generate_plots

generate_plots(
    log_file="train_logs/log_data_2025-12-24_15-30-00.json",
    output_dir="plots/mappo_run1",
    output_file_prefix="mappo",
    timestamp="2025-12-24_15-30-00",
    smoothing_window=10
)
```

## Compare Multiple Algorithms

### Method 1: Command Line (Recommended)

```bash
python utils/compare_algorithms.py \
    --logs train_logs/maddpg_run train_logs/matd3_run train_logs/mappo_run \
    --names MADDPG MATD3 MAPPO \
    --output comparison_plots \
    --smoothing 10
```

### Method 2: Python Script

```python
from utils.comparative_plots import compare_algorithms

compare_algorithms(
    log_dirs=["train_logs/algo1", "train_logs/algo2"],
    algorithm_names=["Algorithm 1", "Algorithm 2"],
    output_dir="comparisons",
    smoothing_window=10
)
```

## Available Plots

### Single-Run Plots

- `reward_vs_episode.png`
- `latency_vs_episode.png`
- `energy_vs_episode.png`
- `fairness_vs_episode.png`
- `offline_rate_vs_episode.png`
- `actor_loss_vs_episode.png` (if available)
- `critic_loss_vs_episode.png` (if available)
- `entropy_loss_vs_episode.png` (MAPPO only)
- `alpha_loss_vs_episode.png` (MASAC only)

### Comparative Plots

- `comparison_reward.png`
- `comparison_latency.png`
- `comparison_energy.png`
- `comparison_fairness.png`
- `comparison_offline_rate.png`
- `comparison_actor_loss.png`
- `comparison_critic_loss.png`
- `comparison_summary.png` (8-panel grid)

## Typical Workflow

1. **Train multiple algorithms**:

   ```bash
   # Edit config.py: MODEL = 'maddpg'
   python main.py

   # Edit config.py: MODEL = 'matd3'
   python main.py

   # Edit config.py: MODEL = 'mappo'
   python main.py
   ```

2. **Generate individual plots** (optional):

   ```python
   for each run:
       generate_plots(log_file, output_dir, prefix, timestamp)
   ```

3. **Generate comparative plots**:
   ```bash
   python utils/compare_algorithms.py \
       --logs train_logs/run1 train_logs/run2 train_logs/run3 \
       --names MADDPG MATD3 MAPPO \
       --output final_comparison
   ```


## Example Directory Structure

```
project/
├── train_logs/
│   ├── maddpg/
│   │   ├── state_images_2025-12-24_17-56-57
│   │   ├── config_2025-12-24_17-56-57.json
│   │   ├── log_data_2025-12-24_17-56-57.json
│   │   └── logs_2025-12-24_17-56-57.txt
│   ├── matd3/
│   └── mappo/
└── train_plots/
    ├── maddpg/
    │   ├── train_reward_2025-12-24_17-56-57.png
    │   └── train_actor_loss_2025-12-24_17-56-57.png
    ├── matd3/
    ├── mappo/
    ├── comparison_reward.png
    ├── comparison_actor_loss.png
    └── comparison_summary.png
```
