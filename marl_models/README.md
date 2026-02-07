# MARL Algorithms - Multi-Agent Reinforcement Learning Models

This folder contains 8 different MARL algorithm implementations for multi-agent coordination:

## 📊 Algorithm Overview

### Base Algorithms (No Attention)

| Algorithm | Type | Best For |
|-----------|------|----------|
| **Random** | None | Random baseline |
| **MADDPG** | Off-policy | Stable baseline |
| **MATD3** | Off-policy | More stable (twin critics) |
| **MAPPO** | On-policy | Fast learning |
| **MASAC** | Off-policy + entropy | Exploration friendly |

### Attention Variants (With Graph Attention Networks)

| Algorithm | Base | Advantage |
|-----------|------|-----------|
| **Attention MADDPG** | MADDPG | Better agent coordination |
| **Attention MATD3** | MATD3 | Stable + coordinated |
| **Attention MAPPO** | MAPPO | Fast + coordinated |
| **Attention MASAC** | MASAC | Exploration + coordination |


## 📁 Folder Structure

```
marl_models/
├── base_model.py          # Base class for all models
├── buffer_and_helpers.py  # Experience replay buffer
├── utils.py               # Model factory & utilities
│
├── maddpg/                # MADDPG implementation
│   ├── agents.py          # Agent class
│   └── maddpg.py          # Algorithm update logic
│
├── matd3/                 # MATD3 implementation
├── mappo/                 # MAPPO implementation
├── masac/                 # MASAC implementation
│
├── attention_maddpg/      # MADDPG + Attention
│   ├── agents.py
│   └── attention_maddpg.py
│
├── attention_matd3/       # MATD3 + Attention
├── attention_mappo/       # MAPPO + Attention
├── attention_masac/       # MASAC + Attention
│
└── random_baseline/       # Random baseline
    └── random_model.py
```

## 🔧 How to Use

### Train with Specific Algorithm

```python
# Method 1: Command line
python train.py --model maddpg --episodes 5000

# Method 2: In Python
from train import train_agent
config = get_config()
config.MODEL = "attention_maddpg"
train_agent(config)
```

### Switch Between Algorithms

Simply change `MODEL` in [config.py](../config.py):

```python
# Choose one:
MODEL = "maddpg"              # Off-policy baseline
MODEL = "matd3"               # Twin delayed DDPG
MODEL = "mappo"               # On-policy PPO
MODEL = "masac"               # Soft Actor-Critic
MODEL = "attention_maddpg"    # MADDPG + attention
MODEL = "attention_matd3"     # MATD3 + attention
MODEL = "attention_mappo"     # MAPPO + attention
MODEL = "attention_masac"     # MASAC + attention
MODEL = "random"              # Random baseline
```

## 🎯 Tuning Hyperparameters

### Stage 1: Reward Optimization
No algorithm-specific tuning, just reward weights:
```bash
python tune.py --stage 1 --model maddpg --episodes 500 --trials 50
```

### Stage 2: Algorithm Hyperparameters
Tunes learning rates, network size, batch size, discount factor:
```bash
python tune.py --stage 2 --model maddpg --episodes 1000 --trials 50
```

### Stage 3: Attention Architecture (Attention Models Only)
Optimize attention dimension and heads:
```bash
python tune.py --stage 3 --model attention_maddpg --episodes 500 --trials 30
```

## 🔍 Comparing Algorithms

```bash
# Train multiple algorithms
python train.py --model maddpg --episodes 5000
python train.py --model masac --episodes 5000
python train.py --model attention_maddpg --episodes 5000

# Compare results
python utils/comparative_plots.py \
  --logs train_logs/maddpg train_logs/masac/ \
  --names MADDPG MASAC \
  --smoothing 10
```

Generates comparison plots showing:
- Reward curves (learning progress)
- Latency (task completion time)
- Energy consumption
- Fairness (equal service)
- Offline rate (battery health)
- Loss curves (training stability)

Refer [Plotting Module](/docs/PLOTTING_MODULE.md) for detailed plotting plan.