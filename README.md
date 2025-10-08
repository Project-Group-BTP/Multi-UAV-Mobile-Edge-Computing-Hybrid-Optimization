# Multi-UAV Assisted Mobile Edge Computing: A Hybrid Optimization Approach

The primary objective of this research is to develop a framework for a multi-UAV-assisted Mobile Edge Computing (MEC) network. We aim to jointly optimize four interdependent components: task offloading decisions, service caching placement, content caching strategies, and UAV trajectories. The goal is to minimize a weighted sum of service latency and system-wide energy consumption while simultaneously maximizing user fairness.

We are aiming to implement a hybrid optimization approach that combines multi-agent deep reinforcement learning with deterministic but adaptive caching policies. We are trying to create a generic framework that can be used with different models for finding the best-suited one for our purpose. Trying to incorporate modern Python practices and type annotations. Developed using Python 3.12.0 and PyTorch 2.2.2.

Currently included MARL models:
- MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
- MATD3 (Multi-Agent Twin Delayed Deep Deterministic Policy Gradient)
- MAPPO (Multi-Agent Proximal Policy Optimization)
- MASAC (Multi-Agent Soft Actor-Critic)
- Random baseline

Directory Structure:

```
.
├── environment
│   ├── comm_model.py
│   ├── env.py
│   ├── uavs.py
│   └── user_equipments.py
├── marl_models
│   ├── base_model.py
│   ├── buffer.py
│   ├── utils.py
│   ├── maddpg
│   │   ├── agents.py
│   │   └── maddpg.py
│   ├── matd3
│   │   ├── agents.py
│   │   └── matd3.py
│   ├── mappo
│   │   ├── agents.py
│   │   └── mappo.py
│   ├── masac
│   │   ├── agents.py
│   │   └── masac.py
│   └── random_baseline
│       └── random_model.py
├── utils
│   ├── logger.py
│   ├── plot_logs.py
│   └── plot_snapshots.py
├── config.py
├── train.py
├── test.py
├── final_main.py
├── visualize.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

*Added a temporary script to run the environment with random actions and visualize the state.*

Run using:

```bash
python visualize.py
```

**Still under rapid development and may be subject to significant changes.**
