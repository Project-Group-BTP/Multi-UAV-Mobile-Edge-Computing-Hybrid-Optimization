# Code Documentation: MAPPO (Standard)

## **1. Overview**

**MAPPO** is an **on-policy** algorithm adapted for multi-agent environments. Unlike DDPG-based methods (which use a Replay Buffer), MAPPO learns a stochastic policy directly from the current trajectory and uses a Value function to reduce variance.

**Key Characteristics:**

* **On-Policy:** Learning happens only on the data collected by the *current* policy. Old data is discarded after every update.
* **Centralized Training, Decentralized Execution (CTDE):**
* **PPO Clipping:** Uses the clipped surrogate objective to prevent the policy from changing too drastically in a single update step.

## 2. File: `marl_models/mappo/agents.py`

This file defines the Neural Networks.

### **Class: `ActorNetwork**`

**Type:** Stochastic Gaussian Policy.

* **`__init__(self, obs_dim, action_dim)`**:
* **Input:** Local observation dimension.
* **Layers:** A standard MLP (`fc1`  `fc2`).
* **Heads:**
* `self.mean`: Outputs the mean action .
* `self.log_std`: Learnable parameter for standard deviation .

* **`forward(self, obs)`**:
* **Input:** `obs` tensor.
* **Output:** Returns a `torch.distributions.Normal` distribution object.
* **Logic:**
1. Pass `obs` through MLP.
2. Compute `mean` (Tanh activation keeps it in ).
3. Clamp `log_std` to prevent numerical instability.
4. Create Normal distribution .

### Class: `CriticNetwork`

**Type:** Value Function .

* **`__init__(self, state_dim)`**:
* **Input:** Global State dimension (`Obs_Dim * Num_Agents`).


* **`forward(self, state)`**:
* **Input:** Flattened global state.
* **Output:** A single scalar value representing how "good" the current state is.


## 3. File: `marl_models/mappo/mappo.py`

This is the main agent class that handles interaction and training.

### Key Methods

#### `get_action_and_value(self, obs, state)`

* **Purpose:** Called during the rollout phase (in `train.py`) to collect data.
* **Arguments:**
* `obs`: Local observations (for Actor).
* `state`: Global state (for Critic).

* **Logic:**
1. **Actor:** Passes `obs` to `ActorNetwork` and gets Distribution and Samples Action.
2. **Log Probs:** Calculates the log-probability of the chosen action (essential for PPO ratio).
3. **Critic:** Passes `state` to `CriticNetwork` and gets Value.

* **Returns:** `(actions, log_probs, values)` to be stored in the buffer.

#### `update(self, batch)`

* **Purpose:** Performs the PPO update using collected trajectories.
* **Input:** A dictionary `batch` from `RolloutBuffer`.
* **Process:**
1. **Advantage Normalization:** Normalizes advantages to stabilize training.
2. **Critic Update:**
* Calculates  using current network.
* **Value Clipping:** Ensures the value function doesn't drift too far from the old values (a PPO trick).

3. **Actor Update:**
* Calculates new `log_probs` for the actions in the batch.
* Uses PPO surrogate loss with clipping.
* **Entropy Bonus:** Adds entropy term to loss to encourage exploration.

## 4. Buffer Interaction & Design

MAPPO interacts differently with memory compared to DDPG/SAC. It uses the `RolloutBuffer` defined in `marl_models/buffer_and_helpers.py`.

### The "Rollout" Lifecycle

1. **Collection Phase (in `train_on_policy` function):**
* The code runs the environment for `STEPS_PER_EPISODE`.
* At each step, it calls `mappo.get_action_and_value()`.
* **Storage:** It stores `(State, Obs, Action, Reward, Done, Value, Log_Prob)` into the `RolloutBuffer` via `buffer.add()`.


2. **GAE Computation Phase:**
* Before updating, the code calls `buffer.compute_returns_and_advantages()`.
* **Logic:** It iterates backwards through the buffer to calculate **Generalized Advantage Estimation (GAE)**.
* This provides a high-quality, lower-variance target for the Actor to learn from.

3. **Learning Phase (in `update`):**
* The buffer yields **Mini-batches** of the collected data (shuffled).
* MAPPO performs `PPO_EPOCHS` (e.g., 10) passes over this same data to squeeze out as much learning as possible.

4. **Cleanup:**
* After `update()`, the `RolloutBuffer` is **cleared** (`buffer.clear()`). Old experiences are never reused.
