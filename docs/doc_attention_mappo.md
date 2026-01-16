# Code Documentation: Attention MAPPO

## 1. Overview

**Attention-MAPPO** adapts the standard MAPPO algorithm to work with the **Graph Attention** architecture.

## 2. File: `agents.py`

This file defines the Neural Networks used by the agent. They inherit the heavy lifting (encoding & attention) from `marl_models/attention.py`.

### Class: `ActorNetwork`

* **`__init__(self, obs_dim, action_dim)`**:
* Initializes the base class to handle the "Graph" inputs (Self, Neighbors, UEs).
* Adds two specific heads for Gaussian Control:
* `self.mean`: A Linear layer outputting the mean action .
* `self.log_std`: A learnable parameter for the standard deviation  (log-scale), clipped to prevent instability.

* **`forward(self, obs)`**:
* **Input:** `obs` (Batch, Obs_Dim)
* **Logic:**
1. Calls `self.get_feature_embedding(obs)` (from Base) to condense the variable number of neighbors/UEs into a single vector.
2. Computes `mean` (via Tanh to keep it in $[-1, 1]$) and `std`.

* **Output:** Returns a `torch.distributions.Normal` object, allowing sampling and log-prob calculation.

### Class: `CriticNetwork`

* **`__init__(self, obs_dim)`**:
* Calls `super().__init__` with `action_dim=0`. Unlike MADDPG's Q-Critic, a PPO V-Critic only looks at the **State**, not the Actions.
* `v_head`: A generic MLP that maps the attention context to a single scalar value.


* **`forward(self, obs_tensor)`**:
* **Input:** `obs_tensor` of shape `(Batch, Num_Agents, Obs_Dim)`.
* **Logic:**
1. **Shared Encoding:** Calls `get_all_embeddings` once to encode the states of *all* agents in the batch.
2. **Agent-Specific Attention:** Loops through every agent index . For each agent, it calls `attend_to_others` to see how the *other* agents affect Agent's value.
3. **Value Prediction:** Passes the context vector through `v_head`.

* **Output:** A tensor of shape `(Batch, Num_Agents)` containing the value estimates for every agent.


## 3. File: `attention_mappo.py`

This file implements the MAPPO training logic.

### Class: `AttentionMAPPO`

#### **Key Methods**

* **`get_action_and_value(self, obs, state)`**
* **Purpose:** Called during rollout collection (interacting with `Env`). Needs to return actions for the Env and Values/LogProbs for the Buffer.
* **Shape Handling (Crucial):**
* The **Actor** treats agents independently. It expects `(Num_Agents, Obs_Dim)`.
* The **Critic** needs to see the whole swarm to compute attention. It expects `(1, Num_Agents, Obs_Dim)` (a batch of size 1).

**Logic:**

1. Unsqueeze observation for the Critic.
2. Sample actions from Actor.
3. Compute Values from Critic.
4. Return everything as NumPy arrays.

**`update(self, batch)`**

* **Purpose:** The PPO Update Loop.
* **Input:** A dictionary batch containing `obs`, `actions`, `old_log_probs`, `advantages`, etc.
* **Shape Distinction:**
* The batch comes from `AttentionRolloutBuffer`, so the shape is `(Batch_Size, Num_Agents, Dim)`. This buffer overrides `get_batches` method of `RolloutBuffer` class to achieve this. To ensure this buffer is used for `attention_mappo`, we dynamically select it in `train.py`.

**Critic Update:**

* Passes the full `(Batch, Num_Agents, Obs)` tensor to the Critic.
* The Critic handles the attention internally across the `Num_Agents` dimension.
* Loss: Standard MSE Loss between Predicted Value and Returns (with clipping).

**Actor Update:**

* **Flattening:** The Actor doesn't care about "other" agents in the same batch; it only applies *local* attention (neighbors provided inside the obs).
* We flatten `(Batch, Num_Agents, ...)` to `(Batch * Num_Agents, ...)`.
* Loss: Standard PPO Surrogate Loss (Clipped Ratio * Advantage).
* Entropy Bonus: Added to encourage exploration.
