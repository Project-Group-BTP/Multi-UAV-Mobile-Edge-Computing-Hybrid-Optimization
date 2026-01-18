# Code Documentation: Attention MADDPG

## 1. Overview

**Attention-MADDPG** combines the deterministic policy gradient framework of **MADDPG** with **Graph Attention Networks (GAT)**.

* **Standard MADDPG:** Concatenates all agents' observations into a single large vector for the Critic
* **Attention MADDPG:**
* **Actor:** Uses attention to process its local "graph" of neighbors and UEs (User Equipments).
* **Critic:** Uses a specialized "Multi-Agent Attention" mechanism (inspired by MAAC) to dynamically focus on other agents that matter most to the current agent's Q-value.

## 2. File: `marl_models/attention_maddpg/agents.py`

This file defines the specific neural networks for this algorithm. Crucially, they inherit the heavy lifting (encoding & attention logic) from `marl_models/attention.py`.

### Class: `ActorNetwork`

**Parent:** `AttentionActorBase`

* **`__init__(self, obs_dim, action_dim)`**:
* Initializes the base attention encoder (which handles Self, Neighbor, and UE embeddings).
* Adds a final output layer `self.out` to map the fused embedding to the action space.

* **`forward(self, obs)`**:
* **Input:** `obs` containing the agent's state and its neighbors/UEs.
* **Logic:**
1. Calls `self.get_feature_embedding(obs)` (defined in Base) to extract a context-aware feature vector.
2. Passes this vector through `self.out`.
3. Applies `tanh` activation to bound actions between `[-1, 1]`.

* **Output:** Deterministic action tensor.

### Class: `CriticNetwork`

**Parent:** `AttentionCriticBase`

* **`__init__(self, obs_dim, action_dim)`**:
* Initializes the base critic attention (which handles attending to *other* agents).
* Adds `self.q_head`: An MLP that takes the context vector and outputs a single scalar (Q-value).

* **`forward(self, obs_tensor, action_tensor, agent_index)`**:
* **Input:** The full swarm's observations and actions.
* **Logic:**
1. Calls `self.get_q_embedding(...)` (defined in Base). This calculates how much every *other* agent influences `agent_index`.
2. Passes the resulting embedding through `self.q_head`.

* **Output:** The Q-value  for the specific agent.


## 3. File: `marl_models/attention_maddpg/attention_maddpg.py`

This is the main algorithm class that orchestrates training.

### Class: `AttentionMADDPG`

**Parent:** `MARLModel`

### **Key Differences from Standard MADDPG**

The most critical change is in how data batches are handled in the `update` function.

* **Standard MADDPG:** Flattens the batch.
* Input shape: `(Batch_Size, Num_Agents * Obs_Dim)`

* **Attention MADDPG:** Preserves dimensions.
* Input shape: `(Batch_Size, Num_Agents, Obs_Dim)`
* *Reason:* The Attention Critic needs to iterate over the `Num_Agents` dimension to treat other agents as "Keys/Values".


### **Methods**

**`update(self, batch)`**

1. **Data Preparation:**
* Retrieves the batch as a tuple of tensors.
* **Crucial:** It explicitly *does not* flatten the observations or actions, keeping them as 3D tensors `(Batch, N, Dim)`.

2. **Critic Update:**
* **Target Q-Value:**
* Feeds `next_obs` into the `Target Actor` to get `next_actions`.
* Feeds `next_obs` and `next_actions` into the `Target Critic`.
* Computes Bellman target: .

* **Current Q-Value:**
* Feeds current `obs` and `actions` into the `Critic`.

* **Loss:** MSE Loss between Current Q and Target Q.

3. **Actor Update:**
* **Action Prediction:** Feeds current `obs` into the `Actor` to get predicted actions (distinct from the sampled actions in the replay buffer).
* **Evaluation:** The `Critic` evaluates these predicted actions.
* **Loss:** Maximizes the Q-value (Gradient Ascent via negative mean).

4. **Target Update:**
* Performs `soft_update` (Polyak averaging) to slowly track the main networks.

## 3. MATD3, MASAC

Attention version of MATD3 and MASAC have been similarly implemented by changing the last layers of Actor networks and making necessary changes in algorithm files to accommodate the twin critics (for MATD3) and entropy terms (for MASAC). The data handling in the `update` functions remains the same as described above for Attention MADDPG.
