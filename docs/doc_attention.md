# Code Documentation: `marl_models/attention.py`

## 1. Overview

This module implements the **Graph Attention Network (GAT)** logic used by the attention-based MARL algorithms (Attention-MADDPG, Attention-MAPPO, etc.).

## 2. Class: `CrossAttentionExtractor`

This is the core building block implementing **Scaled Dot-Product Attention**. It allows a "Query" entity (e.g., the UAV itself) to extract relevant information from a set of "Key/Value" entities (e.g., its neighbors).

### Mathematical Formula

### `__init__(self, self_dim, target_dim)`

Initializes the linear layers required for the attention mechanism.

* **Args:**
* `self_dim`: Dimension of the querying entity's features.
* `target_dim`: Dimension of the target entities' features.


* **Components:**
* `query_layer`: Projects the agent's state into Query space.
* `key_layer`, `value_layer`: Projects neighbor/UE states into Key and Value spaces.
* `scale`: Normalization factor to prevent vanishing gradients.

### `forward(self, self_embedding, target_embeddings, mask=None)`

Performs the forward pass of the attention mechanism.

1. **Projections:** Transforms inputs into Multi-Head Q, K, V matrices.
2. **Scoring:** Computes similarity scores between the agent and every neighbor using Matrix Multiplication.
3. **Masking:** If a `mask` is provided (detecting padding), sets scores of padded entities to -inf so they have zero influence after Softmax.
4. **Softmax:** Converts scores into probabilities (attention weights).
5. **Aggregation:** Computes the weighted sum ofvalues to create the final `context` vector.

## 3. Class: `AttentionActorBase`

This is a base class for the **Actor** (Policy) networks in algorithms like Attention-MADDPG and Attention-MAPPO. It replaces the standard MLP feature extractor.

### **Purpose**

To create a fixed-size embedding vector from a complex, dynamic observation that includes:

1. **Self State:** The UAV's own position, velocity, etc.
2. **Neighbors:** A list of relative positions of other UAVs.
3. **UEs:** A list of requests and positions from ground users.

### **Key Methods**

* **`__init__(self, obs_dim)`**:
* Defines distinct encoders (`nn.Linear` + `LayerNorm` + `ReLU`) for Self, Neighbors, and UEs.
* Initializes two `CrossAttentionExtractor` modules: one for Neighbors and one for UEs.
* Defines a `fusion` layer to combine the outputs.


* **`get_feature_embedding(self, obs_flat)`**:
* **Slicing:** Takes the flattened observation vector provided by the environment and slices it back into its constituent parts: `[Self | Neighbors | UEs]`.
* **Mask Generation:** Automatically detects "padding" (rows of zeros) in the neighbor/UE lists and creates a binary mask to tell the attention layer to ignore them.
* **Encoding & Attention:** Passes valid entities through encoders and then through the attention layers.
* **Fusion:** Concatenates `[Self_Embedding, Neighbor_Context, UE_Context]` and passes them through a final MLP to produce the state representation.

## 4. Class: `AttentionCriticBase`

This is a base class for the **Centralized Critic** networks. It draws inspiration from **MAAC (Multi-Actor Attention-Critic)**.

### **Purpose**

In Multi-Agent RL, a centralized critic needs to evaluate the global state. Simply concatenating all agent states works poorly as the number of agents grows. This class uses attention to let the critic "focus" on the specific agents that are most relevant to the agent being evaluated.

### **Key Methods**

**`get_all_embeddings(self, inputs)`**:
* Encodes the raw state/action pairs of all agents into a hidden representation.


**`attend_to_others(self, embeddings, num_agents, agent_index)`**:
* **Logic:** When evaluating Agent , we treat Agent  as the "Query". All *other* agents are treated as "Keys/Values".
* This answers the question: *"How do the actions of other agents impact Agent ?"*
* Returns a context vector summarizing the influence of the rest of the swarm on Agent .


**`get_q_embedding(self, obs_tensor, action_tensor, agent_index)`**:

A helper method that orchestrates the flow:

1. Concatenates Observations and Actions.
2. Encodes them.
3. Runs `attend_to_others`.
4. Returns the final embedding to be passed to the Q-value head.
