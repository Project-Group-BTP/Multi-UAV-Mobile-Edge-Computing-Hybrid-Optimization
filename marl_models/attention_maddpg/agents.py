import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class CrossAttentionExtractor(nn.Module):
    def __init__(self, self_dim: int, target_dim: int, hidden_dim: int = 64, num_heads: int = 4) -> None:
        super(CrossAttentionExtractor, self).__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        # Query comes from "Self" (UAV)
        self.query_layer: nn.Linear = layer_init(nn.Linear(self_dim, hidden_dim))

        # Key and Value come from "Targets" (Neighbors or UEs)
        self.key_layer: nn.Linear = layer_init(nn.Linear(target_dim, hidden_dim))
        self.value_layer: nn.Linear = layer_init(nn.Linear(target_dim, hidden_dim))
        self.scale: float = hidden_dim ** (-0.5)
        self.out_proj: nn.Linear = layer_init(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, self_embedding: torch.Tensor, target_embeddings: torch.Tensor, mask: torch.Tensor | None = None):
        # self_embedding: (batch, self_dim)
        # target_embeddings: (batch, max_targets, target_dim)
        batch_size: int = self_embedding.shape[0]

        # 1. Linear Projections & Split Heads
        # Q: (batch, 1, hidden) -> (batch, 1, num_heads, head_dim) -> (batch, num_heads, 1, head_dim)
        Q: torch.Tensor = self.query_layer(self_embedding).unsqueeze(1).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # K, V: (batch, num_targets, hidden) -> (batch, num_targets, num_heads, head_dim) -> (batch, num_heads, num_targets, head_dim)
        K: torch.Tensor = self.key_layer(target_embeddings).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V: torch.Tensor = self.value_layer(target_embeddings).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Attention Scores
        # (batch, 1, hidden) @ (batch, hidden, max_targets) -> (batch, 1, max_targets)
        scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Mask padding positions (set score to -infinity so Softmax becomes 0)
            # Mask: (batch, targets) -> (batch, 1, 1, targets)
            mask_expanded: torch.Tensor = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        attn_weights: torch.Tensor = F.softmax(scores, dim=-1)

        # Handle case where all targets are padding (e.g., no neighbors) -> nan check
        attn_weights: torch.Tensor = torch.nan_to_num(attn_weights, nan=0.0)

        # Weighted Sum
        context: torch.Tensor = torch.matmul(attn_weights, V)  # (batch, 1, hidden)

        # (batch, heads, 1, head_dim) -> (batch, 1, heads, head_dim) -> (batch, 1, hidden)
        context = context.transpose(1, 2).reshape(batch_size, 1, -1)

        # 4. Final Projection
        output = self.out_proj(context)
        return output.squeeze(1)


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(ActorNetwork, self).__init__()

        # Define Slices based on Config
        self.num_neighbors: int = config.MAX_UAV_NEIGHBORS
        self.neighbor_feat_dim: int = 2
        self.num_ues: int = config.MAX_ASSOCIATED_UES
        self.ue_feat_dim: int = 6  # (2 pos + 3 req + 1 batt)

        # Calculate start/end indices for slicing the flat input
        # Input structure is assumed to be: [Own_State, Neighbor_Flat, UE_Flat]
        self.neighbor_block_size: int = self.num_neighbors * self.neighbor_feat_dim
        self.ue_block_size: int = self.num_ues * self.ue_feat_dim

        self.own_dim: int = obs_dim - self.neighbor_block_size - self.ue_block_size

        # Feature Encoders (Shared MLPs)
        self.self_encoder: nn.Sequential = nn.Sequential(layer_init(nn.Linear(self.own_dim, 64)), nn.ReLU())
        self.neighbor_encoder: nn.Sequential = nn.Sequential(layer_init(nn.Linear(self.neighbor_feat_dim, 64)), nn.ReLU())
        self.ue_encoder: nn.Sequential = nn.Sequential(layer_init(nn.Linear(self.ue_feat_dim, 64)), nn.ReLU())

        # Cross-Attention Modules
        self.neighbor_attn: CrossAttentionExtractor = CrossAttentionExtractor(self_dim=64, target_dim=64)
        self.ue_attn: CrossAttentionExtractor = CrossAttentionExtractor(self_dim=64, target_dim=64)

        # Action Head
        # Input: Self(64) + Neighbor_Context(64) + UE_Context(64) = 192
        self.fc1: nn.Linear = layer_init(nn.Linear(192, 128))
        self.ln1: nn.LayerNorm = nn.LayerNorm(128)
        self.fc2: nn.Linear = layer_init(nn.Linear(128, 64))
        self.ln2: nn.LayerNorm = nn.LayerNorm(64)
        self.out: nn.Linear = layer_init(nn.Linear(64, action_dim), std=0.01)  # Small std for output

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        batch_size = obs_flat.shape[0]

        # Unflatten / Slice the Input
        own_state: torch.Tensor = obs_flat[:, : self.own_dim]

        neighbor_part: torch.Tensor = obs_flat[:, self.own_dim : self.own_dim + self.neighbor_block_size]
        neighbor_states: torch.Tensor = neighbor_part.reshape(batch_size, self.num_neighbors, self.neighbor_feat_dim)

        ue_part: torch.Tensor = obs_flat[:, self.own_dim + self.neighbor_block_size :]
        ue_states: torch.Tensor = ue_part.reshape(batch_size, self.num_ues, self.ue_feat_dim)

        # Generate Masks (0 for padding, 1 for real)
        # If absolute sum of features is almost 0, treat as padding.
        neighbor_mask: torch.Tensor = (torch.abs(neighbor_states).sum(dim=-1) > 1e-5).float()
        ue_mask: torch.Tensor = (torch.abs(ue_states).sum(dim=-1) > 1e-5).float()

        # Encoding & Attention
        self_emb: torch.Tensor = self.self_encoder(own_state)  # (batch, 64)

        neighbor_embs: torch.Tensor = self.neighbor_encoder(neighbor_states)  # (batch, N, 64)
        ue_embs: torch.Tensor = self.ue_encoder(ue_states)  # (batch, M, 64)

        # Attention
        neighbor_context: torch.Tensor = self.neighbor_attn(self_emb, neighbor_embs, mask=neighbor_mask)
        ue_context: torch.Tensor = self.ue_attn(self_emb, ue_embs, mask=ue_mask)
        # Fusion & Action
        combined: torch.Tensor = torch.cat([self_emb, neighbor_context, ue_context], dim=1)

        x: torch.Tensor = F.relu(self.ln1(self.fc1(combined)))
        x = F.relu(self.ln2(self.fc2(x)))
        return torch.tanh(self.out(x))


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(CriticNetwork, self).__init__()

        # 1. Feature Extraction (Shared across all agents)
        # Input: The concatenated [Obs, Action] of a SINGLE agent
        # We need to calculate the dimensions correctly based on your config
        input_dim: int = obs_dim + action_dim

        self.state_action_encoder: nn.Sequential = nn.Sequential(layer_init(nn.Linear(input_dim, 128)), nn.ReLU(), layer_init(nn.Linear(128, 64)), nn.LayerNorm(64), nn.ReLU())

        # 2. Attention Mechanism (MAAC Style)
        # "Me" (Query) attends to "Others" (Key/Value)
        self.attention: CrossAttentionExtractor = CrossAttentionExtractor(self_dim=64, target_dim=64)

        # 3. Final Q-Value Head
        # Input: Me_Embedding(64) + Weighted_Others(64) = 128
        self.q_head: nn.Sequential = nn.Sequential(layer_init(nn.Linear(128, 128)), nn.LayerNorm(128), nn.ReLU(), layer_init(nn.Linear(128, 1)))  # Output is a single Q-value

    def forward(self, obs_tensor: torch.Tensor, action_tensor: torch.Tensor, agent_index: int) -> torch.Tensor:
        """
        Calculates Q-value for a specific agent 'i' by attending to all other agents.

        Args:
            obs_tensor: (Batch, Num_Agents, Obs_Dim)
            action_tensor: (Batch, Num_Agents, Action_Dim)
            agent_index: The index 'i' of the agent we are critiquing
        """
        num_agents: int = obs_tensor.shape[1]

        # 1. Encode EVERYONE (Batch, Num_Agents, 64)
        # Concatenate State+Action for all agents at once
        inputs: torch.Tensor = torch.cat([obs_tensor, action_tensor], dim=2)
        embeddings: torch.Tensor = self.state_action_encoder(inputs)

        # 2. Extract "Me" (The agent we are calculating Q for)
        me_embedding: torch.Tensor = embeddings[:, agent_index, :]  # (Batch, 64)

        # 3. Extract "Others"
        # We need to exclude 'agent_index' from the attention targets
        # Create a mask of everyone EXCEPT agent_index
        mask: torch.Tensor = torch.ones(num_agents, dtype=torch.bool, device=obs_tensor.device)
        mask[agent_index] = False

        # Select others: (Batch, Num_Agents-1, 64)
        others_embeddings: torch.Tensor = embeddings[:, mask, :]
        # 4. Attention: "Me" asks "How do the others affect my value?"
        # Note: We don't need a padding mask here because Num_Agents is fixed (10)
        others_context: torch.Tensor = self.attention(me_embedding, others_embeddings)

        # 5. Final Q Calculation
        combined: torch.Tensor = torch.cat([me_embedding, others_context], dim=1)
        q_value: torch.Tensor = self.q_head(combined)
        return q_value
