import config
from marl_models.buffer_and_helpers import layer_init
import torch
import torch.nn as nn
from torch.nn import functional as F


class CrossAttentionExtractor(nn.Module):
    """
    Cross-Attention Module used in actors and critics of attention-based models
    Standard Scaled Dot-Product Attention.
    Inputs:
        - self_embedding: The 'Query' (Agent's own state)
        - target_embeddings: The 'Keys/Values' (Neighbors or UEs)
    """

    def __init__(self, self_dim: int, target_dim: int) -> None:
        super().__init__()
        self.head_dim: int = config.ATTN_HIDDEN_DIM // config.ATTN_NUM_HEADS
        assert config.ATTN_HIDDEN_DIM % config.ATTN_NUM_HEADS == 0, "hidden_dim must be divisible by num_heads"
        self.query_layer: nn.Linear = layer_init(nn.Linear(self_dim, config.ATTN_HIDDEN_DIM))

        self.key_layer: nn.Linear = layer_init(nn.Linear(target_dim, config.ATTN_HIDDEN_DIM))
        self.value_layer: nn.Linear = layer_init(nn.Linear(target_dim, config.ATTN_HIDDEN_DIM))
        self.scale: float = config.ATTN_HIDDEN_DIM ** (-0.5)  # Scaling factor for dot-product attention (1 / sqrt(d_k))
        self.out_proj: nn.Linear = layer_init(nn.Linear(config.ATTN_HIDDEN_DIM, config.ATTN_HIDDEN_DIM))

    def forward(self, self_embedding: torch.Tensor, target_embeddings: torch.Tensor, mask: torch.Tensor | None = None):
        # self_embedding: (batch, self_dim)
        # target_embeddings: (batch, max_targets, target_dim)
        batch_size: int = self_embedding.shape[0]

        # Linear Projections & Split Heads
        # Q: (batch, 1, hidden) -> (batch, 1, num_heads, head_dim) -> (batch, num_heads, 1, head_dim)
        Q: torch.Tensor = self.query_layer(self_embedding).unsqueeze(1).view(batch_size, 1, config.ATTN_NUM_HEADS, self.head_dim).transpose(1, 2)

        # K, V: (batch, num_targets, hidden) -> (batch, num_targets, num_heads, head_dim) -> (batch, num_heads, num_targets, head_dim)
        K: torch.Tensor = self.key_layer(target_embeddings).view(batch_size, -1, config.ATTN_NUM_HEADS, self.head_dim).transpose(1, 2)
        V: torch.Tensor = self.value_layer(target_embeddings).view(batch_size, -1, config.ATTN_NUM_HEADS, self.head_dim).transpose(1, 2)
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

        # Final Projection
        output = self.out_proj(context)
        return output.squeeze(1)


class AttentionActorBase(nn.Module):
    """
    Base class for Attention-based Actors (Shared by MADDPG, MATD3, MASAC, MAPPO).
    Handles encoding, attention, and feature fusion.
    """

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.num_neighbors: int = config.MAX_UAV_NEIGHBORS
        self.neighbor_obs_dim: int = config.NEIGHBOR_OBS_DIM
        self.num_ues: int = config.MAX_ASSOCIATED_UES
        self.ue_obs_dim: int = config.UE_OBS_DIM
        self.hidden_dim: int = config.ATTN_HIDDEN_DIM
        self.mlp_dim: int = config.MLP_HIDDEN_DIM

        # Slicing flattened input
        self.neighbor_block_size: int = self.num_neighbors * self.neighbor_obs_dim
        self.ue_block_size: int = self.num_ues * self.ue_obs_dim
        self.own_dim: int = obs_dim - self.neighbor_block_size - self.ue_block_size

        # Feature Encoders (The "Embedding" Layers)
        # These project raw inputs (x, y, battery) into the shared hidden_dim
        # nn.Linear instead of nn.Embedding since inputs are continuous
        self.self_encoder: nn.Sequential = nn.Sequential(layer_init(nn.Linear(self.own_dim, self.hidden_dim)), nn.LayerNorm(self.hidden_dim), nn.ReLU())
        self.neighbor_encoder: nn.Sequential = nn.Sequential(layer_init(nn.Linear(self.neighbor_obs_dim, self.hidden_dim)), nn.LayerNorm(self.hidden_dim), nn.ReLU())
        self.ue_encoder: nn.Sequential = nn.Sequential(layer_init(nn.Linear(self.ue_obs_dim, self.hidden_dim)), nn.LayerNorm(self.hidden_dim), nn.ReLU())

        # Cross-Attention Modules
        self.neighbor_attn: CrossAttentionExtractor = CrossAttentionExtractor(self_dim=self.hidden_dim, target_dim=self.hidden_dim)
        self.ue_attn: CrossAttentionExtractor = CrossAttentionExtractor(self_dim=self.hidden_dim, target_dim=self.hidden_dim)

        # Fusion Layer
        self.fusion_dim: int = self.hidden_dim * 3
        self.fc1: nn.Linear = layer_init(nn.Linear(self.fusion_dim, self.mlp_dim))
        self.ln1: nn.LayerNorm = nn.LayerNorm(self.mlp_dim)
        self.fc2: nn.Linear = layer_init(nn.Linear(self.mlp_dim, self.hidden_dim))
        self.ln2: nn.LayerNorm = nn.LayerNorm(self.hidden_dim)

    def get_feature_embedding(self, obs_flat: torch.Tensor) -> torch.Tensor:
        batch_size: int = obs_flat.shape[0]
        own_state: torch.Tensor = obs_flat[:, : self.own_dim]
        neighbor_part: torch.Tensor = obs_flat[:, self.own_dim : self.own_dim + self.neighbor_block_size]
        neighbor_states: torch.Tensor = neighbor_part.reshape(batch_size, self.num_neighbors, self.neighbor_obs_dim)
        ue_part: torch.Tensor = obs_flat[:, self.own_dim + self.neighbor_block_size :]
        ue_states: torch.Tensor = ue_part.reshape(batch_size, self.num_ues, self.ue_obs_dim)

        # Generate Masks (0 for padding, 1 for real)
        # If absolute sum of features is 0 (or close), it's padding.
        neighbor_mask: torch.Tensor = (torch.abs(neighbor_states).sum(dim=-1) > 1e-5).float()
        ue_mask: torch.Tensor = (torch.abs(ue_states).sum(dim=-1) > 1e-5).float()

        # Encoding (Creating Embeddings)
        self_emb: torch.Tensor = self.self_encoder(own_state)
        neighbor_embs: torch.Tensor = self.neighbor_encoder(neighbor_states)
        ue_embs: torch.Tensor = self.ue_encoder(ue_states)

        # Attention
        neighbor_context: torch.Tensor = self.neighbor_attn(self_emb, neighbor_embs, mask=neighbor_mask)
        ue_context: torch.Tensor = self.ue_attn(self_emb, ue_embs, mask=ue_mask)

        # Fusion
        combined: torch.Tensor = torch.cat([self_emb, neighbor_context, ue_context], dim=1)
        fusion: torch.Tensor = F.relu(self.ln1(self.fc1(combined)))
        fusion = F.relu(self.ln2(self.fc2(fusion)))
        return fusion


class AttentionCriticBase(nn.Module):
    """Base class for Attention-based Critics (Inspired from MAAC)"""

    def __init__(self, obs_dim: int, action_dim: int = 0) -> None:
        super().__init__()
        self.hidden_dim: int = config.ATTN_HIDDEN_DIM
        self.num_heads: int = config.ATTN_NUM_HEADS
        self.mlp_dim: int = config.MLP_HIDDEN_DIM

        # Feature Extraction
        # Input: [Obs + Action] for Q-Critics (MADDPG/MATD3/MASAC)
        # Input: [Obs] for V-Critics (MAPPO)
        input_dim: int = obs_dim + action_dim

        self.state_encoder: nn.Sequential = nn.Sequential(layer_init(nn.Linear(input_dim, self.mlp_dim)), nn.LayerNorm(self.mlp_dim), nn.ReLU(), layer_init(nn.Linear(self.mlp_dim, self.hidden_dim)), nn.LayerNorm(self.hidden_dim), nn.ReLU())

        self.attention: CrossAttentionExtractor = CrossAttentionExtractor(self_dim=self.hidden_dim, target_dim=self.hidden_dim)

        self.fusion_dim: int = self.hidden_dim * 2

    def get_eval_embedding(self, obs_tensor: torch.Tensor, action_tensor: torch.Tensor | None, agent_index: int) -> torch.Tensor:
        """
        Calculates the embedding for agent i by attending to all other agents.
        Args:
            obs_tensor: (Batch, Num_Agents, Obs_Dim)
            action_tensor: (Batch, Num_Agents, Action_Dim) OR None (for MAPPO)
            agent_index: The index of the agent we are critiquing
        """
        num_agents: int = obs_tensor.shape[1]

        inputs: torch.Tensor = obs_tensor  # V-Critic: Just State
        if action_tensor is not None:  # Q-Critic: Concatenate State + Action
            inputs = torch.cat([obs_tensor, action_tensor], dim=2)

        # Encode EVERYONE (Batch, Num_Agents, Hidden)
        embeddings: torch.Tensor = self.state_encoder(inputs)

        # Extract "Me"
        me_embedding: torch.Tensor = embeddings[:, agent_index, :]  # (Batch, Hidden)

        # Extract "Others" (Masking logic)
        # We need to exclude 'agent_index' from the attention targets
        mask: torch.Tensor = torch.ones(num_agents, dtype=torch.bool, device=obs_tensor.device)
        mask[agent_index] = False
        others_embeddings: torch.Tensor = embeddings[:, mask, :]  # (Batch, Num_Agents-1, Hidden)

        # Attention
        # "Me" asks "How do the others affect my value?"
        # No mask needed for 'others' (we assume all agents in training are real)
        others_context: torch.Tensor = self.attention(me_embedding, others_embeddings)

        # Fusion
        combined = torch.cat([me_embedding, others_context], dim=1)
        return combined
