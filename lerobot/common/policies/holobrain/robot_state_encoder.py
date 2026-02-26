"""HoloBrain Robot State Encoder — Phase 1 simplified.

Phase 1: Input (B, 1, state_dim) → output (B, chunk_size, embed_dims).
No URDF, JointGraphAttention degenerates to standard self-attention.
"""

import torch
from torch import nn

from lerobot.common.policies.holobrain.layers import (
    FFN,
    RotaryAttention,
    linear_act_ln,
)


class HoloBrainRobotStateEncoder(nn.Module):
    """Encodes robot state into feature tokens for the action decoder.

    Phase 1 simplification:
      - Input: (B, n_obs_steps, state_dim) — single "joint" treated as 1 token
      - Chunks the temporal dim, applies Transformer layers
      - Output: (B, chunk_size, embed_dims)
    """

    def __init__(
        self,
        state_dim: int = 8,
        embed_dims: int = 256,
        chunk_size: int = 4,
        num_layers: int = 4,
        num_heads: int = 8,
        feedforward_channels: int = 2048,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.chunk_size = chunk_size
        self.state_dim = state_dim

        # Input projection: state_dim * chunk_size → embed_dims
        self.input_fc = nn.Sequential(
            *linear_act_ln(embed_dims, 2, 2, state_dim * chunk_size, act="silu"),
            nn.Linear(embed_dims, embed_dims),
        )

        # Transformer layers: temporal self-attention + FFN
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "norm1": nn.RMSNorm(embed_dims),
                "attn": RotaryAttention(
                    embed_dims, num_heads=num_heads,
                    max_position_embeddings=128,
                ),
                "norm2": nn.RMSNorm(embed_dims),
                "ffn": FFN(embed_dims, feedforward_channels, act="silu"),
            }))

    def forward(
        self,
        robot_state: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            robot_state: (B, n_obs_steps, state_dim)
        Returns:
            (B, chunk_size, embed_dims)
        """
        B = robot_state.shape[0]

        # Pad/repeat state to chunk_size if n_obs_steps < chunk_size
        n_steps = robot_state.shape[1]
        if n_steps < self.chunk_size:
            # Repeat last frame to fill chunk
            pad = robot_state[:, -1:].expand(-1, self.chunk_size - n_steps, -1)
            robot_state = torch.cat([robot_state, pad], dim=1)
            n_steps = self.chunk_size

        num_chunks = n_steps // self.chunk_size
        # Reshape: (B, num_chunks, chunk_size * state_dim)
        x = robot_state.reshape(B, num_chunks, self.chunk_size * self.state_dim)
        x = self.input_fc(x)  # (B, num_chunks, embed_dims)

        # Temporal position ids
        temp_pos = torch.arange(num_chunks, device=x.device)[None].expand(B, -1)

        # Transformer layers
        for layer in self.layers:
            identity = x
            x = layer["norm1"](x)
            x = layer["attn"](
                query=x, key=x, value=x,
                query_pos=temp_pos, key_pos=temp_pos,
                identity=identity,
            )
            identity = x
            x = layer["norm2"](x)
            x = layer["ffn"](x, identity=identity)

        return x  # (B, num_chunks, embed_dims) — typically (B, 1, 256) for Phase 1
