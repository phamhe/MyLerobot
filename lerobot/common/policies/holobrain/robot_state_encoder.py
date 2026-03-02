"""HoloBrain Robot State Encoder.

Phase 1: Input (B, 1, state_dim) → output (B, chunk_size, embed_dims).
Phase 4 (use_urdf=True): Projects EEF state to virtual joint tokens,
applies JointGraphAttention with URDF-derived distances.
"""

import torch
from torch import nn

from lerobot.common.policies.holobrain.layers import (
    FFN,
    JointGraphAttention,
    RotaryAttention,
    linear_act_ln,
)


class HoloBrainRobotStateEncoder(nn.Module):
    """Encodes robot state into feature tokens for the action decoder.

    Phase 1 (use_urdf=False):
      - Input: (B, n_obs_steps, state_dim)
      - Chunks temporal dim, applies Transformer layers
      - Output: (B, num_chunks, embed_dims)

    Phase 4 (use_urdf=True):
      - Input: (B, n_obs_steps, state_dim)
      - Projects to virtual joint tokens: (B, num_joints, embed_dims)
      - Applies JointGraphAttention with URDF joint distances
      - Output: (B, num_joints, 1, embed_dims) — (joint, chunk) layout
    """

    def __init__(
        self,
        state_dim: int = 8,
        embed_dims: int = 256,
        chunk_size: int = 4,
        num_layers: int = 4,
        num_heads: int = 8,
        feedforward_channels: int = 2048,
        use_urdf: bool = False,
        num_joints: int = 7,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.use_urdf = use_urdf
        self.num_joints = num_joints

        if use_urdf:
            # Phase 4: project EEF state to virtual joint tokens
            self.joint_proj = nn.Sequential(
                nn.Linear(state_dim, embed_dims),
                nn.SiLU(),
                nn.Linear(embed_dims, num_joints * embed_dims),
            )

            # Transformer layers with JointGraphAttention
            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                self.layers.append(nn.ModuleDict({
                    "norm1": nn.RMSNorm(embed_dims),
                    "joint_attn": JointGraphAttention(
                        embed_dims, num_heads=num_heads,
                    ),
                    "norm2": nn.RMSNorm(embed_dims),
                    "ffn": FFN(embed_dims, feedforward_channels, act="silu"),
                }))
        else:
            # Phase 1: temporal self-attention
            self.input_fc = nn.Sequential(
                *linear_act_ln(embed_dims, 2, 2, state_dim * chunk_size, act="silu"),
                nn.Linear(embed_dims, embed_dims),
            )

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
        joint_relative_pos: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            robot_state: (B, n_obs_steps, state_dim)
            joint_relative_pos: (num_joints, num_joints) or (B, num_joints, num_joints)
                pairwise joint distances (only used when use_urdf=True)
        Returns:
            Phase 1: (B, num_chunks, embed_dims)
            Phase 4: (B, num_joints, 1, embed_dims)
        """
        if self.use_urdf:
            return self._forward_urdf(robot_state, joint_relative_pos)
        else:
            return self._forward_phase1(robot_state)

    def _forward_urdf(
        self,
        robot_state: torch.Tensor,
        joint_relative_pos: torch.Tensor | None,
    ) -> torch.Tensor:
        """Phase 4: EEF state → virtual joint tokens → JointGraphAttention."""
        B = robot_state.shape[0]

        # Take latest observation if multiple steps
        if robot_state.dim() == 3:
            state = robot_state[:, -1]  # (B, state_dim)
        else:
            state = robot_state

        # Project to virtual joint tokens: (B, state_dim) → (B, num_joints * embed_dims)
        x = self.joint_proj(state)
        x = x.reshape(B, self.num_joints, self.embed_dims)  # (B, num_joints, embed_dims)

        # Expand joint_relative_pos for batch
        if joint_relative_pos is not None and joint_relative_pos.dim() == 2:
            joint_relative_pos = joint_relative_pos.unsqueeze(0).expand(B, -1, -1)

        # Transformer layers with JointGraphAttention
        for layer in self.layers:
            identity = x
            x = layer["norm1"](x)
            x = layer["joint_attn"](
                query=x,
                key=x,
                value=x,
                query_pos=joint_relative_pos,
                identity=identity,
            )
            identity = x
            x = layer["norm2"](x)
            x = layer["ffn"](x, identity=identity)

        # Add chunk dimension: (B, num_joints, embed_dims) → (B, num_joints, 1, embed_dims)
        return x.unsqueeze(2)

    def _forward_phase1(self, robot_state: torch.Tensor) -> torch.Tensor:
        """Phase 1: temporal self-attention (original behavior)."""
        B = robot_state.shape[0]

        # Pad/repeat state to chunk_size if n_obs_steps < chunk_size
        n_steps = robot_state.shape[1]
        if n_steps < self.chunk_size:
            pad = robot_state[:, -1:].expand(-1, self.chunk_size - n_steps, -1)
            robot_state = torch.cat([robot_state, pad], dim=1)
            n_steps = self.chunk_size

        num_chunks = n_steps // self.chunk_size
        x = robot_state.reshape(B, num_chunks, self.chunk_size * self.state_dim)
        x = self.input_fc(x)  # (B, num_chunks, embed_dims)

        temp_pos = torch.arange(num_chunks, device=x.device)[None].expand(B, -1)

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

        return x  # (B, num_chunks, embed_dims)
