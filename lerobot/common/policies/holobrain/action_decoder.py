"""HoloBrain Action Decoder.

Phase 1: 6-layer DiT + TemporalSelfAttention, num_joint=1.
Phase 4 (use_urdf=True): TemporalJointGraphAttention with URDF distances,
    action dims treated as virtual joints.
"""

import torch
import torch.nn.functional as F
from torch import nn

from lerobot.common.policies.holobrain.layers import (
    AdaRMSNorm,
    FFN,
    RotaryAttention,
    ScalarEmbedder,
    TemporalJointGraphAttention,
    TemporalSelfAttention,
    UpsampleHead,
    linear_act_ln,
)


class HoloBrainActionDecoder(nn.Module):
    """DiT-style diffusion Transformer decoder for action prediction.

    Phase 1 (use_urdf=False):
      - num_joint=1, action treated as single token per timestep
      - TemporalSelfAttention for causal temporal attention

    Phase 4 (use_urdf=True):
      - num_joint=action_dim (e.g., 7 for LIBERO)
      - Each action dimension treated as a virtual joint
      - TemporalJointGraphAttention for combined temporal+joint attention
      - URDF joint_relative_pos provides structural attention bias
    """

    def __init__(
        self,
        embed_dims: int = 256,
        action_dim: int = 7,
        pred_steps: int = 64,
        chunk_size: int = 4,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_channels: int = 2048,
        use_urdf: bool = False,
        num_joints: int = 7,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.action_dim = action_dim
        self.pred_steps = pred_steps
        self.chunk_size = chunk_size
        self.num_chunks = pred_steps // chunk_size
        self.use_urdf = use_urdf
        self.num_joints = num_joints if use_urdf else 1

        if use_urdf:
            # Phase 4: each action dim = 1 virtual joint
            # Input per joint per chunk: chunk_size * 1
            self.input_layers = nn.Sequential(
                nn.Linear(chunk_size * 1, embed_dims),
                *linear_act_ln(embed_dims, 2, 2, act="silu"),
            )
        else:
            # Phase 1: all action dims in single token
            self.input_layers = nn.Sequential(
                nn.Linear(chunk_size * action_dim, embed_dims),
                *linear_act_ln(embed_dims, 2, 2, act="silu"),
            )

        # Timestep embedding
        self.t_embed = ScalarEmbedder(embed_dims, frequency_embedding_size=256)

        # Transformer layers
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                "t_norm": AdaRMSNorm(embed_dims, condition_dims=embed_dims, zero=True),
                "norm_img": nn.RMSNorm(embed_dims),
                "img_cross_attn": RotaryAttention(
                    embed_dims, num_heads=num_heads,
                    max_position_embeddings=512,
                ),
                "norm_ffn": nn.RMSNorm(embed_dims),
                "ffn": FFN(embed_dims, feedforward_channels, act="silu"),
            })
            if use_urdf:
                layer["temp_joint_attn"] = TemporalJointGraphAttention(
                    embed_dims, num_heads=num_heads,
                    max_position_embeddings=128,
                )
            else:
                layer["temp_self_attn"] = TemporalSelfAttention(
                    embed_dims, num_heads=num_heads,
                    max_position_embeddings=128,
                )
            self.decoder_layers.append(layer)

        # Output head
        mid_size = self.num_chunks * 2
        if use_urdf:
            # Per-joint output: each joint predicts 1 action dimension
            self.head = UpsampleHead(
                upsample_sizes=[mid_size, pred_steps],
                input_dim=embed_dims,
                dims=[embed_dims, embed_dims],
                out_dim=1,
                num_output_layers=2,
            )
        else:
            self.head = UpsampleHead(
                upsample_sizes=[mid_size, pred_steps],
                input_dim=embed_dims,
                dims=[embed_dims, embed_dims],
                out_dim=action_dim,
                num_output_layers=2,
            )

    def forward(
        self,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        img_features: torch.Tensor,
        state_features: torch.Tensor | None = None,
        joint_relative_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            noisy_action: (B, pred_steps, action_dim) noisy action sequence
            timesteps: (B,) diffusion timesteps
            img_features: (B, N_img, embed_dims) flattened image features
            state_features:
              Phase 1: (B, N_state, embed_dims)
              Phase 4: (B, num_joints, num_hist_chunk, embed_dims)
            joint_relative_pos: (num_joints, num_joints) or (B, num_joints, num_joints)
        Returns:
            (B, pred_steps, action_dim) denoised prediction
        """
        if self.use_urdf:
            return self._forward_urdf(
                noisy_action, timesteps, img_features,
                state_features, joint_relative_pos
            )
        else:
            return self._forward_phase1(
                noisy_action, timesteps, img_features, state_features
            )

    def _forward_urdf(
        self,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        img_features: torch.Tensor,
        state_features: torch.Tensor | None,
        joint_relative_pos: torch.Tensor | None,
    ) -> torch.Tensor:
        """Phase 4: TemporalJointGraphAttention with URDF distances."""
        B = noisy_action.shape[0]
        num_joint = self.num_joints
        num_chunk = self.num_chunks

        # Reshape action: (B, pred_steps, action_dim) → (B, action_dim, pred_steps, 1)
        # Treat each action dim as a joint, each with 1 scalar value
        x = noisy_action.permute(0, 2, 1).unsqueeze(-1)  # (B, num_joint, pred_steps, 1)
        x = x.reshape(B, num_joint, num_chunk, self.chunk_size * 1)
        x = self.input_layers(x)  # (B, num_joint, num_chunk, embed_dims)

        # Timestep embedding
        t_embed = self.t_embed(timesteps.float())  # (B, embed_dims)

        # Prepare temporal positions
        num_hist_chunk = state_features.shape[2] if state_features is not None else 0
        temp_pos_q = (
            torch.arange(num_chunk, device=x.device)[None].expand(B, -1) + num_hist_chunk
        )
        temp_pos_k = torch.arange(
            num_hist_chunk + num_chunk, device=x.device
        )[None].expand(B, -1)

        # Causal mask
        causal_mask = ~torch.tril(
            torch.ones(num_chunk, num_hist_chunk + num_chunk,
                       dtype=torch.bool, device=x.device),
            diagonal=num_hist_chunk,
        )

        # Expand joint_relative_pos for batch
        if joint_relative_pos is not None and joint_relative_pos.dim() == 2:
            joint_relative_pos = joint_relative_pos.unsqueeze(0).expand(B, -1, -1)

        # Image cross-attention positions
        ica_query_pos = torch.arange(num_chunk, device=x.device)[None, None]
        ica_query_pos = ica_query_pos.tile(B, num_joint, 1).flatten(1, 2) + 1
        # (B, num_joint * num_chunk)

        # Transformer layers
        for layer in self.decoder_layers:
            # 1. AdaRMSNorm (timestep conditioning)
            identity_flat = x.reshape(B, num_joint * num_chunk, -1)
            x_flat, gate_msa, shift_mlp, scale_mlp, gate_mlp = layer["t_norm"](
                identity_flat, t_embed
            )
            x = x_flat.reshape(B, num_joint, num_chunk, -1)

            # 2. TemporalJointGraphAttention
            identity_4d = identity_flat.reshape(B, num_joint, num_chunk, -1)
            if state_features is not None:
                kv = torch.cat([state_features, x], dim=2)
            else:
                kv = x
            x = layer["temp_joint_attn"](
                query=x,
                key=kv,
                joint_distance=joint_relative_pos,
                temporal_pos_q=temp_pos_q,
                temporal_pos_k=temp_pos_k,
                temporal_attn_mask=causal_mask,
                identity=identity_4d,
            )

            # Apply gate_msa — (B, 1, embed_dims) broadcasts to (B, num_joint, num_chunk, embed_dims)
            if gate_msa is not None:
                x = gate_msa[:, :, None, :] * x

            # 3. Image cross-attention (flatten joint+chunk dims)
            x_flat = x.reshape(B, num_joint * num_chunk, -1)
            identity_flat = x_flat
            x_flat = layer["norm_img"](x_flat)
            x_flat = layer["img_cross_attn"](
                query=x_flat,
                key=img_features,
                value=img_features,
                query_pos=ica_query_pos,
                identity=identity_flat,
            )

            # 4. FFN with scale_shift
            identity_flat = x_flat
            x_flat = layer["norm_ffn"](x_flat)
            if scale_mlp is not None:
                x_flat = x_flat * (1 + scale_mlp) + shift_mlp
            x_flat = layer["ffn"](x_flat, identity=identity_flat)
            if gate_mlp is not None:
                x_flat = gate_mlp * x_flat

            x = x_flat.reshape(B, num_joint, num_chunk, -1)

        # Output head: (B, num_joint, num_chunk, embed_dims) → (B, num_joint, pred_steps, 1)
        pred = self.head(x)
        # → (B, pred_steps, num_joint) by permute + squeeze
        pred = pred.squeeze(-1).permute(0, 2, 1)  # (B, pred_steps, num_joint=action_dim)

        return pred

    def _forward_phase1(
        self,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        img_features: torch.Tensor,
        state_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Phase 1: TemporalSelfAttention, num_joint=1 (original behavior)."""
        B = noisy_action.shape[0]

        # Chunk: (B, num_chunks, chunk_size * action_dim)
        x = noisy_action.reshape(B, self.num_chunks, self.chunk_size * self.action_dim)
        x = self.input_layers(x)

        t_embed = self.t_embed(timesteps.float())

        num_hist = state_features.shape[1] if state_features is not None else 0
        temp_pos_q = (
            torch.arange(self.num_chunks, device=x.device)[None].expand(B, -1) + num_hist
        )
        temp_pos_k = torch.arange(
            num_hist + self.num_chunks, device=x.device
        )[None].expand(B, -1)

        causal_mask = ~torch.tril(
            torch.ones(self.num_chunks, num_hist + self.num_chunks,
                       dtype=torch.bool, device=x.device),
            diagonal=num_hist,
        )

        ica_query_pos = torch.arange(self.num_chunks, device=x.device)[None].expand(B, -1) + 1

        for layer in self.decoder_layers:
            identity = x
            x, gate_msa, shift_mlp, scale_mlp, gate_mlp = layer["t_norm"](x, t_embed)

            if state_features is not None:
                kv = torch.cat([state_features, x], dim=1)
            else:
                kv = x
            x = layer["temp_self_attn"](
                query=x, key=kv, value=kv,
                temporal_pos_q=temp_pos_q,
                temporal_pos_k=temp_pos_k,
                temporal_attn_mask=causal_mask,
                identity=identity,
            )

            if gate_msa is not None:
                x = gate_msa * x

            identity = x
            x = layer["norm_img"](x)
            x = layer["img_cross_attn"](
                query=x,
                key=img_features,
                value=img_features,
                query_pos=ica_query_pos,
                identity=identity,
            )

            identity = x
            x = layer["norm_ffn"](x)
            if scale_mlp is not None:
                x = x * (1 + scale_mlp) + shift_mlp
            x = layer["ffn"](x, identity=identity)
            if gate_mlp is not None:
                x = gate_mlp * x

        # UpsampleHead: (B, 1, num_chunks, embed_dims)
        x = x.unsqueeze(1)
        pred = self.head(x)
        pred = pred.squeeze(1)

        return pred
