"""HoloBrain Action Decoder — Phase 1 simplified.

6-layer DiT Transformer decoder with UpsampleHead.
Phase 1 simplifications:
  - No text cross-attention (no language condition)
  - TemporalJointGraphAttention → TemporalSelfAttention (no URDF)
  - Standard Gaussian noise (not local_joint)
  - num_joint=1 (LIBERO treats entire action as single token per timestep)
"""

import torch
import torch.nn.functional as F
from torch import nn

from lerobot.common.policies.holobrain.layers import (
    AdaRMSNorm,
    FFN,
    RotaryAttention,
    ScalarEmbedder,
    TemporalSelfAttention,
    UpsampleHead,
    linear_act_ln,
)


class HoloBrainActionDecoder(nn.Module):
    """DiT-style diffusion Transformer decoder for action prediction.

    Each layer: t_norm → temp_self_attn → gate_msa → norm → img_cross_attn →
                norm → scale_shift → ffn → gate_mlp

    Phase 1: single "joint" (num_joint=1), no text, no URDF.
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
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.action_dim = action_dim
        self.pred_steps = pred_steps
        self.chunk_size = chunk_size
        self.num_chunks = pred_steps // chunk_size

        # Input projection: chunk_size * action_dim → embed_dims
        self.input_layers = nn.Sequential(
            nn.Linear(chunk_size * action_dim, embed_dims),
            *linear_act_ln(embed_dims, 2, 2, act="silu"),
        )

        # Timestep embedding
        self.t_embed = ScalarEmbedder(embed_dims, frequency_embedding_size=256)

        # Transformer layers
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_layers.append(nn.ModuleDict({
                "t_norm": AdaRMSNorm(embed_dims, condition_dims=embed_dims, zero=True),
                "temp_self_attn": TemporalSelfAttention(
                    embed_dims, num_heads=num_heads,
                    max_position_embeddings=128,
                ),
                "norm_img": nn.RMSNorm(embed_dims),
                "img_cross_attn": RotaryAttention(
                    embed_dims, num_heads=num_heads,
                    max_position_embeddings=512,
                ),
                "norm_ffn": nn.RMSNorm(embed_dims),
                "ffn": FFN(embed_dims, feedforward_channels, act="silu"),
            }))

        # Output head: upsample from num_chunks back to pred_steps
        # Two-stage: num_chunks → num_chunks*2 → pred_steps
        mid_size = self.num_chunks * 2
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
    ) -> torch.Tensor:
        """
        Args:
            noisy_action: (B, pred_steps, action_dim) noisy action sequence
            timesteps: (B,) diffusion timesteps
            img_features: (B, N_img, embed_dims) flattened image features
            state_features: (B, N_state, embed_dims) robot state features
        Returns:
            (B, pred_steps, action_dim) denoised prediction
        """
        B = noisy_action.shape[0]

        # Chunk the action sequence: (B, num_chunks, chunk_size * action_dim)
        x = noisy_action.reshape(B, self.num_chunks, self.chunk_size * self.action_dim)
        x = self.input_layers(x)  # (B, num_chunks, embed_dims)

        # Timestep embedding
        t_embed = self.t_embed(timesteps.float())  # (B, embed_dims)

        # Prepare temporal positions for causal attention
        num_hist = state_features.shape[1] if state_features is not None else 0
        temp_pos_q = (
            torch.arange(self.num_chunks, device=x.device)[None].expand(B, -1) + num_hist
        )
        temp_pos_k = torch.arange(
            num_hist + self.num_chunks, device=x.device
        )[None].expand(B, -1)

        # Causal mask: each query chunk can only attend to itself and earlier
        causal_mask = ~torch.tril(
            torch.ones(self.num_chunks, num_hist + self.num_chunks,
                       dtype=torch.bool, device=x.device),
            diagonal=num_hist,
        )  # (num_chunks, num_hist + num_chunks), True = masked

        # Image cross-attention positions
        ica_query_pos = torch.arange(self.num_chunks, device=x.device)[None].expand(B, -1) + 1

        # Transformer layers
        for layer in self.decoder_layers:
            # 1. AdaRMSNorm (timestep conditioning)
            identity = x
            x, gate_msa, shift_mlp, scale_mlp, gate_mlp = layer["t_norm"](x, t_embed)

            # 2. Temporal self-attention with state features as context
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

            # Apply gate_msa
            if gate_msa is not None:
                x = gate_msa * x

            # 3. Image cross-attention
            identity = x
            x = layer["norm_img"](x)
            x = layer["img_cross_attn"](
                query=x,
                key=img_features,
                value=img_features,
                query_pos=ica_query_pos,
                identity=identity,
            )

            # 4. FFN with scale_shift
            identity = x
            x = layer["norm_ffn"](x)
            if scale_mlp is not None:
                x = x * (1 + scale_mlp) + shift_mlp
            x = layer["ffn"](x, identity=identity)
            if gate_mlp is not None:
                x = gate_mlp * x

        # Reshape for UpsampleHead: (B, 1, num_chunks, embed_dims)
        # num_joint=1 for Phase 1
        x = x.unsqueeze(1)
        pred = self.head(x)  # (B, 1, pred_steps, action_dim)
        pred = pred.squeeze(1)  # (B, pred_steps, action_dim)

        return pred
