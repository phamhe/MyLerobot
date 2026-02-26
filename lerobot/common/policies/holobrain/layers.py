"""Custom layers for HoloBrain policy.

Ported from RoboOrchardLab/robo_orchard_lab/models/holobrain/layers.py
with robo_orchard_core dependencies removed. Uses standard PyTorch ops.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def linear_act_ln(
    embed_dims: int,
    in_loops: int,
    out_loops: int,
    input_dims: int | None = None,
    act: str = "relu",
):
    """Build stacked Linear+Act+LayerNorm blocks."""
    act_fn = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[act]
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(act_fn(inplace=True) if act == "relu" else act_fn())
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


# ---------------------------------------------------------------------------
# ScalarEmbedder — sinusoidal frequency embedding + MLP
# ---------------------------------------------------------------------------

class ScalarEmbedder(nn.Module):
    """Encodes scalar values (e.g. diffusion timestep, joint distance) via
    sinusoidal frequency embedding followed by a 2-layer MLP."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.freqs * t[:, None]
        t_freq = torch.cat([torch.cos(t_freq), torch.sin(t_freq)], dim=-1)
        return self.mlp(t_freq)


# ---------------------------------------------------------------------------
# Rotary Position Embedding
# ---------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32) / dim
            )
        )
        freqs = torch.arange(max_position_embeddings)[:, None].float() @ inv_freq[None]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        position_ids = position_ids.to(torch.int32)
        cos = self.cos_cached.to(x)[position_ids]
        sin = self.sin_cached.to(x)[position_ids]
        while x.dim() > cos.dim():
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        return (x * cos) + (rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# RotaryAttention — multi-head attention with RoPE
# ---------------------------------------------------------------------------

class RotaryAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding.
    Used for temporal self-attention and cross-attention (img/text)."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        max_position_embeddings: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.v_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.position_encoder = RotaryEmbedding(
            head_dim, max_position_embeddings=max_position_embeddings
        )
        self._init_weights()

    def _init_weights(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        if self.q_proj.bias is not None:
            constant_(self.q_proj.bias, 0.0)
        if self.v_proj.bias is not None:
            constant_(self.v_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        key_pos: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if identity is None:
            identity = query
        B, N, C = query.shape
        M = key.shape[1]
        if value is None:
            value = key

        q = self.q_proj(query).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3)

        if query_pos is not None:
            q = self.position_encoder(q, query_pos)
        if key_pos is not None:
            k = self.position_encoder(k, key_pos)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            if attn_mask.dim() == 3 and attn_mask.shape[0] == B:
                attn_mask = attn_mask.unsqueeze(1)
            attn = torch.where(attn_mask, float("-inf"), attn)

        if key_padding_mask is not None:
            attn = torch.where(
                key_padding_mask[:, None, None], float("-inf"), attn
            )

        attn = attn.softmax(dim=-1).type_as(v)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x + identity


# ---------------------------------------------------------------------------
# JointGraphAttention — self-attention with scalar distance bias
# Phase 1: simplified, joint_relative_pos is optional
# ---------------------------------------------------------------------------

class JointGraphAttention(nn.Module):
    """Joint-dimension self-attention with optional distance-based position
    encoding via ScalarEmbedder.

    In Phase 1 (num_joint=1), this degenerates to standard self-attention.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = head_dim ** -0.5
        self.embed_dims = embed_dims

        self.q_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.v_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.position_encoder = ScalarEmbedder(embed_dims)
        self._init_weights()

    def _init_weights(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        if self.q_proj.bias is not None:
            constant_(self.q_proj.bias, 0.0)
        if self.v_proj.bias is not None:
            constant_(self.v_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if identity is None:
            identity = query
        if key is None:
            key = query

        B, N, C = query.shape
        M = key.shape[1]

        q = self.q_proj(query).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v_proj(key).reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3)

        if query_pos is not None:
            # query_pos: (B_ctx, N, M) or (N, M) joint distances
            pos_flat = query_pos.flatten()
            pos_emb = self.position_encoder(pos_flat).reshape(-1, N, M, C)
            pos_emb = pos_emb.unflatten(-1, (self.num_heads, -1)).permute(0, 3, 1, 2, 4)
            if B != pos_emb.shape[0]:
                pos_emb = pos_emb.tile(B // pos_emb.shape[0], 1, 1, 1, 1)
            q_exp = q[:, :, :, None] * pos_emb  # b,h,n,m,c
            attn = (q_exp * k.unsqueeze(2)).sum(-1) * self.scale
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1).type_as(v)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x + identity


# ---------------------------------------------------------------------------
# TemporalJointGraphAttention — joint + temporal attention
# Phase 1: simplified to temporal self-attention (no URDF)
# ---------------------------------------------------------------------------

class TemporalSelfAttention(nn.Module):
    """Simplified temporal self-attention for Phase 1.
    Replaces TemporalJointGraphAttention when num_joint=1."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        max_position_embeddings: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.v_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.temporal_position_encoder = RotaryEmbedding(
            head_dim, max_position_embeddings=max_position_embeddings
        )
        self._init_weights()

    def _init_weights(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        if self.q_proj.bias is not None:
            constant_(self.q_proj.bias, 0.0)
        if self.v_proj.bias is not None:
            constant_(self.v_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        temporal_pos_q: torch.Tensor | None = None,
        temporal_pos_k: torch.Tensor | None = None,
        temporal_attn_mask: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, T_q, C)
            key: (B, T_k, C) — if None, uses query (self-attention)
            temporal_pos_q: (B, T_q) position ids for query
            temporal_pos_k: (B, T_k) position ids for key
            temporal_attn_mask: (T_q, T_k) or (B, T_q, T_k) bool mask, True=masked
        """
        if identity is None:
            identity = query
        if key is None:
            key = query
        if value is None:
            value = key

        B, T_q, C = query.shape
        T_k = key.shape[1]

        q = self.q_proj(query).reshape(B, T_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, T_k, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, T_k, self.num_heads, -1).permute(0, 2, 1, 3)

        if temporal_pos_q is not None:
            q = self.temporal_position_encoder(q, temporal_pos_q)
        if temporal_pos_k is not None:
            k = self.temporal_position_encoder(k, temporal_pos_k)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if temporal_attn_mask is not None:
            if temporal_attn_mask.dim() == 2:
                temporal_attn_mask = temporal_attn_mask[None, None]
            elif temporal_attn_mask.dim() == 3:
                temporal_attn_mask = temporal_attn_mask.unsqueeze(1)
            attn = torch.where(temporal_attn_mask, float("-inf"), attn)

        attn = attn.softmax(dim=-1).type_as(v)
        x = (attn @ v).transpose(1, 2).reshape(B, T_q, C)
        x = self.proj(x)
        return x + identity


# ---------------------------------------------------------------------------
# AdaRMSNorm — adaptive RMSNorm conditioned on diffusion timestep
# ---------------------------------------------------------------------------

class AdaRMSNorm(nn.RMSNorm):
    """Adaptive RMS normalization conditioned on a vector (e.g. timestep embed).
    When zero=True, outputs 6 modulation signals (DiT style):
    scale, shift for pre-norm, gate_msa, shift_mlp, scale_mlp, gate_mlp.
    """

    def __init__(
        self,
        normalized_shape: int,
        condition_dims: int,
        zero: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=False)
        self.zero = zero
        n_out = normalized_shape * (6 if zero else 2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dims, n_out),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        x = super().forward(x)
        dims = x.shape[-1]
        ada = self.adaLN_modulation(c).unflatten(-1, (dims, -1))
        if ada.dim() != 4:
            ada = ada[:, None]
        x = x * (1 + ada[..., 0]) + ada[..., 1]
        if self.zero:
            gate_msa, shift_mlp, scale_mlp, gate_mlp = [
                t.squeeze(dim=-1) for t in ada[..., 2:].chunk(4, dim=-1)
            ]
        else:
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    """Feed-forward network with residual connection."""

    def __init__(self, embed_dims: int, feedforward_channels: int = 2048, act: str = "silu"):
        super().__init__()
        act_cls = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[act]
        self.layers = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            act_cls(),
            nn.Linear(feedforward_channels, embed_dims),
        )

    def forward(self, x: torch.Tensor, identity: torch.Tensor | None = None) -> torch.Tensor:
        if identity is None:
            identity = x
        return self.layers(x) + identity


# ---------------------------------------------------------------------------
# UpsampleHead — 1D conv + bilinear upsample to pred_steps
# ---------------------------------------------------------------------------

class UpsampleHead(nn.Module):
    """Upsamples chunk-level predictions to full pred_steps via
    bilinear interpolation + Conv1d."""

    def __init__(
        self,
        upsample_sizes: list[int],
        input_dim: int,
        dims: list[int],
        out_dim: int = 7,
        num_output_layers: int = 2,
    ):
        super().__init__()
        self.upsamples = nn.ModuleList()
        self.convs = nn.ModuleList()
        all_dims = [input_dim] + dims
        for i, size in enumerate(upsample_sizes):
            self.upsamples.append(
                nn.Upsample(size=(size, 1), mode="bilinear", align_corners=True)
            )
            self.convs.append(nn.Conv1d(all_dims[i], all_dims[i + 1], 3, padding=1))

        self.output_layers = nn.Sequential()
        for _ in range(num_output_layers):
            self.output_layers.append(nn.SiLU())
            self.output_layers.append(nn.Linear(all_dims[-1], all_dims[-1]))
        self.output_layers.append(nn.Linear(all_dims[-1], out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_joint, num_chunk, embed_dims)
        Returns:
            (B, num_joint, pred_steps, out_dim)
        """
        bs, num_joint, num_chunk, _ = x.shape
        x = x.flatten(0, 1)  # (B*J, T, C)
        for upsample, conv in zip(self.upsamples, self.convs):
            x = x.permute(0, 2, 1)  # (B*J, C, T)
            x = conv(upsample(x.unsqueeze(-1)).squeeze(-1))  # upsample + conv
            x = x.permute(0, 2, 1)  # (B*J, T', C')
        x = x.unflatten(0, (bs, num_joint))
        x = self.output_layers(x)
        return x
