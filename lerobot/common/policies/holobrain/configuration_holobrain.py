"""HoloBrain policy configuration for LeRobot.

Inherits from PreTrainedConfig for compatibility with LeRobot's framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.common.policies.pretrained import PreTrainedConfig


@dataclass
class HoloBrainConfig(PreTrainedConfig):
    """Configuration for HoloBrainPolicy.

    Architecture (Phase 1 simplified):
      - Vision: ResNet18 (dual cameras, crop 84x84)
      - Robot State Encoder: linear + 4-layer Transformer (standard self-attn)
      - Action Decoder: 6-layer DiT Transformer + UpsampleHead
      - Loss: SmoothL1(pos) + SmoothL1(rot) + BCE(gripper) + timestep weight

    Diffusion:
      - Training: DDPMScheduler, 1000 steps, squaredcos_cap_v2, prediction_type="sample"
      - Inference: DPMSolverMultistepScheduler, 10 steps
    """

    # --- Model architecture ---
    embed_dims: int = 256
    num_decoder_layers: int = 6
    num_encoder_layers: int = 4
    num_heads: int = 8
    feedforward_channels: int = 2048

    # --- Observation / action dimensions ---
    action_dim: int = 7       # LIBERO: delta_pos(3) + delta_rot(3) + gripper(1)
    state_dim: int = 8        # LIBERO robot state
    pred_steps: int = 64      # action prediction horizon
    chunk_size: int = 4       # temporal chunking for decoder
    n_action_steps: int = 8   # steps actually executed per inference

    # --- Diffusion ---
    num_train_timesteps: int = 1000
    num_inference_timesteps: int = 10
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "sample"

    # --- Vision backbone ---
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] = (84, 84)
    crop_is_random: bool = True
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    pretrained_backbone_weights: str | None = None

    # --- URDF / Joint Graph ---
    use_urdf: bool = False        # Enable URDF-based joint graph attention
    num_joints: int = 7           # Number of joints (Franka Panda = 7)
    # When use_urdf=True:
    #   - State encoder projects EEF state to virtual joint tokens
    #   - Action decoder uses TemporalJointGraphAttention
    #   - joint_relative_pos computed as |i-j| (linear chain)

    # --- Loss ---
    smooth_l1_beta: float = 0.04
    timestep_loss_weight: float = 0.0
    pos_weight: float = 1.0
    rot_weight: float = 1.0
    grip_weight: float = 1.0
    loss_type: str = "mse"  # "mse" or "smooth_l1"
    fk_loss_weight: float = 0.0   # FK loss weight (0 = disabled)

    # --- Training ---
    lr: float = 1e-4
    weight_decay: float = 5e-4
    grad_clip_max_norm: float = 10.0
    warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()
        assert self.pred_steps % self.chunk_size == 0, (
            f"pred_steps ({self.pred_steps}) must be divisible by "
            f"chunk_size ({self.chunk_size})"
        )

    # --- Abstract method implementations (required by PreTrainedConfig) ---

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.pred_steps))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self):
        from lerobot.common.optim.optimizers import AdamConfig
        return AdamConfig(lr=self.lr, weight_decay=self.weight_decay)

    def get_scheduler_preset(self):
        from lerobot.common.optim.schedulers import DiffuserSchedulerConfig
        return DiffuserSchedulerConfig(name="cosine", num_warmup_steps=self.warmup_steps)

    def validate_features(self) -> None:
        if self.input_features:
            assert len(self.image_features) > 0, "At least one image feature required"
            assert self.robot_state_feature is not None, "Robot state feature required"
        if self.output_features:
            assert self.action_feature is not None, "Action feature required"

    # --- Convenience properties ---

    @property
    def num_chunks(self) -> int:
        return self.pred_steps // self.chunk_size

    @property
    def image_features(self) -> dict:
        if not self.input_features:
            return {}
        return {
            k: v for k, v in self.input_features.items()
            if hasattr(v, "type") and _feature_type_name(v.type) == "VISUAL"
        }

    @property
    def robot_state_feature(self):
        if not self.input_features:
            return None
        for k, v in self.input_features.items():
            if hasattr(v, "type") and _feature_type_name(v.type) == "STATE":
                return v
        return None

    @property
    def action_feature(self):
        if not self.output_features:
            return None
        for k, v in self.output_features.items():
            if hasattr(v, "type") and _feature_type_name(v.type) == "ACTION":
                return v
        return None


def _feature_type_name(ft) -> str:
    """Extract feature type name from FeatureType enum or string."""
    if hasattr(ft, "value"):
        return str(ft.value).upper()
    if hasattr(ft, "name"):
        return str(ft.name).upper()
    return str(ft).upper()
