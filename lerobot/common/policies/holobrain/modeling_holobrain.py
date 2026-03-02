"""HoloBrain Policy — main policy class for LeRobot.

Integrates RGB encoder, Robot State Encoder, Action Decoder, and diffusion
schedulers into a unified policy interface compatible with LeRobot.

Phase 4 adds URDF-based joint graph attention when use_urdf=True.
"""

from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn.functional as F
import torchvision
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from torch import Tensor, nn

from lerobot.common.policies.holobrain.action_decoder import HoloBrainActionDecoder
from lerobot.common.policies.holobrain.configuration_holobrain import HoloBrainConfig
from lerobot.common.policies.holobrain.loss import HoloBrainActionLoss
from lerobot.common.policies.holobrain.robot_state_encoder import HoloBrainRobotStateEncoder
from lerobot.common.policies.holobrain.utils import compute_joint_relative_pos
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy


class HoloBrainPolicy(PreTrainedPolicy):
    """HoloBrain-0 VLA+Diffusion policy ported to LeRobot.

    Phase 1 (use_urdf=False):
      - ResNet18 vision encoder
      - Robot state encoder (4-layer Transformer, no URDF)
      - Action decoder (6-layer DiT + UpsampleHead, TemporalSelfAttention)

    Phase 4 (use_urdf=True):
      - Same vision encoder
      - Robot state encoder with JointGraphAttention (virtual joints)
      - Action decoder with TemporalJointGraphAttention (URDF distances)
      - Each action dim treated as a virtual joint
    """

    config_class = HoloBrainConfig
    name = "holobrain"

    def __init__(
        self,
        config: HoloBrainConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        self.config = config

        # --- Normalization ---
        if config.input_features and config.normalization_mapping:
            self.normalize_inputs = Normalize(
                config.input_features, config.normalization_mapping, dataset_stats
            )
            self.normalize_targets = Normalize(
                config.output_features, config.normalization_mapping, dataset_stats
            )
            self.unnormalize_outputs = Unnormalize(
                config.output_features, config.normalization_mapping, dataset_stats
            )
            self._has_normalizer = True
        else:
            self._has_normalizer = False

        # --- Vision encoder ---
        self.rgb_encoder = _HoloBrainRgbEncoder(config)

        # --- Robot state encoder ---
        self.state_encoder = HoloBrainRobotStateEncoder(
            state_dim=config.state_dim,
            embed_dims=config.embed_dims,
            chunk_size=config.chunk_size,
            num_layers=config.num_encoder_layers,
            num_heads=config.num_heads,
            feedforward_channels=config.feedforward_channels,
            use_urdf=config.use_urdf,
            num_joints=config.num_joints,
        )

        # --- Action decoder ---
        self.action_decoder = HoloBrainActionDecoder(
            embed_dims=config.embed_dims,
            action_dim=config.action_dim,
            pred_steps=config.pred_steps,
            chunk_size=config.chunk_size,
            num_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            feedforward_channels=config.feedforward_channels,
            use_urdf=config.use_urdf,
            num_joints=config.num_joints,
        )

        # --- URDF joint distances ---
        if config.use_urdf:
            joint_rel_pos = compute_joint_relative_pos(config.num_joints)
            self.register_buffer("joint_relative_pos", joint_rel_pos)
        else:
            self.joint_relative_pos = None

        # --- Diffusion schedulers ---
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
            clip_sample=False,
        )
        self.inference_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
        )

        # --- Loss ---
        self.loss_fn = HoloBrainActionLoss(
            loss_type=config.loss_type,
            smooth_l1_beta=config.smooth_l1_beta,
            timestep_loss_weight=config.timestep_loss_weight,
            pos_weight=config.pos_weight,
            rot_weight=config.rot_weight,
            grip_weight=config.grip_weight,
        )

        # --- Action queue for inference ---
        self._action_queue: deque | None = None
        self.reset()

    def reset(self):
        """Clear action queue. Call on env.reset()."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return self.parameters()

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Training forward pass. Returns (loss, None)."""
        if self._has_normalizer:
            batch = self.normalize_inputs(batch)
            batch = self.normalize_targets(batch)

        # Stack images if multiple cameras
        if self.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[k] for k in self.config.image_features], dim=-4
            )

        # Extract tensors
        images = batch["observation.images"]
        state = batch["observation.state"]
        action = batch["action"]

        # Handle temporal dimension
        if images.dim() == 6:
            images = images[:, -1]
        if state.dim() == 3:
            pass
        elif state.dim() == 2:
            state = state.unsqueeze(1)

        # Encode images
        img_features = self.rgb_encoder(images)

        # Encode state
        state_features = self.state_encoder(
            state, joint_relative_pos=self.joint_relative_pos
        )

        # Pad action to pred_steps if needed
        B = action.shape[0]
        if action.shape[1] < self.config.pred_steps:
            pad_len = self.config.pred_steps - action.shape[1]
            action = F.pad(action, (0, 0, 0, pad_len), mode="replicate")
        elif action.shape[1] > self.config.pred_steps:
            action = action[:, :self.config.pred_steps]

        # Diffusion training
        noise = torch.randn_like(action)
        timesteps = torch.randint(
            0, self.config.num_train_timesteps, (B,),
            device=action.device,
        ).long()
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)

        pred_action = self.action_decoder(
            noisy_action, timesteps,
            img_features=img_features,
            state_features=state_features,
            joint_relative_pos=self.joint_relative_pos,
        )

        loss = self.loss_fn(pred_action, action, timesteps)
        return loss, None

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for environment interaction."""
        if self._has_normalizer:
            batch = self.normalize_inputs(batch)

        if len(self._action_queue) == 0:
            if self.config.image_features:
                batch = dict(batch)
                batch["observation.images"] = torch.stack(
                    [batch[k] for k in self.config.image_features], dim=-4
                )

            images = batch["observation.images"]
            state = batch["observation.state"]

            if images.dim() == 6:
                images = images[:, -1]
            if state.dim() == 2:
                state = state.unsqueeze(1)

            img_features = self.rgb_encoder(images)
            state_features = self.state_encoder(
                state, joint_relative_pos=self.joint_relative_pos
            )

            B = images.shape[0]
            device = images.device

            action = torch.randn(
                B, self.config.pred_steps, self.config.action_dim,
                device=device,
            )

            self.inference_scheduler.set_timesteps(
                self.config.num_inference_timesteps, device=device
            )
            for t in self.inference_scheduler.timesteps:
                pred = self.action_decoder(
                    action, t.expand(B),
                    img_features=img_features,
                    state_features=state_features,
                    joint_relative_pos=self.joint_relative_pos,
                )
                action = self.inference_scheduler.step(
                    pred, t, action
                ).prev_sample

            actions = action[:, :self.config.n_action_steps]

            if self._has_normalizer:
                actions = self.unnormalize_outputs({"action": actions})["action"]

            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()


# ---------------------------------------------------------------------------
# RGB Encoder — ResNet18 backbone + spatial softmax
# ---------------------------------------------------------------------------

class _HoloBrainRgbEncoder(nn.Module):
    """ResNet18-based image encoder with crop augmentation and spatial softmax."""

    def __init__(self, config: HoloBrainConfig):
        super().__init__()
        self.config = config

        weights = config.pretrained_backbone_weights
        if weights:
            backbone = getattr(torchvision.models, config.vision_backbone)(
                weights=weights
            )
        else:
            backbone = getattr(torchvision.models, config.vision_backbone)(
                weights=None
            )

        if config.use_group_norm and not weights:
            backbone = _replace_bn_with_gn(backbone)

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        with torch.no_grad():
            dummy = torch.zeros(1, 3, *config.crop_shape)
            feat = self.backbone(dummy)
            self._feat_channels = feat.shape[1]
            self._feat_h = feat.shape[2]
            self._feat_w = feat.shape[3]

        self.num_kp = config.spatial_softmax_num_keypoints
        self.ss_proj = nn.Conv2d(self._feat_channels, self.num_kp, 1)
        self.feature_dim = self.num_kp * 2
        self.out_proj = nn.Linear(self.feature_dim, config.embed_dims)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N_cam = images.shape[:2]

        if self.training and self.config.crop_is_random:
            images = self._random_crop(images)
        else:
            images = self._center_crop(images)

        x = images.flatten(0, 1)
        x = self.backbone(x)
        x = self._spatial_softmax(x)
        x = x.unflatten(0, (B, N_cam))
        x = self.out_proj(x)

        return x

    def _spatial_softmax(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        heatmap = self.ss_proj(x)
        heatmap = heatmap.reshape(B, self.num_kp, -1)
        heatmap = F.softmax(heatmap, dim=-1)
        heatmap = heatmap.reshape(B, self.num_kp, H, W)

        pos_y = torch.linspace(-1, 1, H, device=x.device)
        pos_x = torch.linspace(-1, 1, W, device=x.device)
        expected_y = (heatmap.sum(-1) * pos_y[None, None]).sum(-1)
        expected_x = (heatmap.sum(-2) * pos_x[None, None]).sum(-1)

        return torch.cat([expected_x, expected_y], dim=-1)

    def _random_crop(self, images: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = images.shape
        ch, cw = self.config.crop_shape
        if H <= ch and W <= cw:
            return images
        images = images.flatten(0, 1)
        images = torchvision.transforms.functional.resize(
            images, max(ch, cw), antialias=True
        ) if min(H, W) < max(ch, cw) else images
        _, _, H, W = images.shape
        top = torch.randint(0, H - ch + 1, (1,)).item()
        left = torch.randint(0, W - cw + 1, (1,)).item()
        images = images[:, :, top:top+ch, left:left+cw]
        return images.unflatten(0, (B, N))

    def _center_crop(self, images: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = images.shape
        ch, cw = self.config.crop_shape
        if H <= ch and W <= cw:
            return images
        images = images.flatten(0, 1)
        images = torchvision.transforms.functional.center_crop(images, [ch, cw])
        return images.unflatten(0, (B, N))


def _replace_bn_with_gn(module: nn.Module, num_groups: int = 16) -> nn.Module:
    """Replace all BatchNorm2d with GroupNorm in a module."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            gn = nn.GroupNorm(
                min(num_groups, num_channels),
                num_channels,
                affine=child.affine,
            )
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child, num_groups)
    return module
