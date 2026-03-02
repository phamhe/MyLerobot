"""HoloBrain Policy — main policy class for LeRobot.

Integrates RGB encoder, Robot State Encoder, Action Decoder, and diffusion
schedulers into a unified policy interface compatible with LeRobot.
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
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy


class HoloBrainPolicy(PreTrainedPolicy):
    """HoloBrain-0 VLA+Diffusion policy ported to LeRobot.

    Phase 1 simplified version:
      - ResNet18 vision encoder (shared across cameras)
      - Robot state encoder (4-layer Transformer, no URDF)
      - Action decoder (6-layer DiT + UpsampleHead)
      - DDPM training / DPMSolver inference
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
        )

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
        images = batch["observation.images"]  # (B, [T], N_cam, C, H, W) or (B, N_cam, C, H, W)
        state = batch["observation.state"]    # (B, [T], state_dim)
        action = batch["action"]              # (B, horizon, action_dim)

        # Handle temporal dimension: take last frame for images, keep all for state
        if images.dim() == 6:
            images = images[:, -1]  # (B, N_cam, C, H, W) — latest obs
        if state.dim() == 3:
            # (B, T, state_dim) — keep all frames for state encoder
            pass
        elif state.dim() == 2:
            state = state.unsqueeze(1)  # (B, 1, state_dim)

        # Encode images
        img_features = self.rgb_encoder(images)  # (B, N_tokens, embed_dims)

        # Encode state
        state_features = self.state_encoder(state)  # (B, N_state, embed_dims)

        # Pad action to pred_steps if needed
        B = action.shape[0]
        if action.shape[1] < self.config.pred_steps:
            pad_len = self.config.pred_steps - action.shape[1]
            action = F.pad(action, (0, 0, 0, pad_len), mode="replicate")
        elif action.shape[1] > self.config.pred_steps:
            action = action[:, :self.config.pred_steps]

        # Diffusion training: add noise and predict
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
        )

        loss = self.loss_fn(pred_action, action, timesteps)
        return loss, None

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for environment interaction.

        Uses action queue: generates full trajectory, caches n_action_steps
        actions, pops one per call.
        """
        if self._has_normalizer:
            batch = self.normalize_inputs(batch)

        if len(self._action_queue) == 0:
            # Stack images
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
            state_features = self.state_encoder(state)

            B = images.shape[0]
            device = images.device

            # Start from random noise
            action = torch.randn(
                B, self.config.pred_steps, self.config.action_dim,
                device=device,
            )

            # DPMSolver iterative denoising
            self.inference_scheduler.set_timesteps(
                self.config.num_inference_timesteps, device=device
            )
            for t in self.inference_scheduler.timesteps:
                pred = self.action_decoder(
                    action, t.expand(B),
                    img_features=img_features,
                    state_features=state_features,
                )
                action = self.inference_scheduler.step(
                    pred, t, action
                ).prev_sample

            # Take first n_action_steps
            actions = action[:, :self.config.n_action_steps]  # (B, n_act, action_dim)

            if self._has_normalizer:
                actions = self.unnormalize_outputs({"action": actions})["action"]

            # Queue up actions (each is (B, action_dim))
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()


# ---------------------------------------------------------------------------
# RGB Encoder — ResNet18 backbone + spatial softmax
# ---------------------------------------------------------------------------

class _HoloBrainRgbEncoder(nn.Module):
    """ResNet18-based image encoder with crop augmentation and spatial softmax.

    Processes multiple camera views and returns flattened feature tokens.
    """

    def __init__(self, config: HoloBrainConfig):
        super().__init__()
        self.config = config

        # Build backbone
        weights = config.pretrained_backbone_weights
        if weights:
            backbone = getattr(torchvision.models, config.vision_backbone)(
                weights=weights
            )
        else:
            backbone = getattr(torchvision.models, config.vision_backbone)(
                weights=None
            )

        # Optionally replace BatchNorm with GroupNorm
        if config.use_group_norm and not weights:
            backbone = _replace_bn_with_gn(backbone)

        # Remove final FC and avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Determine feature dim by forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *config.crop_shape)
            feat = self.backbone(dummy)
            self._feat_channels = feat.shape[1]
            self._feat_h = feat.shape[2]
            self._feat_w = feat.shape[3]

        # Spatial softmax
        self.num_kp = config.spatial_softmax_num_keypoints
        self.ss_proj = nn.Conv2d(self._feat_channels, self.num_kp, 1)

        # Feature dim per camera: num_kp * 2 (x, y coordinates)
        self.feature_dim = self.num_kp * 2

        # Project to embed_dims
        self.out_proj = nn.Linear(self.feature_dim, config.embed_dims)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, N_cam, C, H, W) — multiple camera views
        Returns:
            (B, N_cam, embed_dims) — one token per camera
        """
        B, N_cam = images.shape[:2]

        # Random crop during training
        if self.training and self.config.crop_is_random:
            images = self._random_crop(images)
        else:
            images = self._center_crop(images)

        # Process all cameras together
        x = images.flatten(0, 1)  # (B*N_cam, C, H, W)
        x = self.backbone(x)     # (B*N_cam, feat_c, feat_h, feat_w)

        # Spatial softmax
        x = self._spatial_softmax(x)  # (B*N_cam, num_kp*2)
        x = x.unflatten(0, (B, N_cam))  # (B, N_cam, num_kp*2)
        x = self.out_proj(x)  # (B, N_cam, embed_dims)

        return x

    def _spatial_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial softmax to extract keypoint coordinates."""
        B, C, H, W = x.shape
        heatmap = self.ss_proj(x)  # (B, num_kp, H, W)
        heatmap = heatmap.reshape(B, self.num_kp, -1)
        heatmap = F.softmax(heatmap, dim=-1)
        heatmap = heatmap.reshape(B, self.num_kp, H, W)

        # Compute expected coordinates
        pos_y = torch.linspace(-1, 1, H, device=x.device)
        pos_x = torch.linspace(-1, 1, W, device=x.device)
        expected_y = (heatmap.sum(-1) * pos_y[None, None]).sum(-1)
        expected_x = (heatmap.sum(-2) * pos_x[None, None]).sum(-1)

        return torch.cat([expected_x, expected_y], dim=-1)  # (B, num_kp*2)

    def _random_crop(self, images: torch.Tensor) -> torch.Tensor:
        """Apply random crop to images."""
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
        """Apply center crop to images."""
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
