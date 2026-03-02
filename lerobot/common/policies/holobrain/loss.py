"""HoloBrain action loss — supports MSE and SmoothL1.

Phase 1 loss options:
  - MSE: simple mean squared error over all action dims (matches DP baseline)
  - SmoothL1: per-component (pos/rot/grip) with configurable beta
"""

import torch
import torch.nn.functional as F
from torch import nn


class HoloBrainActionLoss(nn.Module):
    """Compute diffusion training loss for HoloBrain.

    Supports two modes:
      - loss_type="mse": F.mse_loss over entire action (matches DP baseline)
      - loss_type="smooth_l1": per-component SmoothL1 with timestep weighting
    """

    def __init__(
        self,
        loss_type: str = "mse",
        smooth_l1_beta: float = 0.04,
        timestep_loss_weight: float = 0.0,
        pos_weight: float = 1.0,
        rot_weight: float = 1.0,
        grip_weight: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.smooth_l1_beta = smooth_l1_beta
        self.timestep_loss_weight = timestep_loss_weight
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.grip_weight = grip_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, T, action_dim) predicted actions
            target: (B, T, action_dim) ground-truth actions
            timesteps: (B,) diffusion timesteps
        Returns:
            Scalar loss
        """
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)

        # SmoothL1 mode (original)
        pred_pos = pred[..., :3]
        pred_rot = pred[..., 3:6]
        pred_grip = pred[..., 6:7]

        tgt_pos = target[..., :3]
        tgt_rot = target[..., 3:6]
        tgt_grip = target[..., 6:7]

        loss_pos = F.smooth_l1_loss(pred_pos, tgt_pos, beta=self.smooth_l1_beta, reduction="none")
        loss_rot = F.smooth_l1_loss(pred_rot, tgt_rot, beta=self.smooth_l1_beta, reduction="none")
        loss_grip = F.smooth_l1_loss(pred_grip, tgt_grip, beta=self.smooth_l1_beta, reduction="none")

        loss = (
            self.pos_weight * loss_pos.sum(-1)
            + self.rot_weight * loss_rot.sum(-1)
            + self.grip_weight * loss_grip.sum(-1)
        )

        if self.timestep_loss_weight > 0:
            t_weight = self.timestep_loss_weight / (timesteps.float() + 1)
            loss = loss * t_weight[:, None]

        return loss.mean()
