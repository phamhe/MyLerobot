"""HoloBrain action loss — Phase 1 simplified version.

Phase 1 simplifications:
  - SmoothL1 for pos and rot (no Wasserstein distance)
  - BCE for gripper
  - timestep weighting: 1000 / (t + 1)
  - No FK loss, parallel_weight, or mobile loss
"""

import torch
import torch.nn.functional as F
from torch import nn


class HoloBrainActionLoss(nn.Module):
    """Compute diffusion training loss for HoloBrain Phase 1.

    Action is 7-dim: pos(3) + rot(3) + gripper(1).
    Loss = SmoothL1(pos) + SmoothL1(rot) + BCE(gripper), weighted by timestep.
    """

    def __init__(
        self,
        smooth_l1_beta: float = 0.04,
        timestep_loss_weight: float = 1000.0,
        pos_weight: float = 1.0,
        rot_weight: float = 1.0,
        grip_weight: float = 1.0,
    ):
        super().__init__()
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
            pred: (B, T, 7) predicted actions
            target: (B, T, 7) ground-truth actions
            timesteps: (B,) diffusion timesteps
        Returns:
            Scalar loss
        """
        pred_pos = pred[..., :3]
        pred_rot = pred[..., 3:6]
        pred_grip = pred[..., 6:7]

        tgt_pos = target[..., :3]
        tgt_rot = target[..., 3:6]
        tgt_grip = target[..., 6:7]

        loss_pos = F.smooth_l1_loss(pred_pos, tgt_pos, beta=self.smooth_l1_beta, reduction="none")
        loss_rot = F.smooth_l1_loss(pred_rot, tgt_rot, beta=self.smooth_l1_beta, reduction="none")
        loss_grip = F.binary_cross_entropy_with_logits(pred_grip, tgt_grip, reduction="none")

        # Sum over feature dims, keep batch and time
        loss = (
            self.pos_weight * loss_pos.sum(-1)
            + self.rot_weight * loss_rot.sum(-1)
            + self.grip_weight * loss_grip.sum(-1)
        )  # (B, T)

        # Timestep weighting: larger weight for smaller t
        if self.timestep_loss_weight > 0:
            t_weight = self.timestep_loss_weight / (timesteps.float() + 1)  # (B,)
            loss = loss * t_weight[:, None]

        return loss.mean()
