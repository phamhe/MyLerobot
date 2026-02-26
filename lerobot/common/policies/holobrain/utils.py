"""Utility functions for HoloBrain policy.

Ported from RoboOrchardLab with FK-related functions removed for Phase 1.
"""

import torch


def apply_scale_shift(
    robot_state: torch.Tensor,
    joint_scale_shift: torch.Tensor | None = None,
    inverse: bool = False,
    scale_only: bool = False,
) -> torch.Tensor:
    """Applies scale and shift normalization to joint angles.

    Args:
        robot_state: (B, T, J, C) tensor. Channel 0 is joint angle.
        joint_scale_shift: (B, J, 2) with [scale, shift] per joint. None = identity.
        inverse: If True, denormalize: val * scale + shift.
        scale_only: If True, shift is set to 0.
    Returns:
        Normalized/denormalized tensor, same shape.
    """
    if joint_scale_shift is None:
        return robot_state

    if robot_state.shape[0] != joint_scale_shift.shape[0]:
        n = robot_state.shape[0] // joint_scale_shift.shape[0]
        joint_scale_shift = joint_scale_shift.repeat_interleave(n, dim=0)

    scale = joint_scale_shift[:, None, :, 0:1]
    shift = 0 if scale_only else joint_scale_shift[:, None, :, 1:2]

    if not inverse:
        robot_state = torch.cat(
            [(robot_state[..., :1] - shift) / scale, robot_state[..., 1:]],
            dim=-1,
        )
    else:
        robot_state = torch.cat(
            [robot_state[..., :1] * scale + shift, robot_state[..., 1:]],
            dim=-1,
        )
    return robot_state


def inverse_scale_shift(
    robot_state: torch.Tensor,
    joint_scale_shift: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convenience wrapper for apply_scale_shift(inverse=True)."""
    return apply_scale_shift(robot_state, joint_scale_shift, inverse=True)
