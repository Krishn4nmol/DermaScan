"""
Focal Loss for multi-class imbalanced classification.
Lin et al., ICCV 2017 — https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma       : Focusing parameter (default 2.0). Higher = more focus on hard examples.
        alpha       : Class weights tensor of shape (C,). If None, uniform weights.
        reduction   : 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha      # stored; moved to device in forward()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) raw logits
            targets : (B,)   integer class indices
        """
        # Standard cross-entropy gives log(p_t) per sample
        log_pt = F.log_softmax(logits, dim=1)               # (B, C)
        log_pt = log_pt.gather(1, targets.view(-1, 1))      # (B, 1)
        log_pt = log_pt.squeeze(1)                           # (B,)

        pt = log_pt.exp()                                    # (B,) — p_t

        # Focal weight
        focal_weight = (1.0 - pt) ** self.gamma             # (B,)

        # Class-weight alpha
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]  # (B,)
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
