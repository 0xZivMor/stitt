import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Iterable

class FocalLoss(nn.Module):
    """
    Focal Loss implementation.

    Args:
        alpha (float): The balancing factor for the positive class. Default is 0.5.
        gamma (float): The focusing parameter. Default is 2.
        reduce (bool): Whether to reduce the loss. Default is True.

    Attributes:
        alpha (float): The balancing factor for the positive class.
        gamma (float): The focusing parameter.
    """

    def __init__(self, alpha: Union[float, Iterable[float]] = 0.5, gamma: float = 2, reduce: bool = True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Focal Loss.

        Args:
            inputs (torch.Tensor): The input tensor.
            targets (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduce:
            return focal_loss.mean()
        else:
            return focal_loss