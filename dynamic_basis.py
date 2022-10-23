import torch
import math


def dynamic_basis(input, current_epoch, total_epoch, num_basis):
    """
    Args:
        input:  (batch * num_pixels, num_basis, 3)
        current_epoch:
        total_epoch:
        num_basis:
    Returns:
    """
    alpha = current_epoch / total_epoch * num_basis                             # (0, 1]
    k = torch.arange(num_basis, dtype=torch.float32, device=input.device)
    weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(math.pi).cos_()) / 2    # (sin^2(0.5*alpha*pi), 0, ..., 0)
    weight = weight[None, :, None]                                              # (1, num_basis, 1), [0->1, 0, ..., 0]
    weighted_input = input * weight
    return weighted_input
