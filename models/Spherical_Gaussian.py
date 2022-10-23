import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Spherical_Gaussian(nn.Module):
    def __init__(
            self,
            num_basis,
            k_low,
            k_high,
            trainable_k,
    ):
        super(Spherical_Gaussian, self).__init__()
        self.num_basis = num_basis

        self.trainable_k = trainable_k
        if self.trainable_k:
            kh = math.log10(k_high)
            kl = math.log10(k_low)
            self.k = nn.Parameter(torch.linspace(kh, kl, num_basis, dtype=torch.float32)[None, :])  # (1, num_basis)
        else:
            kh = math.log10(k_high)
            kl = math.log10(k_low)
            self.k = torch.linspace(kh, kl, num_basis, dtype=torch.float32)[None, :]

    def forward(self, light, normal, mu, view=None):
        if view is None:
            view = torch.zeros_like(light)
            view[..., 2] = -1
            view = view.detach()
        light, view, normal = F.normalize(light, p=2, dim=-1), F.normalize(view, p=2, dim=-1), F.normalize(normal, p=2,
                                                                                                           dim=-1)
        H = F.normalize((view + light) / 2, p=2, dim=-1)
        if self.trainable_k:
            k = self.k
        else:
            k = self.k.to(light.device)
        roughness = 10 ** k  # range: 1 ~ 1000
        out = torch.abs(mu) * torch.exp(roughness * ((H * normal).sum(dim=-1, keepdim=True) - 1))[
            ..., None]  # (batch, num_basis, 3)
        return out
