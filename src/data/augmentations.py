import random
from typing import List, Callable
import numpy as np
import torch


class MedicalMultiSliceAugmentations:
    """
    Medical imaging augmentations applied consistently across all slices
    of a 2.5D volumetric crop (C=3, H, W).
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        brightness_contrast_prob: float = 0.5,
        brightness_limit: float = 0.15,
        contrast_limit: float = 0.15,
        noise_prob: float = 0.3,
        noise_std: float = 0.02,
    ):
        self.hflip_prob = hflip_prob
        self.brightness_contrast_prob = brightness_contrast_prob
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.noise_prob = noise_prob
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (C, H, W) normalized to [0, 1].
        Returns:
            Augmented tensor of shape (C, H, W) clamped to [0, 1].
        """
        # 1. Random Horizontal Flip (consistent across all channels/slices)
        if random.random() < self.hflip_prob:
            x = torch.flip(x, dims=[-1])

        # 2. Random Brightness & Contrast
        if random.random() < self.brightness_contrast_prob:
            alpha = 1.0 + random.uniform(-self.contrast_limit, self.contrast_limit)
            beta = random.uniform(-self.brightness_limit, self.brightness_limit)
            x = (x - 0.5) * alpha + 0.5 + beta

        # 3. Random Gaussian Noise (simulates scanner radiofrequency noise)
        if random.random() < self.noise_prob:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return torch.clamp(x, 0.0, 1.0)
