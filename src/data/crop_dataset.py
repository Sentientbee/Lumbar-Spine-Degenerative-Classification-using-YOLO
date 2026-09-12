import os
from typing import Optional, List, Tuple, Callable
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data.dataset_builder import SEVERITY_TO_ID, SEVERITIES


class LumbarCropDataset(Dataset):
    """
    Dataset for Stage 2 Severity Classification.
    Loads 3-slice volumetric crops (C=3, H, W) paired with condition severity labels:
    {0: Normal/Mild, 1: Moderate, 2: Severe}.
    """

    def __init__(
        self,
        samples: Optional[List[Tuple[str, int]]] = None,
        crop_size: int = 128,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        is_synthetic: bool = False,
        num_synthetic_samples: int = 120,
    ):
        self.crop_size = crop_size
        self.transform = transform
        self.is_synthetic = is_synthetic

        if is_synthetic:
            # Generate deterministic synthetic indices and labels for testing/demo using isolated RNG
            rng = np.random.default_rng(42)
            self.samples = []
            for idx in range(num_synthetic_samples):
                # Distribute severities realistically: 60% Mild, 25% Moderate, 15% Severe
                p = rng.random()
                label = 0 if p < 0.60 else (1 if p < 0.85 else 2)
                self.samples.append((f"synth_{idx}", label))
        else:
            self.samples = samples or []

    def __len__(self) -> int:
        return len(self.samples)

    def _generate_synthetic_crop(self, label: int) -> np.ndarray:
        """Synthesize a 3-slice cropped volume mimicking spinal canal/foramina at a disc level."""
        h = w = self.crop_size
        crop = np.zeros((3, h, w), dtype=np.float32)

        # Baseline tissue intensity
        crop += 0.2 + np.random.normal(0, 0.02, size=(3, h, w))

        # Disc / Canal region in the center
        y, x = np.mgrid[0:h, 0:w]
        dist = ((x - w // 2) / 25.0) ** 2 + ((y - h // 2) / 18.0) ** 2

        # Intensity changes based on pathology: Severe stenosis has compressed canal
        intensity = 0.8 if label == 0 else (0.5 if label == 1 else 0.25)
        for c in range(3):
            slice_dist = dist + np.random.normal(0, 0.05, size=(h, w))
            mask = slice_dist <= 1.0
            crop[c][mask] = intensity + np.random.normal(0, 0.03, size=crop[c][mask].shape)

        return np.clip(crop, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]

        if self.is_synthetic:
            crop_array = self._generate_synthetic_crop(label)
            x_tensor = torch.from_numpy(crop_array)
        else:
            if path.endswith(".npy"):
                crop_array = np.load(path).astype(np.float32)
                if crop_array.max() > 1.0:
                    crop_array = crop_array / 255.0
                if crop_array.ndim == 2:
                    crop_array = np.stack([crop_array] * 3, axis=0)
                elif crop_array.ndim == 3 and crop_array.shape[-1] == 3 and crop_array.shape[0] != 3:
                    crop_array = crop_array.transpose(2, 0, 1)
                x_tensor = torch.from_numpy(crop_array)
            else:
                # Load 2D image and repeat to 3 slices if not 3D
                pil_img = Image.open(path).convert("L")
                pil_img = pil_img.resize((self.crop_size, self.crop_size))
                arr = np.array(pil_img, dtype=np.float32) / 255.0
                x_tensor = torch.from_numpy(np.stack([arr] * 3, axis=0))

        if self.transform is not None:
            x_tensor = self.transform(x_tensor)

        y_tensor = torch.tensor(label, dtype=torch.long)
        return x_tensor, y_tensor
