import os
from typing import Tuple, List, Optional, Dict
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None

LEVELS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
LEVEL_TO_ID = {lvl: idx for idx, lvl in enumerate(LEVELS)}
ID_TO_LEVEL = {idx: lvl for idx, lvl in enumerate(LEVELS)}

SEVERITIES = ['Normal/Mild', 'Moderate', 'Severe']
SEVERITY_TO_ID = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
ID_TO_SEVERITY = {0: 'Normal/Mild', 1: 'Moderate', 2: 'Severe'}


def normalize_level_name(level_text: str) -> str:
    """Normalize level string (e.g. 'L1/L2' or 'l1_l2') into standard 'l1_l2' format."""
    cleaned = level_text.lower().replace('/', '_').replace('-', '_')
    for lvl in LEVELS:
        if lvl in cleaned:
            return lvl
    return cleaned


def point_to_yolo_bbox(
    x: float,
    y: float,
    img_width: int,
    img_height: int,
    box_size: int = 48
) -> Tuple[float, float, float, float]:
    """
    Convert an anatomical landmark coordinate (x, y) into a normalized YOLO bounding box.
    Uses an anatomically realistic patch size (default 48x48 pixels) to ensure robust IoU.

    Returns:
        (x_center_norm, y_center_norm, width_norm, height_norm)
    """
    x_c = np.clip(x / img_width, 0.0, 1.0)
    y_c = np.clip(y / img_height, 0.0, 1.0)
    w_norm = np.clip(box_size / img_width, 0.01, 1.0)
    h_norm = np.clip(box_size / img_height, 0.01, 1.0)
    return float(x_c), float(y_c), float(w_norm), float(h_norm)


def extract_multislice_roi(
    volume: np.ndarray,
    key_slice_idx: int,
    center_x: int,
    center_y: int,
    crop_size: int = 128,
    num_slices: int = 3
) -> np.ndarray:
    """
    Extract a multi-slice 2.5D volumetric crop centered at an anatomical coordinate.

    Args:
        volume: 3D numpy array of shape (Depth, Height, Width).
        key_slice_idx: Index of the key slice containing the landmark.
        center_x: X coordinate of landmark.
        center_y: Y coordinate of landmark.
        crop_size: Height and width of the crop in pixels.
        num_slices: Number of adjacent slices to stack (default 3: [slice-1, slice, slice+1]).

    Returns:
        Numpy array of shape (num_slices, crop_size, crop_size) as float32 normalized to [0, 1].
    """
    depth, height, width = volume.shape
    half_crop = crop_size // 2

    # Bounding box coordinates with clipping
    x1 = max(0, center_x - half_crop)
    x2 = min(width, center_x + half_crop)
    y1 = max(0, center_y - half_crop)
    y2 = min(height, center_y + half_crop)

    # Slice indices around key slice
    offset = num_slices // 2
    slice_indices = [np.clip(key_slice_idx + i, 0, depth - 1) for i in range(-offset, offset + 1)]

    channels = []
    for s_idx in slice_indices:
        patch = volume[s_idx, y1:y2, x1:x2]
        # Pad if near boundaries to maintain exact crop_size
        if patch.shape != (crop_size, crop_size):
            pad_h = crop_size - patch.shape[0]
            pad_w = crop_size - patch.shape[1]
            patch = np.pad(patch, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode='edge')
            patch = patch[:crop_size, :crop_size]
        channels.append(patch)

    roi = np.stack(channels, axis=0).astype(np.float32) / 255.0  # (C, H, W) in [0, 1]
    return roi
