import os
import glob
from typing import Optional, Tuple, List, Union
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pydicom
    from pydicom.multival import MultiValue
except ImportError:
    pydicom = None
    MultiValue = None


def apply_voi_lut(img: np.ndarray, dcm) -> np.ndarray:
    """
    Applies Value of Interest (VOI) Look-Up Table or Window Center/Width transformation.
    Falls back to robust percentile clipping (1% - 99%) if metadata is missing.
    """
    if hasattr(dcm, "WindowCenter") and hasattr(dcm, "WindowWidth"):
        wc = dcm.WindowCenter
        ww = dcm.WindowWidth
        
        # Handle multi-value window tags
        if isinstance(wc, (list, tuple, MultiValue)) if MultiValue else isinstance(wc, (list, tuple)):
            wc = wc[0]
        if isinstance(ww, (list, tuple, MultiValue)) if MultiValue else isinstance(ww, (list, tuple)):
            ww = ww[0]
            
        try:
            wc = float(wc)
            ww = float(ww)
            img_min = wc - ww / 2.0
            img_max = wc + ww / 2.0
            return np.clip(img, img_min, img_max)
        except (ValueError, TypeError):
            pass

    # Robust percentile fallback for MRI when window tags are missing
    p1, p99 = np.percentile(img, (1.0, 99.0))
    if p99 > p1:
        return np.clip(img, p1, p99)
    return img


def read_dicom_windowed(
    src_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    as_rgb: bool = True
) -> np.ndarray:
    """
    Read a medical DICOM file, applying RescaleSlope/Intercept, VOI LUT windowing,
    PhotometricInterpretation handling (MONOCHROME1 vs MONOCHROME2), and normalization.

    Args:
        src_path: Path to the .dcm file.
        target_size: Optional (width, height) to resize.
        as_rgb: If True, returns (H, W, 3) uint8 image; else (H, W) uint8.

    Returns:
        Processed uint8 numpy array with values in [0, 255].
    """
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM files. Install with `pip install pydicom`.")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"DICOM file not found: {src_path}")

    dcm = pydicom.dcmread(src_path)
    img = dcm.pixel_array.astype(np.float32)

    # 1. Apply Rescale Slope & Intercept
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        img = img * slope + intercept

    # 2. Apply VOI LUT / Windowing
    img = apply_voi_lut(img, dcm)

    # 3. Min-Max Normalization to [0, 255]
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min + 1e-6) * 255.0
    else:
        img = np.zeros_like(img, dtype=np.float32)

    # 4. Handle Photometric Interpretation (Invert MONOCHROME1)
    photometric = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        img = 255.0 - img

    img = np.clip(img, 0, 255).astype(np.uint8)

    # 5. Optional Resize
    if target_size is not None:
        if cv2 is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        elif Image is not None:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(target_size, resample=Image.BILINEAR)
            img = np.array(pil_img)

    # 6. Channel formatting
    if as_rgb:
        img = np.stack([img] * 3, axis=-1)

    return img


def load_dicom_series(
    series_dir: str,
    target_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Load an entire volumetric MRI series from a directory, sorted by InstanceNumber.

    Args:
        series_dir: Directory containing .dcm slice files for a single series.
        target_size: Optional (width, height) to resize each slice.

    Returns:
        Tuple of:
            - 3D volume numpy array of shape (Depth, Height, Width), dtype uint8.
            - List of instance numbers corresponding to each slice.
    """
    if not os.path.isdir(series_dir):
        raise NotADirectoryError(f"Series directory does not exist: {series_dir}")

    dcm_files = glob.glob(os.path.join(series_dir, "*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {series_dir}")

    # Extract instance numbers to sort slices along the anatomical axis
    slices_with_info = []
    for f in dcm_files:
        try:
            dcm = pydicom.dcmread(f, stop_before_pixels=True)
            instance_num = int(getattr(dcm, "InstanceNumber", os.path.splitext(os.path.basename(f))[0]))
        except Exception:
            # Fallback to integer filename if header unreadable
            base = os.path.splitext(os.path.basename(f))[0]
            instance_num = int(base) if base.isdigit() else 0
        slices_with_info.append((instance_num, f))

    slices_with_info.sort(key=lambda x: x[0])

    volume = []
    instance_nums = []
    for inst_num, fpath in slices_with_info:
        slice_img = read_dicom_windowed(fpath, target_size=target_size, as_rgb=False)
        volume.append(slice_img)
        instance_nums.append(inst_num)

    volume_array = np.stack(volume, axis=0)  # Shape: (Depth, Height, Width)
    return volume_array, instance_nums
