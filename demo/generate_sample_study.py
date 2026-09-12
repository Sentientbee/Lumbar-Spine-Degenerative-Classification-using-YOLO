import os
import json
import argparse
import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
except ImportError:
    pydicom = None


def generate_synthetic_lumbar_slice(
    slice_idx: int,
    total_slices: int = 15,
    height: int = 512,
    width: int = 512,
    modality: str = "T2"
) -> np.ndarray:
    """
    Synthesize a realistic sagittal lumbar spine MRI slice.
    Includes spinal curvature, vertebral bodies (L1-S1), intervertebral discs,
    and the spinal canal / CSF column.
    """
    img = np.zeros((height, width), dtype=np.float32)

    # Lateral distance factor (slices near center show spinal canal best)
    center_factor = 1.0 - abs(slice_idx - total_slices / 2.0) / (total_slices / 2.0 + 1e-5)

    # 1. Soft tissue background
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    body_mask = (x_coords > 100) & (x_coords < 420) & (y_coords > 40) & (y_coords < 480)
    img[body_mask] = 40.0 + np.random.normal(0, 5, size=img[body_mask].shape)

    # 2. Vertebral bodies (L1 to S1) - 6 vertebrae centers
    vertebrae_y = [90, 155, 225, 300, 380, 450]
    vertebrae_x = [230, 240, 245, 245, 235, 215]

    for vy, vx in zip(vertebrae_y, vertebrae_x):
        # Rectangular/elliptical vertebral body
        dist = ((x_coords - vx) / 38.0) ** 2 + ((y_coords - vy) / 22.0) ** 2
        vert_mask = dist <= 1.0
        img[vert_mask] = 120.0 + np.random.normal(0, 8, size=img[vert_mask].shape)
        # Cortical bone rim (darker on MRI)
        rim_mask = (dist > 0.85) & (dist <= 1.0)
        img[rim_mask] = 30.0

    # 3. Intervertebral discs (L1/L2 to L5/S1) - 5 disc spaces
    disc_y = [122, 190, 262, 340, 415]
    disc_x = [235, 242, 245, 240, 225]

    for dy, dx in zip(disc_y, disc_x):
        dist = ((x_coords - dx) / 35.0) ** 2 + ((y_coords - dy) / 10.0) ** 2
        disc_mask = dist <= 1.0
        # T2: hydrated discs are moderately bright, T1 darker
        disc_val = 140.0 if modality == "T2" else 80.0
        img[disc_mask] = disc_val + np.random.normal(0, 6, size=img[disc_mask].shape)

    # 4. Spinal canal / Thecal sac (posterior to vertebrae)
    canal_x = [285, 292, 296, 294, 282]
    for dy, cx in zip(disc_y, canal_x):
        canal_dist = ((x_coords - cx) / 14.0) ** 2 + ((y_coords - dy) / 28.0) ** 2
        canal_mask = canal_dist <= 1.0
        # High intensity CSF on T2 when near mid-sagittal
        csf_intensity = (220.0 * center_factor) if modality == "T2" else 50.0
        img[canal_mask] = csf_intensity

    # Noise and Gaussian smoothing
    noise = np.random.normal(0, 3, size=(height, width))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def save_as_dicom(img: np.ndarray, output_path: str, instance_number: int, modality: str = "MR"):
    """Write an image array to a standard DICOM file with full medical tags."""
    if pydicom is None:
        raise ImportError("pydicom required for writing .dcm files.")

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = modality
    ds.PatientName = "DEMO^SYNTHETIC"
    ds.PatientID = "DEMO_PATIENT_001"
    ds.SeriesDescription = "Sagittal T2/STIR"
    ds.InstanceNumber = instance_number
    ds.Rows, ds.Columns = img.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.WindowCenter = 128
    ds.WindowWidth = 256
    ds.PixelData = img.tobytes()

    ds.save_as(output_path, write_like_original=False)


def create_sample_study(output_dir: str, num_slices: int = 15, modality: str = "T2"):
    """Create a complete sample study folder with synthetic MRI slices."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Generating {num_slices} synthetic {modality} MRI slices in: {output_dir}")

    # Landmark coordinates for the 5 lumbar discs
    ground_truth = {
        "modality": modality,
        "study_id": "SYNTHETIC_001",
        "key_slice_index": num_slices // 2,
        "levels": {
            "l1_l2": {"center": (235, 122), "severity": "Normal/Mild"},
            "l2_l3": {"center": (242, 190), "severity": "Normal/Mild"},
            "l3_l4": {"center": (245, 262), "severity": "Moderate"},
            "l4_l5": {"center": (240, 340), "severity": "Severe"},
            "l5_s1": {"center": (225, 415), "severity": "Moderate"},
        }
    }

    for i in range(num_slices):
        slice_img = generate_synthetic_lumbar_slice(i, total_slices=num_slices, modality=modality)

        # Save as PNG image
        png_path = os.path.join(output_dir, f"{i+1:03d}.png")
        Image.fromarray(slice_img).save(png_path)

        # Save as DICOM if pydicom available
        if pydicom is not None:
            dcm_path = os.path.join(output_dir, f"{i+1}.dcm")
            save_as_dicom(slice_img, dcm_path, instance_number=i+1)

    # Write ground truth JSON
    truth_path = os.path.join(output_dir, "ground_truth_diagnosis.json")
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"[+] Successfully created sample study with {num_slices} slices.")
    print(f"[+] Ground truth diagnosis saved to: {truth_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic lumbar spine MRI study for demonstration.")
    parser.add_argument("--output_dir", type=str, default="data/sample_study", help="Directory to save sample study")
    parser.add_argument("--num_slices", type=int, default=15, help="Number of slices to generate")
    parser.add_argument("--modality", type=str, default="T2", choices=["T1", "T2"], help="MRI modality")
    args = parser.parse_args()

    create_sample_study(output_dir=args.output_dir, num_slices=args.num_slices, modality=args.modality)
