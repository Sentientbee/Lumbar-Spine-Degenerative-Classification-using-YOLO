import os
import sys
import json
import argparse
from typing import Dict, Any, Optional
import numpy as np
import torch

from src.data.dicom_reader import load_dicom_series
from src.data.dataset_builder import extract_multislice_roi, LEVELS, SEVERITIES
from src.models.yolo_detector import SpineLevelDetector
from src.models.severity_classifier import LumbarSeverityClassifier


def predict_study(
    series_dir: str,
    yolo_weights: str = "yolo11s.pt",
    classifier_weights: Optional[str] = None,
    crop_size: int = 128,
    conf_threshold: float = 0.25,
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Run the end-to-end two-stage diagnostic pipeline on a lumbar MRI series directory.

    Args:
        series_dir: Directory containing DICOM files for a series.
        yolo_weights: Path to trained YOLO11 level detector weights.
        classifier_weights: Optional path to trained 2.5D severity classifier weights.
        crop_size: Size of the multi-slice ROI crop around detected disc levels.
        conf_threshold: YOLO confidence threshold.
        device: 'cuda', 'cpu', or 'auto'.

    Returns:
        Structured study prediction dictionary.
    """
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"[*] Loading volumetric MRI series from: {series_dir}")
    volume, instance_nums = load_dicom_series(series_dir)
    print(f"[+] Loaded volume shape: {volume.shape} (Depth={volume.shape[0]}, Slices={len(instance_nums)})")

    # -------------------------------------------------------------
    # Stage 1: Anatomical Level Localization using YOLO11
    # -------------------------------------------------------------
    print(f"[*] Stage 1: Running YOLO11 disc level localizer ({yolo_weights})...")
    detector = SpineLevelDetector(model_weights=yolo_weights)
    detected_levels = detector.predict_volume(volume, conf_threshold=conf_threshold)

    print(f"[+] Detected {len(detected_levels)} lumbar disc levels across volume.")

    # -------------------------------------------------------------
    # Stage 2: 2.5D Volumetric Severity Classification
    # -------------------------------------------------------------
    print("[*] Stage 2: Initializing 2.5D severity classifier...")
    classifier = LumbarSeverityClassifier(backbone_name="resnet18", num_classes=3, pretrained=False)

    # Resolve default weights if none specified
    resolved_classifier_weights = classifier_weights
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_weights_path = os.path.join(project_root, "weights", "best_severity_classifier.pt")
    if resolved_classifier_weights is None and os.path.exists(default_weights_path):
        resolved_classifier_weights = default_weights_path

    if resolved_classifier_weights and os.path.exists(resolved_classifier_weights):
        state_dict = torch.load(resolved_classifier_weights, map_location=dev)
        classifier.load_state_dict(state_dict)
        print(f"[+] Loaded classifier checkpoint: {resolved_classifier_weights}")
    elif resolved_classifier_weights:
        print(f"[!] Warning: Specified classifier weights '{resolved_classifier_weights}' not found. Running in baseline mode.")
    else:
        print("[!] No custom classifier checkpoint provided; running in zero-shot / baseline mode.")

    classifier.to(dev)
    classifier.eval()

    study_results: Dict[str, Any] = {
        "series_dir": series_dir,
        "slices_count": int(volume.shape[0]),
        "detected_levels": {},
    }

    # Evaluate each level
    for lvl in LEVELS:
        if lvl in detected_levels:
            det = detected_levels[lvl]
            k_idx = det["key_slice_idx"]
            cx, cy = int(det["center"][0]), int(det["center"][1])

            # Extract 2.5D multi-slice crop (3 slices, H, W)
            roi = extract_multislice_roi(volume, key_slice_idx=k_idx, center_x=cx, center_y=cy, crop_size=crop_size)
            roi_tensor = torch.from_numpy(roi).unsqueeze(0).to(dev)  # (1, 3, crop_size, crop_size)

            probs_dict = classifier.predict_dict(roi_tensor)
            pred_severity = max(probs_dict.items(), key=lambda x: x[1])[0]

            key_inst = instance_nums[k_idx] if 0 <= k_idx < len(instance_nums) else k_idx
            study_results["detected_levels"][lvl] = {
                "detected": True,
                "key_slice_instance": key_inst,
                "detector_confidence": round(float(det["conf"]), 4),
                "bbox": [round(c, 2) for c in det["bbox"]],
                "center": (cx, cy),
                "predicted_severity": pred_severity,
                "probabilities": {k: round(v, 4) for k, v in probs_dict.items()},
            }
        else:
            study_results["detected_levels"][lvl] = {
                "detected": False,
                "note": "Level not detected above confidence threshold",
            }

    return study_results


def main():
    parser = argparse.ArgumentParser(description="End-to-End Lumbar Spine Diagnosis Pipeline (YOLO11 + 2.5D Classifier)")
    parser.add_argument("--series_dir", type=str, required=True, help="Path to directory of DICOM files")
    parser.add_argument("--yolo_weights", type=str, default="yolo11s.pt", help="YOLO11 weights path")
    parser.add_argument("--classifier_weights", type=str, default=None, help="Severity classifier weights path")
    parser.add_argument("--output_json", type=str, default="diagnosis_report.json", help="Output report JSON path")
    args = parser.parse_args()

    results = predict_study(
        series_dir=args.series_dir,
        yolo_weights=args.yolo_weights,
        classifier_weights=args.classifier_weights,
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved diagnosis report to: {args.output_json}")

    print("\n" + "=" * 55)
    print("           LUMBAR SPINE DIAGNOSIS SUMMARY           ")
    print("=" * 55)
    for lvl, info in results["detected_levels"].items():
        if info.get("detected"):
            sev = info["predicted_severity"]
            probs = info["probabilities"]
            print(f"Level {lvl.upper():<6} | Severity: {sev:<11} | P(Mild)={probs['Normal/Mild']:.2f} P(Mod)={probs['Moderate']:.2f} P(Sev)={probs['Severe']:.2f}")
        else:
            print(f"Level {lvl.upper():<6} | Not detected")
    print("=" * 55)


if __name__ == "__main__":
    main()
