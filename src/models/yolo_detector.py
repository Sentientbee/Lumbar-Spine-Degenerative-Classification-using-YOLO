import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.data.dataset_builder import ID_TO_LEVEL, LEVELS


class SpineLevelDetector:
    """
    Stage 1: Anatomical Lumbar Disc Level Detector.
    Uses Ultralytics YOLO11 to localize L1/L2, L2/L3, L3/L4, L4/L5, and L5/S1 levels.
    """

    def __init__(self, model_weights: str = "yolo11s.pt"):
        if YOLO is None:
            raise ImportError("Ultralytics is required. Install with `pip install ultralytics>=8.3.0`.")
        self.model_weights = model_weights
        self.model = YOLO(model_weights)

    def train(
        self,
        data_yaml: str = "configs/yolo_levels.yaml",
        epochs: int = 50,
        batch_size: int = 16,
        imgsz: int = 640,
        project: str = "runs/detect",
        name: str = "level_detector_yolo11",
        device: str = "auto"
    ) -> Any:
        """Train YOLO11 on the 5-class anatomical disc level dataset."""
        if device == "auto":
            import torch
            device = "0,1" if torch.cuda.device_count() > 1 else ("0" if torch.cuda.is_available() else "cpu")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=project,
            name=name,
            device=device,
            plots=True,
            save=True,
        )
        return results

    def predict_slice(
        self,
        slice_img: np.ndarray,
        conf_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Detect disc levels on a single 2D MRI slice (H, W) or (H, W, 3).

        Returns:
            List of detection dicts:
            [{'level': 'l1_l2', 'class_id': 0, 'conf': 0.89, 'bbox': [x1, y1, x2, y2], 'center': (cx, cy)}]
        """
        if slice_img.ndim == 2:
            slice_img = np.stack([slice_img] * 3, axis=-1)

        results = self.model.predict(slice_img, conf=conf_threshold, verbose=False)
        detections = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                level_name = ID_TO_LEVEL.get(cls_id, f"level_{cls_id}")
                detections.append({
                    "level": level_name,
                    "class_id": cls_id,
                    "conf": confidence,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "center": (float(cx), float(cy)),
                })

        return detections

    def predict_volume(
        self,
        volume: np.ndarray,
        conf_threshold: float = 0.25
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run detection across all slices in a 3D MRI volume (Depth, Height, Width).
        Aggregates detections across slices and identifies the key slice with the highest
        confidence detection for each of the 5 lumbar levels.

        Returns:
            Dict mapping level name -> detection details including key_slice_idx, center, and bbox.
        """
        depth = volume.shape[0]
        best_by_level: Dict[str, Dict[str, Any]] = {}

        for s_idx in range(depth):
            slice_img = volume[s_idx]
            slice_dets = self.predict_slice(slice_img, conf_threshold=conf_threshold)

            for det in slice_dets:
                lvl = det["level"]
                if lvl not in best_by_level or det["conf"] > best_by_level[lvl]["conf"]:
                    best_by_level[lvl] = {
                        "level": lvl,
                        "key_slice_idx": s_idx,
                        "conf": det["conf"],
                        "bbox": det["bbox"],
                        "center": det["center"],
                    }

        return best_by_level
