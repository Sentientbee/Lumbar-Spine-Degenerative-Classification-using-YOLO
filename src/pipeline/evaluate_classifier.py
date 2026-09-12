import os
import sys
import argparse
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.crop_dataset import LumbarCropDataset
from src.data.dataset_builder import SEVERITIES
from src.models.severity_classifier import LumbarSeverityClassifier
from src.metrics.competition_loss import RSNALogLoss, compute_rsna_log_loss


def evaluate_classifier(
    weights_path: str = "weights/best_severity_classifier.pt",
    data_dir: Optional[str] = None,
    num_test_samples: int = 60,
    batch_size: int = 16,
    backbone: str = "resnet18",
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Independently evaluate a trained 2.5D severity classifier checkpoint.
    Computes overall accuracy, RSNA sample-weighted log loss, and class-wise metrics.
    """
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"[*] Evaluation Device: {dev}")
    print(f"[*] Loading Model Weights: {weights_path}")

    model = LumbarSeverityClassifier(backbone_name=backbone, num_classes=3, pretrained=False)
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=dev)
        model.load_state_dict(state_dict)
        print(f"[+] Loaded weights from {weights_path}")
    else:
        print(f"[!] Checkpoint not found at {weights_path}. Running with initialized weights.")

    model.to(dev)
    model.eval()

    # Load test dataset
    if data_dir is not None and os.path.exists(data_dir):
        import glob
        img_files = glob.glob(os.path.join(data_dir, "*.npy")) or glob.glob(os.path.join(data_dir, "*.png"))
        samples = [(f, 0) for f in img_files]
        dataset = LumbarCropDataset(samples=samples)
        print(f"[+] Loaded real test dataset from {data_dir} ({len(dataset)} samples)")
    else:
        print(f"[*] Using synthetic evaluation dataset ({num_test_samples} samples across 3 severities).")
        dataset = LumbarCropDataset(is_synthetic=True, num_synthetic_samples=num_test_samples)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = RSNALogLoss().to(dev)
    total_loss = 0.0
    total_samples = 0
    correct = 0

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(dev)
            y_batch = y_batch.to(dev)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            total_loss += loss.item() * len(y_batch)
            total_samples += len(y_batch)
            correct += (preds == y_batch).sum().item()

            all_targets.extend(y_batch.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    targets_arr = np.array(all_targets)
    preds_arr = np.array(all_preds)
    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 3))

    avg_loss = total_loss / max(1, total_samples)
    accuracy = correct / max(1, total_samples)
    rsna_loss = compute_rsna_log_loss(targets_arr, probs_arr) if len(targets_arr) > 0 else avg_loss

    # 3x3 Confusion Matrix
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(targets_arr, preds_arr):
        cm[t, p] += 1

    # Class-wise metrics
    class_metrics = {}
    for i, name in enumerate(SEVERITIES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics[name] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "support": int(support)
        }

    results = {
        "weights_path": weights_path,
        "total_samples": total_samples,
        "accuracy": round(float(accuracy), 4),
        "avg_loss": round(float(avg_loss), 4),
        "rsna_log_loss": round(float(rsna_loss), 4),
        "confusion_matrix": cm.tolist(),
        "class_metrics": class_metrics
    }

    # Print Formatted Clinical Evaluation Summary
    print("\n" + "=" * 65)
    print("        STAGE 2 SEVERITY CLASSIFIER EVALUATION REPORT        ")
    print("=" * 65)
    print(f"Total Test Samples:        {total_samples}")
    print(f"Overall Accuracy:          {accuracy * 100:.2f}%")
    print(f"RSNA Weighted Log Loss:    {rsna_loss:.4f} (Competition Metric: 1x/2x/4x)")
    print("-" * 65)
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | Support")
    print("-" * 65)
    for name, m in class_metrics.items():
        print(f"{name:<15} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f} | {m['support']}")
    print("-" * 65)
    print("Confusion Matrix [Rows: True, Columns: Predicted]:")
    print(f"                 {'Normal/Mild':>12} {'Moderate':>12} {'Severe':>12}")
    for i, name in enumerate(SEVERITIES):
        print(f"  {name:<13} {cm[i, 0]:>12} {cm[i, 1]:>12} {cm[i, 2]:>12}")
    print("=" * 65)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 Lumbar Severity Classifier")
    parser.add_argument("--weights", type=str, default="weights/best_severity_classifier.pt", help="Model weights path")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing test crops")
    parser.add_argument("--samples", type=int, default=60, help="Number of synthetic test samples if no data_dir")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone architecture")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, or auto)")
    args = parser.parse_args()

    evaluate_classifier(
        weights_path=args.weights,
        data_dir=args.data_dir,
        num_test_samples=args.samples,
        batch_size=args.batch_size,
        backbone=args.backbone,
        device=args.device
    )


if __name__ == "__main__":
    main()
