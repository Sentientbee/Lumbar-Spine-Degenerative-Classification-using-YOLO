import os
import sys
import argparse
import time
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.crop_dataset import LumbarCropDataset
from src.data.augmentations import MedicalMultiSliceAugmentations
from src.models.severity_classifier import LumbarSeverityClassifier
from src.metrics.competition_loss import RSNALogLoss, compute_rsna_log_loss, CLASS_WEIGHTS


class TransformedSubset(torch.utils.data.Dataset):
    """Wraps a Dataset or Subset and applies an augmentation transform strictly on retrieval."""
    def __init__(self, subset: torch.utils.data.Dataset, transform: Optional[MedicalMultiSliceAugmentations] = None):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.subset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        total_samples += len(y_batch)

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    all_targets = []
    all_probs = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        total_loss += loss.item() * len(y_batch)
        total_samples += len(y_batch)
        correct += (preds == y_batch).sum().item()

        all_targets.append(y_batch.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / max(1, total_samples)
    accuracy = correct / max(1, total_samples)

    all_targets = np.concatenate(all_targets, axis=0) if all_targets else np.array([])
    all_probs = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
    rsna_loss = compute_rsna_log_loss(all_targets, all_probs) if len(all_targets) > 0 else avg_loss

    return avg_loss, accuracy, rsna_loss


def train_severity_classifier(
    data_dir: Optional[str] = None,
    demo: bool = True,
    backbone: str = "resnet18",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    output_dir: str = "weights",
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Train the 2.5D multi-slice lumbar severity classifier.
    """
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Training Device: {dev}")
    print(f"[*] Stage 2 Backbone: {backbone}")

    # Dataset preparation
    augmenter = MedicalMultiSliceAugmentations()

    if demo or data_dir is None or not os.path.exists(data_dir):
        print("[!] Using synthetic demonstration dataset (120 volumetric samples).")
        full_dataset = LumbarCropDataset(is_synthetic=True, num_synthetic_samples=120)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        raw_train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
        train_ds = TransformedSubset(raw_train_ds, transform=augmenter)
    else:
        # Load from actual image files in data_dir
        import glob
        img_files = glob.glob(os.path.join(data_dir, "*.npy")) or glob.glob(os.path.join(data_dir, "*.png"))
        samples = [(f, 0) for f in img_files]  # Default dummy labels if standalone
        full_dataset = LumbarCropDataset(samples=samples, transform=None)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        raw_train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
        train_ds = TransformedSubset(raw_train_ds, transform=augmenter)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[+] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Initialize model, loss, and optimizer
    model = LumbarSeverityClassifier(backbone_name=backbone, num_classes=3, pretrained=False)
    model.to(dev)

    criterion = RSNALogLoss().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_weights_path = os.path.join(output_dir, "best_severity_classifier.pt")

    print("\n" + "=" * 70)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val Acc':<9} | {'RSNA LogLoss':<12} | Status")
    print("=" * 70)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "rsna_loss": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, dev)
        val_loss, val_acc, rsna_loss = evaluate(model, val_loader, criterion, dev)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["rsna_loss"].append(rsna_loss)

        status = ""
        if rsna_loss < best_val_loss:
            best_val_loss = rsna_loss
            torch.save(model.state_dict(), best_weights_path)
            status = "[BEST] Checkpoint Saved"

        elapsed = time.time() - t0
        print(f"[{epoch:>2}/{epochs:<2}]   | {train_loss:<12.4f} | {val_loss:<10.4f} | {val_acc*100:<8.1f}% | {rsna_loss:<12.4f} | {status} ({elapsed:.1f}s)")

    print("=" * 70)
    print(f"[+] Training complete. Best weights saved to: {best_weights_path}")
    print(f"[+] Best Validation RSNA Weighted Log Loss: {best_val_loss:.4f}\n")

    return {
        "best_weights_path": best_weights_path,
        "best_val_loss": best_val_loss,
        "history": history
    }


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 2.5D Lumbar Severity Classifier")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing volumetric crop files")
    parser.add_argument("--demo", action="store_true", default=True, help="Run on synthetic demo data")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone architecture")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--output_dir", type=str, default="weights", help="Directory to save checkpoints")
    args = parser.parse_args()

    train_severity_classifier(
        data_dir=args.data_dir,
        demo=args.demo,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
