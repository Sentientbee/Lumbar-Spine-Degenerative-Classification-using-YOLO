from typing import Union, List, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Official RSNA 2024 Severity Weights
CLASS_WEIGHTS = {
    0: 1.0,  # Normal/Mild
    1: 2.0,  # Moderate
    2: 4.0,  # Severe
}


class RSNALogLoss(nn.Module):
    """
    Official RSNA 2024 Sample-Weighted Multi-Class Log Loss (PyTorch Module).
    Weights: Normal/Mild = 1.0, Moderate = 2.0, Severe = 4.0.
    """

    def __init__(self, weights: Optional[Dict[int, float]] = None, eps: float = 1e-15):
        super().__init__()
        self.eps = eps
        weight_list = [CLASS_WEIGHTS[0], CLASS_WEIGHTS[1], CLASS_WEIGHTS[2]]
        self.register_buffer("weights", torch.tensor(weight_list, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits (B, 3).
            targets: Ground truth class indices (B,) with values in {0, 1, 2},
                     or one-hot probabilities (B, 3).
        Returns:
            Scalar sample-weighted log loss.
        """
        if targets.numel() == 0 or logits.shape[0] == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)

        if targets.ndim == 1:
            # targets are class indices: (B,)
            sample_weights = self.weights[targets]
            log_p = torch.log(probs.gather(1, targets.unsqueeze(1)).squeeze(1))
            weight_sum = torch.clamp(sample_weights.sum(), min=1e-8)
            weighted_loss = - (sample_weights * log_p).sum() / weight_sum
        else:
            # targets are one-hot / probabilities: (B, 3)
            sample_weights = (targets * self.weights.unsqueeze(0)).sum(dim=-1)
            log_p = (targets * torch.log(probs)).sum(dim=-1)
            weight_sum = torch.clamp(sample_weights.sum(), min=1e-8)
            weighted_loss = - (sample_weights * log_p).sum() / weight_sum

        return weighted_loss


def compute_rsna_log_loss(
    y_true: Union[np.ndarray, List[int]],
    y_pred_probs: np.ndarray,
    eps: float = 1e-15
) -> float:
    """
    Numpy evaluation function for RSNA 2024 Sample-Weighted Multi-class Log Loss.

    Args:
        y_true: Ground truth class indices (N,) with values in {0, 1, 2}.
        y_pred_probs: Predicted probabilities (N, 3), summing to 1 across axis 1.
        eps: Epsilon clipping to prevent log(0).

    Returns:
        Weighted log loss scalar.
    """
    y_true = np.asarray(y_true, dtype=int)
    if len(y_true) == 0:
        return 0.0

    probs = np.clip(np.asarray(y_pred_probs, dtype=float), eps, 1.0 - eps)

    # Normalize probabilities so they sum to 1
    probs = probs / probs.sum(axis=-1, keepdims=True)

    weights = np.array([CLASS_WEIGHTS[c] for c in y_true], dtype=float)
    true_class_probs = probs[np.arange(len(y_true)), y_true]
    log_losses = -np.log(true_class_probs)

    total_weight = np.sum(weights)
    if total_weight <= 0:
        return 0.0

    weighted_loss = float(np.sum(weights * log_losses) / total_weight)
    return weighted_loss
