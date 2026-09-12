from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as models
except ImportError:
    models = None

from src.data.dataset_builder import SEVERITIES, SEVERITY_TO_ID, ID_TO_SEVERITY


class LumbarSeverityClassifier(nn.Module):
    """
    Stage 2: 2.5D Multi-Slice Severity Classifier.
    Takes a 3-slice cropped MRI volume (B, C=3, H, W) centered at an anatomical disc level
    and classifies condition severity into [Normal/Mild, Moderate, Severe].
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 3,
        pretrained: bool = False,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        if models is not None and hasattr(models, backbone_name):
            # Load torchvision backbone
            weights = "DEFAULT" if pretrained else None
            backbone_fn = getattr(models, backbone_name)
            base_model = backbone_fn(weights=weights)

            if hasattr(base_model, "fc"):
                in_features = base_model.fc.in_features
                base_model.fc = nn.Identity()
                self.feature_extractor = base_model
            elif hasattr(base_model, "classifier"):
                if isinstance(base_model.classifier, nn.Sequential):
                    in_features = base_model.classifier[-1].in_features
                    base_model.classifier[-1] = nn.Identity()
                else:
                    in_features = base_model.classifier.in_features
                    base_model.classifier = nn.Identity()
                self.feature_extractor = base_model
            else:
                raise ValueError(f"Unsupported backbone structure: {backbone_name}")
        else:
            # Lightweight standard CNN fallback
            in_features = 256
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 3, H, W) normalized to [0, 1].
        Returns:
            Logits tensor of shape (B, 3).
        """
        feats = self.feature_extractor(x)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        logits = self.classifier(feats)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute softmax probabilities for each severity class."""
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs

    @torch.no_grad()
    def predict_dict(self, x: torch.Tensor) -> Dict[str, float]:
        """Convenience method returning {Normal/Mild: p0, Moderate: p1, Severe: p2} for a single sample."""
        if x.ndim == 3:
            x = x.unsqueeze(0)
        probs = self.predict_proba(x)[0].cpu().numpy()
        return {
            SEVERITIES[0]: float(probs[0]),
            SEVERITIES[1]: float(probs[1]),
            SEVERITIES[2]: float(probs[2]),
        }
