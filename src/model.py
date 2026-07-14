"""
LayerLogic — Model Architecture
================================
ResNet-18 feature extractor + classical ML classifier pipelines.

Architecture:
    Audio .wav  → Mel-Spectrogram → ResNet-18 (frozen) → 512-d features ─┐
                                                                          ├→ concat → 1024-d → PCA → Classifier
    Image .jpg  → Resize/Norm     → ResNet-18 (frozen) → 512-d features ─┘

Classifiers: SVM, XGBoost, Random Forest, Decision Tree

Usage:
    from model import create_feature_extractors, build_classifier_pipelines
"""

import sys
# Fix Windows encoding for Unicode output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torchvision import models
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("  [!] XGBoost not installed. Skipping XGBoost classifier.")


# ─── ResNet-18 Feature Extractor ─────────────────────────────────────────────

class ResNetFeatureExtractor(nn.Module):
    """
    Frozen ResNet-18 backbone for feature extraction.

    Removes the final fully-connected classification layer and replaces it
    with an identity mapping. The output is a 512-dimensional feature vector
    per input image (or mel-spectrogram treated as a 3-channel image).

    This mirrors the approach used in the IIT KGP LPBF monitoring research
    where pretrained CNN backbones (EfficientNet-B0 / ResNet-50) extract
    features from both optical and acoustic modalities.
    """

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()

        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Remove the final FC layer (replaces 512 → 1000 with identity)
        self.feature_dim = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        # Freeze all parameters for feature extraction mode
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Feature vector of shape (B, 512)
        """
        return self.backbone(x)

    @property
    def output_dim(self) -> int:
        return self.feature_dim


def create_feature_extractors(
    pretrained: bool = True,
    freeze: bool = True,
    device: torch.device = None,
) -> tuple:
    """
    Create two separate ResNet-18 feature extractors:
    one for acoustic spectrograms, one for optical images.

    Returns:
        (acoustic_extractor, optical_extractor)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acoustic = ResNetFeatureExtractor(pretrained=pretrained, freeze=freeze).to(device)
    optical = ResNetFeatureExtractor(pretrained=pretrained, freeze=freeze).to(device)

    acoustic.eval()
    optical.eval()

    total_params = sum(p.numel() for p in acoustic.parameters())
    frozen_params = sum(p.numel() for p in acoustic.parameters() if not p.requires_grad)

    print(f"  Feature Extractors: ResNet-18 (pretrained={pretrained})")
    print(f"  Parameters per branch: {total_params:,} total, {frozen_params:,} frozen")
    print(f"  Output dimension: {acoustic.output_dim} per branch → {acoustic.output_dim * 2} fused")
    print(f"  Device: {device}")

    return acoustic, optical


# ─── Classical ML Classifier Pipelines ───────────────────────────────────────

def build_classifier_pipelines(pca_variance: float = 0.90) -> dict:
    """
    Build a dictionary of sklearn Pipeline objects for classification.

    Each pipeline: StandardScaler -> PCA (90% variance) -> Classifier

    This matches the approach from the IIT KGP thesis where fused CNN
    feature vectors are dimensionality-reduced via PCA then classified
    by XGBoost, Random Forest, and Decision Tree.

    Regularization is applied to prevent overfitting on small datasets:
    - PCA retains 90% variance (reduces dimensionality aggressively)
    - XGBoost: max_depth=3, L1/L2 regularization
    - Random Forest: max_depth=8, min_samples_leaf=5
    - Decision Tree: max_depth=5, min_samples_leaf=10

    Args:
        pca_variance: Fraction of variance to retain in PCA (default 0.90)

    Returns:
        dict mapping classifier name -> sklearn Pipeline
    """
    pipelines = {}

    # -- XGBoost (Extreme Gradient Boosting) --
    # max_depth=3 prevents memorization, reg_alpha/lambda add L1/L2 penalty
    if HAS_XGBOOST:
        pipelines["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_variance, random_state=42)),
            ("classifier", XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )),
        ])

    # -- Random Forest --
    # max_depth=8, min_samples_leaf=5 to prevent overfitting
    pipelines["Random Forest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_variance, random_state=42)),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # -- Decision Tree (baseline) --
    # Heavily constrained: max_depth=5, min_samples_leaf=10
    pipelines["Decision Tree"] = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_variance, random_state=42)),
        ("classifier", DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=10,
            random_state=42,
        )),
    ])

    print(f"  Classifier Pipelines: {list(pipelines.keys())}")
    print(f"  PCA variance retained: {pca_variance * 100:.0f}%")

    return pipelines


# ─── Model Info ──────────────────────────────────────────────────────────────

def get_model_info() -> dict:
    """Return model metadata for the API /model-info endpoint."""
    return {
        "project": "LayerLogic",
        "backbone": "ResNet-18 (ImageNet pretrained, frozen)",
        "acoustic_input": "Mel-spectrogram (224×224×3) from .wav",
        "optical_input": "Melt-pool image (224×224×3)",
        "feature_dim": "512 per branch → 1024 fused",
        "dimensionality_reduction": "PCA (95% variance)",
        "classifiers": ["XGBoost", "Random Forest", "Decision Tree"],
        "task": "Binary classification: Healthy vs Defect",
        "reference": "IIT Kharagpur LPBF in-situ monitoring research",
    }


if __name__ == "__main__":
    print("Testing model components...")
    device = torch.device("cpu")

    # Test feature extractors
    acoustic, optical = create_feature_extractors(device=device)

    # Test with random input
    dummy_spec = torch.randn(2, 3, 224, 224)
    dummy_img = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        acoustic_feats = acoustic(dummy_spec)
        optical_feats = optical(dummy_img)

    print(f"\n  Acoustic features: {acoustic_feats.shape}")  # (2, 512)
    print(f"  Optical features:  {optical_feats.shape}")     # (2, 512)

    fused = torch.cat([acoustic_feats, optical_feats], dim=1)
    print(f"  Fused features:    {fused.shape}")             # (2, 1024)

    # Test classifier pipelines
    pipelines = build_classifier_pipelines()
    print(f"\n  ✓ All model components working.")
