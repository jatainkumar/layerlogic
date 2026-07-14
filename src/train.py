"""
LayerLogic — Training & Evaluation Pipeline
=============================================
Extracts features with frozen ResNet-18, trains classical ML classifiers
(XGBoost, RF, DT), evaluates on test set, and generates publication-
quality graphs for the pitch deck.

Generated Outputs (saved to outputs/):
    - best_pipeline.pkl         Best classifier pipeline (PCA + classifier)
    - label_encoder.pkl         Label mapping
    - classifier_comparison.png F1/AUC bar chart for all classifiers
    - training_history.png      Learning curves (F1 vs training set size)
    - roc_curve.png             ROC curves with AUC for all classifiers
    - confusion_matrix.png      Confusion matrix heatmap for best classifier

Usage:
    python src/train.py
"""

import os
import sys
import time

# Fix Windows encoding for Unicode output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import numpy as np
import joblib
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report, precision_score, recall_score,
)
from sklearn.model_selection import learning_curve, StratifiedKFold, cross_val_score

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import get_dataloaders, extract_features_from_loader
from model import create_feature_extractors, build_classifier_pipelines


# ─── Configuration ───────────────────────────────────────────────────────────

PROJ_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJ_ROOT / "outputs"
DATA_DIR = PROJ_ROOT / "data"

BATCH_SIZE = 16
RANDOM_SEED = 42

# Graph styling
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#333355",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#b0b0b0",
    "ytick.color": "#b0b0b0",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
    "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": "#333355",
    "font.family": "sans-serif",
    "font.size": 11,
})

# Color palette for classifiers
COLORS = {
    "XGBoost": "#ff3366",
    "Random Forest": "#51cf66",
    "Decision Tree": "#ffd43b",
}


# ─── Feature Extraction ─────────────────────────────────────────────────────

def extract_all_features(device: torch.device) -> dict:
    """
    Extract features from all splits using frozen ResNet-18.

    Returns:
        dict with keys: train_features, train_labels, val_features, val_labels,
                        test_features, test_labels
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: Feature Extraction (Frozen ResNet-18)")
    print("=" * 60)

    # Create data loaders
    print("\n  Loading dataset...")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(
        data_dir=str(DATA_DIR),
        batch_size=BATCH_SIZE,
        num_workers=0,
        seed=RANDOM_SEED,
    )

    # Create feature extractors
    print("\n  Creating feature extractors...")
    acoustic_model, optical_model = create_feature_extractors(device=device)

    # Extract features from each split
    results = {}
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        print(f"\n  Extracting {split_name} features...")
        start = time.time()
        features, labels = extract_features_from_loader(
            acoustic_model, optical_model, loader, device
        )
        elapsed = time.time() - start
        results[f"{split_name}_features"] = features
        results[f"{split_name}_labels"] = labels
        print(f"    → {features.shape[0]} samples, {features.shape[1]}-d features ({elapsed:.1f}s)")

    return results


# ─── Classifier Training ────────────────────────────────────────────────────

def train_classifiers(data: dict) -> dict:
    """
    Train all classical ML classifiers with Stratified 5-Fold Cross-Validation.

    Uses the combined train+val features for CV, then retrains on the full
    train set for final model selection. Reports mean +/- std for each metric.

    Returns:
        dict mapping classifier name -> {pipeline, metrics, cv_scores}
    """
    print("\n" + "=" * 60)
    print("  PHASE 2: Training Classical ML Classifiers")
    print("  (Stratified 5-Fold Cross-Validation)")
    print("=" * 60)

    X_train = data["train_features"]
    y_train = data["train_labels"]
    X_val = data["val_features"]
    y_val = data["val_labels"]

    # Combine train + val for CV (test set remains fully held out)
    X_cv = np.vstack([X_train, X_val])
    y_cv = np.concatenate([y_train, y_val])

    print(f"\n  CV set: {X_cv.shape[0]} samples x {X_cv.shape[1]} features")
    print(f"  CV strategy: StratifiedKFold(n_splits=5)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = build_classifier_pipelines(pca_variance=0.90)
    results = {}

    for name, pipeline in pipelines.items():
        print(f"\n  Training {name}...")
        start = time.time()

        # Cross-validation scores
        cv_f1 = cross_val_score(pipeline, X_cv, y_cv, cv=skf, scoring="f1", n_jobs=-1)
        cv_acc = cross_val_score(pipeline, X_cv, y_cv, cv=skf, scoring="accuracy", n_jobs=-1)
        cv_auc = cross_val_score(pipeline, X_cv, y_cv, cv=skf, scoring="roc_auc", n_jobs=-1)

        # Retrain on full train set for final pipeline
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        # Evaluate on held-out val set (for graph generation)
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred, average="binary"),
            "precision": precision_score(y_val, y_pred, average="binary"),
            "recall": recall_score(y_val, y_pred, average="binary"),
            "auc": roc_auc_score(y_val, y_proba),
            "train_time": train_time,
            # CV statistics
            "cv_f1_mean": float(np.mean(cv_f1)),
            "cv_f1_std": float(np.std(cv_f1)),
            "cv_acc_mean": float(np.mean(cv_acc)),
            "cv_acc_std": float(np.std(cv_acc)),
            "cv_auc_mean": float(np.mean(cv_auc)),
            "cv_auc_std": float(np.std(cv_auc)),
        }

        results[name] = {
            "pipeline": pipeline,
            "metrics": metrics,
        }

        print(f"    CV F1:  {metrics['cv_f1_mean']:.4f} +/- {metrics['cv_f1_std']:.4f}")
        print(f"    CV AUC: {metrics['cv_auc_mean']:.4f} +/- {metrics['cv_auc_std']:.4f}")
        print(f"    CV Acc: {metrics['cv_acc_mean']:.4f} +/- {metrics['cv_acc_std']:.4f}")
        print(f"    Val F1: {metrics['f1']:.4f}  |  Time: {train_time:.2f}s")

    return results


# ─── Final Evaluation on Test Set ────────────────────────────────────────────

def evaluate_best_on_test(
    best_name: str,
    best_pipeline,
    data: dict,
) -> dict:
    """Evaluate the best classifier on the held-out test set."""
    print("\n" + "=" * 60)
    print(f"  PHASE 3: Final Evaluation — {best_name} on Test Set")
    print("=" * 60)

    X_test = data["test_features"]
    y_test = data["test_labels"]

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="binary"),
        "precision": precision_score(y_test, y_pred, average="binary"),
        "recall": recall_score(y_test, y_pred, average="binary"),
        "auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n  Test Results ({best_name}):")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    F1-Score:  {metrics['f1']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    AUC:       {metrics['auc']:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Defect"]))

    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
    }


# ─── Graph Generation ───────────────────────────────────────────────────────

def generate_classifier_comparison(results: dict, output_path: Path):
    """
    Generate grouped bar chart comparing CV F1-score and CV AUC across classifiers.
    Shows error bars from cross-validation std dev.
    Saved as classifier_comparison.png
    """
    print("  Generating classifier_comparison.png...")

    names = list(results.keys())
    f1_means = [results[n]["metrics"]["cv_f1_mean"] for n in names]
    f1_stds = [results[n]["metrics"]["cv_f1_std"] for n in names]
    auc_means = [results[n]["metrics"]["cv_auc_mean"] for n in names]
    auc_stds = [results[n]["metrics"]["cv_auc_std"] for n in names]
    colors = [COLORS.get(n, "#888888") for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, f1_means, width, yerr=f1_stds,
                   label="CV F1 (mean +/- std)", capsize=5,
                   color=colors, alpha=0.9, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, auc_means, width, yerr=auc_stds,
                   label="CV AUC (mean +/- std)", capsize=5,
                   color=colors, alpha=0.5, edgecolor="white", linewidth=0.5,
                   hatch="///")

    # Value labels on bars
    for i, bar in enumerate(bars1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + f1_stds[i] + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#e0e0e0")
    for i, bar in enumerate(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + auc_stds[i] + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=9, color="#b0b0b0")

    ax.set_xlabel("Classifier", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("LayerLogic -- Classifier Performance (5-Fold CV)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Highlight best
    best_idx = np.argmax(f1_means)
    ax.annotate("Best", xy=(best_idx - width / 2, f1_means[best_idx]),
                xytext=(best_idx - width / 2, f1_means[best_idx] + f1_stds[best_idx] + 0.06),
                ha="center", fontsize=11, color="#ffd43b", fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    -> Saved: {output_path}")


def generate_learning_curves(results: dict, data: dict, output_path: Path):
    """
    Generate learning curves showing F1-score vs training set size.
    This demonstrates convergence and generalisation — replaces epoch curves
    since classical ML classifiers don't have iterative training.
    Saved as training_history.png
    """
    print("  Generating training_history.png (learning curves)...")

    X_train = data["train_features"]
    y_train = data["train_labels"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name, result in results.items():
        pipeline = result["pipeline"]
        color = COLORS.get(name, "#888888")

        try:
            train_sizes, train_scores, val_scores = learning_curve(
                pipeline, X_train, y_train,
                train_sizes=np.linspace(0.2, 1.0, 6),
                cv=3, scoring="f1", n_jobs=-1, random_state=42,
            )

            # Plot training score
            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            axes[0].plot(train_sizes, train_mean, "-o", color=color, label=name,
                         markersize=5, linewidth=2)
            axes[0].fill_between(train_sizes, train_mean - train_std,
                                 train_mean + train_std, alpha=0.15, color=color)

            # Plot validation score
            val_mean = val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)
            axes[1].plot(train_sizes, val_mean, "-s", color=color, label=name,
                         markersize=5, linewidth=2)
            axes[1].fill_between(train_sizes, val_mean - val_std,
                                 val_mean + val_std, alpha=0.15, color=color)

        except Exception as e:
            print(f"    [!] Learning curve failed for {name}: {e}")

    for ax, title in zip(axes, ["Training F1-Score", "Cross-Validation F1-Score"]):
        ax.set_xlabel("Training Set Size", fontsize=12, fontweight="bold")
        ax.set_ylabel("F1-Score", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

    fig.suptitle("LayerLogic — Learning Curves (F1 vs Training Size)",
                 fontsize=15, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    → Saved: {output_path}")


def generate_roc_curves(results: dict, data: dict, output_path: Path):
    """
    Generate ROC curves for all classifiers on the test set.
    Saved as roc_curve.png
    """
    print("  Generating roc_curve.png...")

    X_test = data["test_features"]
    y_test = data["test_labels"]

    fig, ax = plt.subplots(figsize=(8, 8))

    for name, result in results.items():
        pipeline = result["pipeline"]
        color = COLORS.get(name, "#888888")

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)

        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f"{name} (AUC = {auc_val:.3f})")

    # Diagonal baseline
    ax.plot([0, 1], [0, 1], "--", color="#555577", linewidth=1.5,
            label="Random (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=13, fontweight="bold")
    ax.set_title("LayerLogic — ROC Curves (All Classifiers)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="lower right",
              framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    → Saved: {output_path}")


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classifier_name: str,
    output_path: Path,
):
    """
    Generate a beautiful confusion matrix heatmap.
    Saved as confusion_matrix.png
    """
    print("  Generating confusion_matrix.png...")

    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 7))

    # Create annotations with both count and percentage
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    # Custom colormap (dark blue theme)
    cmap = sns.color_palette("rocket", as_cmap=True)

    sns.heatmap(
        cm, annot=annotations, fmt="",
        xticklabels=["Healthy", "Defect"],
        yticklabels=["Healthy", "Defect"],
        cmap=cmap, linewidths=2, linecolor="#1a1a2e",
        ax=ax, annot_kws={"size": 16, "fontweight": "bold"},
        cbar_kws={"label": "Count", "shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(f"LayerLogic — Confusion Matrix ({classifier_name})",
                 fontsize=15, fontweight="bold", pad=15)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    → Saved: {output_path}")


from sklearn.manifold import TSNE

def generate_tsne_clustering(data: dict, output_path: Path):
    """
    Generate t-SNE clustering scatter plot of the extracted 1024-d features.
    This helps visualize the separability of the classes in the feature space.
    Saved as feature_clustering.png
    """
    print("  Generating feature_clustering.png (t-SNE)...")

    # Combine train and test for a complete picture
    X_all = np.vstack([data["train_features"], data["val_features"], data["test_features"]])
    y_all = np.concatenate([data["train_labels"], data["val_labels"], data["test_labels"]])

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X_all)

    fig, ax = plt.subplots(figsize=(8, 7))

    healthy_mask = y_all == 0
    defect_mask = y_all == 1

    ax.scatter(X_2d[healthy_mask, 0], X_2d[healthy_mask, 1],
               c="#00d4ff", label="Healthy", alpha=0.7, edgecolors="white", s=60)
    ax.scatter(X_2d[defect_mask, 0], X_2d[defect_mask, 1],
               c="#ff3366", label="Defect", alpha=0.7, edgecolors="white", s=60)

    ax.set_xlabel("t-SNE Dimension 1", fontsize=13, fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=13, fontweight="bold")
    ax.set_title("LayerLogic -- Feature Space Clustering (t-SNE)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    -> Saved: {output_path}")


# ─── Save Artifacts ──────────────────────────────────────────────────────────

def save_artifacts(
    best_name: str,
    best_pipeline,
    test_metrics: dict,
    all_results: dict,
):
    """Save the best pipeline and metadata."""
    print("\n  Saving artifacts...")

    # Save best pipeline
    pipeline_path = OUTPUT_DIR / "best_pipeline.pkl"
    joblib.dump(best_pipeline, pipeline_path)
    print(f"    → Pipeline: {pipeline_path}")

    # Save label encoder (simple mapping)
    label_map = {"healthy": 0, "defect": 1}
    label_path = OUTPUT_DIR / "label_encoder.pkl"
    joblib.dump(label_map, label_path)
    print(f"    → Labels: {label_path}")

    # Save training summary
    summary = {
        "best_classifier": best_name,
        "test_metrics": test_metrics,
        "all_classifiers": {
            name: result["metrics"]
            for name, result in all_results.items()
        },
    }
    summary_path = OUTPUT_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"    → Summary: {summary_path}")


# ─── Main Training Pipeline ─────────────────────────────────────────────────

def main():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  LayerLogic — Training Pipeline".center(58) + "║")
    print("║" + "  ResNet-18 + Classical ML Classifiers".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    start_time = time.time()

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Phase 1: Extract features
    data = extract_all_features(device)

    # Phase 2: Train classifiers
    results = train_classifiers(data)

    # Select best classifier by CV F1-score (more robust than single split)
    best_name = max(results.keys(), key=lambda n: results[n]["metrics"]["cv_f1_mean"])
    best_pipeline = results[best_name]["pipeline"]
    cv_f1 = results[best_name]['metrics']['cv_f1_mean']
    cv_f1_std = results[best_name]['metrics']['cv_f1_std']
    cv_auc = results[best_name]['metrics']['cv_auc_mean']
    print(f"\n  Best Classifier: {best_name} "
          f"(CV F1={cv_f1:.4f}+/-{cv_f1_std:.4f}, "
          f"CV AUC={cv_auc:.4f})")

    # Phase 3: Evaluate best on test set
    test_results = evaluate_best_on_test(best_name, best_pipeline, data)

    # Phase 4: Generate graphs
    print("\n" + "=" * 60)
    print("  PHASE 4: Generating Publication-Quality Graphs")
    print("=" * 60 + "\n")

    generate_classifier_comparison(results, OUTPUT_DIR / "classifier_comparison.png")
    generate_learning_curves(results, data, OUTPUT_DIR / "training_history.png")
    generate_roc_curves(results, data, OUTPUT_DIR / "roc_curve.png")
    generate_confusion_matrix(
        test_results["y_test"], test_results["y_pred"],
        best_name, OUTPUT_DIR / "confusion_matrix.png"
    )
    generate_tsne_clustering(data, OUTPUT_DIR / "feature_clustering.png")

    # Phase 5: Save artifacts
    save_artifacts(best_name, best_pipeline, test_results["metrics"], results)

    # Done
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  Training pipeline complete in {total_time:.1f}s")
    print(f"  Best: {best_name} -- F1={test_results['metrics']['f1']:.4f}, "
          f"AUC={test_results['metrics']['auc']:.4f}")
    print(f"  Outputs: {OUTPUT_DIR}")
    print(f"\n  Next step:")
    print(f"    python src/api.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
