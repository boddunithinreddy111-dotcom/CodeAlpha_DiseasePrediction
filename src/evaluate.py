"""
Clinical Evaluation and Metrics Visualization Module for Disease Prediction.
Computes Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, and ROC-AUC.
Generates ROC curves, confusion matrices, and feature importance charts.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
TARGET_LABELS = ["Normal / Low Risk", "Heart Disease Detected"]


def calculate_medical_metrics(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Computes clinical diagnostic evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    sensitivity = recall_score(y_test, y_pred, zero_division=0)  # Recall is Sensitivity in medicine
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

    return {
        "Accuracy": acc,
        "Precision": precision,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "Confusion Matrix": cm,
        "Classification Report": classification_report(
            y_test, y_pred, target_names=TARGET_LABELS, zero_division=0
        )
    }


def plot_clinical_confusion_matrix(cm: np.ndarray, model_name: str, output_dir: Path = FIGURES_DIR):
    """
    Plots normalized medical confusion matrix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Reds",
        xticklabels=TARGET_LABELS,
        yticklabels=TARGET_LABELS,
        cbar=True,
        square=True
    )
    plt.title(f"Clinical Confusion Matrix — {model_name}", fontsize=13, pad=12)
    plt.xlabel("Predicted Diagnostic Class", fontsize=11)
    plt.ylabel("Actual True Condition", fontsize=11)
    plt.tight_layout()

    filename = output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[CHART SAVED] {filename}")


def plot_clinical_roc_curve(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str, output_dir: Path = FIGURES_DIR):
    """
    Plots ROC-AUC curve with sensitivity vs (1 - specificity).
    """
    if not hasattr(model, "predict_proba"):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_score = auc(fpr, tpr)

    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color="#DC2626", lw=2.5, label=f"{model_name} (AUC = {roc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title(f"Clinical ROC Curve — {model_name}", fontsize=13, pad=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    filename = output_dir / f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[CHART SAVED] {filename}")


def plot_clinical_feature_importance(model, feature_names: List[str], model_name: str, top_n: int = 12, output_dir: Path = FIGURES_DIR):
    """
    Plots top clinical biomarker importances for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_scores = [importances[i] for i in indices]

    plt.figure(figsize=(9, 5.5))
    sns.barplot(x=top_scores, y=top_features, palette="rocket")
    plt.title(f"Top {top_n} Predictive Biomarkers — {model_name}", fontsize=13, pad=12)
    plt.xlabel("Diagnostic Importance Score", fontsize=11)
    plt.ylabel("Clinical Biomarker / Feature", fontsize=11)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    filename = output_dir / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[CHART SAVED] {filename}")


def plot_medical_model_benchmark(results_df, output_dir: Path = FIGURES_DIR):
    """
    Plots side-by-side comparison bar chart for all evaluated clinical algorithms.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_to_plot = ["Accuracy", "Sensitivity (Recall)", "Specificity", "F1-Score", "ROC-AUC"]
    available_metrics = [m for m in metrics_to_plot if m in results_df.columns]

    df_melted = results_df.melt(
        id_vars=["Model"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="Spectral")
    plt.ylim(0, 1.05)
    plt.title("Algorithm Comparison: Disease Prediction from Medical Data", fontsize=13, pad=12)
    plt.ylabel("Score", fontsize=11)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    filename = output_dir / "medical_model_benchmark_comparison.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[CHART SAVED] {filename}")
