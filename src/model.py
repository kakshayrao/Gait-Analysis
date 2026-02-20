# =============================================================================
# model.py
# =============================================================================
# Trains a Random Forest classifier on the extracted gait features and
# evaluates performance with standard metrics.
#
# Outputs
# -------
#   - Console: Accuracy, Precision, Recall, F1-score
#   - Plots : Confusion matrix, Feature importance  (saved to output/)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def train_and_evaluate(features_df: pd.DataFrame,
                       output_dir: str = "output"):
    """
    Train a Random Forest on the feature DataFrame and print evaluation
    metrics.

    Parameters
    ----------
    features_df : pd.DataFrame
        Must contain feature columns and a 'label' column (0/1).
    output_dir : str
        Directory where plots are saved.

    Returns
    -------
    model : trained RandomForestClassifier
    feature_names : list[str]
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Separate features and label ──────────────────────────────────────
    feature_cols = [c for c in features_df.columns if c != "label"]
    X = features_df[feature_cols].values
    y = features_df["label"].values

    # ── Stratified 80 / 20 split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Train Random Forest ──────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Predictions & Metrics ────────────────────────────────────────────
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 50)
    print("  CLASSIFICATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("=" * 50 + "\n")

    # ── Confusion Matrix Plot ────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Healthy", "Parkinson"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"  → Saved confusion_matrix.png")

    # ── Feature Importance Plot ──────────────────────────────────────────
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        [feature_cols[i] for i in sorted_idx],
        importances[sorted_idx],
        color="#4C72B0",
    )
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("Feature Importance — Random Forest", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"  → Saved feature_importance.png")

    # ── Feature Comparison: Healthy vs Parkinson ─────────────────────────
    _plot_feature_comparison(features_df, feature_cols, output_dir)

    # ── Risk vs Variability scatter ──────────────────────────────────────
    _plot_risk_vs_variability(model, features_df, feature_cols, output_dir)

    return model, feature_cols


# ─── Internal Helpers ────────────────────────────────────────────────────────

def _plot_feature_comparison(features_df, feature_cols, output_dir):
    """Side-by-side box plots comparing each feature between groups."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        if i >= len(axes):
            break
        healthy  = features_df.loc[features_df["label"] == 0, col]
        parkinson = features_df.loc[features_df["label"] == 1, col]
        axes[i].boxplot(
            [healthy, parkinson],
            labels=["Healthy", "Parkinson"],
            patch_artist=True,
            boxprops=dict(facecolor="#A1C9F4"),
        )
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Gait Feature Distributions: Healthy vs Parkinson",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_comparison.png"), dpi=150)
    plt.close()
    print(f"  → Saved feature_comparison.png")


def _plot_risk_vs_variability(model, features_df, feature_cols, output_dir):
    """Scatter plot of predicted Parkinson probability vs stride variability."""
    X = features_df[feature_cols].values
    probs = model.predict_proba(X)[:, 1]  # probability of class 1

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2ca02c" if l == 0 else "#d62728"
              for l in features_df["label"]]
    ax.scatter(features_df["variability"], probs, c=colors, alpha=0.4, s=15)
    ax.set_xlabel("Stride Variability", fontsize=12)
    ax.set_ylabel("Predicted Parkinson Probability", fontsize=12)
    ax.set_title("Fall Risk Indicator vs Gait Variability",
                 fontsize=14, fontweight="bold")
    ax.axhline(0.75, color="red", linestyle="--", linewidth=1,
               label="High-risk threshold (0.75)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_vs_variability.png"), dpi=150)
    plt.close()
    print(f"  → Saved risk_vs_variability.png")
