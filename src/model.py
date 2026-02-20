# =============================================================================
# model.py
# =============================================================================
# Trains multiple classifiers on extracted gait features:
#   1. Random Forest   (ensemble, interpretable)
#   2. XGBoost         (gradient boosting, high accuracy)
#   3. LSTM            (sequence modeling via Keras)
#
# Outputs
# -------
#   - Console : Accuracy, Precision, Recall, F1-score for each model
#   - Plots   : Confusion matrices, Feature importance  (saved to output/)
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay)

# XGBoost
from xgboost import XGBClassifier

# Suppress TensorFlow info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: print metrics for any model
# ─────────────────────────────────────────────────────────────────────────────
def _print_metrics(name, y_true, y_pred):
    """Print classification metrics for a named model."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n  {'─' * 44}")
    print(f"  {name}")
    print(f"  {'─' * 44}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _plot_confusion(name, y_true, y_pred, output_dir, filename):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Healthy", "Parkinson"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"  → Saved {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. RANDOM FOREST + XGBOOST  (traditional ML)
# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(features_df: pd.DataFrame,
                       output_dir: str = "output"):
    """
    Train Random Forest and XGBoost on the feature DataFrame.

    Returns
    -------
    rf_model : trained RandomForestClassifier  (used for live monitoring)
    feature_names : list[str]
    """
    os.makedirs(output_dir, exist_ok=True)

    feature_cols = [c for c in features_df.columns if c not in ("label", "subject_id")]
    X = features_df[feature_cols].values
    y = features_df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Random Forest ────────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    print("\n" + "=" * 50)
    print("  CLASSIFICATION RESULTS")
    print("=" * 50)
    _print_metrics("Random Forest", y_test, rf_pred)
    _plot_confusion("Random Forest", y_test, rf_pred, output_dir,
                    "confusion_matrix_rf.png")

    # ── XGBoost ──────────────────────────────────────────────────────────
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    _print_metrics("XGBoost", y_test, xgb_pred)
    _plot_confusion("XGBoost", y_test, xgb_pred, output_dir,
                    "confusion_matrix_xgb.png")
    print("=" * 50)

    # ── Feature Importance (Random Forest) ───────────────────────────────
    importances = rf.feature_importances_
    sorted_idx  = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        [feature_cols[i] for i in sorted_idx],
        importances[sorted_idx],
        color="#4C72B0",
    )
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("Feature Importance — Random Forest",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"  → Saved feature_importance.png")

    # ── Feature Comparison: Healthy vs Parkinson ─────────────────────────
    _plot_feature_comparison(features_df, feature_cols, output_dir)

    # ── Risk vs Variability scatter ──────────────────────────────────────
    _plot_risk_vs_variability(rf, features_df, feature_cols, output_dir)

    # ── Model Comparison Bar Chart ───────────────────────────────────────
    _plot_model_comparison(
        {"Random Forest": _get_metrics(y_test, rf_pred),
         "XGBoost":       _get_metrics(y_test, xgb_pred)},
        output_dir,
    )

    return rf, feature_cols


def _get_metrics(y_true, y_pred):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. LSTM  (sequence modeling)
# ─────────────────────────────────────────────────────────────────────────────
def train_lstm(features_df: pd.DataFrame, output_dir: str = "output",
               seq_len: int = 10):
    """
    Train an LSTM on temporal sequences of gait features.

    Groups consecutive windows from each subject into sequences of length
    `seq_len`, scales features with StandardScaler, and uses a stacked
    2-layer LSTM for binary PD classification.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Lazy import to avoid slow TF startup if not needed
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import StandardScaler

    # Suppress TF warnings
    tf.get_logger().setLevel("ERROR")

    feature_cols = [c for c in features_df.columns
                    if c not in ("label", "subject_id")]

    # ── Build sequences per subject ──────────────────────────────────────
    sequences, labels = [], []
    for sid, grp in features_df.groupby("subject_id", sort=False):
        X_subj = grp[feature_cols].values
        y_subj = grp["label"].values[0]
        for i in range(len(X_subj) - seq_len + 1):
            sequences.append(X_subj[i : i + seq_len])
            labels.append(y_subj)

    X_all = np.array(sequences, dtype=np.float32)   # (N, seq_len, 7)
    y_all = np.array(labels, dtype=np.int32)

    print(f"\n  LSTM sequences: {X_all.shape[0]}  "
          f"(shape per sample: {X_all.shape[1]}×{X_all.shape[2]})")

    # ── Stratified split (same approach as RF/XGBoost) ───────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    # ── Scale features (fit on train only) ───────────────────────────────
    n_feat = X_train.shape[2]
    scaler = StandardScaler()
    # Reshape to 2D for fitting, then reshape back
    scaler.fit(X_train.reshape(-1, n_feat))
    X_train = scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_test  = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    # Class weights to handle imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    total = n_neg + n_pos
    class_weight = {0: total / (2 * n_neg), 1: total / (2 * n_pos)}
    print(f"  Class weights: {class_weight}")

    # ── Build LSTM Model ─────────────────────────────────────────────────
    model = Sequential([
        LSTM(128, input_shape=(seq_len, len(feature_cols)),
             return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("  Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.15,
        class_weight=class_weight,
        callbacks=[EarlyStopping(patience=8, restore_best_weights=True)],
        verbose=0,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = _print_metrics("LSTM", y_test, y_pred)
    _plot_confusion("LSTM", y_test, y_pred, output_dir,
                    "confusion_matrix_lstm.png")

    # ── Training History Plot ────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("LSTM Training Loss", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history["accuracy"], label="Train Acc")
    ax2.plot(history.history["val_accuracy"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("LSTM Training Accuracy", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_training_history.png"), dpi=150)
    plt.close()
    print(f"  → Saved lstm_training_history.png")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Internal plot helpers
# ─────────────────────────────────────────────────────────────────────────────
def _plot_feature_comparison(features_df, feature_cols, output_dir):
    """Side-by-side box plots comparing each feature between groups."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        if i >= len(axes):
            break
        healthy   = features_df.loc[features_df["label"] == 0, col]
        parkinson = features_df.loc[features_df["label"] == 1, col]
        axes[i].boxplot(
            [healthy, parkinson],
            labels=["Healthy", "Parkinson"],
            patch_artist=True,
            boxprops=dict(facecolor="#A1C9F4"),
        )
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].grid(axis="y", alpha=0.3)

    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Gait Feature Distributions: Healthy vs Parkinson",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_comparison.png"), dpi=150)
    plt.close()
    print(f"  → Saved feature_comparison.png")


def _plot_risk_vs_variability(model, features_df, feature_cols, output_dir):
    """Scatter: predicted Parkinson probability vs stride variability."""
    X = features_df[feature_cols].values
    probs = model.predict_proba(X)[:, 1]

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


def _plot_model_comparison(results: dict, output_dir: str):
    """Bar chart comparing metrics across all trained models."""
    metrics = list(next(iter(results.values())).keys())
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, name in enumerate(model_names):
        vals = [results[name][m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=name,
               color=colors[i % len(colors)])

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150)
    plt.close()
    print(f"  → Saved model_comparison.png")
