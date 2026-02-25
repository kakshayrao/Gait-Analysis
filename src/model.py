# =============================================================================
# model.py
# =============================================================================
# Trains multiple classifiers on extracted gait features:
#   1. Random Forest   (ensemble, interpretable)
#   2. XGBoost         (gradient boosting, high accuracy)
#   3. LSTM            (sequence modeling via Keras)
#
# Key design choices
# ------------------
#   - GroupShuffleSplit by subject_id prevents data leakage
#   - Class-weight balancing handles Healthy / PD imbalance
#   - LSTM model + scaler + metrics are persisted to disk
#
# Outputs
# -------
#   - Console : Accuracy, Precision, Recall, F1-score for each model
#   - Plots   : Confusion matrices, learning curves, model comparison
#   - Files   : lstm_model.keras, lstm_scaler.joblib, lstm_metrics.json
# =============================================================================

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# XGBoost
from xgboost import XGBClassifier

# Suppress TensorFlow info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# -----------------------------------------------------------------------------
# Helper: print metrics for any model
# -----------------------------------------------------------------------------
def _print_metrics(name, y_true, y_pred):
    """Print classification metrics for a named model."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n  {'-' * 44}")
    print(f"  {name}")
    print(f"  {'-' * 44}")
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
    ax.set_title(f"Confusion Matrix -- {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"  -> Saved {filename}")


def _get_metrics(y_true, y_pred):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
    }


def _subject_split(features_df, feature_cols, test_size=0.2, random_state=42):
    """
    Split data by subject_id so that no subject appears in both train
    and test sets.  This prevents data-leakage that inflates metrics.
    """
    X = features_df[feature_cols].values
    y = features_df["label"].values
    groups = features_df["subject_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                           random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    return (X[train_idx], X[test_idx],
            y[train_idx], y[test_idx])


# -----------------------------------------------------------------------------
# 1. RANDOM FOREST + XGBOOST  (traditional ML -- improved)
# -----------------------------------------------------------------------------
def train_and_evaluate(features_df: pd.DataFrame,
                       output_dir: str = "output"):
    """
    Train Random Forest and XGBoost on the feature DataFrame.

    Uses GroupShuffleSplit by subject_id to prevent data leakage and
    class_weight balancing to handle class imbalance.

    Returns
    -------
    rf_model : trained RandomForestClassifier  (used for live monitoring)
    feature_names : list[str]
    """
    os.makedirs(output_dir, exist_ok=True)

    feature_cols = [c for c in features_df.columns
                    if c not in ("label", "subject_id")]
    X_train, X_test, y_train, y_test = _subject_split(
        features_df, feature_cols
    )

    # Class weight ratio for XGBoost
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

    print(f"\n  Train: {len(X_train)}  Test: {len(X_test)}  "
          f"(subject-level split -- no data leakage)")
    print(f"  Class balance -> Healthy: {n_neg}  Parkinson: {n_pos}  "
          f"(scale_pos_weight={scale_pos:.2f})")

    # -- Random Forest ----------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    print("\n" + "=" * 50)
    print("  CLASSIFICATION RESULTS")
    print("=" * 50)
    _print_metrics("Random Forest", y_test, rf_pred)
    _plot_confusion("Random Forest", y_test, rf_pred, output_dir,
                    "confusion_matrix_rf.png")

    # -- XGBoost ----------------------------------------------------------
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos,
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

    # -- Save models ------------------------------------------------------
    joblib.dump(rf, os.path.join(output_dir, "rf_model.pkl"))
    joblib.dump(xgb, os.path.join(output_dir, "xgb_model.pkl"))
    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(list(feature_cols), f)
    print("  -> Saved rf_model.pkl, xgb_model.pkl, feature_names.json")

    # -- Learning Curves (RF + XGBoost) -----------------------------------
    _plot_learning_curves(rf, xgb, X_train, y_train,
                          features_df["subject_id"].values,
                          feature_cols, features_df, output_dir)

    return rf, feature_cols


# -----------------------------------------------------------------------------
# 2. LSTM  (sequence modeling -- with persistence)
# -----------------------------------------------------------------------------
LSTM_MODEL_PATH  = os.path.join("output", "lstm_model.keras")
LSTM_SCALER_PATH = os.path.join("output", "lstm_scaler.joblib")
LSTM_METRICS_PATH = os.path.join("output", "lstm_metrics.json")


def load_cached_lstm():
    """
    Attempt to load a previously-trained LSTM model, scaler, and metrics.

    Returns
    -------
    dict with keys model, scaler, seq_len, accuracy, precision, recall, f1
    or None if files are missing.
    """
    if (os.path.exists(LSTM_MODEL_PATH)
            and os.path.exists(LSTM_SCALER_PATH)
            and os.path.exists(LSTM_METRICS_PATH)):
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        model  = tf.keras.models.load_model(LSTM_MODEL_PATH)
        scaler = joblib.load(LSTM_SCALER_PATH)
        with open(LSTM_METRICS_PATH) as f:
            saved = json.load(f)

        print("  [OK] Loaded cached LSTM model from disk")
        return {
            "model":     model,
            "scaler":    scaler,
            "seq_len":   saved.get("seq_len", 10),
            "accuracy":  saved["accuracy"],
            "precision": saved["precision"],
            "recall":    saved["recall"],
            "f1":        saved["f1"],
        }
    return None


def train_lstm(features_df: pd.DataFrame, output_dir: str = "output",
               seq_len: int = 10, force_retrain: bool = False):
    """
    Train an LSTM on temporal sequences of gait features.

    Groups consecutive windows from each subject into sequences of length
    `seq_len`, scales features with StandardScaler, and uses a stacked
    2-layer LSTM for binary PD classification.

    The trained model, scaler, and metrics are saved to disk so that
    subsequent runs can skip training.
    """
    # -- Try loading from cache first -------------------------------------
    if not force_retrain:
        cached = load_cached_lstm()
        if cached is not None:
            return cached

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

    # -- Build sequences per subject --------------------------------------
    # Track which subject each sequence belongs to for subject-level split
    sequences, labels, seq_subjects = [], [], []
    for sid, grp in features_df.groupby("subject_id", sort=False):
        X_subj = grp[feature_cols].values
        y_subj = grp["label"].values[0]
        for i in range(len(X_subj) - seq_len + 1):
            sequences.append(X_subj[i : i + seq_len])
            labels.append(y_subj)
            seq_subjects.append(sid)

    X_all = np.array(sequences, dtype=np.float32)
    y_all = np.array(labels, dtype=np.int32)
    groups_all = np.array(seq_subjects)

    print(f"\n  LSTM sequences: {X_all.shape[0]}  "
          f"(shape per sample: {X_all.shape[1]}x{X_all.shape[2]})")

    # -- Subject-level split (same as RF/XGBoost -- no data leakage) ------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_all, y_all, groups_all))
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    print(f"  Train: {len(X_train)}  Test: {len(X_test)}  "
          f"(subject-level split -- no data leakage)")

    # -- Scale features (fit on train only) -------------------------------
    n_feat = X_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_feat))
    X_train = scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_test  = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    # Class weights to handle imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    total = n_neg + n_pos
    class_weight = {0: total / (2 * n_neg), 1: total / (2 * n_pos)}
    print(f"  Class weights: {class_weight}")

    # -- Build LSTM Model (scaled for 20 features, moderate regularization) -
    from tensorflow.keras.regularizers import l2

    model = Sequential([
        LSTM(128, input_shape=(seq_len, len(feature_cols)),
             return_sequences=True, kernel_regularizer=l2(0.0005)),
        Dropout(0.3),
        LSTM(64, return_sequences=False, kernel_regularizer=l2(0.0005)),
        Dropout(0.3),
        Dense(64, activation="relu", kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Dropout(0.25),
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
        epochs=100,
        batch_size=128,
        validation_split=0.15,
        class_weight=class_weight,
        callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
        verbose=0,
    )

    # -- Evaluate ---------------------------------------------------------
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = _print_metrics("LSTM", y_test, y_pred)
    _plot_confusion("LSTM", y_test, y_pred, output_dir,
                    "confusion_matrix_lstm.png")

    # -- Training History Plot --------------------------------------------
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
    print(f"  -> Saved lstm_training_history.png")

    # -- Persist model, scaler, and metrics to disk -----------------------
    model.save(LSTM_MODEL_PATH)
    joblib.dump(scaler, LSTM_SCALER_PATH)
    saved_metrics = {
        "accuracy":  metrics["accuracy"],
        "precision": metrics["precision"],
        "recall":    metrics["recall"],
        "f1":        metrics["f1"],
        "seq_len":   seq_len,
    }
    with open(LSTM_METRICS_PATH, "w") as f:
        json.dump(saved_metrics, f, indent=2)
    print(f"  -> Saved lstm_model.keras, lstm_scaler.joblib, lstm_metrics.json")

    metrics["model"] = model
    metrics["scaler"] = scaler
    metrics["seq_len"] = seq_len
    return metrics


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def _plot_learning_curves(rf, xgb, X_train, y_train,
                          all_groups, feature_cols, features_df, output_dir):
    """Generate learning curves for RF and XGBoost."""
    from sklearn.model_selection import GroupKFold

    groups_train = features_df.loc[
        features_df.index.isin(
            features_df.index[:len(X_train)]
        ), "subject_id"
    ].values[:len(X_train)]

    # Use simple 3-fold CV for speed
    train_sizes = np.linspace(0.2, 1.0, 6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, model, name in [(axes[0], rf, "Random Forest"),
                             (axes[1], xgb, "XGBoost")]:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42,
        )
        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        ax.fill_between(train_sizes_abs, train_mean - train_std,
                        train_mean + train_std, alpha=0.15, color="#4C72B0")
        ax.fill_between(train_sizes_abs, val_mean - val_std,
                        val_mean + val_std, alpha=0.15, color="#DD8452")
        ax.plot(train_sizes_abs, train_mean, 'o-', color="#4C72B0",
                label="Training Score")
        ax.plot(train_sizes_abs, val_mean, 'o-', color="#DD8452",
                label="Validation Score")
        ax.set_xlabel("Training Set Size", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"Learning Curve -- {name}",
                     fontsize=13, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.5, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves_rf_xgb.png"), dpi=150)
    plt.close()
    print(f"  -> Saved learning_curves_rf_xgb.png")


def plot_all_model_comparison(results: dict, output_dir: str):
    """
    Bar chart comparing metrics across ALL trained models (RF, XGBoost, LSTM).
    Called from app.py / main.py after all models are trained.
    """
    metrics = list(next(iter(results.values())).keys())
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, name in enumerate(model_names):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=colors[i % len(colors)])
        # Add value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison -- All Models",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150)
    plt.close()
    print(f"  -> Saved model_comparison.png")
