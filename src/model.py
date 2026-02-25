"""
src/model.py
============
ML models for IMU fall-risk detection:
  1. Random Forest  (tabular features)
  2. XGBoost        (tabular features)
  3. LSTM           (raw IMU sequences)

Subject-level GroupShuffleSplit ensures no data leakage.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

OUTPUT_DIR = "output"


# ─── Helper: classification report ────────────────────────────────────────────

def _report(name: str, y_true, y_pred) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n  {'─'*46}")
    print(f"  {name}")
    print(f"  {'─'*46}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1}


def _save_confusion_matrix(y_true, y_pred,
                            model_name: str,
                            class_names: list,
                            save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  -> Saved {os.path.basename(save_path)}")


# ─── 1 & 2: Random Forest + XGBoost ──────────────────────────────────────────

def train_classical_models(features_df, output_dir: str = OUTPUT_DIR) -> dict:
    """
    Train RF and XGBoost on tabular gait features with GroupShuffleSplit.

    Parameters
    ----------
    features_df : DataFrame from build_feature_dataframe()

    Returns
    -------
    dict with metrics for RF and XGBoost
    """
    os.makedirs(output_dir, exist_ok=True)

    feature_cols = [c for c in features_df.columns
                    if c not in ("label", "subject_id", "activity")]
    X = features_df[feature_cols].values.astype(np.float32)
    y = features_df["label"].values
    groups = features_df["subject_id"].values

    # Subject-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    n_train = len(train_idx)
    n_test  = len(test_idx)
    n_pos   = int(y_train.sum())
    n_neg   = len(y_train) - n_pos
    scale_w = n_neg / n_pos if n_pos > 0 else 1.0

    print(f"\n  Train: {n_train}  Test: {n_test}  (subject-level split)")
    print(f"  Class balance -> Low risk: {n_neg}  High risk: {n_pos}  "
          f"(scale_pos_weight={scale_w:.2f})")
    print("\n" + "=" * 50)
    print("  CLASSIFICATION RESULTS")
    print("=" * 50)

    metrics = {}

    # ── Random Forest ──────────────────────────────────────────────────────
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
    metrics["rf"] = _report("Random Forest", y_test, rf_pred)

    _save_confusion_matrix(y_test, rf_pred, "Random Forest",
                           ["Low Risk", "High Risk"],
                           os.path.join(output_dir, "confusion_matrix_rf.png"))

    joblib.dump(rf, os.path.join(output_dir, "rf_model.pkl"))

    # ── XGBoost ────────────────────────────────────────────────────────────
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_w,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    metrics["xgb"] = _report("XGBoost", y_test, xgb_pred)

    _save_confusion_matrix(y_test, xgb_pred, "XGBoost",
                           ["Low Risk", "High Risk"],
                           os.path.join(output_dir, "confusion_matrix_xgb.png"))

    joblib.dump(xgb, os.path.join(output_dir, "xgb_model.pkl"))

    # Save feature names
    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(feature_cols, f)

    # ── Learning curves ────────────────────────────────────────────────────
    _plot_learning_curves(rf, xgb, X_train, y_train, X_test, y_test, output_dir)

    print(f"\n  -> Saved rf_model.pkl, xgb_model.pkl, feature_names.json")
    return metrics, X_test, y_test


def _plot_learning_curves(rf, xgb, X_tr, y_tr, X_te, y_te, output_dir):
    """Plot training vs test accuracy across RF tree count."""
    estimator_range = range(50, 501, 50)
    rf_train_acc, rf_test_acc = [], []
    xgb_train_acc, xgb_test_acc = [], []

    for n in estimator_range:
        # RF partial prediction
        rf_partial = RandomForestClassifier(n_estimators=n,
                                             random_state=42, n_jobs=-1)
        rf_partial.fit(X_tr, y_tr)
        rf_train_acc.append(accuracy_score(y_tr, rf_partial.predict(X_tr)))
        rf_test_acc.append(accuracy_score(y_te, rf_partial.predict(X_te)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, name, tr, te in [
        (axes[0], "Random Forest", rf_train_acc, rf_test_acc),
    ]:
        ax.plot(list(estimator_range), tr, "o-", label="Train", color="#2196F3")
        ax.plot(list(estimator_range), te, "s--", label="Test",  color="#FF5722")
        ax.set_title(f"{name} — Learning Curve")
        ax.set_xlabel("Number of Trees")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(alpha=0.3)

    # XGBoost eval by iteration
    from sklearn.metrics import accuracy_score as acc_fn
    xgb_obj = XGBClassifier(max_depth=8, learning_rate=0.1, subsample=0.8,
                              colsample_bytree=0.8, eval_metric="logloss",
                              verbosity=0, random_state=42, n_jobs=-1)
    xgb_obj.fit(X_tr, y_tr,
                eval_set=[(X_tr, y_tr), (X_te, y_te)],
                verbose=False)
    results = xgb_obj.evals_result()
    tr_log = results["validation_0"]["logloss"]
    te_log = results["validation_1"]["logloss"]

    ax2 = axes[1]
    ax2.plot(tr_log, label="Train loss", color="#2196F3")
    ax2.plot(te_log, label="Test loss",  color="#FF5722", linestyle="--")
    ax2.set_title("XGBoost — Log-loss by Iteration")
    ax2.set_xlabel("Boosting Round")
    ax2.set_ylabel("Log-loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "learning_curves_rf_xgb.png")
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  -> Saved learning_curves_rf_xgb.png")


# ─── 3: LSTM ──────────────────────────────────────────────────────────────────

def train_lstm(X_raw: np.ndarray,
               y_risk: np.ndarray,
               subjects: np.ndarray,
               output_dir: str = OUTPUT_DIR) -> dict:
    """
    Train LSTM on raw IMU windows.

    Parameters
    ----------
    X_raw   : (N, 128, 6)  raw IMU windows
    y_risk  : (N,)         binary fall-risk labels
    subjects: (N,)         subject IDs for GroupShuffleSplit

    Returns
    -------
    dict with LSTM metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalise globally
    X_flat = X_raw.reshape(-1, X_raw.shape[-1])
    mu  = X_flat.mean(axis=0, keepdims=True)
    std = X_flat.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X_raw - mu) / std

    # Subject-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X_norm, y_risk, groups=subjects))
    X_tr, X_te = X_norm[train_idx], X_norm[test_idx]
    y_tr, y_te = y_risk[train_idx], y_risk[test_idx]

    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    total = len(y_tr)
    class_weight = {0: total / (2 * n_neg + 1e-8),
                    1: total / (2 * n_pos + 1e-8)}

    print(f"\n  LSTM sequences: {len(X_norm)}  (shape per sample: {X_norm.shape[1]}x{X_norm.shape[2]})")
    print(f"  Train: {len(train_idx)}  Test: {len(test_idx)}  (subject-level split)")
    print(f"  Class weights: {class_weight}")
    print(f"  Training LSTM model...")

    # Build model
    model = Sequential([
        Input(shape=(X_raw.shape[1], X_raw.shape[2])),
        LSTM(128, return_sequences=True, kernel_regularizer=l2(0.0005)),
        Dropout(0.3),
        LSTM(64, return_sequences=False, kernel_regularizer=l2(0.0005)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-5, verbose=0),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=100,
        batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    y_prob = model.predict(X_te, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = _report("LSTM", y_te, y_pred)

    _save_confusion_matrix(y_te, y_pred, "LSTM",
                           ["Low Risk", "High Risk"],
                           os.path.join(output_dir, "confusion_matrix_lstm.png"))

    # Training history plot
    _plot_lstm_history(history, output_dir)

    # Save model + scaler params
    model.save(os.path.join(output_dir, "lstm_model.keras"))
    np.save(os.path.join(output_dir, "lstm_norm_mu.npy"), mu)
    np.save(os.path.join(output_dir, "lstm_norm_std.npy"), std)
    with open(os.path.join(output_dir, "lstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> Saved lstm_model.keras, lstm_metrics.json")

    return metrics


def _plot_lstm_history(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"],     label="Train", color="#2196F3")
    axes[0].plot(history.history["val_loss"], label="Val",   color="#FF5722",
                 linestyle="--")
    axes[0].set_title("LSTM — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"],     label="Train", color="#2196F3")
    axes[1].plot(history.history["val_accuracy"], label="Val",   color="#FF5722",
                 linestyle="--")
    axes[1].set_title("LSTM — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "lstm_training_history.png")
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  -> Saved lstm_training_history.png")


# ─── Model Comparison Plot ────────────────────────────────────────────────────

def plot_model_comparison(metrics_dict: dict, output_dir: str = OUTPUT_DIR):
    """
    Bar chart comparing accuracy / F1 / recall across all three models.
    """
    models   = []
    accuracy = []
    f1_scores = []
    recalls  = []

    for key in ["rf", "xgb", "lstm"]:
        if key not in metrics_dict:
            continue
        m = metrics_dict[key]
        models.append(m["model"])
        accuracy.append(m["accuracy"])
        f1_scores.append(m["f1"])
        recalls.append(m["recall"])

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width, accuracy,  width, label="Accuracy",  color="#2196F3", alpha=0.85)
    b2 = ax.bar(x,          f1_scores, width, label="F1-Score",  color="#4CAF50", alpha=0.85)
    b3 = ax.bar(x + width,  recalls,   width, label="Recall",    color="#FF9800", alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Fall Risk Detection (UCI HAR)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  -> Saved model_comparison.png")
    return save_path
