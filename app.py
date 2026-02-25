"""
app.py
======
Flask web dashboard for the IMU fall-risk detection project.
Trains all models on first run, then serves results at http://127.0.0.1:5000
"""

import os
import sys
import json
import time
import numpy as np
import base64

from flask import Flask, jsonify, render_template

# ── constants ─────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join("data", "UCI-HAR Dataset")
OUTPUT_DIR = "output"

app = Flask(__name__)

# In-memory cache populated on startup
_cache = {}


# ── helpers ───────────────────────────────────────────────────────────────────

def _img_b64(filename: str) -> str:
    """Read a PNG from output/ and return a base64 data-URI string."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _load_cached_metrics() -> dict:
    """Load RF + XGBoost metrics from pkl model meta + LSTM metrics JSON."""
    import joblib
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

    metrics = {}

    # LSTM metrics from JSON
    lstm_path = os.path.join(OUTPUT_DIR, "lstm_metrics.json")
    if os.path.exists(lstm_path):
        with open(lstm_path) as f:
            metrics["lstm"] = json.load(f)

    return metrics


def _run_pipeline():
    """Full training + analysis pipeline (called once at startup if not cached)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    from src.data_loader import load_uci_har, make_fall_risk_dataset
    print("Loading UCI HAR dataset...")
    X_all, y_all, subjects_all = load_uci_har(DATA_DIR)

    print("\nCreating fall-risk dataset...")
    X_mob, y_risk, subj_mob, y_activity = make_fall_risk_dataset(
        X_all, y_all, subjects_all)

    # ── Feature extraction ─────────────────────────────────────────────────
    from src.feature_extraction import build_feature_dataframe
    print("\nExtracting features...")
    features_df = build_feature_dataframe(X_mob, y_risk, y_activity, subj_mob)
    _cache["features_df"] = features_df

    # ── Train RF + XGBoost ────────────────────────────────────────────────
    from src.model import (train_classical_models, train_lstm,
                            plot_model_comparison)
    print("\nTraining RF + XGBoost models...")
    classical_metrics, X_test, y_test = train_classical_models(features_df, OUTPUT_DIR)

    # ── Time-series analysis ──────────────────────────────────────────────
    from src.time_series import run_full_time_series_analysis
    from src.preprocessing import extract_imu_components

    walking_mask = features_df[features_df["activity"] == 1]
    sample_row   = walking_mask.index[0] if len(walking_mask) > 0 else 0
    sample_win   = X_mob[sample_row]
    _, _, _, _, _, _, acc_mag, _ = extract_imu_components(sample_win)

    print("\nRunning time-series analysis...")
    run_full_time_series_analysis(features_df, acc_mag)

    # ── Train LSTM ────────────────────────────────────────────────────────
    print("\nTraining LSTM model...")
    lstm_metrics = train_lstm(X_mob, y_risk, subj_mob, OUTPUT_DIR)

    # ── Model comparison chart ────────────────────────────────────────────
    all_metrics = {
        "rf":   classical_metrics["rf"],
        "xgb":  classical_metrics["xgb"],
        "lstm": lstm_metrics,
    }
    plot_model_comparison(all_metrics, OUTPUT_DIR)

    # Save all metrics to JSON for the API
    with open(os.path.join(OUTPUT_DIR, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    _cache["metrics"] = all_metrics
    print("\nStartup complete!\n")
    print("  [OK] Open http://127.0.0.1:5000 in your browser\n")


def _models_cached() -> bool:
    """Return True if models already trained and saved to disk."""
    required = ["rf_model.pkl", "xgb_model.pkl",
                 "lstm_model.keras", "lstm_metrics.json",
                 "model_comparison.png"]
    return all(os.path.exists(os.path.join(OUTPUT_DIR, f)) for f in required)


def _load_dataset_into_cache():
    """Load UCI HAR into memory for fast live predictions."""
    if "X_mob" in _cache:
        return
    from src.data_loader import load_uci_har, make_fall_risk_dataset
    from src.feature_extraction import build_feature_dataframe
    X_all, y_all, subjects_all = load_uci_har(DATA_DIR)
    X_mob, y_risk, subj_mob, y_activity = make_fall_risk_dataset(
        X_all, y_all, subjects_all)
    features_df = build_feature_dataframe(X_mob, y_risk, y_activity, subj_mob)
    _cache["X_mob"]       = X_mob
    _cache["y_risk"]      = y_risk
    _cache["subj_mob"]    = subj_mob
    _cache["y_activity"]  = y_activity
    _cache["features_df"] = features_df


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({"ready": bool(_cache.get("metrics"))})


@app.route("/api/subjects")
def api_subjects():
    """List all subject IDs available in the mobile (fall-risk) dataset."""
    _load_dataset_into_cache()
    subj_mob   = _cache["subj_mob"]
    y_activity = _cache["y_activity"]
    unique_subjects = sorted(np.unique(subj_mob).tolist())
    activity_names = {1: "Walking", 2: "Upstairs", 3: "Downstairs"}
    subjects_info = []
    for sid in unique_subjects:
        mask = subj_mob == sid
        acts = y_activity[mask]
        subjects_info.append({
            "id": int(sid),
            "total_windows": int(mask.sum()),
            "walking":    int((acts == 1).sum()),
            "upstairs":   int((acts == 2).sum()),
            "downstairs": int((acts == 3).sum()),
        })
    return jsonify(subjects_info)


@app.route("/api/predict/<int:subject_id>")
def api_predict(subject_id: int):
    """
    Run fall-risk predictions for a specific subject.
    Query param: model = rf | xgb | lstm  (default: rf)
    Returns per-window predictions and probabilities.
    """
    from flask import request as freq
    model_name = freq.args.get("model", "rf").lower()

    _load_dataset_into_cache()
    X_mob      = _cache["X_mob"]
    y_risk     = _cache["y_risk"]
    subj_mob   = _cache["subj_mob"]
    y_activity = _cache["y_activity"]
    features_df = _cache["features_df"]

    mask = subj_mob == subject_id
    if not mask.any():
        return jsonify({"error": f"Subject {subject_id} not found"}), 404

    y_true = y_risk[mask].tolist()
    acts   = y_activity[mask].tolist()
    act_names = {1: "Walking", 2: "Upstairs", 3: "Downstairs"}

    if model_name in ("rf", "xgb"):
        import joblib
        key_file = "rf_model.pkl" if model_name == "rf" else "xgb_model.pkl"
        model = joblib.load(os.path.join(OUTPUT_DIR, key_file))
        feature_cols = [c for c in features_df.columns
                        if c not in ("label", "subject_id", "activity")]
        subj_df = features_df[features_df["subject_id"] == subject_id]
        X_subj  = subj_df[feature_cols].values.astype(np.float32)
        probs   = model.predict_proba(X_subj)[:, 1].tolist()
        preds   = (np.array(probs) >= 0.5).astype(int).tolist()
        acts    = subj_df["activity"].tolist()
        y_true  = subj_df["label"].tolist()
    else:
        # LSTM on raw sequences
        from tensorflow.keras.models import load_model
        lstm_model = load_model(os.path.join(OUTPUT_DIR, "lstm_model.keras"))
        mu  = np.load(os.path.join(OUTPUT_DIR, "lstm_norm_mu.npy"))
        std = np.load(os.path.join(OUTPUT_DIR, "lstm_norm_std.npy"))
        X_subj = X_mob[mask]
        X_norm = (X_subj - mu) / (std + 1e-8)
        probs  = lstm_model.predict(X_norm, verbose=0).ravel().tolist()
        preds  = (np.array(probs) >= 0.5).astype(int).tolist()

    accuracy = float(np.mean(np.array(preds) == np.array(y_true))) if y_true else 0.0

    return jsonify({
        "subject_id":  subject_id,
        "model":       model_name,
        "n_windows":   len(preds),
        "accuracy":    round(accuracy, 4),
        "predictions": preds,
        "probabilities": [round(p, 4) for p in probs],
        "true_labels": y_true,
        "activities":  [act_names.get(a, str(a)) for a in acts],
    })


@app.route("/api/metrics")
def api_metrics():
    metrics = _cache.get("metrics", {})
    if not metrics:
        path = os.path.join(OUTPUT_DIR, "all_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                metrics = json.load(f)
    return jsonify(metrics)


@app.route("/api/images")
def api_images():
    images = {
        "model_comparison":          _img_b64("model_comparison.png"),
        "confusion_rf":              _img_b64("confusion_matrix_rf.png"),
        "confusion_xgb":             _img_b64("confusion_matrix_xgb.png"),
        "confusion_lstm":            _img_b64("confusion_matrix_lstm.png"),
        "lstm_history":              _img_b64("lstm_training_history.png"),
        "learning_curves":           _img_b64("learning_curves_rf_xgb.png"),
        "acf_walking":               _img_b64("acf_walking.png"),
        "stl_walking":               _img_b64("stl_walking.png"),
        "stl_upstairs":              _img_b64("stl_upstairs.png"),
        "stl_downstairs":            _img_b64("stl_downstairs.png"),
        "feature_comparison":        _img_b64("feature_comparison_by_activity.png"),
    }
    return jsonify(images)


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print("[ERROR] Dataset not found. Run: python download_data.py")
        sys.exit(1)

    if _models_cached():
        print("Loading cached models...")
        path = os.path.join(OUTPUT_DIR, "all_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                _cache["metrics"] = json.load(f)
        print("\nStartup complete!")
        print("\n  [OK] Open http://127.0.0.1:5000 in your browser\n")
    else:
        _run_pipeline()

    app.run(debug=False, host="127.0.0.1", port=5000)
