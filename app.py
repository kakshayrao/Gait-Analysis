# =============================================================================
# app.py — Flask Web Frontend
# =============================================================================
# Serves a web dashboard for the Gait Analysis project.
# Pages:
#   /            → Dashboard with all results, plots, and metrics
#   /live        → Interactive live monitoring with subject selector
#   /api/subjects  → JSON list of available subjects
#   /api/live/<filename> → Run live monitoring for a specific subject
# =============================================================================

import os
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")

from flask import Flask, render_template, jsonify, send_from_directory, request

from src.data_loader import load_all_subjects
from src.feature_extraction import (
    build_feature_dataframe,
    smooth_signal,
    normalize_signal,
    compute_window_features,
)
from src.model import train_and_evaluate, train_lstm
from src.time_series_analysis import check_stationarity, decompose_gait
from src.live_monitoring import simulate_live_monitoring

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join("data", "gait-in-parkinsons-disease-1.0.0")
OUTPUT_DIR = os.path.join("output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "rf_model.pkl")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "feature_names.json")
FS = 100
WINDOW = 300
STEP = 150

app = Flask(__name__, static_folder="output", static_url_path="/output")

# ── Globals (loaded once at startup) ─────────────────────────────────────────
subjects = []
rf_model = None
xgb_model = None
feature_names = []
features_df = None
metrics_data = {}


def startup():
    """Load data, train model (or load cached), and prepare metrics."""
    global subjects, rf_model, xgb_model, feature_names, features_df, metrics_data

    print("Loading subjects...")
    subjects[:] = load_all_subjects(DATA_DIR)

    print("Extracting features...")
    features_df_local = build_feature_dataframe(subjects, WINDOW, STEP, FS)
    features_df = features_df_local

    # Train or load model
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        print("Loading cached model...")
        rf_model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH) as f:
            feature_names[:] = json.load(f)
    else:
        print("Training models...")
        rf_model_local, feat_names = train_and_evaluate(features_df, OUTPUT_DIR)
        rf_model = rf_model_local
        feature_names[:] = feat_names
        joblib.dump(rf_model, MODEL_PATH)
        with open(FEATURES_PATH, "w") as f:
            json.dump(list(feat_names), f)

    # Run time-series analysis
    healthy_subj = next((s for s in subjects if s[1] == 0), None)
    parkinson_subj = next((s for s in subjects if s[1] == 1), None)

    for subj, tag in [(healthy_subj, "Healthy"), (parkinson_subj, "Parkinson")]:
        if subj is None:
            continue
        df, _, _ = subj
        from scipy.signal import find_peaks
        sig = normalize_signal(smooth_signal(df["total_force"].values))
        peaks, _ = find_peaks(sig, height=np.mean(sig)*0.5, distance=int(FS*0.4))
        if len(peaks) >= 4:
            intervals = np.diff(peaks) / FS
            result = check_stationarity(intervals, label=tag)
            if result:
                metrics_data[f"adf_{tag.lower()}"] = result
            decompose_gait(intervals, label=tag, output_dir=OUTPUT_DIR)

    # Collect summary metrics for ALL models
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from xgboost import XGBClassifier

    feature_cols = [c for c in features_df.columns if c not in ("label", "subject_id")]
    X = features_df[feature_cols].values
    y = features_df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    def _metrics(y_true, y_pred):
        return {
            "accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        }

    # Random Forest
    y_pred_rf = rf_model.predict(X_test)
    metrics_data["rf"] = _metrics(y_test, y_pred_rf)

    # XGBoost — store globally for live monitoring
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                        eval_metric="logloss", random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    metrics_data["xgb"] = _metrics(y_test, y_pred_xgb)

    # LSTM — use the proper sequence-based training from model.py
    try:
        lstm_result = train_lstm(features_df, OUTPUT_DIR)
        metrics_data["lstm"] = {
            "accuracy":  round(lstm_result["accuracy"], 4),
            "precision": round(lstm_result["precision"], 4),
            "recall":    round(lstm_result["recall"], 4),
            "f1":        round(lstm_result["f1"], 4),
        }
    except Exception as e:
        print(f"  [WARN] LSTM metrics skipped: {e}")
        metrics_data["lstm"] = {"accuracy": "—", "precision": "—", "recall": "—", "f1": "—"}

    metrics_data["dataset"] = {
        "total_files": len(subjects),
        "healthy": sum(1 for _, l, _ in subjects if l == 0),
        "parkinson": sum(1 for _, l, _ in subjects if l == 1),
        "total_windows": len(features_df),
        "healthy_windows": int((features_df["label"] == 0).sum()),
        "parkinson_windows": int((features_df["label"] == 1).sum()),
    }
    print("Startup complete!")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    """Main dashboard page showing all results."""
    return render_template("dashboard.html", metrics=metrics_data)


@app.route("/live")
def live_page():
    """Interactive live monitoring page."""
    return render_template("live.html")


@app.route("/api/subjects")
def get_subjects():
    """Return list of available subjects with labels."""
    result = []
    for df, label, fname in subjects:
        result.append({
            "filename": fname,
            "label": int(label),
            "label_name": "Healthy" if label == 0 else "Parkinson",
            "duration_s": round(len(df) / FS, 1),
            "samples": len(df),
        })
    return jsonify(result)


@app.route("/api/live/<filename>")
def run_live(filename):
    """
    Run live monitoring simulation for a specific subject.
    Accepts ?model=rf|xgb query param (default: rf).
    Returns JSON array of (time, risk_probability) pairs.
    """
    model_choice = request.args.get("model", "rf")
    model = rf_model if model_choice != "xgb" else xgb_model
    if model is None:
        return jsonify({"error": f"Model '{model_choice}' not available"}), 400

    # Find the subject
    subj = next((s for s in subjects if s[2] == filename), None)
    if subj is None:
        return jsonify({"error": "Subject not found"}), 404

    df, label, fname = subj
    df = df.ffill().fillna(0)

    left  = normalize_signal(smooth_signal(df["total_left"].values))
    right = normalize_signal(smooth_signal(df["total_right"].values))
    total = normalize_signal(smooth_signal(df["total_force"].values))

    window_size = 300
    step = 50
    results = []

    for i in range(0, len(total) - window_size + 1, step):
        lw = left[i : i + window_size]
        rw = right[i : i + window_size]
        tw = total[i : i + window_size]

        feats = compute_window_features(lw, rw, tw, FS)
        if feats is None:
            continue

        x = np.array([[feats[f] for f in feature_names]])
        prob = float(model.predict_proba(x)[0][1])
        centre_time = round((i + window_size / 2) / FS, 2)

        results.append({
            "time": centre_time,
            "risk": round(prob, 4),
        })

    label_name = "Healthy" if label == 0 else "Parkinson"
    model_name = "Random Forest" if model_choice == "rf" else "XGBoost"
    return jsonify({
        "filename": fname,
        "label": label_name,
        "model": model_name,
        "duration_s": round(len(df) / FS, 1),
        "data": results,
    })


@app.route("/output/<path:filename>")
def serve_plot(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    startup()
    print("\n  ✓ Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=False, port=5000)
