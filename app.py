# =============================================================================
# app.py -- Flask Web Frontend
# =============================================================================
# Serves a web dashboard for the Gait Analysis project.
# Pages:
#   /            -> Dashboard with all results, plots, and metrics
#   /live        -> Interactive live monitoring with subject selector
#   /api/subjects  -> JSON list of available subjects
#   /api/live/<filename> -> Run live monitoring for a specific subject
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
from src.model import train_and_evaluate, train_lstm, load_cached_lstm, plot_all_model_comparison
from src.time_series_analysis import check_stationarity, decompose_gait
from src.live_monitoring import simulate_live_monitoring

# -- Configuration ------------------------------------------------------------
DATA_DIR   = os.path.join("data", "gait-in-parkinsons-disease-1.0.0")
OUTPUT_DIR = os.path.join("output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "rf_model.pkl")
XGB_MODEL_PATH = os.path.join(OUTPUT_DIR, "xgb_model.pkl")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "feature_names.json")
FS = 100
WINDOW = 300
STEP = 150

app = Flask(__name__, static_folder="output", static_url_path="/output")

# -- Globals (loaded once at startup) -----------------------------------------
subjects = []
rf_model = None
xgb_model = None
lstm_model = None
lstm_scaler = None
lstm_seq_len = 10
feature_names = []
features_df = None
metrics_data = {}


def startup():
    """Load data, train model (or load cached), and prepare metrics."""
    global subjects, rf_model, xgb_model, lstm_model, lstm_scaler
    global lstm_seq_len, feature_names, features_df, metrics_data

    print("Loading subjects...")
    subjects[:] = load_all_subjects(DATA_DIR)

    print("Extracting features...")
    features_df_local = build_feature_dataframe(subjects, WINDOW, STEP, FS)
    features_df = features_df_local

    # -- Train or load RF + XGBoost ---------------------------------------
    if (os.path.exists(MODEL_PATH)
            and os.path.exists(XGB_MODEL_PATH)
            and os.path.exists(FEATURES_PATH)):
        print("Loading cached RF + XGBoost models...")
        rf_model = joblib.load(MODEL_PATH)
        xgb_model = joblib.load(XGB_MODEL_PATH)
        with open(FEATURES_PATH) as f:
            feature_names[:] = json.load(f)
    else:
        print("Training RF + XGBoost models...")
        rf_model_local, feat_names = train_and_evaluate(features_df, OUTPUT_DIR)
        rf_model = rf_model_local
        feature_names[:] = feat_names
        # Models are saved inside train_and_evaluate, also load XGBoost
        xgb_model = joblib.load(XGB_MODEL_PATH)

    # -- Train or load LSTM -----------------------------------------------
    try:
        lstm_result = train_lstm(features_df, OUTPUT_DIR)
        lstm_model = lstm_result["model"]
        lstm_scaler = lstm_result["scaler"]
        lstm_seq_len = lstm_result["seq_len"]
        metrics_data["lstm"] = {
            "accuracy":  round(lstm_result["accuracy"], 4),
            "precision": round(lstm_result["precision"], 4),
            "recall":    round(lstm_result["recall"], 4),
            "f1":        round(lstm_result["f1"], 4),
        }
    except Exception as e:
        print(f"  [WARN] LSTM skipped: {e}")
        metrics_data["lstm"] = {
            "accuracy": "--", "precision": "--", "recall": "--", "f1": "--"
        }

    # -- Run time-series analysis -----------------------------------------
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

    # -- Collect summary metrics for RF + XGBoost -------------------------
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import GroupShuffleSplit

    feature_cols = [c for c in features_df.columns if c not in ("label", "subject_id")]
    X = features_df[feature_cols].values
    y = features_df["label"].values
    groups = features_df["subject_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_test, y_test = X[test_idx], y[test_idx]

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

    # XGBoost
    y_pred_xgb = xgb_model.predict(X_test)
    metrics_data["xgb"] = _metrics(y_test, y_pred_xgb)

    metrics_data["dataset"] = {
        "total_files": len(subjects),
        "healthy": sum(1 for _, l, _ in subjects if l == 0),
        "parkinson": sum(1 for _, l, _ in subjects if l == 1),
        "total_windows": len(features_df),
        "healthy_windows": int((features_df["label"] == 0).sum()),
        "parkinson_windows": int((features_df["label"] == 1).sum()),
    }

    # -- Generate all-3-models comparison chart ----------------------------
    try:
        plot_all_model_comparison(
            {
                "Random Forest": {
                    "Accuracy": metrics_data["rf"]["accuracy"],
                    "Precision": metrics_data["rf"]["precision"],
                    "Recall": metrics_data["rf"]["recall"],
                    "F1": metrics_data["rf"]["f1"],
                },
                "XGBoost": {
                    "Accuracy": metrics_data["xgb"]["accuracy"],
                    "Precision": metrics_data["xgb"]["precision"],
                    "Recall": metrics_data["xgb"]["recall"],
                    "F1": metrics_data["xgb"]["f1"],
                },
                "LSTM": {
                    "Accuracy": metrics_data["lstm"]["accuracy"],
                    "Precision": metrics_data["lstm"]["precision"],
                    "Recall": metrics_data["lstm"]["recall"],
                    "F1": metrics_data["lstm"]["f1"],
                },
            },
            OUTPUT_DIR,
        )
    except Exception as e:
        print(f"  [WARN] Model comparison plot skipped: {e}")

    print("Startup complete!")


# -- Routes -------------------------------------------------------------------

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
    Accepts ?model=rf|xgb|lstm query param (default: rf).
    Returns JSON array of (time, risk_probability) pairs.
    """
    model_choice = request.args.get("model", "rf")

    if model_choice == "lstm":
        if lstm_model is None:
            return jsonify({"error": "LSTM model not available"}), 400
    elif model_choice == "xgb":
        if xgb_model is None:
            return jsonify({"error": "XGBoost model not available"}), 400
    else:
        if rf_model is None:
            return jsonify({"error": "Random Forest model not available"}), 400

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

    if model_choice == "lstm":
        # -- LSTM path: collect features, build sequences, predict --------
        all_feats = []
        all_times = []
        for i in range(0, len(total) - window_size + 1, step):
            lw = left[i : i + window_size]
            rw = right[i : i + window_size]
            tw = total[i : i + window_size]

            feats = compute_window_features(lw, rw, tw, FS)
            if feats is None:
                continue
            all_feats.append([feats[f] for f in feature_names])
            all_times.append(round((i + window_size / 2) / FS, 2))

        # Build sliding sequences of length lstm_seq_len
        if len(all_feats) >= lstm_seq_len:
            feat_array = np.array(all_feats, dtype=np.float32)
            # Scale using the same scaler used during training
            feat_scaled = lstm_scaler.transform(feat_array)

            for j in range(len(feat_scaled) - lstm_seq_len + 1):
                seq = feat_scaled[j : j + lstm_seq_len]
                seq_input = seq.reshape(1, lstm_seq_len, -1)
                prob = float(lstm_model.predict(seq_input, verbose=0)[0][0])
                # Use the centre time of the last window in the sequence
                centre_time = all_times[j + lstm_seq_len - 1]
                results.append({
                    "time": centre_time,
                    "risk": round(prob, 4),
                })
    else:
        # -- RF / XGBoost path --------------------------------------------
        model = rf_model if model_choice == "rf" else xgb_model
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
    model_names = {"rf": "Random Forest", "xgb": "XGBoost", "lstm": "LSTM"}
    return jsonify({
        "filename": fname,
        "label": label_name,
        "model": model_names.get(model_choice, "Random Forest"),
        "duration_s": round(len(df) / FS, 1),
        "data": results,
    })


@app.route("/output/<path:filename>")
def serve_plot(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    startup()
    print("\n  [OK] Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=False, port=5000)
