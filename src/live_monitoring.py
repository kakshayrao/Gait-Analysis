# =============================================================================
# live_monitoring.py
# =============================================================================
# Simulates real-time gait monitoring by sliding a window across a subject's
# vGRF signal, extracting features, and predicting fall risk probability
# using the pre-trained Random Forest model.
#
# Outputs
# -------
#   - Risk Over Time line plot  (saved to output/)
#   - Console alerts when predicted risk > 0.75
# =============================================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.feature_extraction import (
    smooth_signal,
    normalize_signal,
    compute_window_features,
)


def simulate_live_monitoring(model,
                             feature_names: list,
                             signal_left: np.ndarray,
                             signal_right: np.ndarray,
                             signal_total: np.ndarray,
                             subject_label: str = "Unknown",
                             output_dir: str = "output",
                             fs: int = 100,
                             window_size: int = 300,
                             step: int = 50):
    """
    Stream sliding windows across the given subject's signal, predict
    fall risk for each window, and save a risk-over-time plot.

    Parameters
    ----------
    model : trained sklearn classifier with predict_proba
    feature_names : list[str]  — column order used during training
    signal_left, signal_right, signal_total : 1-D numpy arrays
    subject_label : str  — label for the plot title (e.g. "Healthy", "Parkinson")
    output_dir : str
    fs : int — sampling frequency (Hz)
    window_size : int — samples per window
    step : int — hop between consecutive windows
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pre-process (same pipeline as training)
    left  = normalize_signal(smooth_signal(signal_left))
    right = normalize_signal(smooth_signal(signal_right))
    total = normalize_signal(smooth_signal(signal_total))

    time_points = []   # centre-time of each window
    risk_scores = []   # predicted probability of Parkinson (fall risk)
    alert_count = 0

    n_windows = (len(total) - window_size) // step + 1

    print(f"\n{'─' * 55}")
    print(f"  LIVE MONITORING — Subject: {subject_label}")
    print(f"  Signal length : {len(total)} samples "
          f"({len(total) / fs:.1f} s)")
    print(f"  Window        : {window_size} samples "
          f"({window_size / fs:.1f} s), step {step}")
    print(f"  Total windows : {n_windows}")
    print(f"{'─' * 55}")

    for i in range(0, len(total) - window_size + 1, step):
        lw = left[i : i + window_size]
        rw = right[i : i + window_size]
        tw = total[i : i + window_size]

        feats = compute_window_features(lw, rw, tw, fs)
        if feats is None:
            continue

        # Build feature vector in the correct column order
        x = np.array([[feats[f] for f in feature_names]])
        prob = model.predict_proba(x)[0][1]  # P(Parkinson)

        centre_time = (i + window_size / 2) / fs
        time_points.append(centre_time)
        risk_scores.append(prob)

        # Alert
        if prob > 0.75:
            alert_count += 1
            if alert_count <= 5:  # don't flood the console
                print(f"  ⚠  HIGH FALL RISK DETECTED  "
                      f"t={centre_time:6.1f}s  risk={prob:.3f}")

    if alert_count > 5:
        print(f"  ... ({alert_count - 5} more alerts suppressed)")

    mean_risk = np.mean(risk_scores) if risk_scores else 0
    print(f"  Mean risk score: {mean_risk:.3f}")
    print(f"  Alerts triggered: {alert_count}")

    # ── Risk Over Time Plot ──────────────────────────────────────────────
    if not time_points:
        print("  [WARN] No valid windows — skipping plot.")
        return

    fig, (ax_line, ax_bar) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # --- Line plot ---
    ax_line.plot(time_points, risk_scores, linewidth=1.2, color="#1f77b4")
    ax_line.axhline(0.75, color="red", linestyle="--", linewidth=1,
                    label="High-risk threshold")
    ax_line.axhline(0.40, color="orange", linestyle=":", linewidth=1,
                    label="Moderate-risk threshold")
    ax_line.fill_between(time_points, risk_scores, alpha=0.15,
                         color="#1f77b4")
    ax_line.set_xlabel("Time (s)", fontsize=12)
    ax_line.set_ylabel("Predicted Risk Probability", fontsize=12)
    ax_line.set_title(f"Real-Time Fall Risk — {subject_label}",
                      fontsize=14, fontweight="bold")
    ax_line.set_ylim(-0.05, 1.05)
    ax_line.legend(loc="upper right")
    ax_line.grid(alpha=0.3)

    # --- Summary risk bar ---
    bar_color = (
        "#2ca02c" if mean_risk < 0.4
        else "#ff7f0e" if mean_risk < 0.7
        else "#d62728"
    )
    ax_bar.barh(["Risk"], [mean_risk], color=bar_color, height=0.4)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_title("Mean Risk", fontsize=13, fontweight="bold")
    ax_bar.axvline(0.75, color="red", linestyle="--", linewidth=1)
    ax_bar.axvline(0.40, color="orange", linestyle=":", linewidth=1)

    plt.tight_layout()
    safe_label = subject_label.replace(" ", "_").lower()
    save_path = os.path.join(output_dir, f"live_monitoring_{safe_label}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Saved {os.path.basename(save_path)}")
