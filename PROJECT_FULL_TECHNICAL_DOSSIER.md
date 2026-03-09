# IMU Gait Analysis Project — Full Technical Dossier

This document is an exhaustive, report-ready technical reference for the repository.  
Goal: give you enough detail to write a complete long-form project report (30 pages or more) without needing to reverse-engineer the code.

---

## 1) Project Identity

- **Repository name**: `Gait-Analysis`
- **Workspace path**: `c:/Users/kaksh/Desktop/3yr projects/ATSA`
- **Primary objective**: Binary fall-risk detection from smartphone IMU gait windows.
- **Application type**: End-to-end ML pipeline + Flask web dashboard.
- **Dataset**: UCI HAR (Human Activity Recognition Using Smartphones).
- **Model families implemented**:
  - Random Forest (tabular engineered features)
  - XGBoost (tabular engineered features)
  - LSTM (raw sequential IMU data)

---

## 2) What This System Actually Does

The project converts smartphone IMU windows into a **fall-risk classifier**.

1. Loads raw UCI HAR windows (`128 x 6` each).
2. Keeps only mobility activities:
   - WALKING (activity 1)
   - WALKING_UPSTAIRS (activity 2)
   - WALKING_DOWNSTAIRS (activity 3)
3. Maps to fall-risk labels:
   - WALKING -> **Low risk** (`0`)
   - UPSTAIRS + DOWNSTAIRS -> **High risk** (`1`)
4. Builds engineered gait features (13 features/window) for classical ML.
5. Trains RF + XGBoost with subject-level split.
6. Trains LSTM on normalized raw sequences with subject-level split.
7. Generates plots and serialized artifacts under `output/`.
8. Serves dashboard + API through Flask (`app.py`).

---

## 3) Repository Structure (Detailed)

## Root Files

- `app.py`
  - Main Flask app.
  - Triggers training on first run if models absent.
  - Loads cached metrics/models on subsequent runs.
  - Exposes JSON APIs for metrics, images, subjects, and per-subject predictions.
- `main.py`
  - Headless pipeline (no web server), for training + analysis only.
- `download_data.py`
  - Downloads and extracts UCI HAR ZIP to `data/raw/`.
- `README.md`
  - Project overview, setup, model summary.
- `requirements.txt`
  - Python package requirements.

## Source Package (`src/`)

- `src/data_loader.py`
  - Raw data loading, train+test merge, risk mapping.
- `src/preprocessing.py`
  - Bandpass filter helper, signal magnitude helpers, normalization utility.
- `src/feature_extraction.py`
  - Per-window feature engineering (13 features + labels + subject metadata).
- `src/model.py`
  - Classical model training, LSTM training, metrics, confusion matrices, learning curves.
- `src/time_series.py`
  - ACF plots, ADF tests, STL decomposition, activity-wise feature boxplots.
- `src/__init__.py`
  - Present but effectively empty.

## Templates (`templates/`)

- `templates/index.html`
  - **Active dashboard** used by route `/` in `app.py`.
  - Includes modern dark UI, metrics cards, plot sections, live subject predictions.
- `templates/dashboard.html`
  - Legacy template with Parkinson/vGRF wording; not referenced by current Flask routes.
- `templates/live.html`
  - Legacy live page with `/api/live/*` assumptions; not referenced by current Flask routes.

## Data (`data/`)

- `data/UCI-HAR Dataset/`
  - Main extracted dataset used in training.
- `data/raw/UCI_HAR.zip`
  - Downloaded archive cache.
- `data/processed/`
  - Exists, currently empty.

## Output (`output/`)

Contains trained models, metrics, and generated visualizations (full list in section 13).

---

## 4) Dataset and Label Semantics

## Original UCI HAR facts used by this project

- Volunteers: **30**
- Sampling rate: **50 Hz**
- Window length: **128 samples** (`2.56 s`)
- Overlap: **50%**
- Activities in source dataset (1..6):
  1. WALKING
  2. WALKING_UPSTAIRS
  3. WALKING_DOWNSTAIRS
  4. SITTING
  5. STANDING
  6. LAYING

## Counts from current local dataset files

- Total windows: **10,299**
- Activity-wise counts:
  - 1 (WALKING): 1,722
  - 2 (WALKING_UPSTAIRS): 1,544
  - 3 (WALKING_DOWNSTAIRS): 1,406
  - 4 (SITTING): 1,777
  - 5 (STANDING): 1,906
  - 6 (LAYING): 1,944
- Unique subjects: **30**

## Fall-risk dataset after filtering to mobility classes 1/2/3

- Mobile windows total: **4,672**
- Low risk (`0` = WALKING): **1,722**
- High risk (`1` = UPSTAIRS or DOWNSTAIRS): **2,950**

This creates a class imbalance toward high-risk windows.

---

## 5) End-to-End Data Flow

## Pipeline graph

Raw UCI HAR files -> merge train/test -> mobility filtering -> risk relabeling -> (A) engineered features for RF/XGB and (B) normalized sequences for LSTM -> subject-level split -> training/evaluation -> artifacts -> dashboard APIs.

## Data dimensionality

- `X` after loading: `(N, 128, 6)` where `N=10299` before filtering.
- Channels per timestep:
  1. `body_acc_x`
  2. `body_acc_y`
  3. `body_acc_z`
  4. `body_gyro_x`
  5. `body_gyro_y`
  6. `body_gyro_z`

---

## 6) Module Deep Dive: `src/data_loader.py`

## Constants

- `DEFAULT_DATA_DIR = data/UCI-HAR Dataset`
- `ACTIVITY_NAMES` dictionary for 1..6 labels.
- `SIGNAL_FILES` fixed to six channels above.

## Core functions

### `_load_txt(path)`
- Uses `np.loadtxt(..., dtype=np.float32)`.
- Reads whitespace-delimited files.

### `_load_split(data_dir, split)`
- Loads one split (`train` or `test`).
- Builds `X` by stacking six `N x 128` signal files into shape `N x 128 x 6`.
- Loads `y_split.txt` and `subject_split.txt`.

### `load_uci_har(data_dir)`
- Loads train and test split independently.
- Concatenates along axis 0 for full dataset.
- Prints per-activity counts and subject count.

### `make_fall_risk_dataset(X, y, subjects)`
- Keeps only `y in {1,2,3}`.
- Converts to binary risk:
  - `1 -> 0` (low risk)
  - `2,3 -> 1` (high risk)
- Returns:
  - `X_mob`
  - `y_risk`
  - `subj_mob`
  - `y_activity` (original 1..3 labels for analysis/visualization)

---

## 7) Module Deep Dive: `src/preprocessing.py`

## Signal constants

- `FS = 50`
- `WINDOW = 128`

## Filtering functions

### `butter_bandpass(lowcut=0.3, highcut=20.0, fs=50, order=4)`
- Designs SOS Butterworth band-pass filter.
- Normalized with Nyquist frequency (`fs/2`).

### `_SOS`
- Filter coefficients precomputed once at module import.

### `bandpass_filter(signal)`
- Applies zero-phase filtering using `sosfiltfilt`.
- Returns float32.

## IMU helper functions

### `magnitude(x,y,z)`
- Euclidean norm: $\sqrt{x^2 + y^2 + z^2}$

### `extract_imu_components(X_window)`
- Splits one `128 x 6` window into six 1D components + two magnitudes:
  - `acc_mag`, `gyro_mag`

### `normalize_windows(X)`
- Global channel-wise z-score over all windows:
  - reshape to `(N*128, 6)`
  - compute `mu`, `std`
  - return normalized `X`

Note: LSTM path in `src/model.py` performs equivalent normalization internally and saves `mu/std` arrays.

---

## 8) Module Deep Dive: `src/feature_extraction.py`

This module generates the tabular feature matrix for RF/XGBoost.

## Spectral helper functions

### `_spectral_entropy(signal)`
- Computes FFT power spectrum, normalizes to pseudo-PSD.
- Shannon entropy over non-zero bins.
- Normalized by $\log_2(K)$ where `K` is number of FFT bins.

### `_dominant_frequency(signal, fs=50)`
- Uses `rfft` power, zeroes DC bin, returns frequency of max power.

## Main per-window feature function

### `compute_imu_features(window, fs=50)`

Input: one window of shape `128 x 6`.

Step detection:
- Finds peaks in acceleration magnitude using:
  - height threshold = `mean(acc_mag) * 0.8`
  - minimum peak distance = `0.3s * fs`
- If fewer than 2 peaks -> returns `None` (window skipped).

Computed features (13):

1. **stride_time**
   - Mean inter-peak interval (seconds).
2. **cadence**
   - $60 / stride\_time$ (steps/min style proxy based on interval definition).
3. **step_variance**
   - Variance of inter-peak intervals.
4. **symmetry_index**
   - Ratio of first-half signal energy to second-half energy.
5. **sma**
   - Mean signal magnitude area over tri-axial acceleration.
6. **mean_gyro**
   - Mean gyroscope magnitude.
7. **jerk**
   - Mean absolute first difference of acceleration magnitude.
8. **dominant_freq**
   - Peak FFT frequency.
9. **spectral_entropy**
   - Normalized spectral entropy.
10. **autocorr_lag1**
   - Pearson correlation of signal with one-sample lag.
11. **rms_acc**
   - Root-mean-square acceleration magnitude.
12. **tilt_angle**
   - Mean tilt estimate from $\arctan2(|a_z|, \sqrt{a_x^2+a_y^2})$, in degrees.
13. **stride_cv**
   - Coefficient of variation of inter-peak intervals.

### `build_feature_dataframe(X, y_risk, y_activity, subjects)`
- Iterates all windows.
- Skips windows with insufficient step peaks.
- Appends metadata columns:
  - `label`
  - `activity`
  - `subject_id`
- Prints retained/skipped counts and class counts.

---

## 9) Module Deep Dive: `src/model.py`

## Global setup

- Uses `matplotlib` with `Agg` backend for headless image generation.
- Serialization:
  - `joblib` for RF/XGB models.
  - `.keras` for LSTM.

## Shared evaluation helper

### `_report(name, y_true, y_pred)`
Returns metrics dict:
- accuracy
- precision
- recall
- f1

### `_save_confusion_matrix(...)`
- Saves confusion matrix PNG using `ConfusionMatrixDisplay`.

## Classical model training

### `train_classical_models(features_df, output_dir='output')`

Feature matrix:
- Uses all columns except `label`, `subject_id`, `activity`.

Split strategy:
- `GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)`
- Group key: `subject_id`
- Prevents same-subject leakage between train and test.

Class imbalance handling:
- Computes `scale_pos_weight = n_neg / n_pos` for XGBoost.
- RF uses `class_weight='balanced'`.

Random Forest hyperparameters:
- `n_estimators=500`
- `max_features='sqrt'`
- `min_samples_split=3`
- `class_weight='balanced'`
- `n_jobs=-1`
- `random_state=42`

XGBoost hyperparameters:
- `n_estimators=500`
- `max_depth=8`
- `learning_rate=0.1`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `scale_pos_weight=...`
- `eval_metric='logloss'`
- `random_state=42`

Saved outputs:
- `rf_model.pkl`
- `xgb_model.pkl`
- `feature_names.json`
- `confusion_matrix_rf.png`
- `confusion_matrix_xgb.png`
- `learning_curves_rf_xgb.png`

### `_plot_learning_curves(...)`
- RF: retrains RF with estimators `50..500` step 50 and plots train/test accuracy.
- XGB: plots train/test logloss per boosting round using `evals_result()`.

## LSTM training

### `train_lstm(X_raw, y_risk, subjects, output_dir='output')`

Normalization:
- Channel-wise global mean/std from all windows.
- Saves normalization arrays to `.npy`.

Split:
- Same group split strategy as classical models.

Class weighting:
- Computes inverse-frequency-like weights:
  - `class_weight[0] = total/(2*n_neg)`
  - `class_weight[1] = total/(2*n_pos)`

Architecture:
- Input `(128,6)`
- `LSTM(128, return_sequences=True, kernel_regularizer=l2(0.0005))`
- `Dropout(0.3)`
- `LSTM(64, return_sequences=False, kernel_regularizer=l2(0.0005))`
- `Dropout(0.3)`
- `Dense(32, relu)`
- `Dense(1, sigmoid)`

Compile:
- Optimizer: `adam`
- Loss: `binary_crossentropy`
- Metric: `accuracy`

Callbacks:
- EarlyStopping on `val_loss` with `patience=15`, `restore_best_weights=True`
- ReduceLROnPlateau with factor 0.5, patience 5, `min_lr=1e-5`

Fit settings:
- `epochs=100`
- `batch_size=64`
- validation on held-out subject split.

Saved outputs:
- `lstm_model.keras`
- `lstm_norm_mu.npy`
- `lstm_norm_std.npy`
- `lstm_metrics.json`
- `confusion_matrix_lstm.png`
- `lstm_training_history.png`

## Final model comparison

### `plot_model_comparison(metrics_dict, output_dir)`
- Compares accuracy, F1, recall for RF/XGB/LSTM.
- Saves `model_comparison.png`.

---

## 10) Module Deep Dive: `src/time_series.py`

## `plot_acf_steps(acc_mag_signal, label='Gait', nlags=40)`
- Plots autocorrelation for gait regularity patterns.
- Saves default `acf_<label>.png`.

## `run_adf_test(signal, label='Signal')`
- Uses `adfuller(..., autolag='AIC')`.
- Returns dictionary with statistic, p-value, critical values, stationarity flag (`p<0.05`).

## `plot_stl_decomposition(stride_series, label='Gait', period=10)`
- Requires at least `2*period` observations.
- Decomposes into observed, trend, seasonal, residual with robust STL.
- Saves `stl_<label>.png`.

## `plot_feature_comparison(features_df, features_to_plot=None)`
- Activity-wise boxplots.
- Default plotted features:
  - stride_time
  - cadence
  - sma
  - jerk
  - mean_gyro
  - spectral_entropy
- Saves `feature_comparison_by_activity.png`.

## `run_full_time_series_analysis(features_df, X_sample)`
Executes all analyses:
1. ACF for sample walking signal.
2. ADF for stride-time series by activity (if enough samples).
3. STL decomposition by activity.
4. Feature comparison plot.

---

## 11) Entry Point Behavior

## `main.py` (offline/headless)

Execution order:
1. Ensure output directory exists.
2. Validate dataset path (`data/UCI-HAR Dataset`).
3. Load and relabel data.
4. Extract features.
5. Train RF + XGB.
6. Run time-series analysis.
7. Train LSTM.
8. Generate combined comparison chart.

Use case:
- Batch training on server/CLI.
- Regenerating all outputs without web UI.

## `app.py` (interactive dashboard)

Startup logic:
- If required model artifacts exist (`_models_cached()`), skip retraining and load cached metrics.
- Else run full `_run_pipeline()`.

In-memory cache `_cache` stores:
- `metrics`
- `features_df`
- mobile dataset arrays (`X_mob`, `y_risk`, `subj_mob`, `y_activity`)

This accelerates API response time after startup.

---

## 12) Flask API Reference (Current)

Base host in code: `127.0.0.1:5000`

### `GET /`
- Returns `templates/index.html`.

### `GET /api/status`
- Response: `{ "ready": true|false }`
- Used by front-end polling before showing dashboard.

### `GET /api/subjects`
- Returns per-subject metadata array:
  - `id`
  - `total_windows`
  - `walking`
  - `upstairs`
  - `downstairs`

### `GET /api/predict/<subject_id>?model=rf|xgb|lstm`
- Runs per-window predictions for selected subject.
- Response fields:
  - `subject_id`
  - `model`
  - `n_windows`
  - `accuracy` (subject-level against true window labels)
  - `predictions` (0/1)
  - `probabilities` (risk prob, rounded 4 decimals)
  - `true_labels`
  - `activities` (Walking/Upstairs/Downstairs)

### `GET /api/metrics`
- Returns combined metrics from `all_metrics.json` or cache.

### `GET /api/images`
- Returns base64 data URIs for all generated PNGs used in dashboard.

---

## 13) Output Artifact Inventory (Current Workspace)

File sizes are from current local output directory:

- `acf_walking.png` — 38,739 bytes
- `all_metrics.json` — 554 bytes
- `confusion_matrix_lstm.png` — 20,499 bytes
- `confusion_matrix_rf.png` — 22,330 bytes
- `confusion_matrix_xgb.png` — 22,073 bytes
- `feature_comparison_by_activity.png` — 109,165 bytes
- `feature_names.json` — 181 bytes
- `learning_curves_rf_xgb.png` — 71,963 bytes
- `lstm_metrics.json` — 156 bytes
- `lstm_model.keras` — 1,489,785 bytes
- `lstm_norm_mu.npy` — 152 bytes
- `lstm_norm_std.npy` — 152 bytes
- `lstm_training_history.png` — 62,988 bytes
- `model_comparison.png` — 38,989 bytes
- `rf_model.pkl` — 20,384,009 bytes
- `stl_downstairs.png` — 263,851 bytes
- `stl_upstairs.png` — 282,142 bytes
- `stl_walking.png` — 274,850 bytes
- `xgb_model.pkl` — 1,050,930 bytes

Interpretation:
- RF model is largest serialized artifact.
- LSTM model is compact relative to RF pkl here.
- STL images are large due to 4-panel decomposition plots.

---

## 14) Current Metrics Snapshot

From `output/all_metrics.json`:

- Random Forest:
  - Accuracy: 0.9015
  - Precision: 0.9273
  - Recall: 0.9193
  - F1: 0.9233

- XGBoost:
  - Accuracy: 0.8747
  - Precision: 0.9176
  - Recall: 0.8851
  - F1: 0.9011

- LSTM:
  - Accuracy: 0.9606
  - Precision: 0.9476
  - Recall: 0.9939
  - F1: 0.9702

Observed ranking in this run:
- LSTM > RF > XGBoost (across major classification metrics).

---

## 15) Dashboard/UI Behavior (`templates/index.html`)

The active UI is a single-page dark themed dashboard with these sections:

1. **Loading screen**
   - Shows spinner and waits until `/api/status.ready == true`.
2. **Dataset banner**
   - Briefs UCI HAR windowing and risk mapping.
3. **Model metric cards**
   - Auto-populated from `/api/metrics`.
4. **Model comparison image**
5. **Confusion matrix trio**
6. **Learning curves (RF/XGB + LSTM)**
7. **Time-series images (ACF + feature comparison)**
8. **STL decomposition images for three activities**
9. **Live monitoring panel**
   - Subject dropdown from `/api/subjects`
   - Model dropdown
   - Run button triggers `/api/predict/{subject}`
   - Animated probability bars with tooltip per window
   - Aggregated stats (accuracy, low/high predictions, walking/stairs mix)

Polling behavior:
- `poll()` requests `/api/status` every 3 sec until ready.

---

## 16) Legacy/Unused Assets (Important for Report Accuracy)

Two templates appear to be from an older Parkinson/vGRF project stage:

- `templates/dashboard.html`
- `templates/live.html`

Why considered legacy in current app state:
- `app.py` only routes `"/"` to `index.html`.
- No routes to `/live` or Parkinson-specific APIs shown in those files.
- These templates reference endpoint shapes (`/api/live/<filename>`, subject labels) not implemented in current `app.py`.

Report recommendation:
- Mention them as historical UI artifacts or experimental/previous version remnants.
- State clearly they are not active in current runtime path.

---

## 17) Reproducibility and Determinism Notes

- Random seeds set for split/models in many places (`random_state=42`) but full determinism is not guaranteed due to:
  - TensorFlow runtime non-determinism on some hardware/backends.
  - Multi-threading in tree models.
- Train/test split uses only one GroupShuffleSplit split (single holdout), not cross-validation.
- Metrics may vary if data filtering skips different windows due to peak-detection behavior under code changes.

---

## 18) Dependency Stack

From `requirements.txt`:

- `flask>=2.3`
- `numpy>=1.24`
- `scipy>=1.10`
- `scikit-learn>=1.3`
- `xgboost>=2.0`
- `tensorflow>=2.13`
- `statsmodels>=0.14`
- `pandas>=2.0`
- `matplotlib>=3.7`
- `joblib>=1.3`

Dependency roles:
- Flask: API + UI serving.
- NumPy/SciPy: numeric signal processing.
- Pandas: tabular feature management.
- scikit-learn: RF, metrics, splitting.
- XGBoost: boosted tree classifier.
- TensorFlow Keras: sequence model.
- statsmodels: ADF/STL/ACF.
- Matplotlib: artifact plots.
- Joblib: model persistence.

---

## 19) Methodological Strengths

- Subject-level split prevents leakage from same person across train/test.
- Combined modeling strategy (feature-based + sequence-based) provides methodological breadth.
- Time-series diagnostics (ACF/ADF/STL) add interpretable temporal insight.
- Dashboard consolidates quantitative and qualitative outputs in one interface.

---

## 20) Methodological Limitations (for Discussion Chapter)

- Binary relabeling collapses nuanced activities into coarse risk categories.
- Window-level labels are activity-derived, not directly clinical fall labels.
- No external validation dataset beyond UCI HAR.
- No calibration analysis (e.g., reliability curves/Brier score).
- No uncertainty intervals/confidence bounds reported.
- Single random split; no grouped K-fold CV in current implementation.

---

## 21) Operational Notes

- First `python app.py` run can be slow because all models are trained.
- Subsequent runs faster due to cached artifacts.
- If `data/UCI-HAR Dataset` is missing, `app.py` exits with explicit error.
- `download_data.py` can recover dataset if deleted.

---

## 22) Security / Robustness Observations

- API does not expose arbitrary file reads; image serving uses fixed known filenames from `output/`.
- Input surface on prediction endpoint is constrained to integer subject id and known model options (with default behavior).
- No authentication layer (acceptable for local academic dashboard but should be considered for deployment).

---

## 23) Suggested 30-Page Report Chapter Plan

Use this exact structure to expand into a full report:

1. **Abstract (1 page)**
2. **Problem Motivation and Scope (2 pages)**
3. **Literature Context (2–3 pages)**
4. **Dataset Description and Suitability (3 pages)**
5. **Data Engineering Pipeline (3 pages)**
6. **Feature Engineering Theory and Formulas (4 pages)**
7. **Model Design (RF, XGB, LSTM) and Hyperparameters (4 pages)**
8. **Experimental Protocol (splits, metrics, leakage control) (2 pages)**
9. **Results (tables + plots) (3 pages)**
10. **Time-Series Analysis Findings (ACF/ADF/STL) (2 pages)**
11. **Dashboard System Design and API Contract (2 pages)**
12. **Limitations, Threats to Validity, and Ethics (1–2 pages)**
13. **Future Work and Deployment Considerations (1 page)**
14. **Conclusion (1 page)**
15. **Appendix (code snippets, endpoint examples, artifact inventory) (3+ pages)**

---

## 24) Ready-to-Use Tables for Report

## Table A: Activity to risk mapping

| Original Activity ID | Activity Name | Risk Label |
|---|---|---|
| 1 | WALKING | 0 (Low) |
| 2 | WALKING_UPSTAIRS | 1 (High) |
| 3 | WALKING_DOWNSTAIRS | 1 (High) |
| 4 | SITTING | Excluded |
| 5 | STANDING | Excluded |
| 6 | LAYING | Excluded |

## Table B: Current model performance

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.9015 | 0.9273 | 0.9193 | 0.9233 |
| XGBoost | 0.8747 | 0.9176 | 0.8851 | 0.9011 |
| LSTM | 0.9606 | 0.9476 | 0.9939 | 0.9702 |

## Table C: Core engineered features

| Feature | Meaning |
|---|---|
| stride_time | Mean inter-step interval |
| cadence | Step rate proxy from stride_time |
| step_variance | Variability in inter-step intervals |
| symmetry_index | Energy ratio first vs second half of window |
| sma | Signal magnitude area of acceleration |
| mean_gyro | Mean angular velocity magnitude |
| jerk | Mean absolute acceleration derivative |
| dominant_freq | Peak frequency in FFT |
| spectral_entropy | Complexity in frequency distribution |
| autocorr_lag1 | Lag-1 autocorrelation |
| rms_acc | RMS acceleration magnitude |
| tilt_angle | Mean orientation proxy |
| stride_cv | Relative variability of stride intervals |

---

## 25) Exact Commands for Reproduction

## Environment setup

```bash
pip install -r requirements.txt
```

## Dataset fetch (if missing)

```bash
python download_data.py
```

## Train everything (headless)

```bash
python main.py
```

## Run dashboard

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.

---

## 26) What to Cite in Academic Writing

For dataset usage:
- Anguita et al., ESANN 2013 (as provided inside UCI HAR README).

For methodological references you can discuss:
- Random Forest ensemble learning.
- Gradient boosting/XGBoost for tabular classification.
- LSTM for temporal sequence modeling.
- ADF and STL for stationarity/seasonality analysis.

---

## 27) Gaps to Clarify in Final Report (if needed)

If your evaluator expects stricter ML rigor, include (as future work or extension section):

- Grouped K-fold cross-validation.
- Confidence intervals over repeated splits.
- Probability calibration and threshold analysis.
- Subject-wise error analysis and fairness checks.
- External validation on another gait dataset.

---

## 28) Final Summary

This codebase is a complete **sensor-based fall-risk modeling workflow** with:
- robust data loading and relabeling,
- interpretable feature engineering,
- classical + deep learning baselines,
- temporal diagnostics,
- and a working local dashboard for visual analysis and live per-subject inference.

It is strong for an academic applied-ML project and already contains enough implementation and analysis depth to support a long technical report.
