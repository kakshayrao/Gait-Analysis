# 🚶 IMU Gait Analysis — Fall Risk Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-black?logo=flask&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20HAR-informational)
![License](https://img.shields.io/badge/License-MIT-green)

> **Time-series gait analysis using smartphone IMU data to detect fall risk.**  
> Classifies walking patterns from 30 subjects into **Low Risk** (flat walking) vs **High Risk** (stairs) using Random Forest, XGBoost, and LSTM models — with a live interactive dashboard.

## 🗂 Dataset

**[UCI HAR (Human Activity Recognition)](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)**

| Property | Value |
|---|---|
| Subjects | 30 |
| Sampling Rate | 50 Hz |
| Window Size | 128 samples (2.56 s, 50% overlap) |
| IMU Channels | 6 (body_acc x/y/z, body_gyro x/y/z) |
| Activities Used | Walking, Walking Upstairs, Walking Downstairs |
| Activities Excluded | Sitting, Standing, Laying |

**Fall Risk Labelling:**
- 🟢 **Low Risk** → `WALKING` (flat ground)
- 🔴 **High Risk** → `WALKING_UPSTAIRS` + `WALKING_DOWNSTAIRS`

---

## ✨ Features

### 🧠 Models
- **Random Forest** — trained on 13 engineered gait features
- **XGBoost** — same tabular features, class-weighted
- **LSTM** — trained on raw IMU sequences (128 × 6), with EarlyStopping

### 📐 Gait Features Extracted
Stride time, Cadence, Step variance, Symmetry index, Signal Magnitude Area (SMA), Mean angular velocity, Jerk, Dominant frequency (FFT), Spectral entropy, Autocorrelation (lag-1), RMS acceleration, Tilt angle, Stride coefficient of variation

### 📉 Time-Series Analysis
- **ACF** plots (gait step autocorrelation)
- **ADF Test** (stationarity check per activity)
- **STL Decomposition** (trend + seasonality of stride-time series)
- **Per-activity feature comparison** (box plots)

### 🔴 Live Monitoring
Select any of the 30 subjects + choose a model → animated per-window fall-risk bar chart with per-window hover tooltips and activity colour strip.

---

## 🗃 Project Structure

```
ATSA/
├── data/
│   ├── raw/                    # Zip download cache
│   └── UCI-HAR Dataset/        # Extracted dataset (auto-populated)
├── src/
│   ├── data_loader.py          # Loads UCI HAR raw signals
│   ├── preprocessing.py        # Bandpass filter, magnitude, normalisation
│   ├── feature_extraction.py   # 13 IMU gait features per window
│   ├── time_series.py          # ACF, ADF, STL, feature comparison plots
│   └── model.py                # RF, XGBoost, LSTM training + evaluation
├── output/                     # Saved models + plots (auto-generated)
├── templates/
│   └── index.html              # Flask dashboard UI
├── app.py                      # Flask web app (train + serve)
├── main.py                     # Headless training pipeline
├── download_data.py            # Auto-download UCI HAR dataset
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/kakshayrao/Gait-Analysis.git
cd Gait-Analysis
pip install -r requirements.txt
```

### 2. Get the dataset

The dataset is already included at `data/UCI-HAR Dataset/`. If it's missing, run:

```bash
python download_data.py
```

### 3. Run the dashboard (trains models on first launch)

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

**First run** trains all three models (~5–10 min depending on hardware).  
**Subsequent runs** load cached models instantly.

### 4. Headless training only (no web server)

```bash
python main.py
```

All outputs saved to `output/`.

---

## 📦 Requirements

```
flask>=2.3
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
xgboost>=2.0
tensorflow>=2.13
statsmodels>=0.14
pandas>=2.0
matplotlib>=3.7
joblib>=1.3
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📁 Output Files

| File | Description |
|---|---|
| `output/rf_model.pkl` | Trained Random Forest |
| `output/xgb_model.pkl` | Trained XGBoost |
| `output/lstm_model.keras` | Trained LSTM |
| `output/model_comparison.png` | Accuracy/F1/Recall bar chart |
| `output/confusion_matrix_*.png` | Confusion matrices (RF, XGB, LSTM) |
| `output/lstm_training_history.png` | Loss & accuracy curves |
| `output/acf_walking.png` | Autocorrelation of gait steps |
| `output/stl_*.png` | STL decomposition per activity |
| `output/feature_comparison_by_activity.png` | Box plots by activity |

---

## 🧪 Methodology

```
Raw IMU signals (50 Hz)
        ↓
  128-sample windows (2.56 s, 50% overlap)
        ↓
  Fall-risk labelling  [Walking→Low | Upstairs/Downstairs→High]
        ↓
    ┌───────────────────┬──────────────────────┐
    │  Feature path     │   Sequence path       │
    │  (RF / XGBoost)   │   (LSTM)              │
    │  13 gait features │   raw (128 × 6)       │
    └──────────┬────────┴───────────┬──────────┘
               │                   │
     GroupShuffleSplit (subject-level, no leakage)
               │
          Evaluation → Accuracy / F1 / Recall / Precision
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) — HAR Dataset
- Davide Anguita et al., *"A Public Domain Dataset for Human Activity Recognition Using Smartphones"*, ESANN 2013.
