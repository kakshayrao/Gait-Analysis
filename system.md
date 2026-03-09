┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                       │
│  data/UCI-HAR Dataset/                                                   │
│    ├── train/  (7352 × 128 windows, 6 subjects)                          │
│    │   ├── Inertial Signals/  (body_acc_x/y/z, body_gyro_x/y/z .txt)    │
│    │   ├── y_train.txt        (activity labels 1-6)                      │
│    │   └── subject_train.txt  (subject IDs 1-30)                         │
│    └── test/   (2947 × 128 windows)                                      │
│        ├── Inertial Signals/                                             │
│        ├── y_test.txt                                                    │
│        └── subject_test.txt                                              │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              1. DATA LOADING (src/data_loader.py)                        │
│  • Load 6 raw IMU channels from .txt files                               │
│  • Stack into (10299, 128, 6) windows  [N, timesteps, channels]          │
│  • Merge train/test splits                                              │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│         2. FALL-RISK LABELLING (src/data_loader.py)                      │
│  Activity 1 (WALKING) → LOW_RISK                                         │
│  Activity 2,3 (UP/DOWN stairs) → HIGH_RISK                               │
│  (Discard sitting/standing/laying)                                       │
│  Result: 4672 × 128 × 6 windows with binary labels                       │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
        ┌────────────┴──────────────┐
        ▼                           ▼
  ┌──────────────────┐       ┌──────────────────────┐
  │ DATA BRANCH A    │       │ DATA BRANCH B        │
  │ (Feature Path)   │       │ (Sequence Path)      │
  │                  │       │                      │
  └──────┬───────────┘       └──────┬───────────────┘
         │                          │
         ▼                          │
┌──────────────────────────     │
│ 3. FEATURE EXTRACTION       │
│    (src/feature_extraction.py) │
│                              │
│ • FFT spectral features    │
│ • Step detection (peaks)   │
│ • Acceleration/gyro stats  │
│ • Symmetry, jerk, entropy  │
│                            │
│ Output: 13 features/window │
│ Shape: (4672, 13)          │
└─────────┬──────────────────│
          │                  │
          ▼                  │
┌─────────────────────────┐ │
│ 4A. TRAIN RF/XGBoost    │ │
│    (src/model.py)        │ │
│                          │ │
│ • GroupShuffleSplit     │ │
│   (subject-level split) │ │
│ • RandomForest          │ │
│ • XGBoost (class-wt)    │ │
│                         │ │
│ Outputs:               │ │
│ ├─ rf_model.pkl       │ │
│ ├─ xgb_model.pkl      │ │
│ └─ metrics (acc, F1)  │ │
└────────┬───────────────│ │
         │               │ │
         ▼               ▼
         ...             │
                         │
                    (Raw 128×6)
                         │
                         ▼
                  ┌──────────────────────┐
                  │ 4B. TRAIN LSTM        │
                  │    (src/model.py)     │
                  │                       │
                  │ • Sequential model    │
                  │ • LSTM layer          │
                  │ • Dense + Dropout     │
                  │ • EarlyStopping       │
                  │ • ReduceLROnPlateau   │
                  │                       │
                  │ Output:               │
                  │ └─ lstm_model.keras   │
                  │ └─ lstm_metrics.json  │
                  └──────────┬────────────┘
                             │
         ┌───────────────────┴─────────────────┐
         │                                     │
         ▼                                     ▼
┌──────────────────────────────┐    ┌────────────────────────────┐
│ 5. TIME-SERIES ANALYSIS      │    │ 6. PLOTTING & METRICS      │
│    (src/time_series.py)      │    │    (src/model.py)          │
│                              │    │                            │
│ • ACF plots (autocorr)       │    │ • Confusion matrices       │
│ • ADF tests (stationarity)   │    │ • Model comparison chart   │
│ • STL decomposition          │    │ • Training histories       │
│ • Feature box-plots          │    │                            │
│                              │    └────────────┬───────────────┘
└──────────┬───────────────────┘                 │
           │                                    │
           └─────────────────┬──────────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │ OUTPUT/ Directory   │
                    │ ├─ *.pkl models    │
                    │ ├─ *.keras LSTM    │
                    │ ├─ *.json metrics  │
                    │ └─ *.png plots     │
                    └────────────────────┘