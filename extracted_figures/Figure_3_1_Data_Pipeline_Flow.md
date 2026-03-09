# Figure 3.1 — End-to-End Data Flow Diagram

```mermaid
flowchart TD
    A[UCI HAR Raw Files\ntrain/test + inertial signals] --> B[Load + Merge Splits\nX shape: N x 128 x 6]
    B --> C[Filter Activities\nKeep 1/2/3 only]
    C --> D[Risk Relabeling\n1 -> Low 0\n2/3 -> High 1]

    D --> E[Feature Path\n13 engineered gait features/window]
    D --> F[Sequence Path\nRaw IMU windows 128 x 6]

    E --> G[GroupShuffleSplit by subject_id\nNo subject leakage]
    F --> H[Global channel normalization\n(mu, std)]
    H --> I[GroupShuffleSplit by subject_id\nNo subject leakage]

    G --> J[Random Forest Training]
    G --> K[XGBoost Training]
    I --> L[LSTM Training]

    J --> M[Metrics + Confusion Matrix]
    K --> M
    L --> M

    M --> N[Model Comparison Plot\nAccuracy / F1 / Recall]
    M --> O[Saved Artifacts in output/\nmodels + JSON + PNGs]

    O --> P[Flask API Layer\n/api/metrics /api/images /api/predict]
    P --> Q[Dashboard UI\nLive Subject Risk View]
```
