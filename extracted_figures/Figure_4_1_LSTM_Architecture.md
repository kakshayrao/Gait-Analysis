# Figure 4.1 — LSTM Network Architecture

```mermaid
flowchart TD
    A[Input Sequence\nShape: 128 x 6\n(IMU window)] --> B[LSTM Layer 1\nUnits: 128\nreturn_sequences=True\nL2 regularization: 0.0005]
    B --> C[Dropout\nRate: 0.30]
    C --> D[LSTM Layer 2\nUnits: 64\nreturn_sequences=False\nL2 regularization: 0.0005]
    D --> E[Dropout\nRate: 0.30]
    E --> F[Dense Layer\nUnits: 32\nActivation: ReLU]
    F --> G[Output Layer\nUnits: 1\nActivation: Sigmoid\nOutput: Fall-risk probability]

    H[Loss: Binary Cross-Entropy]:::meta --> G
    I[Optimizer: Adam]:::meta --> G
    J[Metric: Accuracy]:::meta --> G
    K[Class Weights Applied]:::meta --> G
    L[EarlyStopping + ReduceLROnPlateau]:::meta --> G

    classDef meta fill:#f5f5f5,stroke:#888,color:#222,stroke-width:1px;
```
