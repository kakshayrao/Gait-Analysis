# Gait-Based Parkinson Detection and Real-Time Fall Risk Monitoring

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Project Overview
This project implements a technical pipeline for the automated detection of Parkinson's Disease (PD) using Vertical Ground Reaction Force (vGRF) signals. By analyzing stride-to-stride dynamics, the system extracts high-fidelity biomechanical features to classify subjects and provide a simulated real-time fall risk monitoring system.

The core objective is to demonstrate how time-series analysis of gait patterns can be translated into actionable clinical insights, specifically identifying gait degradation and instability associated with PD.


## Technical Methodology

### 1. Data Acquisition
Utilizes the **PhysioNet Gait in Parkinson's Disease (gaitpdb)** dataset.
- **Sensors**: 8 vGRF sensors per foot (16 channels total).
- **Sampling Rate**: 100 Hz.
- **Classes**: Healthy Control (Co) vs. Parkinson's Disease (Pt).

### 2. Signal Processing & Feature Engineering
Raw signals undergo a multi-stage preprocessing pipeline:
- **Smoothing**: Moving average filter for high-frequency noise removal.
- **Normalization**: Z-score or Min-Max scaling per subject to handle individual force variations.
- **Segmentation**: Sliding window approach (3.0s windows with 50% overlap).
- **Peak Detection**: Identification of heel-strike events to derive temporal gait cycles.

**Extracted Features:**
- **Stride Time**: Mean duration of one complete gait cycle.
- **Cadence**: Steps per minute derived from frequency analysis.
- **Variability**: Standard deviation of stride intervals (hallmark indicator of PD).
- **Symmetry**: Ratio of Root Mean Square (RMS) forces between left and right foot.
- **Force Statistics**: Mean, Standard Deviation, and Coefficient of Variation (CV) of vertical force.

### 3. Machine Learning Architecture
A **Random Forest Classifier** is employed for its interpretability and robustness to high-dimensional time-series features.
- **Model**: Ensemble of 100 decision trees.
- **Validation**: Stratified 80/20 Train-Test split.
- **Evaluation**: Accuracy, Precision, Recall, and F1-score computation.

### 4. Real-Time Fall Risk Simulation
The trained model is integrated into a monitoring loop that simulates live data streaming:
- **Dynamic Inference**: Processes incoming 3s windows sequentially.
- **Risk Scoring**: Maps model class probability to a "Fall Risk" metric.
- **Alert Logic**: Triggers alerts when the predicted Parkinson probability exceeds 0.75.


Upon execution, the system will:
1. Load and parse the 309 subject files.
2. Extract temporal and statistical gait features.
3. Train the Random Forest classifier.
4. Generate 10 comparative and performance plots in the `output/` directory.
5. Simulate real-time monitoring on selected subjects.


## Clinical Relevance
The system highlights several key metrics used in clinical gait analysis:
- **Instability**: Higher stride variability strongly correlates with increased fall risk in PD patients.
- **Bradykinesia**: Reduced cadence and increased stride time reflect the slowing of movement.
- **Continuous Monitoring**: Demonstrates the feasibility of using wearable insole sensors for long-term monitoring of disease progression and intervention efficacy.
