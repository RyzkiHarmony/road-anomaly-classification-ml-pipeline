# 🛣️ Road Anomaly Detection — ML Pipeline

![CI Pipeline](https://github.com/YOUR_USERNAME/ml_pipelines/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-ee4c2c.svg)
![ONNX](https://img.shields.io/badge/export-ONNX-informational.svg)
![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

An **end-to-end ML pipeline** for real-time road anomaly detection (potholes & speed bumps) on Android edge devices. The system uses raw IMU sensor data (accelerometer + gyroscope) collected via smartphone mounted on a motorcycle, and exports a production-ready model to ONNX for zero-latency on-device inference.

> **Context:** This pipeline was developed as the core ML system for an undergraduate thesis (*Skripsi*) at UDINUS. The final model achieves **Macro F1-Score = 0.86** on the holdout test set after Optuna hyperparameter optimization.

---

## 📐 System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Raw Sensor Data│────▶│  Sensor Fusion   │────▶│  Windowing &       │
│  (100Hz CSV)    │     │  (Causal Filter) │     │  Signal Extraction │
└─────────────────┘     └──────────────────┘     └────────────────────┘
                                                             │
                                                             ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  ONNX Export    │◀────│  CNN Training    │◀────│  Optuna HPO        │
│  (Android Ready)│     │  (Focal Loss)    │     │  (10 Trials)       │
└─────────────────┘     └──────────────────┘     └────────────────────┘
```

**Key Pipeline Stages:**
1. **Sensor Fusion** — Separates gravity from linear acceleration using a causal low-pass filter (`scipy.signal.lfilter`) with zero lookahead, ensuring real-time viability on mobile.
2. **Signal Preprocessing** — Strict resampling to 100 Hz (10ms interval) with a maximum gap tolerance of 50ms. Applies **Global Z-Score Normalization** anchored on training set statistics (not per-instance), preventing inference-time distribution shift.
3. **Physics-Aware Augmentation** — During training: **Time Warping** (simulates variable motor speed), **Channel Dropout** (simulates sensor orientation faults), and **Temporal Jitter** (simulates mounting vibration).
4. **Hyperparameter Optimization** — Optuna with Median Pruner searches across LR, Dropout, Batch Size, Channels, Focal Loss Gamma, and Weight Decay. The objective is a **Robust Score** (`mean_F1 − std_F1`) to penalize unstable cross-fold behavior.
5. **ONNX Export** — PyTorch model wrapped with `MobileInferenceWrapper` (global scaler baked in as buffers) and exported via TorchScript to ONNX for on-device Android inference.
6. **Android Synchronization** — Processing logic is kept 1-to-1 between this pipeline and the Kotlin Android app, including the 50ms dropout gap tolerance and `eps=1e-6` Z-Score normalization.

---

## 📊 Model Performance (Holdout Test Set — 20%)

Results on 941 samples from **5 trip-isolated** test routes (never seen during training):

| Metric | Baseline 1D-CNN | Optuna Tuned 1D-CNN | Δ Improvement |
|:---|:---:|:---:|:---:|
| **Non-Event F1** | 0.98 | 0.98 | — |
| **Pothole F1** | 0.72 | **0.78** | +0.06 ✅ |
| **Pothole Recall** | 0.59 | **0.71** | +0.12 ✅ |
| **Speed Bump F1** | 0.79 | **0.81** | +0.02 ✅ |
| **Macro F1-Score** | 0.83 | **0.86** | +0.03 ✅ |
| **Accuracy** | 96% | **96%** | — |

> **Optuna Best Parameters:** `LR=0.001253`, `Batch=16`, `Channels=48`, `Dropout=0.2094`, `WeightDecay=2.7e-5`, `Gamma=1.67`

---

## 🚀 Quickstart

```bash
# 1. Clone and create venv
git clone https://github.com/YOUR_USERNAME/ml_pipelines.git
cd ml_pipelines
python -m venv .venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. Install all dependencies
make install

# 3. Build datasets from raw CSV recordings
python src/dataset/build_cnn_data.py

# 4. Train the model
make train

# 5. Export to ONNX for Android
make export-onnx
```

---

## 🗂️ Project Structure

```
ml_pipelines/
├── src/
│   ├── cnn_model/
│   │   ├── train.py            # Main training script (K-Fold + Holdout)
│   │   ├── optuna_tune.py      # Hyperparameter optimization
│   │   ├── export_onnx.py      # ONNX export with MobileInferenceWrapper
│   │   ├── model.py            # InceptionTime1D architecture
│   │   └── analysis/           # PR-Curve, convergence visualization
│   ├── xgboost_model/          # Classical ML baseline (XGBoost + Isotonic)
│   ├── dataset/                # Dataset builders & ground truth labelers
│   ├── utils/                  # Shared config, logger, feature utils
│   └── tests/                  # Unit tests (pytest)
├── evaluation/
│   ├── models/                 # Saved .pth, .onnx, scaler params
│   └── reports/                # Confusion matrices, PR curves
├── data/
│   └── processed/              # Windowed features & signals (generated)
├── log/                        # Training run logs
├── Makefile                    # Developer automation commands
├── ruff.toml                   # Code quality configuration
└── requirements.txt            # All dependencies (pinned versions)
```

---

## 🔬 Key Technical Decisions

| Decision | Rationale |
|:---|:---|
| **Causal Filter (no lookahead)** | Ensures zero-latency real-time inference on device. Non-causal filters would require future samples, making live detection impossible. |
| **Trip-Based Train/Test Split** | Prevents data leakage. Samples from the same trip share temporal correlations; shuffling would produce artificially inflated metrics. |
| **Global Z-Score (not per-instance)** | Anchoring normalization on training set statistics ensures the inference distribution matches training, preventing silent accuracy degradation at deployment. |
| **Focal Loss (γ=1.67)** | Addresses the extreme class imbalance (~1:15 ratio of anomaly to normal road). Down-weights easy Non-Event samples so the model focuses on learning the minority anomaly pattern. |
| **Robust Score Objective** | `mean(F1) − std(F1)` penalizes high-variance solutions in Optuna. A model that scores 0.70 in all folds is preferred over one that scores 0.90 in one fold and 0.50 in another. |
| **MobileInferenceWrapper (ONNX)** | Bakes the global scaler (mean, std) as PyTorch buffers into the ONNX graph, so the Android app only sends raw sensor windows — no separate preprocessing code required. |

---

## 🛠️ Developer Commands

```bash
make help         # List all available commands
make install      # Install all Python dependencies
make lint         # Run Ruff linter + auto-fix
make test         # Run unit tests with pytest
make train        # Train 1D-CNN (local)
make tune         # Run Optuna HPO (local)
make export-onnx  # Export model to ONNX (local)
make clean        # Remove __pycache__ and .pytest_cache
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
