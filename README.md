# LayerLogic

### Real-Time Defect Detection in Metal Additive Manufacturing via Multimodal Deep Learning

> **Team SteelSync** · IIT Kharagpur Platinum Jubilee Innovation Challenge — Stage 2  
> Track: Deep Tech for the World · Domain: AI/ML × Advanced Manufacturing (Industry 4.0)

---

## Overview

LayerLogic is a multimodal AI pipeline that detects micro-defects (porosity, keyhole collapse, lack-of-fusion) during Laser Powder Bed Fusion (LPBF) metal 3D printing. It fuses two data streams — **acoustic emissions** and **melt-pool imagery** — through pretrained CNN feature extractors and classical ML classifiers to output a real-time defect probability score.

**Architecture:**
```
Audio (.wav) → Mel-Spectrogram → ResNet-18 → 512-d features ─┐
                                                                ├→ Concat → PCA → XGBoost/RF/DT → Defect Score
Image (.jpg) → Resize/Norm     → ResNet-18 → 512-d features ─┘
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Data

Downloads real melt-pool images from Zenodo (EOS M290, 316L steel) and generates physics-informed synthetic acoustic files. Falls back to fully synthetic data if download fails.

```bash
python src/data_setup.py
```

### 3. Train the Model

Extracts features with frozen ResNet-18, trains 3 classifiers (XGBoost, Random Forest, Decision Tree), evaluates on test set, and generates publication-quality graphs.

```bash
python src/train.py
```

**Outputs saved to `outputs/`:**
- `best_pipeline.pkl` — Best trained classifier pipeline
- `training_summary.json` — Metrics for all classifiers
- `classifier_comparison.png` — F1/AUC bar chart
- `training_history.png` — Learning curves
- `roc_curve.png` — ROC curves with AUC
- `confusion_matrix.png` — Confusion matrix heatmap

### 4. Launch the Application

**Start the FastAPI backend** (Terminal 1):
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

**Start the Streamlit frontend** (Terminal 2):
```bash
streamlit run src/app.py
```

Open your browser to `http://localhost:8501` to see the full pitch-deck app with interactive demo and live replay mode.

---

## Project Structure

```
layerlogic/
├── src/
│   ├── data_setup.py     # Data acquisition + synthetic generation
│   ├── dataset.py        # PyTorch Dataset + feature extraction
│   ├── model.py          # ResNet-18 extractor + classifier pipelines
│   ├── train.py          # Training, evaluation, graph generation
│   ├── api.py            # FastAPI inference server
│   └── app.py            # Streamlit web application
├── data/                 # Generated/downloaded data
├── outputs/              # Trained models + graphs
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Deep Learning | PyTorch (ResNet-18, ImageNet pretrained) |
| Audio Processing | Librosa (Mel-spectrograms) |
| Computer Vision | OpenCV, PIL |
| Classical ML | Scikit-learn (RF, DT), XGBoost |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Visualization | Matplotlib, Seaborn |

---

## Research References

- Shevchik, Wasmer et al. (EMPA) — Acoustic emission + spectral CNN for LPBF
- Scime & Beuth (Carnegie Mellon) — Multi-scale CNN for powder-bed anomaly detection
- IIT Kharagpur (Subir Chowdhury School) — Multimodal monitoring with EfficientNet/ResNet + XGBoost
- NIST AM-Bench — Benchmark additive manufacturing datasets

---

*Built with ❤️ for zero-defect manufacturing.*
