# AI-Driven Malaria Screening and Diagnosis Support System

[![CI](https://github.com/Creator/Busta/actions/workflows/ci.yml/badge.svg)](https://github.com/Creator/Busta/actions/workflows/ci.yml)

---

## Project Overview

An end-to-end deep learning pipeline for classifying thin blood-smear microscopy images as *Parasitized* or *Uninfected* using the NIH Cell Image dataset. The system provides a baseline CNN, a MobileNetV2 transfer-learning model with two-stage fine-tuning, a rate-limited Flask REST API with Grad-CAM explainability, and a single-page web UI — all containerised with Docker.

---

## System Architecture

```
NIH Data (data/raw/cell_images/)
        │
        ▼
  Preprocessor
  (resize 224×224 · median denoise · [0,1] normalize)
        │
        ├──────────────────────────┐
        ▼                          ▼
  Baseline CNN              MobileNetV2
  (3-block conv)        (frozen base → fine-tune)
        │                          │
        └──────────┬───────────────┘
                   ▼
             Flask API  (:5000)
         /api/predict  /api/gradcam  /api/health
                   │
                   ▼
              Web UI  (/)
         Diagnose tab · Explain (Grad-CAM) tab
```

---

## Setup

**Python requirement:** 3.9 – 3.12 (TensorFlow has no wheel for Python 3.13+).

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt   # dev/test tools

# 3. Place the NIH dataset at:
#    data/raw/cell_images/Parasitized/   ← .png files
#    data/raw/cell_images/Uninfected/    ← .png files
```

Download the dataset from [Kaggle — Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria).

---

## Running the Pipeline

All commands assume the working directory is `Busta/` (the project root containing `src/`).

```bash
# Train baseline CNN (saves model + metrics to models_artifacts/)
python scripts/run_training.py

# Train MobileNetV2 transfer model (2-stage fine-tuning)
python scripts/run_transfer_training.py --epochs 10 --fine-tune-epochs 5

# Re-run evaluation on the held-out test split (saves PNG charts + metrics.json)
python scripts/run_evaluation.py

# Benchmark all .keras models and produce a comparison table
python scripts/run_benchmark.py

# Start the Flask API server (requires a trained model)
python scripts/run_api.py
```

---

## API Reference

| Method | Path | Description | Request | Response |
|--------|------|-------------|---------|----------|
| GET | `/api/health` | Liveness check | — | `{"status":"ok","model_loaded":true}` |
| POST | `/api/predict` | Binary classification | `multipart/form-data` field `image` (PNG/JPG/JPEG/BMP, ≤ 8 MB) | `{"prediction":"Parasitized","confidence":0.92,"label":1}` |
| POST | `/api/gradcam` | Grad-CAM heatmap overlay | `multipart/form-data` field `image` (PNG/JPG/JPEG/BMP, ≤ 8 MB) | PNG image bytes |

**Rate limits:** `/api/predict` and `/api/gradcam` are capped at **10 requests / minute** per IP address. Exceeding the limit returns HTTP 429 `{"error":"rate limit exceeded"}`.

**Curl example:**
```bash
curl -X POST http://localhost:5000/api/predict -F "image=@cell.png"
```

---

## Benchmark Results

Run `python scripts/run_benchmark.py` after training to populate real numbers.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| *(fill after training)* | — | — | — | — | — |

> Fill with results from `scripts/run_benchmark.py` after training.

---

## Docker

```bash
# Build the image
docker build -t malaria-api .

# Run with docker-compose (mounts models_artifacts/ as a volume)
docker-compose up
```

The API is then available at `http://localhost:5000`. Place a trained `.keras` file in `models_artifacts/` before starting the container.

---

## CI

The GitHub Actions workflow at `.github/workflows/ci.yml` runs on every push and pull-request to `main`:

- Lints with **ruff** (`src/`, `tests/`, `scripts/`)
- Runs all TensorFlow-free tests with **pytest** (TF tests are excluded from CI to keep runtime under 3 minutes)

Replace `<owner>/<repo>` in the badge URL at the top of this file with your GitHub repository path.

---

## Known Limitations

- **Python 3.9–3.12 required for TF**: TensorFlow publishes no wheel for Python 3.13+. Tests that do not require TF pass on any Python ≥ 3.9.
- **File-level split leakage**: Train/test splitting is done at the image-file level. If the NIH dataset contains multiple images from the same patient, the same patient may appear in both splits, making evaluation metrics optimistic. A patient-level split would require external patient-ID metadata.
- **8 MB upload cap**: The API rejects uploads larger than 8 MB at the route level. Very high-resolution microscopy images must be resized before submission.
