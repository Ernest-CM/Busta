# AI-Driven Malaria Screening and Diagnosis Support System

Modular ML pipeline for malaria cell-image classification using the NIH thin-blood-smear dataset.

## Project structure

```
src/data/           dataset ingestion and split utilities
src/preprocessing/  deterministic preprocessing (resize, denoise, normalize)
src/models/         CNN definition, training loop, inference loader
src/evaluation/     metrics and evaluation utilities
src/api/            Flask prediction API
src/utils/          shared config and I/O helpers
tests/              unit + integration tests
scripts/            runnable entry points
```

## Label mapping (fixed contract)

| Folder name  | Label |
|---|---|
| Parasitized  | 1 |
| Uninfected   | 0 |

## Preprocessor output contract

| Property | Value |
|---|---|
| Shape | `(N, H, W, 3)` batch / `(H, W, 3)` single |
| dtype | `float32` |
| Color order | RGB |
| Value range | `[0.0, 1.0]` |

---

## Python version

TensorFlow currently has no wheel for Python 3.13+. Training and the API server require **Python 3.9 – 3.12**. Tests that only exercise preprocessing and metrics run on any Python ≥ 3.9 (TensorFlow tests are auto-skipped via `pytest.importorskip` when TF is unavailable).

## Setup

All commands below assume the working directory is `Busta/` (the project root containing `src/`).

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # dev/test dependencies
```

---

## Data

Download the [NIH Malaria Cell Images dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) and place it at:

```
data/raw/cell_images/
    Parasitized/   ← .png files here
    Uninfected/    ← .png files here
```

---

## Training

```bash
python scripts/run_training.py
```

Artifacts are saved to `models_artifacts/`:

| File | Contents |
|---|---|
| `baseline_cnn.keras` | Saved Keras model |
| `training_history.json` | Per-epoch loss/accuracy/AUC |
| `evaluation_metrics.json` | Test-set accuracy, precision, recall, F1, ROC-AUC |
| `confusion_matrix.csv` | 2×2 confusion matrix |

---

## API server

Requires a trained model at `models_artifacts/baseline_cnn.keras`.

```bash
python scripts/run_api.py
```

Server listens on `http://127.0.0.1:5000`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/predict` | Predict from uploaded image |

**Predict example (curl):**

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -F "image=@/path/to/cell_image.png"
```

**Response:**

```json
{
  "prediction": 1,
  "label": "Parasitized",
  "probability_parasitized": 0.9123
}
```

---

## Tests

```bash
pytest tests/ -v
```

Expected output: all tests in `test_preprocessor.py`, `test_metrics.py`, and `test_model.py` pass.

The `test_training_smoke` test runs a full training cycle on 12 synthetic images at 32×32 resolution (1 epoch). It is slow on first run due to TensorFlow initialisation (~10–30 s depending on hardware).

---

## Lint

```bash
ruff check src/ tests/ scripts/
```

---

## Risks and TODOs

- **Patient-level leakage** — the current split is file-level. If multiple cells from the same patient appear in both splits, evaluation may be optimistic. Extend to patient-level split when patient IDs are available.
- **No augmentation** — augmentation (flips, rotations) will be added in the next training phase to improve generalisation.
- **Class imbalance** — the NIH dataset is roughly balanced; if a domain-shifted subset is used, consider weighted loss or oversampling.
- **GPU/CPU determinism** — `tf.config.experimental.enable_op_determinism()` may raise on some CPU builds. If training fails, set `seed` only via `tf.keras.utils.set_random_seed`.
