import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.evaluation.metrics import EvaluationResult, compute_classification_metrics


def evaluate_predictions(y_true, y_pred, y_score=None) -> EvaluationResult:
    """Thin wrapper kept for backwards compatibility."""
    return compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score)


def run_evaluation(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir,
) -> Dict[str, float | None]:
    """
    Run full evaluation: compute metrics, save confusion matrix PNG,
    ROC curve PNG, and metrics.json to output_dir. Returns the metrics dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_score = model.predict(x_test, verbose=0).reshape(-1)
    y_pred = (y_score >= 0.5).astype(np.int64)
    result = compute_classification_metrics(y_true=y_test, y_pred=y_pred, y_score=y_score)

    (output_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2), encoding="utf-8"
    )

    _save_confusion_matrix_png(result.confusion_matrix, output_dir / "confusion_matrix.png")

    if result.metrics["roc_auc"] is not None:
        _save_roc_curve_png(y_test, y_score, result.metrics["roc_auc"], output_dir / "roc_curve.png")

    return result.metrics


def load_test_split(processed_dir) -> tuple[np.ndarray, np.ndarray]:
    """
    Load X_test.npy and y_test.npy saved by the training pipeline.
    Raises FileNotFoundError with a clear message if either file is missing.
    """
    processed_dir = Path(processed_dir)
    x_path = processed_dir / "X_test.npy"
    y_path = processed_dir / "y_test.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Test split not found in {processed_dir}. "
            "Run scripts/run_training.py first to generate X_test.npy and y_test.npy."
        )
    return np.load(x_path), np.load(y_path)


def _save_confusion_matrix_png(cm: np.ndarray, path: Path) -> None:
    import matplotlib  # lazy — not a hard dependency at import time
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Uninfected", "Parasitized"])
    ax.set_yticklabels(["Uninfected", "Parasitized"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _save_roc_curve_png(
    y_true: np.ndarray, y_score: np.ndarray, auc: float, path: Path
) -> None:
    import matplotlib  # lazy
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#4f46e5", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
