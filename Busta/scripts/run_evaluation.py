from pathlib import Path

from src.evaluation.evaluate import load_test_split, run_evaluation
from src.models.predict import load_model_for_inference

MODEL_PATH = Path("models_artifacts") / "baseline_cnn.keras"
PROCESSED_DIR = Path("data") / "processed"
OUTPUT_DIR = Path("models_artifacts") / "eval"


if __name__ == "__main__":
    model = load_model_for_inference(MODEL_PATH)
    x_test, y_test = load_test_split(PROCESSED_DIR)
    metrics = run_evaluation(model=model, x_test=x_test, y_test=y_test, output_dir=OUTPUT_DIR)

    rows = [
        ("Accuracy",  metrics.get("accuracy")),
        ("Precision", metrics.get("precision")),
        ("Recall",    metrics.get("recall")),
        ("F1 Score",  metrics.get("f1")),
        ("ROC-AUC",   metrics.get("roc_auc")),
    ]
    print(f"\n{'Metric':<12}  {'Value':>8}")
    print("-" * 22)
    for name, val in rows:
        display = f"{val:.4f}" if val is not None else "N/A"
        print(f"{name:<12}  {display:>8}")
    print(f"\nArtifacts saved to: {OUTPUT_DIR.resolve()}")
