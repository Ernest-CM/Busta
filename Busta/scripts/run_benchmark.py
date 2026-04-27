import argparse
import json
import sys
from pathlib import Path

from src.evaluation.evaluate import load_test_split, run_evaluation
from src.models.predict import load_model_for_inference

_DEFAULT_MODELS_DIR = Path("models_artifacts")
_DEFAULT_PROCESSED_DIR = Path("data") / "processed"
_DEFAULT_OUTPUT_DIR = Path("models_artifacts") / "benchmark"


def _fixed_width_table(headers: list, rows: list) -> str:
    """Render a plain fixed-width table — no external dependencies required."""
    col_widths = [
        max(len(str(h)), *(len(str(r[i])) for r in rows))
        for i, h in enumerate(headers)
    ]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def _row(cells):
        return "| " + " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    lines = [sep, _row(headers), sep, *(_row(r) for r in rows), sep]
    return "\n".join(lines)


def run_benchmark(
    processed_dir,
    output_dir,
    models_dir=None,
) -> list:
    """
    Evaluate every .keras model found in models_dir against the saved test split.

    Returns a list of result dicts (one per model).
    Raises FileNotFoundError if no .keras files are found or the test split is missing.
    """
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    models_dir = Path(models_dir) if models_dir is not None else _DEFAULT_MODELS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    keras_files = sorted(models_dir.glob("*.keras"))
    if not keras_files:
        raise FileNotFoundError(f"No .keras files found in {models_dir}. Train a model first.")

    x_test, y_test = load_test_split(processed_dir)

    results = []
    for keras_path in keras_files:
        print(f"Evaluating {keras_path.name} …", flush=True)
        model = load_model_for_inference(keras_path)
        stem = keras_path.stem
        metrics = run_evaluation(
            model=model,
            x_test=x_test,
            y_test=y_test,
            output_dir=output_dir / stem,
        )
        results.append({
            "model_name": stem,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "auc": metrics.get("roc_auc"),
        })

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]
    rows = [
        [
            r["model_name"],
            f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "N/A",
            f"{r['precision']:.4f}" if r["precision"] is not None else "N/A",
            f"{r['recall']:.4f}" if r["recall"] is not None else "N/A",
            f"{r['f1']:.4f}" if r["f1"] is not None else "N/A",
            f"{r['auc']:.4f}" if r["auc"] is not None else "N/A",
        ]
        for r in results
    ]
    print("\n" + _fixed_width_table(headers, rows))

    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nComparison saved to: {comparison_path.resolve()}")

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark all trained models against the held-out test split.")
    parser.add_argument(
        "--processed-dir", default=str(_DEFAULT_PROCESSED_DIR),
        help="Directory containing X_test.npy / y_test.npy (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir", default=str(_DEFAULT_OUTPUT_DIR),
        help="Where to write per-model evaluation artefacts (default: models_artifacts/benchmark/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        run_benchmark(args.processed_dir, args.output_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
