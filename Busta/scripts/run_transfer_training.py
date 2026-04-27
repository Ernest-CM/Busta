import argparse
from pathlib import Path

from src.models.train import train_transfer_model
from src.utils.config import TRANSFER_BASE_MODEL

DATA_DIR = Path("data") / "raw" / "cell_images"
OUTPUT_DIR = Path("models_artifacts")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transfer learning malaria classifier.")
    parser.add_argument("--base", default=TRANSFER_BASE_MODEL,
                        help="Base architecture (mobilenetv2 / efficientnetb0)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Max epochs for stage 1 (frozen base)")
    parser.add_argument("--fine-tune-epochs", type=int, default=5, dest="fine_tune_epochs",
                        help="Max epochs for stage 2 (fine-tuning)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        f"Starting transfer learning — base: {args.base}, "
        f"stage-1 epochs: {args.epochs}, stage-2 epochs: {args.fine_tune_epochs}"
    )

    _, history = train_transfer_model(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        base=args.base,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
    )

    s1_len = args.epochs
    if len(history.get("val_accuracy", [])) < s1_len:
        s1_len = len(history.get("val_accuracy", []))

    val_acc = history.get("val_accuracy", [])
    val_auc = history.get("val_auc", [])

    s1_end = min(args.epochs, len(val_acc))
    print(f"\nStage 1 final val_accuracy : {val_acc[s1_end - 1]:.4f}" if s1_end else "\nStage 1: no history recorded")
    print(f"Stage 1 final val_auc      : {val_auc[s1_end - 1]:.4f}" if s1_end and val_auc else "")

    if len(val_acc) > s1_end:
        print(f"Stage 2 final val_accuracy : {val_acc[-1]:.4f}")
        print(f"Stage 2 final val_auc      : {val_auc[-1]:.4f}" if val_auc else "")

    print(f"\nModel saved to: {(OUTPUT_DIR / f'transfer_{args.base}.keras').resolve()}")


if __name__ == "__main__":
    main()
