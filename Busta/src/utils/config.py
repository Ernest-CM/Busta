from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple

LABEL_MAP: Dict[str, int] = {
    "Parasitized": 1,
    "Uninfected": 0,
}

VALID_IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: Tuple[int, int] = (224, 224)
    color_mode: Literal["rgb"] = "rgb"
    normalize_mode: Literal["zero_one"] = "zero_one"
    denoise_mode: Literal["median", "gaussian", "none"] = "median"
    denoise_kernel_size: int = 3
    on_error: Literal["raise", "skip"] = "raise"


@dataclass(frozen=True)
class DataConfig:
    dataset_root: Path
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass(frozen=True)
class TrainConfig:
    model_dir: Path = Path("models_artifacts")
    model_name: str = "baseline_cnn.keras"
    history_name: str = "training_history.json"
    evaluation_name: str = "evaluation_metrics.json"
    confusion_matrix_name: str = "confusion_matrix.csv"
    seed: int = 42
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 1e-3
    val_size: float = 0.2


