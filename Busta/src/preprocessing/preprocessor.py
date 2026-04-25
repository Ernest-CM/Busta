from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from src.data.ingestion import ImageRecord, discover_image_records, split_records
from src.utils.config import LABEL_MAP, DataConfig, PreprocessConfig


class Preprocessor:
    """
    Preprocess thin blood smear images for CNN ingestion.

    Output contract:
    - Single image: numpy array of shape (H, W, 3), dtype float32, RGB color order.
    - Batch output: numpy array of shape (N, H, W, 3), dtype float32, RGB.
    - Pixel value range after normalization: [0.0, 1.0].
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def preprocess_image(self, image_path: Path | str) -> np.ndarray:
        image = self._read_image(Path(image_path))
        return self.preprocess_array(image)

    def preprocess_array(self, image_rgb: np.ndarray) -> np.ndarray:
        image = image_rgb
        image = self._resize(image)
        image = self._denoise(image)
        image = self._normalize(image)
        return image

    def preprocess_records(
        self,
        records: Iterable[ImageRecord],
    ) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
        processed_images: List[np.ndarray] = []
        labels: List[int] = []
        skipped_paths: List[Path] = []

        for record in records:
            try:
                processed = self.preprocess_image(record.path)
                processed_images.append(processed)
                labels.append(record.label)
            except Exception:
                if self.config.on_error == "skip":
                    skipped_paths.append(record.path)
                else:
                    raise

        if not processed_images:
            return (
                np.empty((0, *self.config.image_size, 3), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                skipped_paths,
            )

        x = np.stack(processed_images).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)
        return x, y, skipped_paths

    def load_dataset(self, data_config: DataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        records = discover_image_records(
            dataset_root=data_config.dataset_root,
            label_map=LABEL_MAP,
        )
        train_records, test_records = split_records(
            records=records,
            test_size=data_config.test_size,
            random_state=data_config.random_state,
            stratify=data_config.stratify,
        )
        x_train, y_train, _ = self.preprocess_records(train_records)
        x_test, y_test, _ = self.preprocess_records(test_records)
        return x_train, y_train, x_test, y_test

    def _read_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image file: {image_path}")
        # OpenCV loads BGR by default; convert to RGB for model consistency.
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        # Uniform size is required so all samples match the fixed CNN input tensor shape.
        width, height = self.config.image_size[1], self.config.image_size[0]
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        # Mild denoising helps suppress microscopy artifacts (stain speckles/light noise).
        kernel_size = self.config.denoise_kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1

        if self.config.denoise_mode == "median":
            return cv2.medianBlur(image, kernel_size)
        if self.config.denoise_mode == "gaussian":
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        if self.config.denoise_mode == "none":
            return image
        raise ValueError(f"Unsupported denoise mode: {self.config.denoise_mode}")

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        # Scaling to [0,1] stabilizes gradients and improves CNN convergence.
        if self.config.normalize_mode != "zero_one":
            raise ValueError(f"Unsupported normalization mode: {self.config.normalize_mode}")
        return image.astype(np.float32) / 255.0


def run_preprocessing_pipeline(
    dataset_root: Path | str,
    image_size: Sequence[int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preprocess_cfg = PreprocessConfig(image_size=(int(image_size[0]), int(image_size[1])))
    data_cfg = DataConfig(dataset_root=Path(dataset_root))
    return Preprocessor(config=preprocess_cfg).load_dataset(data_cfg)

