from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from sklearn.model_selection import train_test_split

from src.utils.config import LABEL_MAP, VALID_IMAGE_EXTENSIONS


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label_name: str
    label: int


def discover_image_records(
    dataset_root: Path,
    label_map: Dict[str, int] = LABEL_MAP,
    valid_extensions: Sequence[str] = VALID_IMAGE_EXTENSIONS,
) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for label_name, label in label_map.items():
        class_dir = dataset_root / label_name
        if not class_dir.exists():
            continue
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in valid_extensions:
                records.append(ImageRecord(path=path, label_name=label_name, label=label))
    return sorted(records, key=lambda x: str(x.path))


def split_records(
    records: Iterable[ImageRecord],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """
    Split records at file-level.

    Note:
    This reduces class imbalance risk via stratification, but patient-level leakage can still
    happen if multiple cells from the same patient are present across splits.
    """
    records_list = list(records)
    if not records_list:
        return [], []

    labels = [r.label for r in records_list]
    stratify_labels = labels if stratify else None
    train_records, test_records = train_test_split(
        records_list,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    return list(train_records), list(test_records)

