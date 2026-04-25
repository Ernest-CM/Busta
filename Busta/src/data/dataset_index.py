from pathlib import Path
from typing import Dict

from src.data.ingestion import discover_image_records
from src.utils.config import LABEL_MAP


def summarize_dataset(dataset_root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {name: 0 for name in LABEL_MAP}
    for record in discover_image_records(dataset_root=dataset_root):
        counts[record.label_name] += 1
    return counts

