from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from torch.utils.data import Dataset

from .labels import apply_label_rules

REQUIRED_COLUMNS = [
    "id",
    "prompt",
    "consensus_combined_grade",
    "combined_category",
]


@dataclass(frozen=True)
class DataSummary:
    total_rows: int
    usable_rows: int


def load_and_validate_csv(data_csv: Path) -> pd.DataFrame:
    if not data_csv.exists():
        raise FileNotFoundError(f"CSV not found: {data_csv}")

    df = pd.read_csv(data_csv)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    if df["id"].isna().any():
        raise ValueError("Found rows with missing id.")

    if df["prompt"].isna().any():
        raise ValueError("Found rows with missing prompt.")

    return df


def find_image_for_id(images_dir: Path, sample_id: object) -> Path:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    sid = str(sample_id).strip()
    matches = sorted(images_dir.glob(f"{sid}.*"))

    if not matches:
        raise FileNotFoundError(f"No image found for id={sid} in {images_dir}")
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous images for id={sid}. Found {len(matches)} matches: "
            f"{[m.name for m in matches]}"
        )

    return matches[0]


def attach_image_paths(df: pd.DataFrame, images_dir: Path) -> pd.DataFrame:
    work = df.copy()
    paths: List[str] = []
    for sample_id in work["id"].tolist():
        image_path = find_image_for_id(images_dir=images_dir, sample_id=sample_id)
        paths.append(str(image_path))

    work["image_path"] = paths
    return work


def build_labeled_dataframe(data_csv: Path, images_dir: Path) -> pd.DataFrame:
    raw = load_and_validate_csv(data_csv)
    labeled = apply_label_rules(raw)
    labeled = attach_image_paths(labeled, images_dir)
    return labeled


class RawMultimodalDataset(Dataset):
    """Simple row-backed dataset for later training and caching modules."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        return {
            "id": row["id"],
            "prompt": row["prompt"],
            "image_path": row["image_path"],
            "binary_label": int(row["binary_label"]),
            "unsafe_subclass_label": int(row["unsafe_subclass_label"]),
            "composite_9way_label": int(row["composite_9way_label"]),
        }
