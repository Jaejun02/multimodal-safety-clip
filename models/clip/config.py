from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from . import paths


@dataclass(frozen=True)
class SplitConfig:
    data_csv: Path = paths.DATA_DIR / "vlsu_mod.csv"
    images_dir: Path = paths.DATA_DIR / "vlsu_images"
    out_dir: Path = paths.SPLITS_DIR
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    min_class_count: int = 3
    sanity_check: bool = False
    sanity_labels_only: bool = False

    def as_dict(self) -> Dict[str, Any]:
        cfg = asdict(self)
        cfg["data_csv"] = str(self.data_csv)
        cfg["images_dir"] = str(self.images_dir)
        cfg["out_dir"] = str(self.out_dir)
        return cfg


@dataclass(frozen=True)
class CacheConfig:
    splits_dir: Path = paths.SPLITS_DIR
    out_dir: Path = paths.CACHES_DIR
    images_dir: Path = paths.DATA_DIR / "vlsu_images"
    model_name: str = "ViT-B-16"
    pretrained: str = "laion2b_s34b_b88k"
    batch_size: int = 64
    device: str | None = None
    split_names: tuple[str, ...] = ("train", "val", "test")
    max_samples: int | None = None
    overwrite: bool = False
    augmented: bool = False

    def as_dict(self) -> Dict[str, Any]:
        cfg = asdict(self)
        cfg["splits_dir"] = str(self.splits_dir)
        cfg["out_dir"] = str(self.out_dir)
        cfg["images_dir"] = str(self.images_dir)
        cfg["split_names"] = list(self.split_names)
        return cfg


@dataclass(frozen=True)
class TrainConfig:
    cache_dir: Path = paths.CACHES_DIR
    run_dir: Path = paths.RUNS_DIR / "dev_smoke"
    model_name: str = "ViT-B-16"
    pretrained: str = "laion2b_s34b_b88k"
    fusion: str = "interaction"
    head_type: str = "medium"
    output_mode: str = "hierarchical"
    proj_dim: int = 512
    batch_size: int = 32
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    class_weighting: str = "effective"
    beta: float = 0.999
    loss_type: str = "ce"
    focal_gamma: float = 2.0
    lambda_unsafe: float = 1.0
    seed: int = 42
    device: str | None = None
    num_workers: int = 0
    train_split: str = "train"
    val_split: str = "val"
    skip_val: bool = False
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    n_bins: int = 15
    early_stopping: bool = False
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 0.0
    log_batch_loss: bool = False
    log_batch_every: int = 10
    sanity_check: bool = False
    model_smoke_test: bool = False
    loss_smoke_test: bool = False
    overwrite_run: bool = False

    def as_dict(self) -> Dict[str, Any]:
        cfg = asdict(self)
        cfg["cache_dir"] = str(self.cache_dir)
        cfg["run_dir"] = str(self.run_dir)
        return cfg


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.8f}"
        )
    for name, ratio in (
        ("train_ratio", train_ratio),
        ("val_ratio", val_ratio),
        ("test_ratio", test_ratio),
    ):
        if ratio <= 0:
            raise ValueError(f"{name} must be > 0, got {ratio}")
