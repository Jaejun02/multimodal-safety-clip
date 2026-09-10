from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SplitConfig, validate_split_ratios
from .dataset import attach_image_paths, load_and_validate_csv
from .labels import REDUCED_CLASSES, SAFE_LABEL, apply_label_rules, class_counts, label_maps_payload
from .seed import set_global_seed
from .utils import ensure_dir, write_config_snapshot, write_json


def _print_counts(df: pd.DataFrame) -> None:
    binary = class_counts(df["binary_label"])
    composite = class_counts(df["composite_9way_label"])

    unsafe_only = df[df["binary_label"] != SAFE_LABEL]
    unsafe_subclass = class_counts(unsafe_only["unsafe_subclass_label"])

    print("\nBinary counts (0=safe, 1=unsafe):")
    print(binary)

    print("\nComposite 9-way counts (0=safe, 1..8=unsafe classes):")
    print(composite)

    print("\nUnsafe subclass counts (0..7):")
    print(unsafe_subclass)

    print("\nUnsafe subclass names:")
    for idx, name in enumerate(REDUCED_CLASSES):
        print(f"  {idx}: {name}")


def _validate_min_class_counts(df: pd.DataFrame, min_class_count: int) -> None:
    counts = class_counts(df["composite_9way_label"])
    bad = {k: v for k, v in counts.items() if v < min_class_count}
    if bad:
        raise ValueError(
            "Insufficient samples for one or more composite classes. "
            f"min_class_count={min_class_count}, offending_counts={bad}"
        )


def _split_frame(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strat = df["composite_9way_label"]

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=strat,
    )

    relative_test = test_ratio / (val_ratio + test_ratio)
    temp_strat = temp_df["composite_9way_label"]
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_strat,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _split_count_payload(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dict[str, Dict[str, int]]:
    return {
        "train": class_counts(train_df["composite_9way_label"]),
        "val": class_counts(val_df["composite_9way_label"]),
        "test": class_counts(test_df["composite_9way_label"]),
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create stratified train/val/test data splits.")
    p.add_argument("--data-csv", type=Path, default=SplitConfig().data_csv)
    p.add_argument("--images-dir", type=Path, default=SplitConfig().images_dir)
    p.add_argument("--out-dir", type=Path, default=SplitConfig().out_dir)
    p.add_argument("--seed", type=int, default=SplitConfig().seed)
    p.add_argument("--train-ratio", type=float, default=SplitConfig().train_ratio)
    p.add_argument("--val-ratio", type=float, default=SplitConfig().val_ratio)
    p.add_argument("--test-ratio", type=float, default=SplitConfig().test_ratio)
    p.add_argument("--min-class-count", type=int, default=SplitConfig().min_class_count)
    p.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run full sanity checks, including image path resolution, and exit.",
    )
    p.add_argument(
        "--sanity-labels-only",
        action="store_true",
        help="Run label sanity checks only and exit.",
    )
    return p


def main() -> None:
    args = _parser().parse_args()

    validate_split_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    set_global_seed(args.seed)

    cfg = SplitConfig(
        data_csv=args.data_csv,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_class_count=args.min_class_count,
        sanity_check=bool(args.sanity_check),
        sanity_labels_only=bool(args.sanity_labels_only),
    )

    raw = load_and_validate_csv(cfg.data_csv)
    labeled = apply_label_rules(raw)

    print(f"Loaded rows: {len(raw)}")
    dropped_unsafe = int(labeled["dropped_unsafe_rows"].iloc[0]) if len(labeled) > 0 else 0
    print(f"Dropped unsafe rows with missing combined_category: {dropped_unsafe}")
    print(f"Usable rows after filtering borderline: {len(labeled)}")
    _print_counts(labeled)

    _validate_min_class_counts(labeled, cfg.min_class_count)

    if cfg.sanity_labels_only:
        print("\nLabel sanity checks passed.")
        return

    labeled = attach_image_paths(labeled, cfg.images_dir)

    if cfg.sanity_check:
        missing = labeled[~labeled["image_path"].map(lambda p: Path(p).exists())]
        if len(missing) > 0:
            raise FileNotFoundError(
                f"Found {len(missing)} rows with missing images after resolution."
            )
        print("\nFull sanity checks passed (labels + image paths).")
        return

    train_df, val_df, test_df = _split_frame(
        labeled,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )

    post_counts = _split_count_payload(train_df, val_df, test_df)

    ensure_dir(cfg.out_dir)
    train_df.to_csv(cfg.out_dir / "train.csv", index=False)
    val_df.to_csv(cfg.out_dir / "val.csv", index=False)
    test_df.to_csv(cfg.out_dir / "test.csv", index=False)

    write_json(cfg.out_dir / "label_map.json", label_maps_payload())
    write_json(
        cfg.out_dir / "class_counts.json",
        {
            "binary": class_counts(labeled["binary_label"]),
            "composite_9way": class_counts(labeled["composite_9way_label"]),
            "unsafe_subclass": class_counts(
                labeled[labeled["binary_label"] == 1]["unsafe_subclass_label"]
            ),
        },
    )
    write_json(cfg.out_dir / "split_counts.json", post_counts)
    write_config_snapshot(cfg.out_dir / "split_config.json", cfg)

    print("\nSaved split files:")
    print(f"  - {cfg.out_dir / 'train.csv'}")
    print(f"  - {cfg.out_dir / 'val.csv'}")
    print(f"  - {cfg.out_dir / 'test.csv'}")
    print("\nSaved metadata:")
    print(f"  - {cfg.out_dir / 'label_map.json'}")
    print(f"  - {cfg.out_dir / 'class_counts.json'}")
    print(f"  - {cfg.out_dir / 'split_counts.json'}")
    print(f"  - {cfg.out_dir / 'split_config.json'}")


if __name__ == "__main__":
    main()
