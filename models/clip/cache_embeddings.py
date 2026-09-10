from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .clip_backbone import (
    encode_image_batch,
    encode_text_batch,
    load_clip_backbone,
    preprocess_images,
)
from .config import CacheConfig
from .dataset import attach_image_paths
from .utils import ensure_dir, utc_now_iso, write_config_snapshot, write_json

REQUIRED_SPLIT_COLUMNS = [
    "id",
    "prompt",
    "binary_label",
    "unsafe_subclass_label",
    "composite_9way_label",
]


def _safe_token(value: str) -> str:
    return value.replace("/", "-").replace("\\", "-").replace(":", "-").replace(" ", "_")


def _feature_fingerprint(df: pd.DataFrame) -> str:
    payload_cols = [
        "id",
        "prompt",
        "binary_label",
        "unsafe_subclass_label",
        "composite_9way_label",
    ]
    available = [c for c in payload_cols if c in df.columns]
    payload = df[available].to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_split_frame(split_csv: Path, images_dir: Path) -> pd.DataFrame:
    if not split_csv.exists():
        raise FileNotFoundError(f"Split file not found: {split_csv}")

    frame = pd.read_csv(split_csv)

    missing = [c for c in REQUIRED_SPLIT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Split CSV missing required columns {missing}: {split_csv}")

    if "image_path" not in frame.columns:
        frame = attach_image_paths(frame, images_dir)

    if frame["image_path"].isna().any():
        raise ValueError(f"Split contains missing image_path values: {split_csv}")

    return frame


def _load_images(image_paths: Sequence[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path_str in image_paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Image path missing: {path}")
        with Image.open(path) as img:
            if img.mode == "P" and isinstance(img.info.get("transparency"), bytes):
                converted = img.convert("RGBA").convert("RGB")
            else:
                converted = img.convert("RGB")
            images.append(converted)
    return images


def _encode_split(
    frame: pd.DataFrame,
    batch_size: int,
    backbone,
) -> tuple[torch.Tensor, torch.Tensor]:
    text_chunks: list[torch.Tensor] = []
    image_chunks: list[torch.Tensor] = []

    total = len(frame)
    for start in tqdm(range(0, total, batch_size), desc="Encoding", unit="batch"):
        batch = frame.iloc[start : start + batch_size]

        texts = batch["prompt"].tolist()
        image_paths = batch["image_path"].tolist()
        images = _load_images(image_paths)

        image_batch = preprocess_images(backbone, images)

        text_features = encode_text_batch(backbone, texts, normalize=True).cpu()
        image_features = encode_image_batch(backbone, image_batch, normalize=True).cpu()

        text_chunks.append(text_features)
        image_chunks.append(image_features)

    text_all = torch.cat(text_chunks, dim=0)
    image_all = torch.cat(image_chunks, dim=0)
    return text_all, image_all


def _save_split_cache(
    split_name: str,
    frame: pd.DataFrame,
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    split_out_dir: Path,
    split_csv: Path,
    model_name: str,
    pretrained: str,
    device: str,
    augmented: bool,
) -> None:
    ensure_dir(split_out_dir)

    torch.save(text_features, split_out_dir / "text_features.pt")
    torch.save(image_features, split_out_dir / "image_features.pt")

    frame[["id"]].to_csv(split_out_dir / "ids.csv", index=False)

    label_cols = [
        "id",
        "binary_label",
        "unsafe_subclass_label",
        "composite_9way_label",
    ]
    optional_cols = ["consensus_combined_grade", "combined_category", "reduced_unsafe_class"]
    for col in optional_cols:
        if col in frame.columns:
            label_cols.append(col)
    frame[label_cols].to_csv(split_out_dir / "labels_snapshot.csv", index=False)

    metadata = {
        "created_at_utc": utc_now_iso(),
        "split_name": split_name,
        "model_name": model_name,
        "pretrained": pretrained,
        "device": device,
        "rows": int(len(frame)),
        "text_feature_dim": int(text_features.shape[1]),
        "image_feature_dim": int(image_features.shape[1]),
        "normalized_features": True,
        "augmented": bool(augmented),
        "split_csv": str(split_csv),
        "split_fingerprint_sha256": _feature_fingerprint(frame),
    }
    write_json(split_out_dir / "cache_metadata.json", metadata)


def _run_dry_load(cfg: CacheConfig) -> None:
    backbone = load_clip_backbone(
        model_name=cfg.model_name,
        pretrained=cfg.pretrained,
        device=cfg.device,
    )
    trainable_params = sum(p.numel() for p in backbone.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in backbone.model.parameters())

    print(f"model_name={backbone.model_name}")
    print(f"pretrained={backbone.pretrained}")
    print(f"device={backbone.device}")
    print(f"total_params={total_params}")
    print(f"trainable_params={trainable_params}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache frozen OpenCLIP text/image embeddings.")
    parser.add_argument("--splits-dir", type=Path, default=CacheConfig().splits_dir)
    parser.add_argument("--out-dir", type=Path, default=CacheConfig().out_dir)
    parser.add_argument("--images-dir", type=Path, default=CacheConfig().images_dir)
    parser.add_argument("--model-name", type=str, default=CacheConfig().model_name)
    parser.add_argument("--pretrained", type=str, default=CacheConfig().pretrained)
    parser.add_argument("--batch-size", type=int, default=CacheConfig().batch_size)
    parser.add_argument("--device", type=str, default=CacheConfig().device)
    parser.add_argument(
        "--split-names",
        nargs="+",
        default=list(CacheConfig().split_names),
        help="Split names to process, e.g. train val test",
    )
    parser.add_argument("--max-samples", type=int, default=CacheConfig().max_samples)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run-backbone",
        action="store_true",
        help="Only load the configured OpenCLIP backbone and print summary.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"batch-size must be > 0, got {args.batch_size}")

    cfg = CacheConfig(
        splits_dir=args.splits_dir,
        out_dir=args.out_dir,
        images_dir=args.images_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        device=args.device,
        split_names=tuple(args.split_names),
        max_samples=args.max_samples,
        overwrite=bool(args.overwrite),
        augmented=False,
    )

    if args.dry_run_backbone:
        _run_dry_load(cfg)
        return

    cache_root = cfg.out_dir / f"{_safe_token(cfg.model_name)}__{_safe_token(cfg.pretrained)}"
    ensure_dir(cache_root)
    write_config_snapshot(cache_root / "cache_config.json", cfg)

    backbone = load_clip_backbone(
        model_name=cfg.model_name,
        pretrained=cfg.pretrained,
        device=cfg.device,
    )

    for split_name in cfg.split_names:
        split_csv = cfg.splits_dir / f"{split_name}.csv"
        split_out_dir = cache_root / split_name

        if (
            not cfg.overwrite
            and (split_out_dir / "text_features.pt").exists()
            and (split_out_dir / "image_features.pt").exists()
        ):
            print(f"Skipping split '{split_name}' because cache already exists at {split_out_dir}")
            continue

        frame = _load_split_frame(split_csv=split_csv, images_dir=cfg.images_dir)
        if cfg.max_samples is not None:
            frame = frame.head(cfg.max_samples).copy()

        if len(frame) == 0:
            raise ValueError(f"Split '{split_name}' has no rows after filtering.")

        print(f"\nEncoding split='{split_name}' rows={len(frame)}")
        text_features, image_features = _encode_split(
            frame=frame,
            batch_size=cfg.batch_size,
            backbone=backbone,
        )

        _save_split_cache(
            split_name=split_name,
            frame=frame,
            text_features=text_features,
            image_features=image_features,
            split_out_dir=split_out_dir,
            split_csv=split_csv,
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
            device=str(backbone.device),
            augmented=cfg.augmented,
        )
        print(f"Saved cache for split '{split_name}' to {split_out_dir}")


if __name__ == "__main__":
    main()
