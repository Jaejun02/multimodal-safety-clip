from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .fusion_heads import FusionClassifier
from .labels import IGNORE_INDEX
from .metrics import (
    compute_binary_metrics,
    compute_flat9_metrics,
    compute_multiclass_metrics_from_predictions,
    compute_reliability_metrics,
    compute_unsafe_subclass_metrics,
)
from .paths import PROJECT_ROOT
from .torch_runtime import cuda_available_safely
from .utils import ensure_dir, utc_now_iso, write_json

REQUIRED_LABEL_COLUMNS = [
    "binary_label",
    "unsafe_subclass_label",
    "composite_9way_label",
]


def _safe_token(value: str) -> str:
    return value.replace("/", "-").replace("\\", "-").replace(":", "-").replace(" ", "_")


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if cuda_available_safely() else "cpu")


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing run config: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "config" not in payload or not isinstance(payload["config"], dict):
        raise ValueError(f"Invalid run config format in {cfg_path}")
    return payload["config"]


def _load_split_cache(
    cache_root: Path,
    split_name: str,
    max_samples: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    split_dir = cache_root / split_name
    text_path = split_dir / "text_features.pt"
    image_path = split_dir / "image_features.pt"
    labels_path = split_dir / "labels_snapshot.csv"

    if not text_path.exists() or not image_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing split artifacts for split='{split_name}' under {split_dir}. "
            "Need text_features.pt, image_features.pt, labels_snapshot.csv"
        )

    text_features = torch.load(text_path, map_location="cpu").float()
    image_features = torch.load(image_path, map_location="cpu").float()
    labels_df = pd.read_csv(labels_path)

    missing = [c for c in REQUIRED_LABEL_COLUMNS if c not in labels_df.columns]
    if missing:
        raise ValueError(f"labels_snapshot.csv missing required columns {missing}: {labels_path}")

    n = len(labels_df)
    if text_features.shape[0] != n or image_features.shape[0] != n:
        raise ValueError(
            f"Feature/label length mismatch for split='{split_name}': "
            f"text={text_features.shape[0]}, image={image_features.shape[0]}, labels={n}"
        )

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max-samples must be > 0 when provided, got {max_samples}")
        n = min(n, max_samples)
        text_features = text_features[:n]
        image_features = image_features[:n]
        labels_df = labels_df.iloc[:n].reset_index(drop=True)

    binary_targets = torch.tensor(labels_df["binary_label"].to_numpy(), dtype=torch.long)
    unsafe_targets = torch.tensor(labels_df["unsafe_subclass_label"].to_numpy(), dtype=torch.long)
    flat_targets = torch.tensor(labels_df["composite_9way_label"].to_numpy(), dtype=torch.long)

    return text_features, image_features, binary_targets, unsafe_targets, flat_targets, labels_df


def _resolve_cache_root(cache_dir: Path, model_name: str, pretrained: str) -> Path:
    direct_train = cache_dir / "train"
    direct_text = direct_train / "text_features.pt"
    direct_image = direct_train / "image_features.pt"
    if direct_text.exists() and direct_image.exists():
        return cache_dir

    model_cache = cache_dir / f"{_safe_token(model_name)}__{_safe_token(pretrained)}"
    model_text = model_cache / "train" / "text_features.pt"
    model_image = model_cache / "train" / "image_features.pt"
    if model_text.exists() and model_image.exists():
        return model_cache

    raise FileNotFoundError(
        "Could not resolve cache root. Expected either '\n"
        f"  - {direct_train} (split folders directly under --cache-dir) or '\n"
        f"  - {model_cache} (model/pretrained nested cache root)."
    )


def _make_loader(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    binary_targets: torch.Tensor,
    unsafe_targets: torch.Tensor,
    flat_targets: torch.Tensor,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        text_features,
        image_features,
        binary_targets,
        unsafe_targets,
        flat_targets,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cuda_available_safely(),
    )


def _safe_reliability(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int,
    ignore_index: int | None = None,
) -> dict[str, Any]:
    try:
        rel = compute_reliability_metrics(
            logits=logits,
            targets=targets,
            n_bins=n_bins,
            ignore_index=ignore_index,
        )
    except ValueError:
        return {
            "ece": None,
            "brier": None,
            "bin_edges": [],
            "bin_counts": [],
            "bin_accuracy": [],
            "bin_confidence": [],
        }

    return {
        "ece": rel.ece,
        "brier": rel.brier,
        "bin_edges": rel.bin_edges,
        "bin_counts": rel.bin_counts,
        "bin_accuracy": rel.bin_accuracy,
        "bin_confidence": rel.bin_confidence,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained fusion head run.")
    parser.add_argument("--run-dir", type=Path, required=False, default=Path("output/clip/runs/dev_smoke"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-bins", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--metrics-smoke-test", action="store_true")
    return parser


def _run_metrics_smoke_test() -> None:
    gen = torch.Generator().manual_seed(42)

    binary_logits = torch.randn(32, 2, generator=gen)
    binary_targets = torch.randint(0, 2, (32,), generator=gen)
    binary_metrics = compute_binary_metrics(binary_logits=binary_logits, binary_targets=binary_targets)

    flat_logits = torch.randn(32, 9, generator=gen)
    flat_targets = torch.randint(0, 9, (32,), generator=gen)
    flat_metrics = compute_flat9_metrics(flat_logits=flat_logits, flat_targets=flat_targets)

    unsafe_logits = torch.randn(32, 8, generator=gen)
    unsafe_targets = torch.randint(0, 8, (32,), generator=gen)
    unsafe_targets[:8] = torch.arange(8, dtype=torch.long)
    unsafe_metrics = compute_unsafe_subclass_metrics(
        unsafe_logits=unsafe_logits,
        unsafe_targets=unsafe_targets,
        ignore_index=IGNORE_INDEX,
    )

    print("metrics_smoke_test=ok")
    print(f"binary_f1={binary_metrics.f1:.6f}")
    print(f"flat_macro_f1={flat_metrics.macro_f1:.6f}")
    print(f"unsafe_macro_f1={unsafe_metrics.macro_f1:.6f}")


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.metrics_smoke_test:
        _run_metrics_smoke_test()
        return

    if args.batch_size <= 0:
        raise ValueError(f"batch-size must be > 0, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"num-workers must be >= 0, got {args.num_workers}")

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run-dir does not exist: {run_dir}")

    cfg = _load_run_config(run_dir)

    cache_dir = args.cache_dir
    if cache_dir is None:
        if "cache_dir" not in cfg:
            raise ValueError("run config does not contain cache_dir; pass --cache-dir explicitly")
        cache_dir = _resolve_path(str(cfg["cache_dir"]), PROJECT_ROOT)
    else:
        cache_dir = cache_dir.resolve()

    split_name = args.split
    ckpt_path = run_dir / f"{args.checkpoint}.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    ckpt_cfg = checkpoint.get("config", cfg)

    cache_root = _resolve_cache_root(
        cache_dir=cache_dir,
        model_name=str(ckpt_cfg.get("model_name", cfg.get("model_name", ""))),
        pretrained=str(ckpt_cfg.get("pretrained", cfg.get("pretrained", ""))),
    )

    text_feat, image_feat, binary_targets, unsafe_targets, flat_targets, labels_df = _load_split_cache(
        cache_root=cache_root,
        split_name=split_name,
        max_samples=args.max_samples,
    )

    device = _resolve_device(args.device)

    output_mode = str(ckpt_cfg["output_mode"])
    fusion = str(ckpt_cfg["fusion"])
    head_type = str(ckpt_cfg["head_type"])
    proj_dim = int(ckpt_cfg.get("proj_dim", 512))

    model = FusionClassifier(
        text_dim=int(text_feat.shape[1]),
        image_dim=int(image_feat.shape[1]),
        fusion_type=fusion,
        head_type=head_type,
        output_mode=output_mode,
        proj_dim=proj_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    loader = _make_loader(
        text_features=text_feat,
        image_features=image_feat,
        binary_targets=binary_targets,
        unsafe_targets=unsafe_targets,
        flat_targets=flat_targets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    binary_logits_chunks: list[torch.Tensor] = []
    unsafe_logits_chunks: list[torch.Tensor] = []
    flat_logits_chunks: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in loader:
            bt, it, _, _, _ = batch
            bt = bt.to(device, non_blocking=True)
            it = it.to(device, non_blocking=True)
            outputs = model(text_feat=bt, image_feat=it)

            if output_mode == "flat":
                flat_logits_chunks.append(outputs["flat_logits"].detach().cpu())
            else:
                binary_logits_chunks.append(outputs["binary_logits"].detach().cpu())
                unsafe_logits_chunks.append(outputs["unsafe_logits"].detach().cpu())

    metrics_payload: dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(ckpt_path),
        "split": split_name,
        "cache_dir": str(cache_dir),
        "cache_root": str(cache_root),
        "num_samples": int(len(labels_df)),
        "output_mode": output_mode,
        "fusion": fusion,
        "head_type": head_type,
    }

    pred_df = labels_df.copy()

    eval_n_bins = int(args.n_bins) if args.n_bins is not None else int(cfg.get("n_bins", 15))

    if output_mode == "flat":
        flat_logits_all = torch.cat(flat_logits_chunks, dim=0)
        flat_probs = torch.softmax(flat_logits_all, dim=1)
        flat_pred = flat_probs.argmax(dim=1)

        flat_metrics = compute_flat9_metrics(flat_logits_all, flat_targets)
        rel = _safe_reliability(flat_logits_all, flat_targets, n_bins=eval_n_bins)

        metrics_payload["flat9_metrics"] = {
            "accuracy": flat_metrics.accuracy,
            "balanced_accuracy": flat_metrics.balanced_accuracy,
            "macro_f1": flat_metrics.macro_f1,
            "weighted_f1": flat_metrics.weighted_f1,
            "per_class_precision": flat_metrics.per_class_precision,
            "per_class_recall": flat_metrics.per_class_recall,
            "per_class_f1": flat_metrics.per_class_f1,
            "per_class_support": flat_metrics.per_class_support,
            "confusion_matrix": flat_metrics.confusion_matrix,
        }
        metrics_payload["flat9_reliability"] = rel

        pred_df["flat9_pred"] = flat_pred.numpy()
        pred_df["flat9_confidence"] = flat_probs.max(dim=1).values.numpy()

    else:
        binary_logits_all = torch.cat(binary_logits_chunks, dim=0)
        unsafe_logits_all = torch.cat(unsafe_logits_chunks, dim=0)

        binary_probs = torch.softmax(binary_logits_all, dim=1)
        unsafe_probs = torch.softmax(unsafe_logits_all, dim=1)

        binary_pred = (binary_probs[:, 1] >= 0.5).long()
        unsafe_pred = unsafe_probs.argmax(dim=1)

        binary_metrics = compute_binary_metrics(binary_logits_all, binary_targets)
        unsafe_metrics = compute_unsafe_subclass_metrics(
            unsafe_logits=unsafe_logits_all,
            unsafe_targets=unsafe_targets,
            ignore_index=IGNORE_INDEX,
        )
        binary_rel = _safe_reliability(binary_logits_all, binary_targets, n_bins=eval_n_bins)
        unsafe_rel = _safe_reliability(
            unsafe_logits_all,
            unsafe_targets,
            n_bins=eval_n_bins,
            ignore_index=IGNORE_INDEX,
        )

        derived_flat_pred = torch.where(binary_pred == 0, torch.zeros_like(unsafe_pred), unsafe_pred + 1)
        derived_flat_metrics = compute_multiclass_metrics_from_predictions(
            predictions=derived_flat_pred,
            targets=flat_targets,
            num_classes=9,
            ignore_index=None,
        )

        metrics_payload["binary_metrics"] = {
            "accuracy": binary_metrics.accuracy,
            "precision": binary_metrics.precision,
            "recall": binary_metrics.recall,
            "f1": binary_metrics.f1,
            "auroc": binary_metrics.auroc,
            "pr_auc": binary_metrics.pr_auc,
            "brier": binary_metrics.brier,
        }
        metrics_payload["unsafe_metrics"] = {
            "accuracy": unsafe_metrics.accuracy,
            "balanced_accuracy": unsafe_metrics.balanced_accuracy,
            "macro_f1": unsafe_metrics.macro_f1,
            "weighted_f1": unsafe_metrics.weighted_f1,
            "per_class_precision": unsafe_metrics.per_class_precision,
            "per_class_recall": unsafe_metrics.per_class_recall,
            "per_class_f1": unsafe_metrics.per_class_f1,
            "per_class_support": unsafe_metrics.per_class_support,
            "confusion_matrix": unsafe_metrics.confusion_matrix,
        }
        metrics_payload["binary_reliability"] = binary_rel
        metrics_payload["unsafe_reliability"] = unsafe_rel
        metrics_payload["derived_flat9_metrics_from_hierarchical"] = {
            "accuracy": derived_flat_metrics.accuracy,
            "balanced_accuracy": derived_flat_metrics.balanced_accuracy,
            "macro_f1": derived_flat_metrics.macro_f1,
            "weighted_f1": derived_flat_metrics.weighted_f1,
            "per_class_precision": derived_flat_metrics.per_class_precision,
            "per_class_recall": derived_flat_metrics.per_class_recall,
            "per_class_f1": derived_flat_metrics.per_class_f1,
            "per_class_support": derived_flat_metrics.per_class_support,
            "confusion_matrix": derived_flat_metrics.confusion_matrix,
        }

        pred_df["binary_pred"] = binary_pred.numpy()
        pred_df["binary_prob_unsafe"] = binary_probs[:, 1].numpy()
        pred_df["unsafe_pred"] = unsafe_pred.numpy()
        pred_df["unsafe_confidence"] = unsafe_probs.max(dim=1).values.numpy()
        pred_df["derived_flat9_pred"] = derived_flat_pred.numpy()

    out_dir = args.out_dir.resolve() if args.out_dir is not None else run_dir / "evaluation"
    ensure_dir(out_dir)

    metrics_path = out_dir / f"metrics_{split_name}_{args.checkpoint}.json"
    preds_path = out_dir / f"predictions_{split_name}_{args.checkpoint}.csv"

    write_json(metrics_path, metrics_payload)
    pred_df.to_csv(preds_path, index=False)

    print("evaluation_complete=ok")
    print(f"run_dir={run_dir}")
    print(f"split={split_name}")
    print(f"checkpoint={args.checkpoint}")
    print(f"metrics_json={metrics_path}")
    print(f"predictions_csv={preds_path}")


if __name__ == "__main__":
    main()
