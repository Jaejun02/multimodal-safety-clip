from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig
from .fusion_heads import FusionClassifier
from .labels import IGNORE_INDEX
from .losses import (
    build_binary_class_weights,
    build_flat9_class_weights,
    build_unsafe_class_weights,
    compute_flat_loss,
    compute_hierarchical_loss,
)
from .metrics import (
    compute_binary_metrics,
    compute_flat9_metrics,
    compute_multiclass_metrics_from_predictions,
    compute_reliability_metrics,
    compute_unsafe_subclass_metrics,
    final_metric_policy,
    search_metric_policy,
)
from .seed import set_global_seed
from .torch_runtime import cuda_available_safely
from .utils import ensure_dir, utc_now_iso, write_config_snapshot, write_json


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


def _load_split_cache(
    cache_root: Path,
    split_name: str,
    max_samples: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    split_dir = cache_root / split_name
    text_path = split_dir / "text_features.pt"
    image_path = split_dir / "image_features.pt"
    labels_path = split_dir / "labels_snapshot.csv"

    if not text_path.exists() or not image_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing cache artifacts for split='{split_name}' under {split_dir}. "
            "Need text_features.pt, image_features.pt, labels_snapshot.csv"
        )

    text_features = torch.load(text_path, map_location="cpu")
    image_features = torch.load(image_path, map_location="cpu")
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
            raise ValueError(f"max_samples must be > 0 when provided, got {max_samples}")
        n = min(n, max_samples)
        text_features = text_features[:n]
        image_features = image_features[:n]
        labels_df = labels_df.iloc[:n].reset_index(drop=True)

    binary_targets = torch.tensor(labels_df["binary_label"].to_numpy(), dtype=torch.long)
    unsafe_targets = torch.tensor(labels_df["unsafe_subclass_label"].to_numpy(), dtype=torch.long)
    flat_targets = torch.tensor(labels_df["composite_9way_label"].to_numpy(), dtype=torch.long)

    return text_features.float(), image_features.float(), binary_targets, unsafe_targets, flat_targets


def _make_loader(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    binary_targets: torch.Tensor,
    unsafe_targets: torch.Tensor,
    flat_targets: torch.Tensor,
    batch_size: int,
    shuffle: bool,
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
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=cuda_available_safely(),
    )


def _dictify(obj: Any) -> dict[str, Any]:
    return asdict(obj)


def _monitor_name_for_output_mode(output_mode: str) -> str:
    if output_mode == "flat":
        return "flat_macro_f1"
    return "derived_flat_macro_f1"


def _monitor_value_from_metrics(metrics: dict[str, Any], output_mode: str) -> float:
    if output_mode == "flat":
        return float(metrics["flat_macro_f1"])
    if "derived_flat_macro_f1" in metrics:
        return float(metrics["derived_flat_macro_f1"])
    return float(metrics["unsafe_macro_f1"])


def _run_model_smoke_test(cfg: TrainConfig) -> None:
    batch_size = max(cfg.batch_size, 2)
    text_dim = 512
    image_dim = 512

    model = FusionClassifier(
        text_dim=text_dim,
        image_dim=image_dim,
        fusion_type=cfg.fusion,
        head_type=cfg.head_type,
        output_mode=cfg.output_mode,
        proj_dim=cfg.proj_dim,
    )

    text_feat = torch.randn(batch_size, text_dim)
    image_feat = torch.randn(batch_size, image_dim)
    outputs = model(text_feat=text_feat, image_feat=image_feat)

    print("model_smoke_test=ok")
    print(f"fusion={cfg.fusion}")
    print(f"head_type={cfg.head_type}")
    print(f"output_mode={cfg.output_mode}")
    for key, value in outputs.items():
        print(f"{key}_shape={tuple(value.shape)}")


def _run_loss_smoke_test(cfg: TrainConfig) -> None:
    batch_size = max(cfg.batch_size, 16)

    if cfg.output_mode == "flat":
        flat_logits = torch.randn(batch_size, 9)
        flat_targets = torch.randint(0, 9, (batch_size,), dtype=torch.long)
        flat_weights = build_flat9_class_weights(
            flat_targets.tolist(),
            strategy=cfg.class_weighting,
            beta=cfg.beta,
            device=flat_logits.device,
        )
        flat_loss = compute_flat_loss(
            flat_logits,
            flat_targets,
            class_weights=flat_weights,
            loss_type=cfg.loss_type,
            focal_gamma=cfg.focal_gamma,
        )
        print("loss_smoke_test=ok")
        print(f"output_mode=flat")
        print(f"flat_loss={float(flat_loss):.6f}")
        return

    binary_logits = torch.randn(batch_size, 2)
    unsafe_logits = torch.randn(batch_size, 8)

    binary_targets = torch.randint(0, 2, (batch_size,), dtype=torch.long)
    unsafe_targets = torch.randint(0, 8, (batch_size,), dtype=torch.long)

    if batch_size >= 8:
        binary_targets[:8] = 1
        unsafe_targets[:8] = torch.arange(8, dtype=torch.long)

    safe_mask = binary_targets == 0
    unsafe_targets[safe_mask] = IGNORE_INDEX

    binary_weights = build_binary_class_weights(
        binary_targets.tolist(),
        strategy=cfg.class_weighting,
        beta=cfg.beta,
        device=binary_logits.device,
    )
    unsafe_weights = build_unsafe_class_weights(
        unsafe_targets.tolist(),
        strategy=cfg.class_weighting,
        beta=cfg.beta,
        device=unsafe_logits.device,
        ignore_index=IGNORE_INDEX,
    )
    out = compute_hierarchical_loss(
        binary_logits=binary_logits,
        unsafe_logits=unsafe_logits,
        binary_targets=binary_targets,
        unsafe_targets=unsafe_targets,
        lambda_unsafe=cfg.lambda_unsafe,
        binary_class_weights=binary_weights,
        unsafe_class_weights=unsafe_weights,
        ignore_index=IGNORE_INDEX,
        loss_type=cfg.loss_type,
        focal_gamma=cfg.focal_gamma,
    )

    print("loss_smoke_test=ok")
    print(f"output_mode=hierarchical")
    print(f"binary_loss={float(out.binary_loss):.6f}")
    print(f"unsafe_loss={float(out.unsafe_loss):.6f}")
    print(f"total_loss={float(out.total_loss):.6f}")


def _epoch_pass(
    model: FusionClassifier,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
    epoch: int,
    output_mode: str,
    optimizer: torch.optim.Optimizer | None,
    lambda_unsafe: float,
    loss_type: str,
    focal_gamma: float,
    flat_class_weights: torch.Tensor | None,
    binary_class_weights: torch.Tensor | None,
    unsafe_class_weights: torch.Tensor | None,
    n_bins: int,
    include_uncalibrated_reliability: bool,
    log_batch_loss: bool,
    log_batch_every: int,
    batch_log_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)

    sum_total_loss = 0.0
    sum_binary_loss = 0.0
    sum_unsafe_loss = 0.0
    sum_flat_loss = 0.0
    seen = 0

    flat_logits_chunks: list[torch.Tensor] = []
    flat_targets_chunks: list[torch.Tensor] = []
    binary_logits_chunks: list[torch.Tensor] = []
    binary_targets_chunks: list[torch.Tensor] = []
    unsafe_logits_chunks: list[torch.Tensor] = []
    unsafe_targets_chunks: list[torch.Tensor] = []

    for batch_idx, batch in enumerate(loader, start=1):
        text_feat, image_feat, binary_targets, unsafe_targets, flat_targets = batch
        text_feat = text_feat.to(device, non_blocking=True)
        image_feat = image_feat.to(device, non_blocking=True)
        binary_targets = binary_targets.to(device, non_blocking=True)
        unsafe_targets = unsafe_targets.to(device, non_blocking=True)
        flat_targets = flat_targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(text_feat=text_feat, image_feat=image_feat)

        if output_mode == "flat":
            flat_logits = outputs["flat_logits"]
            loss = compute_flat_loss(
                flat_logits=flat_logits,
                flat_targets=flat_targets,
                class_weights=flat_class_weights,
                loss_type=loss_type,
                focal_gamma=focal_gamma,
            )
            total_loss_t = loss
            binary_loss_t = torch.tensor(0.0, device=device)
            unsafe_loss_t = torch.tensor(0.0, device=device)

            flat_logits_chunks.append(flat_logits.detach().cpu())
            flat_targets_chunks.append(flat_targets.detach().cpu())
        else:
            hier = compute_hierarchical_loss(
                binary_logits=outputs["binary_logits"],
                unsafe_logits=outputs["unsafe_logits"],
                binary_targets=binary_targets,
                unsafe_targets=unsafe_targets,
                lambda_unsafe=lambda_unsafe,
                binary_class_weights=binary_class_weights,
                unsafe_class_weights=unsafe_class_weights,
                ignore_index=IGNORE_INDEX,
                loss_type=loss_type,
                focal_gamma=focal_gamma,
            )
            total_loss_t = hier.total_loss
            binary_loss_t = hier.binary_loss
            unsafe_loss_t = hier.unsafe_loss
            loss = total_loss_t

            binary_logits_chunks.append(outputs["binary_logits"].detach().cpu())
            binary_targets_chunks.append(binary_targets.detach().cpu())
            unsafe_logits_chunks.append(outputs["unsafe_logits"].detach().cpu())
            unsafe_targets_chunks.append(unsafe_targets.detach().cpu())
            flat_targets_chunks.append(flat_targets.detach().cpu())

        if is_train:
            loss.backward()
            optimizer.step()

        if log_batch_loss and batch_log_rows is not None and (batch_idx % log_batch_every == 0):
            lr_value = None
            if optimizer is not None and len(optimizer.param_groups) > 0:
                lr_value = float(optimizer.param_groups[0].get("lr", 0.0))

            batch_log_rows.append(
                {
                    "epoch": epoch,
                    "split": split_name,
                    "batch_idx": batch_idx,
                    "batch_size": int(text_feat.shape[0]),
                    "loss_total": float(total_loss_t.detach().cpu()),
                    "loss_binary": float(binary_loss_t.detach().cpu()) if output_mode != "flat" else 0.0,
                    "loss_unsafe": float(unsafe_loss_t.detach().cpu()) if output_mode != "flat" else 0.0,
                    "loss_flat": float(loss.detach().cpu()) if output_mode == "flat" else 0.0,
                    "lr": lr_value,
                }
            )

        bs = int(text_feat.shape[0])
        seen += bs
        sum_total_loss += float(total_loss_t.detach().cpu()) * bs
        sum_binary_loss += float(binary_loss_t.detach().cpu()) * bs
        sum_unsafe_loss += float(unsafe_loss_t.detach().cpu()) * bs
        sum_flat_loss += float(loss.detach().cpu()) * bs if output_mode == "flat" else 0.0

    if seen == 0:
        raise ValueError("Empty loader encountered during epoch pass")

    result: dict[str, Any] = {
        "loss_total": sum_total_loss / seen,
    }

    if output_mode == "flat":
        result["loss_flat"] = sum_flat_loss / seen
        flat_logits_all = torch.cat(flat_logits_chunks, dim=0)
        flat_targets_all = torch.cat(flat_targets_chunks, dim=0)

        flat_metrics = compute_flat9_metrics(flat_logits_all, flat_targets_all)
        result.update(
            {
                "flat_accuracy": flat_metrics.accuracy,
                "flat_balanced_accuracy": flat_metrics.balanced_accuracy,
                "flat_macro_f1": flat_metrics.macro_f1,
                "flat_weighted_f1": flat_metrics.weighted_f1,
                "flat_per_class_precision": flat_metrics.per_class_precision,
                "flat_per_class_recall": flat_metrics.per_class_recall,
                "flat_per_class_f1": flat_metrics.per_class_f1,
                "flat_per_class_support": flat_metrics.per_class_support,
                "flat_confusion_matrix": flat_metrics.confusion_matrix,
            }
        )

        if include_uncalibrated_reliability:
            rel = compute_reliability_metrics(
                logits=flat_logits_all,
                targets=flat_targets_all,
                n_bins=n_bins,
            )
            result.update(
                {
                    "flat_ece": rel.ece,
                    "flat_brier": rel.brier,
                    "flat_reliability_bin_edges": rel.bin_edges,
                    "flat_reliability_bin_counts": rel.bin_counts,
                    "flat_reliability_bin_accuracy": rel.bin_accuracy,
                    "flat_reliability_bin_confidence": rel.bin_confidence,
                }
            )

        return result

    result["loss_binary"] = sum_binary_loss / seen
    result["loss_unsafe"] = sum_unsafe_loss / seen

    binary_logits_all = torch.cat(binary_logits_chunks, dim=0)
    binary_targets_all = torch.cat(binary_targets_chunks, dim=0)
    unsafe_logits_all = torch.cat(unsafe_logits_chunks, dim=0)
    unsafe_targets_all = torch.cat(unsafe_targets_chunks, dim=0)
    flat_targets_all = torch.cat(flat_targets_chunks, dim=0)

    binary_metrics = compute_binary_metrics(binary_logits_all, binary_targets_all)
    unsafe_metrics = compute_unsafe_subclass_metrics(
        unsafe_logits=unsafe_logits_all,
        unsafe_targets=unsafe_targets_all,
        ignore_index=IGNORE_INDEX,
    )

    binary_probs_all = torch.softmax(binary_logits_all, dim=1)
    unsafe_probs_all = torch.softmax(unsafe_logits_all, dim=1)
    binary_pred_all = (binary_probs_all[:, 1] >= 0.5).long()
    unsafe_pred_all = unsafe_probs_all.argmax(dim=1)
    derived_flat_pred_all = torch.where(
        binary_pred_all == 0,
        torch.zeros_like(unsafe_pred_all),
        unsafe_pred_all + 1,
    )
    derived_flat_metrics = compute_multiclass_metrics_from_predictions(
        predictions=derived_flat_pred_all,
        targets=flat_targets_all,
        num_classes=9,
        ignore_index=None,
    )

    result.update(
        {
            "binary_accuracy": binary_metrics.accuracy,
            "binary_precision": binary_metrics.precision,
            "binary_recall": binary_metrics.recall,
            "binary_f1": binary_metrics.f1,
            "binary_auroc": binary_metrics.auroc,
            "binary_pr_auc": binary_metrics.pr_auc,
            "binary_brier": binary_metrics.brier,
            "unsafe_accuracy": unsafe_metrics.accuracy,
            "unsafe_balanced_accuracy": unsafe_metrics.balanced_accuracy,
            "unsafe_macro_f1": unsafe_metrics.macro_f1,
            "unsafe_weighted_f1": unsafe_metrics.weighted_f1,
            "unsafe_per_class_precision": unsafe_metrics.per_class_precision,
            "unsafe_per_class_recall": unsafe_metrics.per_class_recall,
            "unsafe_per_class_f1": unsafe_metrics.per_class_f1,
            "unsafe_per_class_support": unsafe_metrics.per_class_support,
            "unsafe_confusion_matrix": unsafe_metrics.confusion_matrix,
            "derived_flat_accuracy": derived_flat_metrics.accuracy,
            "derived_flat_balanced_accuracy": derived_flat_metrics.balanced_accuracy,
            "derived_flat_macro_f1": derived_flat_metrics.macro_f1,
            "derived_flat_weighted_f1": derived_flat_metrics.weighted_f1,
            "derived_flat_per_class_precision": derived_flat_metrics.per_class_precision,
            "derived_flat_per_class_recall": derived_flat_metrics.per_class_recall,
            "derived_flat_per_class_f1": derived_flat_metrics.per_class_f1,
            "derived_flat_per_class_support": derived_flat_metrics.per_class_support,
            "derived_flat_confusion_matrix": derived_flat_metrics.confusion_matrix,
        }
    )

    if include_uncalibrated_reliability:
        binary_rel = compute_reliability_metrics(
            logits=binary_logits_all,
            targets=binary_targets_all,
            n_bins=n_bins,
        )
        unsafe_rel = compute_reliability_metrics(
            logits=unsafe_logits_all,
            targets=unsafe_targets_all,
            n_bins=n_bins,
            ignore_index=IGNORE_INDEX,
        )
        result.update(
            {
                "binary_ece": binary_rel.ece,
                "unsafe_ece": unsafe_rel.ece,
                "unsafe_brier": unsafe_rel.brier,
                "binary_reliability_bin_edges": binary_rel.bin_edges,
                "binary_reliability_bin_counts": binary_rel.bin_counts,
                "unsafe_reliability_bin_edges": unsafe_rel.bin_edges,
                "unsafe_reliability_bin_counts": unsafe_rel.bin_counts,
            }
        )

    return result


def _to_loggable(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, (list, dict)):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train fusion heads on cached OpenCLIP embeddings.")
    p.add_argument("--cache-dir", type=Path, default=TrainConfig().cache_dir)
    p.add_argument("--run-dir", type=Path, default=TrainConfig().run_dir)
    p.add_argument("--model-name", type=str, default=TrainConfig().model_name)
    p.add_argument("--pretrained", type=str, default=TrainConfig().pretrained)
    p.add_argument("--fusion", choices=["concat", "interaction", "interaction_only"], default=TrainConfig().fusion)
    p.add_argument("--head-type", choices=["linear", "small", "medium", "large"], default=TrainConfig().head_type)
    p.add_argument("--output-mode", choices=["flat", "hierarchical"], default=TrainConfig().output_mode)
    p.add_argument("--proj-dim", type=int, default=TrainConfig().proj_dim)
    p.add_argument("--batch-size", type=int, default=TrainConfig().batch_size)
    p.add_argument("--epochs", type=int, default=TrainConfig().epochs)
    p.add_argument("--lr", type=float, default=TrainConfig().lr)
    p.add_argument("--weight-decay", type=float, default=TrainConfig().weight_decay)
    p.add_argument("--class-weighting", choices=["none", "inverse", "effective"], default=TrainConfig().class_weighting)
    p.add_argument("--beta", type=float, default=TrainConfig().beta)
    p.add_argument("--loss-type", choices=["ce", "focal"], default=TrainConfig().loss_type)
    p.add_argument("--focal-gamma", type=float, default=TrainConfig().focal_gamma)
    p.add_argument("--lambda-unsafe", type=float, default=TrainConfig().lambda_unsafe)
    p.add_argument("--seed", type=int, default=TrainConfig().seed)
    p.add_argument("--device", type=str, default=TrainConfig().device)
    p.add_argument("--num-workers", type=int, default=TrainConfig().num_workers)
    p.add_argument("--train-split", type=str, default=TrainConfig().train_split)
    p.add_argument("--val-split", type=str, default=TrainConfig().val_split)
    p.add_argument("--skip-val", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=TrainConfig().max_train_samples)
    p.add_argument("--max-val-samples", type=int, default=TrainConfig().max_val_samples)
    p.add_argument("--n-bins", type=int, default=TrainConfig().n_bins)
    p.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=TrainConfig().early_stopping)
    p.add_argument("--early-stopping-patience", type=int, default=TrainConfig().early_stopping_patience)
    p.add_argument("--early-stopping-min-delta", type=float, default=TrainConfig().early_stopping_min_delta)
    p.add_argument("--log-batch-loss", action="store_true", default=TrainConfig().log_batch_loss)
    p.add_argument("--log-batch-every", type=int, default=TrainConfig().log_batch_every)
    p.add_argument("--overwrite-run", action="store_true")
    p.add_argument("--sanity-check", action="store_true")
    p.add_argument("--model-smoke-test", action="store_true")
    p.add_argument("--loss-smoke-test", action="store_true")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    cfg = TrainConfig(
        cache_dir=args.cache_dir,
        run_dir=args.run_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        fusion=args.fusion,
        head_type=args.head_type,
        output_mode=args.output_mode,
        proj_dim=args.proj_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weighting=args.class_weighting,
        beta=args.beta,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        lambda_unsafe=args.lambda_unsafe,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        train_split=args.train_split,
        val_split=args.val_split,
        skip_val=bool(args.skip_val),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        n_bins=args.n_bins,
        early_stopping=bool(args.early_stopping),
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        log_batch_loss=bool(args.log_batch_loss),
        log_batch_every=args.log_batch_every,
        sanity_check=bool(args.sanity_check),
        model_smoke_test=bool(args.model_smoke_test),
        loss_smoke_test=bool(args.loss_smoke_test),
        overwrite_run=bool(args.overwrite_run),
    )

    if cfg.model_smoke_test:
        _run_model_smoke_test(cfg)
        return

    if cfg.loss_smoke_test:
        _run_loss_smoke_test(cfg)
        return

    if cfg.batch_size <= 0:
        raise ValueError(f"batch-size must be > 0, got {cfg.batch_size}")
    if cfg.epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {cfg.epochs}")
    if cfg.num_workers < 0:
        raise ValueError(f"num-workers must be >= 0, got {cfg.num_workers}")
    if cfg.log_batch_every <= 0:
        raise ValueError(f"log-batch-every must be > 0, got {cfg.log_batch_every}")
    if cfg.focal_gamma < 0:
        raise ValueError(f"focal-gamma must be >= 0, got {cfg.focal_gamma}")
    if cfg.early_stopping_patience < 1:
        raise ValueError(f"early-stopping-patience must be >= 1, got {cfg.early_stopping_patience}")
    if cfg.early_stopping_min_delta < 0:
        raise ValueError(f"early-stopping-min-delta must be >= 0, got {cfg.early_stopping_min_delta}")

    set_global_seed(cfg.seed)
    device = _resolve_device(cfg.device)

    if cfg.run_dir.exists() and any(cfg.run_dir.iterdir()) and not cfg.overwrite_run:
        raise FileExistsError(
            f"run-dir already exists and is non-empty: {cfg.run_dir}. "
            "Use --overwrite-run to reuse it."
        )
    ensure_dir(cfg.run_dir)

    cache_root = _resolve_cache_root(cfg.cache_dir, cfg.model_name, cfg.pretrained)
    train_t, train_i, train_b, train_u, train_f = _load_split_cache(
        cache_root=cache_root,
        split_name=cfg.train_split,
        max_samples=cfg.max_train_samples,
    )

    has_val = not cfg.skip_val
    if has_val:
        try:
            val_t, val_i, val_b, val_u, val_f = _load_split_cache(
                cache_root=cache_root,
                split_name=cfg.val_split,
                max_samples=cfg.max_val_samples,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Validation split cache not found for split='{cfg.val_split}'. "
                "Either generate val cache or pass --skip-val."
            ) from exc

    train_loader = _make_loader(
        train_t,
        train_i,
        train_b,
        train_u,
        train_f,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    if has_val:
        val_loader = _make_loader(
            val_t,
            val_i,
            val_b,
            val_u,
            val_f,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
    else:
        val_loader = None

    model = FusionClassifier(
        text_dim=int(train_t.shape[1]),
        image_dim=int(train_i.shape[1]),
        fusion_type=cfg.fusion,
        head_type=cfg.head_type,
        output_mode=cfg.output_mode,
        proj_dim=cfg.proj_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    flat_class_weights = None
    binary_class_weights = None
    unsafe_class_weights = None

    if cfg.output_mode == "flat":
        flat_class_weights = build_flat9_class_weights(
            train_f.tolist(),
            strategy=cfg.class_weighting,
            beta=cfg.beta,
            device=device,
        )
    else:
        binary_class_weights = build_binary_class_weights(
            train_b.tolist(),
            strategy=cfg.class_weighting,
            beta=cfg.beta,
            device=device,
        )
        unsafe_class_weights = build_unsafe_class_weights(
            train_u.tolist(),
            strategy=cfg.class_weighting,
            beta=cfg.beta,
            device=device,
            ignore_index=IGNORE_INDEX,
        )

    policy = final_metric_policy() if cfg.sanity_check else search_metric_policy()

    write_config_snapshot(cfg.run_dir / "run_config.json", cfg)
    write_json(
        cfg.run_dir / "class_counts.json",
        {
            "train_binary": pd.Series(train_b.numpy()).value_counts().sort_index().to_dict(),
            "train_unsafe_subclass_non_ignore": pd.Series(train_u[train_u != IGNORE_INDEX].numpy())
            .value_counts()
            .sort_index()
            .to_dict(),
            "train_composite_9way": pd.Series(train_f.numpy()).value_counts().sort_index().to_dict(),
        },
    )

    if cfg.sanity_check:
        sanity = _epoch_pass(
            model=model,
            loader=train_loader,
            device=device,
            split_name="train",
            epoch=1,
            output_mode=cfg.output_mode,
            optimizer=None,
            lambda_unsafe=cfg.lambda_unsafe,
            loss_type=cfg.loss_type,
            focal_gamma=cfg.focal_gamma,
            flat_class_weights=flat_class_weights,
            binary_class_weights=binary_class_weights,
            unsafe_class_weights=unsafe_class_weights,
            n_bins=cfg.n_bins,
            include_uncalibrated_reliability=True,
            log_batch_loss=False,
            log_batch_every=1,
            batch_log_rows=None,
        )
        if not torch.isfinite(torch.tensor(float(sanity["loss_total"]))):
            raise ValueError("Sanity check failed: non-finite loss")
        write_json(
            cfg.run_dir / "sanity_check.json",
            {
                "created_at_utc": utc_now_iso(),
                "metrics": _to_loggable(sanity),
            },
        )
        print("sanity_check=ok")
        print(f"loss_total={float(sanity['loss_total']):.6f}")
        return

    log_rows: list[dict[str, Any]] = []
    batch_log_rows: list[dict[str, Any]] = []
    monitor_name = _monitor_name_for_output_mode(cfg.output_mode)
    best_score = float("-inf")
    best_epoch = -1
    no_improve_epochs = 0
    epochs_ran = 0
    stopped_early = False

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _epoch_pass(
            model=model,
            loader=train_loader,
            device=device,
            split_name="train",
            epoch=epoch,
            output_mode=cfg.output_mode,
            optimizer=optimizer,
            lambda_unsafe=cfg.lambda_unsafe,
            loss_type=cfg.loss_type,
            focal_gamma=cfg.focal_gamma,
            flat_class_weights=flat_class_weights,
            binary_class_weights=binary_class_weights,
            unsafe_class_weights=unsafe_class_weights,
            n_bins=cfg.n_bins,
            include_uncalibrated_reliability=policy.include_uncalibrated_reliability,
            log_batch_loss=cfg.log_batch_loss,
            log_batch_every=cfg.log_batch_every,
            batch_log_rows=batch_log_rows,
        )

        train_row = {
            "epoch": epoch,
            "split": "train",
            **train_metrics,
        }
        log_rows.append(_to_loggable(train_row))

        print(f"epoch={epoch} split=train loss_total={float(train_metrics['loss_total']):.6f}")

        if val_loader is not None:
            with torch.inference_mode():
                val_metrics = _epoch_pass(
                    model=model,
                    loader=val_loader,
                    device=device,
                    split_name="val",
                    epoch=epoch,
                    output_mode=cfg.output_mode,
                    optimizer=None,
                    lambda_unsafe=cfg.lambda_unsafe,
                    loss_type=cfg.loss_type,
                    focal_gamma=cfg.focal_gamma,
                    flat_class_weights=flat_class_weights,
                    binary_class_weights=binary_class_weights,
                    unsafe_class_weights=unsafe_class_weights,
                    n_bins=cfg.n_bins,
                    include_uncalibrated_reliability=policy.include_uncalibrated_reliability,
                    log_batch_loss=cfg.log_batch_loss,
                    log_batch_every=cfg.log_batch_every,
                    batch_log_rows=batch_log_rows,
                )
            val_row = {
                "epoch": epoch,
                "split": "val",
                **val_metrics,
            }
            log_rows.append(_to_loggable(val_row))
            monitor = _monitor_value_from_metrics(val_metrics, cfg.output_mode)
            print(
                f"epoch={epoch} split=val loss_total={float(val_metrics['loss_total']):.6f} "
                f"{monitor_name}={monitor:.6f}"
            )
        else:
            monitor = _monitor_value_from_metrics(train_metrics, cfg.output_mode)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.as_dict(),
            "monitor_name": monitor_name,
            "monitor_value": monitor,
        }

        torch.save(state, cfg.run_dir / "last.ckpt")
        if monitor > (best_score + cfg.early_stopping_min_delta):
            best_score = monitor
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(state, cfg.run_dir / "best.ckpt")
        else:
            no_improve_epochs += 1

        epochs_ran = epoch
        if cfg.early_stopping and no_improve_epochs >= cfg.early_stopping_patience:
            stopped_early = True
            print(
                "early_stopping=triggered "
                f"epoch={epoch} no_improve_epochs={no_improve_epochs} "
                f"monitor={monitor_name}"
            )
            break

    train_log = pd.DataFrame(log_rows)
    train_log.to_csv(cfg.run_dir / "train_log.csv", index=False)

    batch_log_path = cfg.run_dir / "train_batch_log.csv"
    if cfg.log_batch_loss and len(batch_log_rows) > 0:
        pd.DataFrame(batch_log_rows).to_csv(batch_log_path, index=False)

    run_summary = {
        "created_at_utc": utc_now_iso(),
        "run_dir": str(cfg.run_dir),
        "cache_root": str(cache_root),
        "output_mode": cfg.output_mode,
        "fusion": cfg.fusion,
        "head_type": cfg.head_type,
        "epochs": cfg.epochs,
        "epochs_ran": int(epochs_ran),
        "best_epoch": int(best_epoch),
        "best_monitor_name": monitor_name,
        "best_monitor_value": float(best_score),
        "stopped_early": bool(stopped_early),
        "early_stopping": {
            "enabled": bool(cfg.early_stopping),
            "patience": int(cfg.early_stopping_patience),
            "min_delta": float(cfg.early_stopping_min_delta),
        },
        "metric_policy": _dictify(policy),
        "artifacts": {
            "config": str(cfg.run_dir / "run_config.json"),
            "train_log": str(cfg.run_dir / "train_log.csv"),
            "train_batch_log": str(batch_log_path) if cfg.log_batch_loss else None,
            "best_ckpt": str(cfg.run_dir / "best.ckpt"),
            "last_ckpt": str(cfg.run_dir / "last.ckpt"),
            "class_counts": str(cfg.run_dir / "class_counts.json"),
        },
    }
    write_json(cfg.run_dir / "run_summary.json", run_summary)

    print("training_complete=ok")
    print(f"best_epoch={best_epoch}")
    print(f"best_{monitor_name}={best_score:.6f}")
    print(f"run_dir={cfg.run_dir}")


if __name__ == "__main__":
    main()
