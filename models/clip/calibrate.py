from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .fusion_heads import FusionClassifier
from .labels import IGNORE_INDEX
from .metrics import (
    compute_binary_metrics,
    compute_flat9_metrics,
    compute_reliability_metrics,
    compute_unsafe_subclass_metrics,
)
from .paths import CALIBRATION_DIR, PROJECT_ROOT
from .torch_runtime import cuda_available_safely
from .utils import ensure_dir, utc_now_iso, write_json

REQUIRED_LABEL_COLUMNS = [
    "binary_label",
    "unsafe_subclass_label",
    "composite_9way_label",
]


@dataclass(frozen=True)
class LoadedSplit:
    text_features: torch.Tensor
    image_features: torch.Tensor
    binary_targets: torch.Tensor
    unsafe_targets: torch.Tensor
    flat_targets: torch.Tensor
    labels_df: pd.DataFrame


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
) -> LoadedSplit:
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

    return LoadedSplit(
        text_features=text_features,
        image_features=image_features,
        binary_targets=binary_targets,
        unsafe_targets=unsafe_targets,
        flat_targets=flat_targets,
        labels_df=labels_df,
    )


def _make_loader(
    loaded: LoadedSplit,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        loaded.text_features,
        loaded.image_features,
        loaded.binary_targets,
        loaded.unsafe_targets,
        loaded.flat_targets,
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


def _fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int | None = None,
    max_iter: int = 50,
) -> float:
    logits = logits.detach().float()
    targets = targets.detach().long()

    if ignore_index is not None:
        valid = targets != ignore_index
        logits = logits[valid]
        targets = targets[valid]

    if logits.shape[0] == 0:
        return 1.0

    device = logits.device
    log_temp = nn.Parameter(torch.zeros((), device=device))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temp).clamp(min=1e-4, max=1e4)
        loss = F.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temp.detach()).clamp(min=1e-4, max=1e4).cpu())


def _scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(float(temperature), 1e-4)


def _evaluate_flat_logits(
    flat_logits: torch.Tensor,
    flat_targets: torch.Tensor,
    n_bins: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    flat_probs = torch.softmax(flat_logits, dim=1)
    flat_pred = flat_probs.argmax(dim=1)
    flat_metrics = compute_flat9_metrics(flat_logits, flat_targets)
    rel = _safe_reliability(flat_logits, flat_targets, n_bins=n_bins)

    payload = {
        "flat9_metrics": {
            "accuracy": flat_metrics.accuracy,
            "balanced_accuracy": flat_metrics.balanced_accuracy,
            "macro_f1": flat_metrics.macro_f1,
            "weighted_f1": flat_metrics.weighted_f1,
            "per_class_precision": flat_metrics.per_class_precision,
            "per_class_recall": flat_metrics.per_class_recall,
            "per_class_f1": flat_metrics.per_class_f1,
            "per_class_support": flat_metrics.per_class_support,
            "confusion_matrix": flat_metrics.confusion_matrix,
        },
        "flat9_reliability": rel,
    }
    return payload, flat_pred, flat_probs.max(dim=1).values


def _evaluate_hierarchical_logits(
    binary_logits: torch.Tensor,
    unsafe_logits: torch.Tensor,
    binary_targets: torch.Tensor,
    unsafe_targets: torch.Tensor,
    flat_targets: torch.Tensor,
    n_bins: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    binary_probs = torch.softmax(binary_logits, dim=1)
    unsafe_probs = torch.softmax(unsafe_logits, dim=1)
    derived_flat_probs = torch.cat([binary_probs[:, :1], binary_probs[:, 1:2] * unsafe_probs], dim=1)
    derived_flat_logits = torch.log(derived_flat_probs.clamp_min(1e-12))

    binary_pred = (binary_probs[:, 1] >= 0.5).long()
    unsafe_pred = unsafe_probs.argmax(dim=1)
    derived_flat_pred = torch.where(binary_pred == 0, torch.zeros_like(unsafe_pred), unsafe_pred + 1)

    binary_metrics = compute_binary_metrics(binary_logits, binary_targets)
    unsafe_metrics = compute_unsafe_subclass_metrics(
        unsafe_logits=unsafe_logits,
        unsafe_targets=unsafe_targets,
        ignore_index=IGNORE_INDEX,
    )
    derived_flat_metrics = compute_flat9_metrics(
        flat_logits=derived_flat_logits,
        targets=flat_targets,
    )

    payload = {
        "binary_metrics": {
            "accuracy": binary_metrics.accuracy,
            "precision": binary_metrics.precision,
            "recall": binary_metrics.recall,
            "f1": binary_metrics.f1,
            "auroc": binary_metrics.auroc,
            "pr_auc": binary_metrics.pr_auc,
            "brier": binary_metrics.brier,
        },
        "unsafe_metrics": {
            "accuracy": unsafe_metrics.accuracy,
            "balanced_accuracy": unsafe_metrics.balanced_accuracy,
            "macro_f1": unsafe_metrics.macro_f1,
            "weighted_f1": unsafe_metrics.weighted_f1,
            "per_class_precision": unsafe_metrics.per_class_precision,
            "per_class_recall": unsafe_metrics.per_class_recall,
            "per_class_f1": unsafe_metrics.per_class_f1,
            "per_class_support": unsafe_metrics.per_class_support,
            "confusion_matrix": unsafe_metrics.confusion_matrix,
        },
        "binary_reliability": _safe_reliability(binary_logits, binary_targets, n_bins=n_bins),
        "unsafe_reliability": _safe_reliability(
            unsafe_logits,
            unsafe_targets,
            n_bins=n_bins,
            ignore_index=IGNORE_INDEX,
        ),
        "derived_flat9_reliability": _safe_reliability(
            derived_flat_logits,
            flat_targets,
            n_bins=n_bins,
        ),
        "derived_flat9_metrics_from_hierarchical": {
            "accuracy": derived_flat_metrics.accuracy,
            "balanced_accuracy": derived_flat_metrics.balanced_accuracy,
            "macro_f1": derived_flat_metrics.macro_f1,
            "weighted_f1": derived_flat_metrics.weighted_f1,
            "per_class_precision": derived_flat_metrics.per_class_precision,
            "per_class_recall": derived_flat_metrics.per_class_recall,
            "per_class_f1": derived_flat_metrics.per_class_f1,
            "per_class_support": derived_flat_metrics.per_class_support,
            "confusion_matrix": derived_flat_metrics.confusion_matrix,
        },
    }
    predictions = {
        "binary_pred": binary_pred,
        "binary_prob_unsafe": binary_probs[:, 1],
        "unsafe_pred": unsafe_pred,
        "unsafe_confidence": unsafe_probs.max(dim=1).values,
        "derived_flat9_pred": derived_flat_pred,
    }
    return payload, predictions


def _delta_summary(output_mode: str, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if output_mode == "flat":
        before_metrics = before["flat9_metrics"]
        after_metrics = after["flat9_metrics"]
        before_rel = before["flat9_reliability"]
        after_rel = after["flat9_reliability"]
        return {
            "flat_macro_f1_delta": float(after_metrics["macro_f1"] - before_metrics["macro_f1"]),
            "flat_weighted_f1_delta": float(after_metrics["weighted_f1"] - before_metrics["weighted_f1"]),
            "flat_ece_delta": None
            if before_rel["ece"] is None or after_rel["ece"] is None
            else float(after_rel["ece"] - before_rel["ece"]),
            "flat_brier_delta": None
            if before_rel["brier"] is None or after_rel["brier"] is None
            else float(after_rel["brier"] - before_rel["brier"]),
        }

    before_binary = before["binary_metrics"]
    after_binary = after["binary_metrics"]
    before_unsafe = before["unsafe_metrics"]
    after_unsafe = after["unsafe_metrics"]
    before_flat = before["derived_flat9_metrics_from_hierarchical"]
    after_flat = after["derived_flat9_metrics_from_hierarchical"]
    before_flat_rel = before["derived_flat9_reliability"]
    after_flat_rel = after["derived_flat9_reliability"]
    before_binary_rel = before["binary_reliability"]
    after_binary_rel = after["binary_reliability"]
    before_unsafe_rel = before["unsafe_reliability"]
    after_unsafe_rel = after["unsafe_reliability"]
    return {
        "binary_f1_delta": float(after_binary["f1"] - before_binary["f1"]),
        "unsafe_macro_f1_delta": float(after_unsafe["macro_f1"] - before_unsafe["macro_f1"]),
        "derived_flat_macro_f1_delta": float(after_flat["macro_f1"] - before_flat["macro_f1"]),
        "derived_flat_ece_delta": None
        if before_flat_rel["ece"] is None or after_flat_rel["ece"] is None
        else float(after_flat_rel["ece"] - before_flat_rel["ece"]),
        "binary_ece_delta": None
        if before_binary_rel["ece"] is None or after_binary_rel["ece"] is None
        else float(after_binary_rel["ece"] - before_binary_rel["ece"]),
        "unsafe_ece_delta": None
        if before_unsafe_rel["ece"] is None or after_unsafe_rel["ece"] is None
        else float(after_unsafe_rel["ece"] - before_unsafe_rel["ece"]),
    }


def _collect_logits(
    model: FusionClassifier,
    loader: DataLoader,
    output_mode: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    binary_logits_chunks: list[torch.Tensor] = []
    unsafe_logits_chunks: list[torch.Tensor] = []
    flat_logits_chunks: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in loader:
            text_feat, image_feat, _, _, _ = batch
            text_feat = text_feat.to(device, non_blocking=True)
            image_feat = image_feat.to(device, non_blocking=True)
            outputs = model(text_feat=text_feat, image_feat=image_feat)

            if output_mode == "flat":
                flat_logits_chunks.append(outputs["flat_logits"].detach().cpu())
            else:
                binary_logits_chunks.append(outputs["binary_logits"].detach().cpu())
                unsafe_logits_chunks.append(outputs["unsafe_logits"].detach().cpu())

    if output_mode == "flat":
        return {"flat_logits": torch.cat(flat_logits_chunks, dim=0)}
    return {
        "binary_logits": torch.cat(binary_logits_chunks, dim=0),
        "unsafe_logits": torch.cat(unsafe_logits_chunks, dim=0),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit temperature scaling and compare calibrated metrics.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--fit-split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=["val", "test"],
        help="Splits to report after fitting temperatures.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-bins", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"batch-size must be > 0, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"num-workers must be >= 0, got {args.num_workers}")

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run-dir does not exist: {run_dir}")

    cfg = _load_run_config(run_dir)
    ckpt_path = run_dir / f"{args.checkpoint}.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    ckpt_cfg = checkpoint.get("config", cfg)

    cache_dir = args.cache_dir
    if cache_dir is None:
        if "cache_dir" not in cfg:
            raise ValueError("run config does not contain cache_dir; pass --cache-dir explicitly")
        cache_dir = _resolve_path(str(cfg["cache_dir"]), PROJECT_ROOT)
    else:
        cache_dir = cache_dir.resolve()

    cache_root = _resolve_cache_root(
        cache_dir=cache_dir,
        model_name=str(ckpt_cfg.get("model_name", cfg.get("model_name", ""))),
        pretrained=str(ckpt_cfg.get("pretrained", cfg.get("pretrained", ""))),
    )
    device = _resolve_device(args.device)

    output_mode = str(ckpt_cfg["output_mode"])
    fusion = str(ckpt_cfg["fusion"])
    head_type = str(ckpt_cfg["head_type"])
    proj_dim = int(ckpt_cfg.get("proj_dim", 512))

    fit_loaded = _load_split_cache(cache_root=cache_root, split_name=args.fit_split, max_samples=args.max_samples)
    fit_loader = _make_loader(fit_loaded, batch_size=args.batch_size, num_workers=args.num_workers)

    model = FusionClassifier(
        text_dim=int(fit_loaded.text_features.shape[1]),
        image_dim=int(fit_loaded.image_features.shape[1]),
        fusion_type=fusion,
        head_type=head_type,
        output_mode=output_mode,
        proj_dim=proj_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    fit_logits = _collect_logits(model=model, loader=fit_loader, output_mode=output_mode, device=device)

    if output_mode == "flat":
        flat_temperature = _fit_temperature(
            logits=fit_logits["flat_logits"],
            targets=fit_loaded.flat_targets,
        )
        temperatures = {"flat_temperature": flat_temperature}
    else:
        binary_temperature = _fit_temperature(
            logits=fit_logits["binary_logits"],
            targets=fit_loaded.binary_targets,
        )
        unsafe_temperature = _fit_temperature(
            logits=fit_logits["unsafe_logits"],
            targets=fit_loaded.unsafe_targets,
            ignore_index=IGNORE_INDEX,
        )
        temperatures = {
            "binary_temperature": binary_temperature,
            "unsafe_temperature": unsafe_temperature,
        }

    eval_n_bins = int(args.n_bins) if args.n_bins is not None else int(cfg.get("n_bins", 15))
    split_payloads: dict[str, Any] = {}

    out_dir = args.out_dir.resolve() if args.out_dir is not None else CALIBRATION_DIR / run_dir.name
    ensure_dir(out_dir)

    for split_name in args.eval_splits:
        loaded = _load_split_cache(cache_root=cache_root, split_name=split_name, max_samples=args.max_samples)
        loader = _make_loader(loaded, batch_size=args.batch_size, num_workers=args.num_workers)
        logits = _collect_logits(model=model, loader=loader, output_mode=output_mode, device=device)

        pred_df = loaded.labels_df.copy()

        if output_mode == "flat":
            uncalibrated, flat_pred, flat_conf = _evaluate_flat_logits(
                flat_logits=logits["flat_logits"],
                flat_targets=loaded.flat_targets,
                n_bins=eval_n_bins,
            )
            calibrated_logits = _scale_logits(logits["flat_logits"], temperatures["flat_temperature"])
            calibrated, flat_pred_cal, flat_conf_cal = _evaluate_flat_logits(
                flat_logits=calibrated_logits,
                flat_targets=loaded.flat_targets,
                n_bins=eval_n_bins,
            )
            pred_df["flat9_pred_uncalibrated"] = flat_pred.numpy()
            pred_df["flat9_confidence_uncalibrated"] = flat_conf.numpy()
            pred_df["flat9_pred_calibrated"] = flat_pred_cal.numpy()
            pred_df["flat9_confidence_calibrated"] = flat_conf_cal.numpy()
        else:
            uncalibrated, uncal_pred = _evaluate_hierarchical_logits(
                binary_logits=logits["binary_logits"],
                unsafe_logits=logits["unsafe_logits"],
                binary_targets=loaded.binary_targets,
                unsafe_targets=loaded.unsafe_targets,
                flat_targets=loaded.flat_targets,
                n_bins=eval_n_bins,
            )
            calibrated_binary_logits = _scale_logits(logits["binary_logits"], temperatures["binary_temperature"])
            calibrated_unsafe_logits = _scale_logits(logits["unsafe_logits"], temperatures["unsafe_temperature"])
            calibrated, cal_pred = _evaluate_hierarchical_logits(
                binary_logits=calibrated_binary_logits,
                unsafe_logits=calibrated_unsafe_logits,
                binary_targets=loaded.binary_targets,
                unsafe_targets=loaded.unsafe_targets,
                flat_targets=loaded.flat_targets,
                n_bins=eval_n_bins,
            )

            pred_df["binary_pred_uncalibrated"] = uncal_pred["binary_pred"].numpy()
            pred_df["binary_prob_unsafe_uncalibrated"] = uncal_pred["binary_prob_unsafe"].numpy()
            pred_df["unsafe_pred_uncalibrated"] = uncal_pred["unsafe_pred"].numpy()
            pred_df["unsafe_confidence_uncalibrated"] = uncal_pred["unsafe_confidence"].numpy()
            pred_df["derived_flat9_pred_uncalibrated"] = uncal_pred["derived_flat9_pred"].numpy()

            pred_df["binary_pred_calibrated"] = cal_pred["binary_pred"].numpy()
            pred_df["binary_prob_unsafe_calibrated"] = cal_pred["binary_prob_unsafe"].numpy()
            pred_df["unsafe_pred_calibrated"] = cal_pred["unsafe_pred"].numpy()
            pred_df["unsafe_confidence_calibrated"] = cal_pred["unsafe_confidence"].numpy()
            pred_df["derived_flat9_pred_calibrated"] = cal_pred["derived_flat9_pred"].numpy()

        split_payloads[split_name] = {
            "num_samples": int(len(loaded.labels_df)),
            "uncalibrated": uncalibrated,
            "calibrated": calibrated,
            "delta": _delta_summary(output_mode=output_mode, before=uncalibrated, after=calibrated),
        }
        pred_df.to_csv(out_dir / f"predictions_{split_name}_{args.checkpoint}.csv", index=False)

    summary = {
        "created_at_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(ckpt_path),
        "cache_dir": str(cache_dir),
        "cache_root": str(cache_root),
        "fit_split": args.fit_split,
        "eval_splits": list(args.eval_splits),
        "output_mode": output_mode,
        "fusion": fusion,
        "head_type": head_type,
        "temperatures": temperatures,
        "splits": split_payloads,
    }

    summary_path = out_dir / "calibration_summary.json"
    write_json(summary_path, summary)

    print("calibration_complete=ok")
    print(f"summary_json={summary_path}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
