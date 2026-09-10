from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .labels import IGNORE_INDEX


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    pr_auc: float
    brier: float


@dataclass(frozen=True)
class MultiClassMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: list[float]
    per_class_recall: list[float]
    per_class_f1: list[float]
    per_class_support: list[int]
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class ReliabilityMetrics:
    ece: float
    brier: float
    bin_edges: list[float]
    bin_counts: list[int]
    bin_accuracy: list[float]
    bin_confidence: list[float]


@dataclass(frozen=True)
class MetricPolicy:
    # Core discrimination metrics are always tracked for every variant.
    include_core_metrics: bool
    # Lightweight confidence diagnostics can be tracked for every variant.
    include_uncalibrated_reliability: bool
    # Temperature fitting and reliability diagrams are reserved for final model.
    include_calibration_workflow: bool


def search_metric_policy() -> MetricPolicy:
    return MetricPolicy(
        include_core_metrics=True,
        include_uncalibrated_reliability=True,
        include_calibration_workflow=False,
    )


def final_metric_policy() -> MetricPolicy:
    return MetricPolicy(
        include_core_metrics=True,
        include_uncalibrated_reliability=True,
        include_calibration_workflow=True,
    )


def _to_numpy_1d(values: torch.Tensor | np.ndarray | Iterable[int]) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")
    return arr


def _to_numpy_2d(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    return arr


def _safe_metric(metric_fn, *args, **kwargs) -> float:
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return float("nan")


def _binary_probabilities(binary_logits: torch.Tensor) -> np.ndarray:
    if binary_logits.ndim != 2 or binary_logits.shape[1] != 2:
        raise ValueError(
            f"binary_logits must have shape [N, 2], got {tuple(binary_logits.shape)}"
        )
    probs = F.softmax(binary_logits, dim=1)[:, 1]
    return probs.detach().cpu().numpy()


def compute_binary_metrics(
    binary_logits: torch.Tensor,
    binary_targets: torch.Tensor,
) -> BinaryMetrics:
    targets = _to_numpy_1d(binary_targets).astype(int)
    probs = _binary_probabilities(binary_logits)
    preds = (probs >= 0.5).astype(int)

    if len(targets) != len(probs):
        raise ValueError(
            f"Targets and probabilities must have same length, got {len(targets)} and {len(probs)}"
        )

    return BinaryMetrics(
        accuracy=float(accuracy_score(targets, preds)),
        precision=float(precision_score(targets, preds, zero_division=0)),
        recall=float(recall_score(targets, preds, zero_division=0)),
        f1=float(f1_score(targets, preds, zero_division=0)),
        auroc=_safe_metric(roc_auc_score, targets, probs),
        pr_auc=_safe_metric(average_precision_score, targets, probs),
        brier=float(np.mean((probs - targets) ** 2)),
    )


def _filter_ignore(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [N, C], got {tuple(logits.shape)}")

    target_np = _to_numpy_1d(targets).astype(int)
    if logits.shape[0] != target_np.shape[0]:
        raise ValueError(
            f"logits batch and targets length mismatch: {logits.shape[0]} vs {target_np.shape[0]}"
        )

    mask = target_np != ignore_index
    logits_np = _to_numpy_2d(logits)[mask]
    target_np = target_np[mask]
    return logits_np, target_np


def compute_multiclass_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> MultiClassMetrics:
    if ignore_index is None:
        logits_np = _to_numpy_2d(logits)
        target_np = _to_numpy_1d(targets).astype(int)
    else:
        logits_np, target_np = _filter_ignore(logits, targets, ignore_index=ignore_index)

    if len(target_np) == 0:
        raise ValueError("No valid samples available after ignore_index filtering")

    pred_np = logits_np.argmax(axis=1)

    labels = list(range(num_classes))
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        target_np,
        pred_np,
        labels=labels,
        average=None,
        zero_division=0,
    )
    cm = confusion_matrix(target_np, pred_np, labels=labels)
    return MultiClassMetrics(
        accuracy=float(accuracy_score(target_np, pred_np)),
        balanced_accuracy=float(balanced_accuracy_score(target_np, pred_np)),
        macro_f1=float(f1_score(target_np, pred_np, labels=labels, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(target_np, pred_np, labels=labels, average="weighted", zero_division=0)
        ),
        per_class_precision=per_precision.astype(float).tolist(),
        per_class_recall=per_recall.astype(float).tolist(),
        per_class_f1=per_f1.astype(float).tolist(),
        per_class_support=per_support.astype(int).tolist(),
        confusion_matrix=cm.astype(int).tolist(),
    )


def compute_multiclass_metrics_from_predictions(
    predictions: torch.Tensor | np.ndarray | Iterable[int],
    targets: torch.Tensor | np.ndarray | Iterable[int],
    num_classes: int,
    ignore_index: int | None = None,
) -> MultiClassMetrics:
    pred_np = _to_numpy_1d(predictions).astype(int)
    target_np = _to_numpy_1d(targets).astype(int)

    if pred_np.shape[0] != target_np.shape[0]:
        raise ValueError(
            f"predictions and targets length mismatch: {pred_np.shape[0]} vs {target_np.shape[0]}"
        )

    if ignore_index is not None:
        valid = target_np != ignore_index
        pred_np = pred_np[valid]
        target_np = target_np[valid]

    if len(target_np) == 0:
        raise ValueError("No valid samples available after ignore_index filtering")

    if (pred_np < 0).any() or (pred_np >= num_classes).any():
        bad = pred_np[(pred_np < 0) | (pred_np >= num_classes)]
        raise ValueError(
            f"Predictions contain class ids outside [0, {num_classes - 1}]: {bad[:10].tolist()}"
        )

    labels = list(range(num_classes))
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        target_np,
        pred_np,
        labels=labels,
        average=None,
        zero_division=0,
    )
    cm = confusion_matrix(target_np, pred_np, labels=labels)

    return MultiClassMetrics(
        accuracy=float(accuracy_score(target_np, pred_np)),
        balanced_accuracy=float(balanced_accuracy_score(target_np, pred_np)),
        macro_f1=float(f1_score(target_np, pred_np, labels=labels, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(target_np, pred_np, labels=labels, average="weighted", zero_division=0)
        ),
        per_class_precision=per_precision.astype(float).tolist(),
        per_class_recall=per_recall.astype(float).tolist(),
        per_class_f1=per_f1.astype(float).tolist(),
        per_class_support=per_support.astype(int).tolist(),
        confusion_matrix=cm.astype(int).tolist(),
    )


def compute_unsafe_subclass_metrics(
    unsafe_logits: torch.Tensor,
    unsafe_targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> MultiClassMetrics:
    return compute_multiclass_metrics(
        logits=unsafe_logits,
        targets=unsafe_targets,
        num_classes=8,
        ignore_index=ignore_index,
    )


def compute_flat9_metrics(
    flat_logits: torch.Tensor,
    flat_targets: torch.Tensor,
) -> MultiClassMetrics:
    return compute_multiclass_metrics(
        logits=flat_logits,
        targets=flat_targets,
        num_classes=9,
        ignore_index=None,
    )


def _ece_from_confidence(
    confidence: np.ndarray,
    correctness: np.ndarray,
    n_bins: int,
) -> tuple[float, list[float], list[int], list[float], list[float]]:
    if n_bins <= 0:
        raise ValueError(f"n_bins must be > 0, got {n_bins}")

    if len(confidence) == 0:
        raise ValueError("Cannot compute ECE with empty inputs")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidence, bin_edges[1:-1], right=True)

    total = float(len(confidence))
    ece = 0.0
    bin_counts: list[int] = []
    bin_acc: list[float] = []
    bin_conf: list[float] = []

    for i in range(n_bins):
        mask = bin_ids == i
        count = int(mask.sum())
        bin_counts.append(count)

        if count == 0:
            bin_acc.append(float("nan"))
            bin_conf.append(float("nan"))
            continue

        acc_i = float(correctness[mask].mean())
        conf_i = float(confidence[mask].mean())
        bin_acc.append(acc_i)
        bin_conf.append(conf_i)
        ece += (count / total) * abs(acc_i - conf_i)

    return float(ece), bin_edges.tolist(), bin_counts, bin_acc, bin_conf


def compute_reliability_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
    ignore_index: int | None = None,
) -> ReliabilityMetrics:
    logits_np = _to_numpy_2d(logits)
    target_np = _to_numpy_1d(targets).astype(int)

    if logits_np.shape[0] != target_np.shape[0]:
        raise ValueError(
            f"logits batch and targets length mismatch: {logits_np.shape[0]} vs {target_np.shape[0]}"
        )

    if ignore_index is not None:
        valid = target_np != ignore_index
        logits_np = logits_np[valid]
        target_np = target_np[valid]

    if len(target_np) == 0:
        raise ValueError("No valid samples available for reliability metrics")

    probs = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correctness = (preds == target_np).astype(np.float32)

    ece, edges, counts, acc, conf = _ece_from_confidence(
        confidence=confidence,
        correctness=correctness,
        n_bins=n_bins,
    )

    one_hot = np.eye(probs.shape[1], dtype=np.float32)[target_np]
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    return ReliabilityMetrics(
        ece=ece,
        brier=brier,
        bin_edges=edges,
        bin_counts=counts,
        bin_accuracy=acc,
        bin_confidence=conf,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Metrics module smoke test.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=15)
    return parser


def _coverage_targets(num_classes: int, batch_size: int, gen: torch.Generator) -> torch.Tensor:
    if batch_size < num_classes:
        raise ValueError(
            f"batch-size must be >= {num_classes} to cover all classes, got {batch_size}"
        )

    base = torch.arange(num_classes, dtype=torch.long)
    extra = torch.randint(0, num_classes, (batch_size - num_classes,), generator=gen)
    targets = torch.cat([base, extra], dim=0)
    perm = torch.randperm(batch_size, generator=gen)
    return targets[perm]


def main() -> None:
    args = _parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    gen = torch.Generator().manual_seed(args.seed)

    binary_logits = torch.randn(args.batch_size, 2, generator=gen)
    binary_targets = _coverage_targets(num_classes=2, batch_size=args.batch_size, gen=gen)
    binary_metrics = compute_binary_metrics(binary_logits=binary_logits, binary_targets=binary_targets)

    flat_logits = torch.randn(args.batch_size, 9, generator=gen)
    flat_targets = _coverage_targets(num_classes=9, batch_size=args.batch_size, gen=gen)
    flat_metrics = compute_flat9_metrics(flat_logits=flat_logits, flat_targets=flat_targets)
    flat_reliability = compute_reliability_metrics(
        logits=flat_logits,
        targets=flat_targets,
        n_bins=args.n_bins,
    )

    unsafe_logits = torch.randn(args.batch_size, 8, generator=gen)
    unsafe_targets = torch.randint(0, 8, (args.batch_size,), generator=gen)
    # Keep first 8 rows unsafe and class-covered so smoke metrics are stable.
    if args.batch_size >= 8:
        binary_targets[:8] = 1
        unsafe_targets[:8] = torch.arange(8, dtype=torch.long)
    safe_mask = binary_targets == 0
    unsafe_targets[safe_mask] = IGNORE_INDEX
    unsafe_metrics = compute_unsafe_subclass_metrics(
        unsafe_logits=unsafe_logits,
        unsafe_targets=unsafe_targets,
        ignore_index=IGNORE_INDEX,
    )

    print("binary_metrics")
    print(f"  accuracy={binary_metrics.accuracy:.6f}")
    print(f"  precision={binary_metrics.precision:.6f}")
    print(f"  recall={binary_metrics.recall:.6f}")
    print(f"  f1={binary_metrics.f1:.6f}")
    print(f"  auroc={binary_metrics.auroc:.6f}")
    print(f"  pr_auc={binary_metrics.pr_auc:.6f}")
    print(f"  brier={binary_metrics.brier:.6f}")

    print("flat9_metrics")
    print(f"  accuracy={flat_metrics.accuracy:.6f}")
    print(f"  balanced_accuracy={flat_metrics.balanced_accuracy:.6f}")
    print(f"  macro_f1={flat_metrics.macro_f1:.6f}")
    print(f"  weighted_f1={flat_metrics.weighted_f1:.6f}")
    print(f"  per_class_precision={flat_metrics.per_class_precision}")
    print(f"  per_class_recall={flat_metrics.per_class_recall}")
    print(f"  per_class_support={flat_metrics.per_class_support}")
    print(f"  confusion_matrix_shape=({len(flat_metrics.confusion_matrix)}, {len(flat_metrics.confusion_matrix[0])})")

    print("unsafe_subclass_metrics")
    print(f"  accuracy={unsafe_metrics.accuracy:.6f}")
    print(f"  balanced_accuracy={unsafe_metrics.balanced_accuracy:.6f}")
    print(f"  macro_f1={unsafe_metrics.macro_f1:.6f}")
    print(f"  weighted_f1={unsafe_metrics.weighted_f1:.6f}")
    print(f"  per_class_precision={unsafe_metrics.per_class_precision}")
    print(f"  per_class_recall={unsafe_metrics.per_class_recall}")
    print(f"  per_class_support={unsafe_metrics.per_class_support}")
    print(
        f"  confusion_matrix_shape=({len(unsafe_metrics.confusion_matrix)}, {len(unsafe_metrics.confusion_matrix[0])})"
    )

    print("reliability_metrics")
    print(f"  ece={flat_reliability.ece:.6f}")
    print(f"  brier={flat_reliability.brier:.6f}")
    print(f"  bins={len(flat_reliability.bin_counts)}")


if __name__ == "__main__":
    main()
