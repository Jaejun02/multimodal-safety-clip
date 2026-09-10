from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .labels import IGNORE_INDEX, effective_number_weights, inverse_frequency_weights

WeightStrategy = Literal["none", "inverse", "effective"]


@dataclass(frozen=True)
class HierarchicalLossOutput:
    total_loss: torch.Tensor
    binary_loss: torch.Tensor
    unsafe_loss: torch.Tensor


def _to_numpy_int(labels: Iterable[int]) -> np.ndarray:
    return np.array(list(labels), dtype=int)


def build_class_weights(
    labels: Iterable[int],
    num_classes: int,
    strategy: WeightStrategy = "effective",
    beta: float = 0.999,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    if strategy == "none":
        return None

    labels_arr = _to_numpy_int(labels)
    if labels_arr.size == 0:
        raise ValueError("Cannot build class weights from empty labels")

    if strategy == "inverse":
        weights = inverse_frequency_weights(labels_arr, num_classes=num_classes)
    elif strategy == "effective":
        weights = effective_number_weights(labels_arr, num_classes=num_classes, beta=beta)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    weight_t = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        weight_t = weight_t.to(device)
    return weight_t


def build_binary_class_weights(
    binary_labels: Iterable[int],
    strategy: WeightStrategy = "effective",
    beta: float = 0.999,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    return build_class_weights(
        labels=binary_labels,
        num_classes=2,
        strategy=strategy,
        beta=beta,
        device=device,
    )


def build_unsafe_class_weights(
    unsafe_labels: Iterable[int],
    strategy: WeightStrategy = "effective",
    beta: float = 0.999,
    device: torch.device | str | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor | None:
    labels_arr = _to_numpy_int(unsafe_labels)
    labels_arr = labels_arr[labels_arr != ignore_index]
    return build_class_weights(
        labels=labels_arr,
        num_classes=8,
        strategy=strategy,
        beta=beta,
        device=device,
    )


def build_flat9_class_weights(
    flat9_labels: Iterable[int],
    strategy: WeightStrategy = "effective",
    beta: float = 0.999,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    return build_class_weights(
        labels=flat9_labels,
        num_classes=9,
        strategy=strategy,
        beta=beta,
        device=device,
    )


def _focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None,
    gamma: float,
    ignore_index: int | None = None,
) -> torch.Tensor:
    if gamma < 0:
        raise ValueError(f"focal gamma must be >= 0, got {gamma}")

    ce_kwargs: dict[str, object] = {"weight": class_weights, "reduction": "none"}
    if ignore_index is not None:
        ce_kwargs["ignore_index"] = int(ignore_index)

    ce = F.cross_entropy(logits, targets.long(), **ce_kwargs)
    pt = torch.exp(-ce)
    focal = ((1.0 - pt) ** gamma) * ce

    if ignore_index is None:
        return focal.mean()

    valid = targets.long() != int(ignore_index)
    if not torch.any(valid):
        return ce.new_tensor(0.0)
    return focal[valid].mean()


def compute_flat_loss(
    flat_logits: torch.Tensor,
    flat_targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    loss_type: Literal["ce", "focal"] = "ce",
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    if loss_type == "ce":
        return F.cross_entropy(flat_logits, flat_targets.long(), weight=class_weights)
    if loss_type == "focal":
        return _focal_cross_entropy(
            logits=flat_logits,
            targets=flat_targets,
            class_weights=class_weights,
            gamma=focal_gamma,
            ignore_index=None,
        )
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def compute_hierarchical_loss(
    binary_logits: torch.Tensor,
    unsafe_logits: torch.Tensor,
    binary_targets: torch.Tensor,
    unsafe_targets: torch.Tensor,
    lambda_unsafe: float = 1.0,
    binary_class_weights: torch.Tensor | None = None,
    unsafe_class_weights: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    loss_type: Literal["ce", "focal"] = "ce",
    focal_gamma: float = 2.0,
) -> HierarchicalLossOutput:
    if lambda_unsafe < 0:
        raise ValueError(f"lambda_unsafe must be >= 0, got {lambda_unsafe}")

    if loss_type == "ce":
        binary_loss = F.cross_entropy(
            binary_logits,
            binary_targets.long(),
            weight=binary_class_weights,
        )
        unsafe_loss = F.cross_entropy(
            unsafe_logits,
            unsafe_targets.long(),
            weight=unsafe_class_weights,
            ignore_index=ignore_index,
        )
    elif loss_type == "focal":
        binary_loss = _focal_cross_entropy(
            logits=binary_logits,
            targets=binary_targets,
            class_weights=binary_class_weights,
            gamma=focal_gamma,
            ignore_index=None,
        )
        unsafe_loss = _focal_cross_entropy(
            logits=unsafe_logits,
            targets=unsafe_targets,
            class_weights=unsafe_class_weights,
            gamma=focal_gamma,
            ignore_index=ignore_index,
        )
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    total_loss = binary_loss + (lambda_unsafe * unsafe_loss)

    return HierarchicalLossOutput(
        total_loss=total_loss,
        binary_loss=binary_loss,
        unsafe_loss=unsafe_loss,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Loss module smoke test.")
    parser.add_argument("--output-mode", choices=["flat", "hierarchical"], default="hierarchical")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--class-weighting", choices=["none", "inverse", "effective"], default="effective")
    parser.add_argument("--beta", type=float, default=0.999)
    parser.add_argument("--lambda-unsafe", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
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

    if args.output_mode == "flat":
        logits = torch.randn(args.batch_size, 9, generator=gen)
        targets = _coverage_targets(num_classes=9, batch_size=args.batch_size, gen=gen)
        weights = build_flat9_class_weights(
            targets.tolist(),
            strategy=args.class_weighting,
            beta=args.beta,
            device=logits.device,
        )
        loss = compute_flat_loss(logits, targets, class_weights=weights)
        print(f"output_mode=flat")
        print(f"class_weighting={args.class_weighting}")
        print(f"flat_logits_shape={tuple(logits.shape)}")
        print(f"flat_loss={float(loss):.6f}")
        return

    binary_logits = torch.randn(args.batch_size, 2, generator=gen)
    unsafe_logits = torch.randn(args.batch_size, 8, generator=gen)

    binary_targets = _coverage_targets(num_classes=2, batch_size=args.batch_size, gen=gen)
    unsafe_targets = torch.randint(0, 8, (args.batch_size,), generator=gen)

    # Keep the first 8 samples unsafe so unsafe class weights can cover all 8 classes.
    binary_targets[:8] = 1
    unsafe_targets[:8] = torch.arange(8, dtype=torch.long)
    safe_mask = binary_targets == 0
    unsafe_targets[safe_mask] = IGNORE_INDEX

    binary_weights = build_binary_class_weights(
        binary_targets.tolist(),
        strategy=args.class_weighting,
        beta=args.beta,
        device=binary_logits.device,
    )
    unsafe_weights = build_unsafe_class_weights(
        unsafe_targets.tolist(),
        strategy=args.class_weighting,
        beta=args.beta,
        device=unsafe_logits.device,
        ignore_index=IGNORE_INDEX,
    )

    out = compute_hierarchical_loss(
        binary_logits=binary_logits,
        unsafe_logits=unsafe_logits,
        binary_targets=binary_targets,
        unsafe_targets=unsafe_targets,
        lambda_unsafe=args.lambda_unsafe,
        binary_class_weights=binary_weights,
        unsafe_class_weights=unsafe_weights,
        ignore_index=IGNORE_INDEX,
    )

    print("output_mode=hierarchical")
    print(f"class_weighting={args.class_weighting}")
    print(f"binary_logits_shape={tuple(binary_logits.shape)}")
    print(f"unsafe_logits_shape={tuple(unsafe_logits.shape)}")
    print(f"binary_loss={float(out.binary_loss):.6f}")
    print(f"unsafe_loss={float(out.unsafe_loss):.6f}")
    print(f"total_loss={float(out.total_loss):.6f}")


if __name__ == "__main__":
    main()
