from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

from .torch_runtime import cuda_available_safely


@dataclass(frozen=True)
class ClipBackbone:
    model_name: str
    pretrained: str
    device: torch.device
    model: torch.nn.Module
    preprocess: Callable
    tokenizer: Callable


def default_device() -> torch.device:
    return torch.device("cuda" if cuda_available_safely() else "cpu")


def load_clip_backbone(
    model_name: str,
    pretrained: str,
    device: str | torch.device | None = None,
) -> ClipBackbone:
    resolved_device = torch.device(device) if device is not None else default_device()

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=resolved_device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return ClipBackbone(
        model_name=model_name,
        pretrained=pretrained,
        device=resolved_device,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
    )


def preprocess_images(backbone: ClipBackbone, images: Sequence[Image.Image]) -> torch.Tensor:
    tensors = []
    for image in images:
        if image.mode == "P" and isinstance(image.info.get("transparency"), bytes):
            rgb_image = image.convert("RGBA").convert("RGB")
        else:
            rgb_image = image.convert("RGB")
        tensors.append(backbone.preprocess(rgb_image))
    return torch.stack(tensors, dim=0)


def encode_text_batch(
    backbone: ClipBackbone,
    texts: Sequence[str],
    normalize: bool = True,
) -> torch.Tensor:
    if not texts:
        raise ValueError("texts must not be empty")

    tokens = backbone.tokenizer(list(texts)).to(backbone.device)

    with torch.inference_mode():
        text_features = backbone.model.encode_text(tokens)

    if normalize:
        text_features = F.normalize(text_features, dim=-1)

    return text_features


def encode_image_batch(
    backbone: ClipBackbone,
    image_batch: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    if image_batch.ndim != 4:
        raise ValueError(
            f"image_batch must have shape [B, C, H, W], got {tuple(image_batch.shape)}"
        )

    image_batch = image_batch.to(backbone.device, non_blocking=True)

    with torch.inference_mode():
        image_features = backbone.model.encode_image(image_batch)

    if normalize:
        image_features = F.normalize(image_features, dim=-1)

    return image_features


def encode_multimodal_batch(
    backbone: ClipBackbone,
    texts: Sequence[str],
    image_batch: torch.Tensor,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    text_features = encode_text_batch(backbone=backbone, texts=texts, normalize=normalize)
    image_features = encode_image_batch(
        backbone=backbone,
        image_batch=image_batch,
        normalize=normalize,
    )
    return text_features, image_features


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenCLIP backbone loader and smoke check.")
    parser.add_argument("--model-name", type=str, default="ViT-B-16")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b88k")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--dry-run-load",
        action="store_true",
        help="Load frozen OpenCLIP model and print a small summary.",
    )
    parser.add_argument(
        "--debug-shapes",
        action="store_true",
        help=(
            "During --dry-run-load, run a tiny modality sanity check and print "
            "tokenizer/preprocess/feature shapes."
        ),
    )
    return parser


def _run_debug_shapes(backbone: ClipBackbone) -> None:
    sample_texts = ["safe example", "unsafe example"]
    sample_images = [
        Image.new("RGB", (64, 64), color=(128, 128, 128)),
        Image.new("RGB", (512, 320), color=(96, 96, 96)),
    ]

    tokenized = backbone.tokenizer(sample_texts)
    preprocessed = preprocess_images(backbone, sample_images)
    text_features = encode_text_batch(backbone, sample_texts, normalize=True)
    image_features = encode_image_batch(backbone, preprocessed, normalize=True)

    print("debug_text_count=", len(sample_texts))
    print("debug_tokenized_shape=", tuple(tokenized.shape))
    print("debug_preprocessed_shape=", tuple(preprocessed.shape))
    print("debug_text_features_shape=", tuple(text_features.shape))
    print("debug_image_features_shape=", tuple(image_features.shape))


def main() -> None:
    args = _parser().parse_args()

    if not args.dry_run_load:
        print("Nothing to run. Use --dry-run-load to load and verify backbone setup.")
        return

    backbone = load_clip_backbone(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
    )

    trainable_params = sum(p.numel() for p in backbone.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in backbone.model.parameters())

    print(f"model_name={backbone.model_name}")
    print(f"pretrained={backbone.pretrained}")
    print(f"device={backbone.device}")
    print(f"total_params={total_params}")
    print(f"trainable_params={trainable_params}")

    if args.debug_shapes:
        _run_debug_shapes(backbone)


if __name__ == "__main__":
    main()
