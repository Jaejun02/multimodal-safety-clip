from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

FusionType = Literal["concat", "interaction", "interaction_only"]
HeadType = Literal["linear", "small", "medium", "large"]
OutputMode = Literal["flat", "hierarchical"]


class BaseFusion(nn.Module):
    output_dim: int


class ConcatFusion(BaseFusion):
    def __init__(self, text_dim: int, image_dim: int) -> None:
        super().__init__()
        self.output_dim = text_dim + image_dim

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        return torch.cat([text_feat, image_feat], dim=-1)


class InteractionFusion(BaseFusion):
    def __init__(self, text_dim: int, image_dim: int, proj_dim: int = 512) -> None:
        super().__init__()
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.image_proj = nn.Linear(image_dim, proj_dim)
        self.output_dim = proj_dim * 4

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        text_norm = F.normalize(text_feat, dim=-1)
        image_norm = F.normalize(image_feat, dim=-1)

        t_proj = self.text_proj(text_norm)
        i_proj = self.image_proj(image_norm)

        return torch.cat(
            [
                t_proj,
                i_proj,
                torch.abs(t_proj - i_proj),
                t_proj * i_proj,
            ],
            dim=-1,
        )


class InteractionOnlyFusion(BaseFusion):
    def __init__(self, text_dim: int, image_dim: int, proj_dim: int = 512) -> None:
        super().__init__()
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.image_proj = nn.Linear(image_dim, proj_dim)
        self.output_dim = proj_dim * 2

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        text_norm = F.normalize(text_feat, dim=-1)
        image_norm = F.normalize(image_feat, dim=-1)

        t_proj = self.text_proj(text_norm)
        i_proj = self.image_proj(image_norm)

        return torch.cat(
            [
                torch.abs(t_proj - i_proj),
                t_proj * i_proj,
            ],
            dim=-1,
        )


@dataclass(frozen=True)
class HeadSpec:
    trunk: nn.Module
    hidden_dim: int


def build_trunk(head_type: HeadType, in_dim: int) -> HeadSpec:
    if head_type == "linear":
        return HeadSpec(trunk=nn.Identity(), hidden_dim=in_dim)

    if head_type == "small":
        return HeadSpec(
            trunk=nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 512),
                nn.GELU(),
                nn.Dropout(0.2),
            ),
            hidden_dim=512,
        )

    if head_type == "medium":
        return HeadSpec(
            trunk=nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 1024),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Dropout(0.2),
            ),
            hidden_dim=512,
        )

    if head_type == "large":
        return HeadSpec(
            trunk=nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 2048),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Dropout(0.3),
            ),
            hidden_dim=512,
        )

    raise ValueError(f"Unsupported head_type: {head_type}")


def build_fusion(
    fusion_type: FusionType,
    text_dim: int,
    image_dim: int,
    proj_dim: int = 512,
) -> BaseFusion:
    if fusion_type == "concat":
        return ConcatFusion(text_dim=text_dim, image_dim=image_dim)
    if fusion_type == "interaction":
        return InteractionFusion(text_dim=text_dim, image_dim=image_dim, proj_dim=proj_dim)
    if fusion_type == "interaction_only":
        return InteractionOnlyFusion(text_dim=text_dim, image_dim=image_dim, proj_dim=proj_dim)
    raise ValueError(f"Unsupported fusion_type: {fusion_type}")


class FusionClassifier(nn.Module):
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        fusion_type: FusionType = "interaction",
        head_type: HeadType = "medium",
        output_mode: OutputMode = "hierarchical",
        proj_dim: int = 512,
    ) -> None:
        super().__init__()
        self.output_mode = output_mode
        self.fusion = build_fusion(
            fusion_type=fusion_type,
            text_dim=text_dim,
            image_dim=image_dim,
            proj_dim=proj_dim,
        )
        self.fused_dim = self.fusion.output_dim

        head_spec = build_trunk(head_type=head_type, in_dim=self.fused_dim)
        self.trunk = head_spec.trunk
        hidden_dim = head_spec.hidden_dim

        if output_mode == "flat":
            self.flat_head = nn.Linear(hidden_dim, 9)
            self.binary_head = None
            self.unsafe_head = None
        elif output_mode == "hierarchical":
            self.binary_head = nn.Linear(hidden_dim, 2)
            self.unsafe_head = nn.Linear(hidden_dim, 8)
            self.flat_head = None
        else:
            raise ValueError(f"Unsupported output_mode: {output_mode}")

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor):
        fused = self.fusion(text_feat, image_feat)
        hidden = self.trunk(fused)

        if self.output_mode == "flat":
            return {"flat_logits": self.flat_head(hidden)}

        return {
            "binary_logits": self.binary_head(hidden),
            "unsafe_logits": self.unsafe_head(hidden),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fusion/head module shape smoke test.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--text-dim", type=int, default=512)
    parser.add_argument("--image-dim", type=int, default=512)
    parser.add_argument(
        "--fusion-type",
        choices=["concat", "interaction", "interaction_only"],
        default="interaction",
    )
    parser.add_argument("--head-type", choices=["linear", "small", "medium", "large"], default="medium")
    parser.add_argument("--output-mode", choices=["flat", "hierarchical"], default="hierarchical")
    parser.add_argument("--proj-dim", type=int, default=512)
    return parser


def main() -> None:
    args = _parser().parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    model = FusionClassifier(
        text_dim=args.text_dim,
        image_dim=args.image_dim,
        fusion_type=args.fusion_type,
        head_type=args.head_type,
        output_mode=args.output_mode,
        proj_dim=args.proj_dim,
    )

    text_feat = torch.randn(args.batch_size, args.text_dim)
    image_feat = torch.randn(args.batch_size, args.image_dim)

    outputs = model(text_feat=text_feat, image_feat=image_feat)

    print(f"fusion_type={args.fusion_type}")
    print(f"head_type={args.head_type}")
    print(f"output_mode={args.output_mode}")
    print(f"fused_dim={model.fused_dim}")

    for key, value in outputs.items():
        print(f"{key}_shape={tuple(value.shape)}")


if __name__ == "__main__":
    main()
