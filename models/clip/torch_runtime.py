from __future__ import annotations

import warnings

import torch


def cuda_available_safely() -> bool:
    """Return CUDA availability without surfacing known driver mismatch warnings."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="CUDA initialization: The NVIDIA driver on your system is too old.*",
                category=UserWarning,
            )
            return bool(torch.cuda.is_available())
    except Exception:
        return False
