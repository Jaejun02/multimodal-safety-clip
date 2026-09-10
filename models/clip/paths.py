from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "clip"
SPLITS_DIR = OUTPUT_DIR / "splits"
CACHES_DIR = OUTPUT_DIR / "caches"
RUNS_DIR = OUTPUT_DIR / "runs"
CALIBRATION_DIR = OUTPUT_DIR / "calibration"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"


def rel_to_root(path: Path) -> str:
    """Return a stable project-root-relative path string."""
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
