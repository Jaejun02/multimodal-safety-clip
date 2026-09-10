from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_serializable_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        raw = asdict(obj)
    elif isinstance(obj, dict):
        raw = dict(obj)
    else:
        raise TypeError(f"Unsupported type for serialization: {type(obj)}")

    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=True)


def write_config_snapshot(path: Path, config: Any) -> None:
    payload = {
        "created_at_utc": utc_now_iso(),
        "config": to_serializable_dict(config),
    }
    write_json(path, payload)
