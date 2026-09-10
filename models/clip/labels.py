from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

REDUCED_CATEGORY_MAP = {
    "C1: Slurs, Hate Speech, Hate Symbols": "Hate, Harassment & Discrimination",
    "C2: Discrimination and Unequal Treatment": "Hate, Harassment & Discrimination",
    "C3: Drug Abuse": "Substance Abuse & Addictive Behavior",
    "C4: Self-Harm and Suicide": "Self-Harm & Suicide",
    "C5: Animal Violence and Gore": "Violence, Weapons & Gore",
    "C6: Adult Explicit Sexual Material": "Sexual Content",
    "C7: Adult Racy Material": "Sexual Content",
    "C8: Warfare and Armed Conflicts": "Conflict, Terrorism & Extremism",
    "C9: Interpersonal Violence": "Violence, Weapons & Gore",
    "C10: Weapons and Dangerous Objects": "Violence, Weapons & Gore",
    "C11: Gore and Graphic Content": "Violence, Weapons & Gore",
    "C12: Terrorism and Violent Extremism": "Conflict, Terrorism & Extremism",
    "C13: Jailbreaks": "Platform Abuse, Fraud & Evasion",
    "C14: Inauthentic Practices/Fraud": "Platform Abuse, Fraud & Evasion",
    "C15: Human Exploitation": "Exploitation & Abuse",
}

REDUCED_CLASSES = [
    "Hate, Harassment & Discrimination",
    "Substance Abuse & Addictive Behavior",
    "Self-Harm & Suicide",
    "Violence, Weapons & Gore",
    "Sexual Content",
    "Conflict, Terrorism & Extremism",
    "Platform Abuse, Fraud & Evasion",
    "Exploitation & Abuse",
]

SAFE_LABEL = 0
UNSAFE_LABEL = 1
IGNORE_INDEX = -100


def _clean_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def apply_label_rules(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work["consensus_combined_grade"] = work["consensus_combined_grade"].map(_clean_str)
    work["combined_category"] = work["combined_category"].map(_clean_str)

    work = work[work["consensus_combined_grade"].str.lower() != "borderline"].copy()

    valid_binary = {"safe", "unsafe"}
    bad_binary = sorted(
        set(work["consensus_combined_grade"].str.lower().unique()) - valid_binary
    )
    if bad_binary:
        raise ValueError(
            "Found invalid consensus_combined_grade values after filtering borderline: "
            f"{bad_binary}"
        )

    work["binary_label"] = work["consensus_combined_grade"].str.lower().map(
        {"safe": SAFE_LABEL, "unsafe": UNSAFE_LABEL}
    )

    reduced_to_index = {name: idx for idx, name in enumerate(REDUCED_CLASSES)}

    def map_unsafe_subclass(row: pd.Series) -> int:
        if int(row["binary_label"]) == SAFE_LABEL:
            return IGNORE_INDEX

        raw_category = _clean_str(row["combined_category"])
        if not raw_category:
            return IGNORE_INDEX

        reduced_name = REDUCED_CATEGORY_MAP.get(raw_category)
        if reduced_name is None:
            raise ValueError(
                f"Unsafe row id={row.get('id')} has unmapped combined_category: {raw_category}"
            )

        return reduced_to_index[reduced_name]

    work["unsafe_subclass_label"] = work.apply(map_unsafe_subclass, axis=1).astype(int)

    unresolved_unsafe_mask = (
        (work["binary_label"] == UNSAFE_LABEL)
        & (work["unsafe_subclass_label"] == IGNORE_INDEX)
    )
    dropped_unsafe = int(unresolved_unsafe_mask.sum())
    work = work.loc[~unresolved_unsafe_mask].copy()
    work["dropped_unsafe_rows"] = dropped_unsafe

    work["composite_9way_label"] = np.where(
        work["binary_label"] == SAFE_LABEL,
        0,
        work["unsafe_subclass_label"] + 1,
    ).astype(int)

    work["reduced_unsafe_class"] = np.where(
        work["unsafe_subclass_label"] == IGNORE_INDEX,
        "",
        [REDUCED_CLASSES[idx] for idx in work["unsafe_subclass_label"].clip(lower=0)],
    )

    return work


def class_counts(series: pd.Series) -> Dict[int, int]:
    counts = series.value_counts().sort_index()
    return {int(k): int(v) for k, v in counts.items()}


def inverse_frequency_weights(labels: Iterable[int], num_classes: int) -> np.ndarray:
    labels_arr = np.array(list(labels), dtype=int)
    counts = np.bincount(labels_arr, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Cannot compute inverse-frequency weights; missing classes: {missing}")

    weights = 1.0 / counts
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def effective_number_weights(
    labels: Iterable[int], num_classes: int, beta: float = 0.999
) -> np.ndarray:
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0,1), got {beta}")

    labels_arr = np.array(list(labels), dtype=int)
    counts = np.bincount(labels_arr, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Cannot compute effective-number weights; missing classes: {missing}")

    effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    weights = 1.0 / effective_num
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def label_maps_payload() -> Dict[str, object]:
    return {
        "safe_label": SAFE_LABEL,
        "unsafe_label": UNSAFE_LABEL,
        "ignore_index": IGNORE_INDEX,
        "reduced_category_map": REDUCED_CATEGORY_MAP,
        "reduced_classes": REDUCED_CLASSES,
    }
