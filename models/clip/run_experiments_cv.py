from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from .paths import ANALYSIS_DIR, CALIBRATION_DIR, OUTPUT_DIR, PROJECT_ROOT, RUNS_DIR, SPLITS_DIR
from .utils import ensure_dir, utc_now_iso, write_json


def _utc_now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_token(value: str) -> str:
    return value.replace("/", "-").replace("\\", "-").replace(":", "-").replace(" ", "_")


def _stream_command(cmd: list[str], log_path: Path) -> int:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"$ {' '.join(cmd)}\n")
        logf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        return proc.wait()


def _run(cmd: list[str], dry_run: bool, log_path: Path | None = None) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return

    if log_path is None:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        return

    rc = _stream_command(cmd, log_path=log_path)
    if rc != 0:
        raise subprocess.CalledProcessError(returncode=rc, cmd=cmd)


def _contains_oom(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    patterns = [
        "out of memory",
        "cuda out of memory",
        "cudnn_status_alloc_failed",
        "resource exhausted",
    ]
    return any(p in text for p in patterns)


def _replace_arg(cmd: list[str], arg_name: str, arg_value: str) -> list[str]:
    out = list(cmd)
    if arg_name not in out:
        raise ValueError(f"Argument not found in command: {arg_name}")
    idx = out.index(arg_name)
    if idx + 1 >= len(out):
        raise ValueError(f"Missing value for argument: {arg_name}")
    out[idx + 1] = arg_value
    return out


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_seed_list(seed_list_raw: str) -> list[int]:
    values: list[int] = []
    for token in seed_list_raw.split(","):
        item = token.strip()
        if item == "":
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid seed in seed list: '{item}'") from exc
    if len(values) == 0:
        raise ValueError("Seed list must contain at least one integer.")
    return values


def _nan_safe_mean(values: Iterable[float]) -> float:
    usable = [float(v) for v in values if not math.isnan(float(v))]
    if len(usable) == 0:
        return float("nan")
    return float(statistics.fmean(usable))


def _nan_safe_pstdev(values: Iterable[float]) -> float:
    usable = [float(v) for v in values if not math.isnan(float(v))]
    if len(usable) <= 1:
        return 0.0
    return float(statistics.pstdev(usable))


def _model_short(model_name: str) -> str:
    mapping = {
        "ViT-B-16": "B16",
        "ViT-L-14": "L14",
    }
    return mapping.get(model_name, _safe_token(model_name))


def _pretrained_short(pretrained: str) -> str:
    mapping = {
        "laion2b_s34b_b88k": "laion_b88k",
        "laion2b_s32b_b82k": "laion_b82k",
        "datacomp_xl_s13b_b90k": "datacomp_xl",
    }
    return mapping.get(pretrained, _safe_token(pretrained))


@dataclass(frozen=True)
class MethodSpec:
    model_name: str
    pretrained: str
    fusion: str
    head_type: str
    output_mode: str
    loss_type: str
    class_weighting: str
    focal_gamma: float
    lambda_unsafe: float

    def canonical(self) -> "MethodSpec":
        focal_gamma = self.focal_gamma if self.loss_type == "focal" else 0.0
        lambda_unsafe = self.lambda_unsafe if self.output_mode == "hierarchical" else 1.0
        return MethodSpec(
            model_name=self.model_name,
            pretrained=self.pretrained,
            fusion=self.fusion,
            head_type=self.head_type,
            output_mode=self.output_mode,
            loss_type=self.loss_type,
            class_weighting=self.class_weighting,
            focal_gamma=round(float(focal_gamma), 4),
            lambda_unsafe=round(float(lambda_unsafe), 4),
        )

    def signature(self) -> str:
        spec = self.canonical()
        parts = [
            _model_short(spec.model_name),
            _pretrained_short(spec.pretrained),
            f"fu-{spec.fusion}",
            f"hd-{spec.head_type}",
            f"om-{spec.output_mode}",
            f"ls-{spec.loss_type}",
            f"cw-{spec.class_weighting}",
        ]
        if spec.loss_type == "focal":
            parts.append(f"fg-{str(spec.focal_gamma).replace('.', 'p')}")
        if spec.output_mode == "hierarchical":
            parts.append(f"lu-{str(spec.lambda_unsafe).replace('.', 'p')}")
        return "__".join(parts)

    def describe(self) -> str:
        spec = self.canonical()
        return (
            f"{spec.model_name}/{spec.pretrained} | fusion={spec.fusion} | head={spec.head_type} | "
            f"output={spec.output_mode} | loss={spec.loss_type} | class_weighting={spec.class_weighting} | "
            f"focal_gamma={spec.focal_gamma:.2f} | lambda_unsafe={spec.lambda_unsafe:.2f}"
        )


@dataclass(frozen=True)
class BeamNode:
    spec: MethodSpec
    lineage: tuple[str, ...]


@dataclass(frozen=True)
class CVRunResult:
    stage: str
    spec_signature: str
    run_name: str
    run_dir: str
    fold_index: int
    seed: int
    val_macro_f1: float
    val_weighted_f1: float
    val_binary_f1: float | None
    val_unsafe_macro_f1: float | None
    val_metrics_path: str


@dataclass(frozen=True)
class CandidateAggregate:
    stage: str
    spec_signature: str
    model_name: str
    pretrained: str
    fusion: str
    head_type: str
    output_mode: str
    loss_type: str
    class_weighting: str
    focal_gamma: float
    lambda_unsafe: float
    fold_count: int
    seed_values: list[int]
    run_count: int
    val_macro_f1_mean: float
    val_macro_f1_std: float
    val_weighted_f1_mean: float
    val_weighted_f1_std: float
    val_binary_f1_mean: float | None
    val_unsafe_macro_f1_mean: float | None
    best_run_name: str
    best_val_macro_f1: float
    best_val_weighted_f1: float

    def ranking_key(self) -> tuple[float, float, float, float]:
        return (
            self.val_macro_f1_mean,
            self.val_weighted_f1_mean,
            -self.val_macro_f1_std,
            -self.val_weighted_f1_std,
        )


@dataclass(frozen=True)
class FinalRunResult:
    finalist_rank: int
    spec_signature: str
    run_name: str
    run_dir: str
    seed: int
    val_macro_f1: float
    val_weighted_f1: float
    test_macro_f1: float
    test_weighted_f1: float
    val_metrics_path: str
    test_metrics_path: str


@dataclass(frozen=True)
class StageBlueprint:
    name: str
    description: str
    expander: Callable[[BeamNode], list[BeamNode]]
    seed_values: list[int]
    beam_width: int
    beam_tolerance: float
    max_keep: int


class ExperimentRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.python = sys.executable
        self.timestamp = _utc_now_token()
        self.plan_tag = args.plan_tag or self.timestamp

        self.analysis_root = ANALYSIS_DIR / "model_search_cv"
        self.stage_dir = self.analysis_root / "stages" / self.plan_tag
        self.logs_dir = self.analysis_root / "logs" / self.plan_tag
        self.cv_cache_dir = OUTPUT_DIR / "cv_caches" / self.plan_tag
        self.calibration_root = CALIBRATION_DIR / self.plan_tag
        self.summary_path = self.analysis_root / f"summary_{self.plan_tag}.json"
        self.data_profile_path = self.analysis_root / f"data_profile_{self.plan_tag}.json"

        ensure_dir(self.analysis_root)
        ensure_dir(self.stage_dir)
        ensure_dir(self.logs_dir)
        ensure_dir(self.cv_cache_dir)
        ensure_dir(self.calibration_root)

        self.search_seed_values = _parse_seed_list(self.args.seed_list)
        self.robust_seed_values = _parse_seed_list(self.args.robust_seed_list)
        self.final_fit_seed_values = _parse_seed_list(self.args.final_fit_seed_list)

        self.scout_backbone = ("ViT-B-16", "laion2b_s34b_b88k")
        self.backbone_grid = [
            ("ViT-B-16", "laion2b_s34b_b88k"),
            ("ViT-B-16", "datacomp_xl_s13b_b90k"),
            ("ViT-L-14", "laion2b_s32b_b82k"),
            ("ViT-L-14", "datacomp_xl_s13b_b90k"),
        ]

        self.results: list[CVRunResult] = []
        self.aggregates: list[dict[str, Any]] = []
        self.stage_summaries: list[dict[str, Any]] = []
        self.finalists: list[dict[str, Any]] = []
        self._prepared_caches: set[tuple[str, str]] = set()
        self._prepared_cv_caches: set[tuple[str, str]] = set()
        self._source_train_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, pd.DataFrame]] = {}
        self._fold_manifest: list[dict[str, Any]] = []

    def _train_batch_size_for_model(self, model_name: str) -> int:
        upper = model_name.upper()
        if ("L-14" in upper) or ("VIT-L" in upper):
            return int(self.args.batch_size_l14)
        return int(self.args.batch_size)

    def _cache_batch_size_for_model(self, model_name: str) -> int:
        upper = model_name.upper()
        if ("L-14" in upper) or ("VIT-L" in upper):
            return int(self.args.cache_batch_size_l14)
        return int(self.args.cache_batch_size)

    def _run_with_oom_retry(
        self,
        cmd: list[str],
        log_path: Path,
        batch_arg: str,
        start_batch: int,
        min_batch: int,
    ) -> int:
        if log_path.exists() and not self.args.dry_run:
            log_path.unlink()

        batch = int(start_batch)
        attempt = 1
        while True:
            run_cmd = _replace_arg(cmd, batch_arg, str(batch))
            try:
                if not self.args.dry_run:
                    ensure_dir(log_path.parent)
                    with log_path.open("a", encoding="utf-8") as logf:
                        logf.write(f"oom_retry_attempt={attempt} {batch_arg}={batch}\n")
                _run(run_cmd, dry_run=self.args.dry_run, log_path=log_path)
                return batch
            except subprocess.CalledProcessError:
                if self.args.dry_run:
                    raise
                if (not self.args.oom_retry) or (not _contains_oom(log_path)):
                    raise
                next_batch = batch // 2
                if next_batch < min_batch:
                    raise
                print(
                    f"oom_retry=attempt{attempt} arg={batch_arg} batch={batch} -> retry_with={next_batch}"
                )
                batch = next_batch
                attempt += 1

    def clear_output(self) -> None:
        if self.args.clear_scope == "none":
            return

        targets: list[Path] = [
            self.stage_dir,
            self.logs_dir,
            self.cv_cache_dir,
            self.calibration_root,
        ]
        if self.summary_path.exists():
            targets.append(self.summary_path)
        if self.data_profile_path.exists():
            targets.append(self.data_profile_path)

        for run_dir in RUNS_DIR.glob(f"{self.plan_tag}__*"):
            targets.append(run_dir)

        for target in targets:
            if not target.exists():
                continue
            if target.is_dir():
                print(f"clear-scope=plan removing {target}")
                if not self.args.dry_run:
                    shutil.rmtree(target)
            else:
                print(f"clear-scope=plan removing {target}")
                if not self.args.dry_run:
                    target.unlink()

        ensure_dir(self.stage_dir)
        ensure_dir(self.logs_dir)
        ensure_dir(self.cv_cache_dir)
        ensure_dir(self.calibration_root)

    def ensure_splits(self) -> None:
        if not self.args.bootstrap_split:
            return

        required = [SPLITS_DIR / f"{name}.csv" for name in (self.args.train_split, self.args.val_split, self.args.test_split)]
        if all(p.exists() for p in required):
            print("bootstrap_split=skip (required split CSVs already exist)")
            return

        cmd = [
            self.python,
            "-m",
            "models.clip.split_data",
            "--data-csv",
            str(self.args.data_csv),
            "--images-dir",
            str(self.args.images_dir),
            "--out-dir",
            str(SPLITS_DIR),
            "--seed",
            str(self.args.base_seed),
            "--train-ratio",
            str(self.args.train_ratio),
            "--val-ratio",
            str(self.args.val_ratio),
            "--test-ratio",
            str(self.args.test_ratio),
            "--min-class-count",
            str(self.args.min_class_count),
        ]
        split_log = self.logs_dir / "bootstrap__split_data.log"
        _run(cmd, dry_run=self.args.dry_run, log_path=split_log)

    def ensure_cache_for(self, model_name: str, pretrained: str) -> None:
        if not self.args.bootstrap_cache:
            return

        key = (model_name, pretrained)
        if key in self._prepared_caches:
            return

        safe_model = _safe_token(model_name)
        safe_pretrained = _safe_token(pretrained)
        cache_root = Path(self.args.cache_dir) / f"{safe_model}__{safe_pretrained}"

        all_present = True
        for split_name in (self.args.train_split, self.args.val_split, self.args.test_split):
            if not (cache_root / split_name / "text_features.pt").exists() or not (
                cache_root / split_name / "image_features.pt"
            ).exists():
                all_present = False
                break

        if all_present:
            print(f"bootstrap_cache=skip ({model_name}, {pretrained}) already cached")
            self._prepared_caches.add(key)
            return

        cmd = [
            self.python,
            "-m",
            "models.clip.cache_embeddings",
            "--splits-dir",
            str(SPLITS_DIR),
            "--out-dir",
            str(self.args.cache_dir),
            "--images-dir",
            str(self.args.images_dir),
            "--model-name",
            model_name,
            "--pretrained",
            pretrained,
            "--batch-size",
            str(self._cache_batch_size_for_model(model_name)),
            "--split-names",
            self.args.train_split,
            self.args.val_split,
            self.args.test_split,
        ]
        cache_log = self.logs_dir / f"bootstrap__cache__{safe_model}__{safe_pretrained}.log"
        self._run_with_oom_retry(
            cmd=cmd,
            log_path=cache_log,
            batch_arg="--batch-size",
            start_batch=self._cache_batch_size_for_model(model_name),
            min_batch=self.args.min_cache_batch_size,
        )
        self._prepared_caches.add(key)

    def _build_fold_manifest(self) -> list[dict[str, Any]]:
        if len(self._fold_manifest) > 0:
            return self._fold_manifest

        train_csv = SPLITS_DIR / f"{self.args.train_split}.csv"
        if not train_csv.exists():
            raise FileNotFoundError(f"Missing train split CSV: {train_csv}")

        train_df = pd.read_csv(train_csv)
        if "composite_9way_label" not in train_df.columns:
            raise ValueError(f"Train split CSV missing composite_9way_label: {train_csv}")

        splitter = StratifiedKFold(
            n_splits=self.args.cv_folds,
            shuffle=True,
            random_state=self.args.cv_seed,
        )

        manifest: list[dict[str, Any]] = []
        labels = train_df["composite_9way_label"].to_numpy()
        ids = train_df["id"].tolist()
        for fold_index, (train_idx, val_idx) in enumerate(splitter.split(ids, labels), start=1):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]
            manifest.append(
                {
                    "fold_index": fold_index,
                    "train_indices": [int(i) for i in train_idx.tolist()],
                    "val_indices": [int(i) for i in val_idx.tolist()],
                    "train_counts": fold_train["composite_9way_label"].value_counts().sort_index().astype(int).to_dict(),
                    "val_counts": fold_val["composite_9way_label"].value_counts().sort_index().astype(int).to_dict(),
                    "train_size": int(len(fold_train)),
                    "val_size": int(len(fold_val)),
                }
            )

        self._fold_manifest = manifest
        return manifest

    def _write_data_profile(self) -> dict[str, Any]:
        fold_manifest = self._build_fold_manifest()
        split_counts_path = SPLITS_DIR / "split_counts.json"
        split_counts = _read_json(split_counts_path) if split_counts_path.exists() else {}
        payload = {
            "created_at_utc": utc_now_iso(),
            "plan_tag": self.plan_tag,
            "split_counts": split_counts,
            "cv_folds": self.args.cv_folds,
            "cv_seed": self.args.cv_seed,
            "fold_manifest": fold_manifest,
        }
        if not self.args.dry_run:
            write_json(self.data_profile_path, payload)
        return payload

    def _load_source_train_cache(self, model_name: str, pretrained: str) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        key = (model_name, pretrained)
        if key in self._source_train_cache:
            return self._source_train_cache[key]

        safe_model = _safe_token(model_name)
        safe_pretrained = _safe_token(pretrained)
        split_dir = Path(self.args.cache_dir) / f"{safe_model}__{safe_pretrained}" / self.args.train_split
        text_features = torch.load(split_dir / "text_features.pt", map_location="cpu").float()
        image_features = torch.load(split_dir / "image_features.pt", map_location="cpu").float()
        labels_df = pd.read_csv(split_dir / "labels_snapshot.csv")

        train_csv = pd.read_csv(SPLITS_DIR / f"{self.args.train_split}.csv")
        if "id" in labels_df.columns and "id" in train_csv.columns:
            cache_ids = labels_df["id"].tolist()
            split_ids = train_csv["id"].tolist()
            if cache_ids != split_ids:
                raise ValueError(
                    "Cached train labels are not aligned with split train.csv. "
                    f"cache_root={split_dir.parent}"
                )

        self._source_train_cache[key] = (text_features, image_features, labels_df)
        return self._source_train_cache[key]

    def _write_fold_split_artifacts(
        self,
        split_dir: Path,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        labels_df: pd.DataFrame,
        split_name: str,
        model_name: str,
        pretrained: str,
        fold_index: int,
    ) -> None:
        ensure_dir(split_dir)
        torch.save(text_features, split_dir / "text_features.pt")
        torch.save(image_features, split_dir / "image_features.pt")
        labels_df.to_csv(split_dir / "labels_snapshot.csv", index=False)
        if "id" in labels_df.columns:
            labels_df[["id"]].to_csv(split_dir / "ids.csv", index=False)

        metadata = {
            "created_at_utc": utc_now_iso(),
            "split_name": split_name,
            "fold_index": int(fold_index),
            "model_name": model_name,
            "pretrained": pretrained,
            "rows": int(len(labels_df)),
            "text_feature_dim": int(text_features.shape[1]),
            "image_feature_dim": int(image_features.shape[1]),
        }
        write_json(split_dir / "cache_metadata.json", metadata)

    def ensure_cv_cache_for(self, model_name: str, pretrained: str) -> None:
        key = (model_name, pretrained)
        if key in self._prepared_cv_caches:
            return

        self.ensure_cache_for(model_name, pretrained)
        source_split_dir = Path(self.args.cache_dir) / f"{_safe_token(model_name)}__{_safe_token(pretrained)}" / self.args.train_split
        if self.args.dry_run and not (source_split_dir / "text_features.pt").exists():
            self._prepared_cv_caches.add(key)
            return

        fold_manifest = self._build_fold_manifest()
        text_features, image_features, labels_df = self._load_source_train_cache(model_name, pretrained)

        root = self.cv_cache_dir / f"{_safe_token(model_name)}__{_safe_token(pretrained)}"
        ensure_dir(root)

        for fold in fold_manifest:
            fold_index = int(fold["fold_index"])
            fold_root = root / f"fold_{fold_index:02d}"
            train_dir = fold_root / "train"
            val_dir = fold_root / "val"

            train_text = train_dir / "text_features.pt"
            val_text = val_dir / "text_features.pt"
            if train_text.exists() and val_text.exists():
                continue

            train_indices = torch.tensor(fold["train_indices"], dtype=torch.long)
            val_indices = torch.tensor(fold["val_indices"], dtype=torch.long)
            train_labels = labels_df.iloc[train_indices.tolist()].reset_index(drop=True)
            val_labels = labels_df.iloc[val_indices.tolist()].reset_index(drop=True)

            if not self.args.dry_run:
                self._write_fold_split_artifacts(
                    split_dir=train_dir,
                    text_features=text_features.index_select(0, train_indices),
                    image_features=image_features.index_select(0, train_indices),
                    labels_df=train_labels,
                    split_name="train",
                    model_name=model_name,
                    pretrained=pretrained,
                    fold_index=fold_index,
                )
                self._write_fold_split_artifacts(
                    split_dir=val_dir,
                    text_features=text_features.index_select(0, val_indices),
                    image_features=image_features.index_select(0, val_indices),
                    labels_df=val_labels,
                    split_name="val",
                    model_name=model_name,
                    pretrained=pretrained,
                    fold_index=fold_index,
                )
                write_json(
                    fold_root / "fold_info.json",
                    {
                        "created_at_utc": utc_now_iso(),
                        "fold_index": fold_index,
                        "model_name": model_name,
                        "pretrained": pretrained,
                        "train_size": int(len(train_labels)),
                        "val_size": int(len(val_labels)),
                        "train_counts": fold["train_counts"],
                        "val_counts": fold["val_counts"],
                    },
                )

        self._prepared_cv_caches.add(key)

    def _extract_metrics(self, spec: MethodSpec, metrics_payload: dict[str, Any]) -> tuple[float, float, float | None, float | None]:
        if spec.output_mode == "flat":
            block = metrics_payload["flat9_metrics"]
            return (
                float(block["macro_f1"]),
                float(block["weighted_f1"]),
                None,
                None,
            )

        flat_block = metrics_payload["derived_flat9_metrics_from_hierarchical"]
        binary_block = metrics_payload["binary_metrics"]
        unsafe_block = metrics_payload["unsafe_metrics"]
        return (
            float(flat_block["macro_f1"]),
            float(flat_block["weighted_f1"]),
            float(binary_block["f1"]),
            float(unsafe_block["macro_f1"]),
        )

    def _cv_run_name(self, stage: str, spec: MethodSpec, fold_index: int, seed: int) -> str:
        return f"{self.plan_tag}__{stage}__{spec.signature()}__f{fold_index:02d}__s{seed}"

    def _final_run_name(self, finalist_rank: int, spec: MethodSpec, seed: int) -> str:
        return f"{self.plan_tag}__finalist{finalist_rank}__{spec.signature()}__s{seed}"

    def _train_eval_cv_fold(self, stage: str, spec: MethodSpec, fold_index: int, seed: int) -> CVRunResult:
        self.ensure_cv_cache_for(spec.model_name, spec.pretrained)

        run_name = self._cv_run_name(stage=stage, spec=spec, fold_index=fold_index, seed=seed)
        run_dir = RUNS_DIR / run_name
        metrics_path = run_dir / "evaluation" / "metrics_val_best.json"
        if metrics_path.exists() and not self.args.rerun_completed:
            metrics_payload = _read_json(metrics_path)
            macro_f1, weighted_f1, binary_f1, unsafe_macro_f1 = self._extract_metrics(spec, metrics_payload)
            return CVRunResult(
                stage=stage,
                spec_signature=spec.signature(),
                run_name=run_name,
                run_dir=str(run_dir),
                fold_index=fold_index,
                seed=seed,
                val_macro_f1=macro_f1,
                val_weighted_f1=weighted_f1,
                val_binary_f1=binary_f1,
                val_unsafe_macro_f1=unsafe_macro_f1,
                val_metrics_path=str(metrics_path),
            )

        fold_cache_root = self.cv_cache_dir / f"{_safe_token(spec.model_name)}__{_safe_token(spec.pretrained)}" / f"fold_{fold_index:02d}"

        train_cmd = [
            self.python,
            "-m",
            "models.clip.train",
            "--run-dir",
            str(run_dir),
            "--cache-dir",
            str(fold_cache_root),
            "--model-name",
            spec.model_name,
            "--pretrained",
            spec.pretrained,
            "--fusion",
            spec.fusion,
            "--head-type",
            spec.head_type,
            "--output-mode",
            spec.output_mode,
            "--epochs",
            str(self.args.epochs),
            "--batch-size",
            str(self._train_batch_size_for_model(spec.model_name)),
            "--lr",
            str(self.args.lr),
            "--weight-decay",
            str(self.args.weight_decay),
            "--class-weighting",
            spec.class_weighting,
            "--beta",
            str(self.args.beta),
            "--loss-type",
            spec.loss_type,
            "--focal-gamma",
            str(spec.focal_gamma),
            "--lambda-unsafe",
            str(spec.lambda_unsafe),
            "--seed",
            str(seed),
            "--num-workers",
            str(self.args.num_workers),
            "--train-split",
            "train",
            "--val-split",
            "val",
            "--early-stopping-patience",
            str(self.args.early_stopping_patience),
            "--early-stopping-min-delta",
            str(self.args.early_stopping_min_delta),
            "--overwrite-run",
        ]
        train_cmd.append("--early-stopping" if self.args.early_stopping else "--no-early-stopping")
        train_log = self.logs_dir / f"{run_name}__train.log"
        self._run_with_oom_retry(
            cmd=train_cmd,
            log_path=train_log,
            batch_arg="--batch-size",
            start_batch=self._train_batch_size_for_model(spec.model_name),
            min_batch=self.args.min_train_batch_size,
        )

        eval_cmd = [
            self.python,
            "-m",
            "models.clip.evaluate",
            "--run-dir",
            str(run_dir),
            "--split",
            "val",
            "--checkpoint",
            "best",
            "--num-workers",
            str(self.args.num_workers),
        ]
        eval_log = self.logs_dir / f"{run_name}__eval_val.log"
        _run(eval_cmd, dry_run=self.args.dry_run, log_path=eval_log)

        if self.args.dry_run:
            return CVRunResult(
                stage=stage,
                spec_signature=spec.signature(),
                run_name=run_name,
                run_dir=str(run_dir),
                fold_index=fold_index,
                seed=seed,
                val_macro_f1=float("nan"),
                val_weighted_f1=float("nan"),
                val_binary_f1=None,
                val_unsafe_macro_f1=None,
                val_metrics_path=str(metrics_path),
            )

        metrics_payload = _read_json(metrics_path)
        macro_f1, weighted_f1, binary_f1, unsafe_macro_f1 = self._extract_metrics(spec, metrics_payload)
        return CVRunResult(
            stage=stage,
            spec_signature=spec.signature(),
            run_name=run_name,
            run_dir=str(run_dir),
            fold_index=fold_index,
            seed=seed,
            val_macro_f1=macro_f1,
            val_weighted_f1=weighted_f1,
            val_binary_f1=binary_f1,
            val_unsafe_macro_f1=unsafe_macro_f1,
            val_metrics_path=str(metrics_path),
        )

    def _aggregate_candidate(self, stage: str, spec: MethodSpec, runs: list[CVRunResult]) -> CandidateAggregate:
        best_run = sorted(runs, key=lambda r: (r.val_macro_f1, r.val_weighted_f1), reverse=True)[0]
        binary_vals = [r.val_binary_f1 for r in runs if r.val_binary_f1 is not None]
        unsafe_vals = [r.val_unsafe_macro_f1 for r in runs if r.val_unsafe_macro_f1 is not None]
        fold_count = len({r.fold_index for r in runs})
        seed_values = sorted({int(r.seed) for r in runs})

        return CandidateAggregate(
            stage=stage,
            spec_signature=spec.signature(),
            model_name=spec.model_name,
            pretrained=spec.pretrained,
            fusion=spec.fusion,
            head_type=spec.head_type,
            output_mode=spec.output_mode,
            loss_type=spec.loss_type,
            class_weighting=spec.class_weighting,
            focal_gamma=spec.focal_gamma,
            lambda_unsafe=spec.lambda_unsafe,
            fold_count=fold_count,
            seed_values=seed_values,
            run_count=len(runs),
            val_macro_f1_mean=_nan_safe_mean(r.val_macro_f1 for r in runs),
            val_macro_f1_std=_nan_safe_pstdev(r.val_macro_f1 for r in runs),
            val_weighted_f1_mean=_nan_safe_mean(r.val_weighted_f1 for r in runs),
            val_weighted_f1_std=_nan_safe_pstdev(r.val_weighted_f1 for r in runs),
            val_binary_f1_mean=_nan_safe_mean(binary_vals) if len(binary_vals) > 0 else None,
            val_unsafe_macro_f1_mean=_nan_safe_mean(unsafe_vals) if len(unsafe_vals) > 0 else None,
            best_run_name=best_run.run_name,
            best_val_macro_f1=best_run.val_macro_f1,
            best_val_weighted_f1=best_run.val_weighted_f1,
        )

    def evaluate_nodes(self, stage: str, nodes: list[BeamNode], seed_values: list[int]) -> list[dict[str, Any]]:
        deduped: dict[str, BeamNode] = {}
        for node in nodes:
            deduped[node.spec.signature()] = BeamNode(spec=node.spec.canonical(), lineage=node.lineage)

        evaluations: list[dict[str, Any]] = []
        for node in deduped.values():
            runs: list[CVRunResult] = []
            for fold in self._build_fold_manifest():
                fold_index = int(fold["fold_index"])
                for seed in seed_values:
                    result = self._train_eval_cv_fold(stage=stage, spec=node.spec, fold_index=fold_index, seed=seed)
                    self.results.append(result)
                    runs.append(result)

            aggregate = self._aggregate_candidate(stage=stage, spec=node.spec, runs=runs)
            evaluations.append(
                {
                    "node": node,
                    "aggregate": aggregate,
                    "runs": runs,
                }
            )
        return evaluations

    def _select_beam(
        self,
        evaluations: list[dict[str, Any]],
        beam_width: int,
        beam_tolerance: float,
        max_keep: int,
    ) -> list[BeamNode]:
        if len(evaluations) == 0:
            return []

        ranked = sorted(
            evaluations,
            key=lambda item: item["aggregate"].ranking_key(),
            reverse=True,
        )
        best_score = ranked[0]["aggregate"].val_macro_f1_mean
        keep: list[dict[str, Any]] = ranked[:beam_width]
        for item in ranked[beam_width:]:
            if len(keep) >= max_keep:
                break
            if item["aggregate"].val_macro_f1_mean >= (best_score - beam_tolerance):
                keep.append(item)
        return [item["node"] for item in keep]

    def _stage_summary_payload(
        self,
        blueprint: StageBlueprint,
        evaluations: list[dict[str, Any]],
        selected_nodes: list[BeamNode],
    ) -> dict[str, Any]:
        selected = {node.spec.signature() for node in selected_nodes}
        ranked = sorted(
            evaluations,
            key=lambda item: item["aggregate"].ranking_key(),
            reverse=True,
        )
        payload = {
            "created_at_utc": utc_now_iso(),
            "plan_tag": self.plan_tag,
            "stage": blueprint.name,
            "description": blueprint.description,
            "seed_values": blueprint.seed_values,
            "beam_width": blueprint.beam_width,
            "beam_tolerance": blueprint.beam_tolerance,
            "max_keep": blueprint.max_keep,
            "results": [],
        }

        for rank, item in enumerate(ranked, start=1):
            aggregate = item["aggregate"]
            node = item["node"]
            payload["results"].append(
                {
                    "rank": rank,
                    "selected_for_next_stage": aggregate.spec_signature in selected,
                    "lineage": list(node.lineage),
                    "spec_description": node.spec.describe(),
                    "aggregate": asdict(aggregate),
                }
            )
        return payload

    def _save_stage_summary(self, blueprint: StageBlueprint, payload: dict[str, Any]) -> None:
        self.stage_summaries.append(payload)
        self.aggregates.extend(payload["results"])
        if not self.args.dry_run:
            write_json(self.stage_dir / f"{blueprint.name}.json", payload)

    def _make_stage_blueprints(self) -> list[StageBlueprint]:
        baseline = MethodSpec(
            model_name=self.scout_backbone[0],
            pretrained=self.scout_backbone[1],
            fusion="concat",
            head_type="linear",
            output_mode="flat",
            loss_type="ce",
            class_weighting="effective",
            focal_gamma=0.0,
            lambda_unsafe=1.0,
        )

        architecture_variants = [
            ("concat", "linear"),
            ("concat", "small"),
            ("interaction", "linear"),
            ("interaction", "small"),
            ("interaction", "medium"),
            ("interaction_only", "small"),
        ]

        def stage1_expander(_: BeamNode) -> list[BeamNode]:
            return [
                BeamNode(
                    spec=replace(baseline, fusion=fusion, head_type=head).canonical(),
                    lineage=(f"stage1:{fusion}/{head}/flat",),
                )
                for fusion, head in architecture_variants
            ]

        def stage2_expander(node: BeamNode) -> list[BeamNode]:
            out: list[BeamNode] = []
            for output_mode in ("flat", "hierarchical"):
                spec = replace(node.spec, output_mode=output_mode).canonical()
                out.append(
                    BeamNode(
                        spec=spec,
                        lineage=node.lineage + (f"stage2:output={output_mode}",),
                    )
                )
            return out

        def stage3_expander(node: BeamNode) -> list[BeamNode]:
            out: list[BeamNode] = [
                BeamNode(
                    spec=replace(node.spec, loss_type="ce", focal_gamma=0.0).canonical(),
                    lineage=node.lineage + ("stage3:loss=ce",),
                )
            ]
            for gamma in (1.0, 2.0):
                out.append(
                    BeamNode(
                        spec=replace(node.spec, loss_type="focal", focal_gamma=gamma).canonical(),
                        lineage=node.lineage + (f"stage3:loss=focal_gamma={gamma:.1f}",),
                    )
                )
            return out

        def stage4_expander(node: BeamNode) -> list[BeamNode]:
            return [
                BeamNode(
                    spec=replace(node.spec, class_weighting=weighting).canonical(),
                    lineage=node.lineage + (f"stage4:class_weighting={weighting}",),
                )
                for weighting in ("none", "inverse", "effective")
            ]

        def stage5_expander(node: BeamNode) -> list[BeamNode]:
            if node.spec.output_mode != "hierarchical":
                return [BeamNode(spec=node.spec.canonical(), lineage=node.lineage + ("stage5:lambda=flat_passthrough",))]
            return [
                BeamNode(
                    spec=replace(node.spec, lambda_unsafe=value).canonical(),
                    lineage=node.lineage + (f"stage5:lambda_unsafe={value:.2f}",),
                )
                for value in (0.5, 1.0, 1.5)
            ]

        def stage6_expander(node: BeamNode) -> list[BeamNode]:
            return [
                BeamNode(
                    spec=replace(node.spec, model_name=model_name, pretrained=pretrained).canonical(),
                    lineage=node.lineage + (f"stage6:backbone={model_name}/{pretrained}",),
                )
                for model_name, pretrained in self.backbone_grid
            ]

        return [
            StageBlueprint(
                name="stage1_architecture",
                description="Compare fusion/head architecture families on a scout backbone using flat output.",
                expander=stage1_expander,
                seed_values=self.search_seed_values,
                beam_width=self.args.beam_width,
                beam_tolerance=self.args.beam_tolerance,
                max_keep=self.args.beam_max_keep,
            ),
            StageBlueprint(
                name="stage2_output_mode",
                description="Let the architecture survivors branch into flat and hierarchical formulations.",
                expander=stage2_expander,
                seed_values=self.search_seed_values,
                beam_width=self.args.beam_width,
                beam_tolerance=self.args.beam_tolerance,
                max_keep=self.args.beam_max_keep,
            ),
            StageBlueprint(
                name="stage3_loss_type",
                description="Compare cross-entropy against focal loss variants on the current beam.",
                expander=stage3_expander,
                seed_values=self.search_seed_values,
                beam_width=self.args.beam_width,
                beam_tolerance=self.args.beam_tolerance,
                max_keep=self.args.beam_max_keep,
            ),
            StageBlueprint(
                name="stage4_class_weighting",
                description="Compare class-weighting strategies for the surviving candidates.",
                expander=stage4_expander,
                seed_values=self.search_seed_values,
                beam_width=self.args.beam_width,
                beam_tolerance=self.args.beam_tolerance,
                max_keep=self.args.beam_max_keep,
            ),
            StageBlueprint(
                name="stage5_hierarchical_weighting",
                description="Sweep lambda_unsafe for hierarchical models while flat models pass through unchanged.",
                expander=stage5_expander,
                seed_values=self.search_seed_values,
                beam_width=self.args.beam_width,
                beam_tolerance=self.args.beam_tolerance,
                max_keep=self.args.beam_max_keep,
            ),
            StageBlueprint(
                name="stage6_backbone_pretraining",
                description="Run the beam survivors across the four backbone/pretraining combinations.",
                expander=stage6_expander,
                seed_values=self.search_seed_values,
                beam_width=self.args.beam_width,
                beam_tolerance=self.args.beam_tolerance,
                max_keep=self.args.beam_max_keep,
            ),
        ]

    def _run_stage(self, blueprint: StageBlueprint, beam: list[BeamNode]) -> list[BeamNode]:
        expanded: list[BeamNode] = []
        if len(beam) == 0:
            seed_node = BeamNode(
                spec=MethodSpec(
                    self.scout_backbone[0],
                    self.scout_backbone[1],
                    "concat",
                    "linear",
                    "flat",
                    "ce",
                    "effective",
                    0.0,
                    1.0,
                ),
                lineage=(),
            )
            expanded = blueprint.expander(seed_node)
        else:
            for node in beam:
                expanded.extend(blueprint.expander(node))

        evaluations = self.evaluate_nodes(stage=blueprint.name, nodes=expanded, seed_values=blueprint.seed_values)
        selected = self._select_beam(
            evaluations=evaluations,
            beam_width=blueprint.beam_width,
            beam_tolerance=blueprint.beam_tolerance,
            max_keep=blueprint.max_keep,
        )
        payload = self._stage_summary_payload(blueprint=blueprint, evaluations=evaluations, selected_nodes=selected)
        self._save_stage_summary(blueprint, payload)
        return selected

    def _run_robustness_stage(self, beam: list[BeamNode]) -> list[BeamNode]:
        blueprint = StageBlueprint(
            name="stage7_robustness",
            description="Re-score the strongest methodologies with a larger seed budget before final fitting.",
            expander=lambda node: [node],
            seed_values=self.robust_seed_values,
            beam_width=self.args.finalist_count,
            beam_tolerance=self.args.beam_tolerance,
            max_keep=self.args.finalist_count,
        )
        evaluations = self.evaluate_nodes(stage=blueprint.name, nodes=beam, seed_values=blueprint.seed_values)
        selected = self._select_beam(
            evaluations=evaluations,
            beam_width=blueprint.beam_width,
            beam_tolerance=blueprint.beam_tolerance,
            max_keep=blueprint.max_keep,
        )
        payload = self._stage_summary_payload(blueprint=blueprint, evaluations=evaluations, selected_nodes=selected)
        self._save_stage_summary(blueprint, payload)
        return selected

    def _train_eval_standard(self, finalist_rank: int, spec: MethodSpec, seed: int) -> FinalRunResult:
        self.ensure_cache_for(spec.model_name, spec.pretrained)

        run_name = self._final_run_name(finalist_rank=finalist_rank, spec=spec, seed=seed)
        run_dir = RUNS_DIR / run_name
        val_metrics_path = run_dir / "evaluation" / "metrics_val_best.json"
        test_metrics_path = run_dir / "evaluation" / "metrics_test_best.json"

        if val_metrics_path.exists() and test_metrics_path.exists() and not self.args.rerun_completed:
            val_payload = _read_json(val_metrics_path)
            test_payload = _read_json(test_metrics_path)
            val_macro, val_weighted, _, _ = self._extract_metrics(spec, val_payload)
            test_macro, test_weighted, _, _ = self._extract_metrics(spec, test_payload)
            return FinalRunResult(
                finalist_rank=finalist_rank,
                spec_signature=spec.signature(),
                run_name=run_name,
                run_dir=str(run_dir),
                seed=seed,
                val_macro_f1=val_macro,
                val_weighted_f1=val_weighted,
                test_macro_f1=test_macro,
                test_weighted_f1=test_weighted,
                val_metrics_path=str(val_metrics_path),
                test_metrics_path=str(test_metrics_path),
            )

        train_cmd = [
            self.python,
            "-m",
            "models.clip.train",
            "--run-dir",
            str(run_dir),
            "--cache-dir",
            str(self.args.cache_dir),
            "--model-name",
            spec.model_name,
            "--pretrained",
            spec.pretrained,
            "--fusion",
            spec.fusion,
            "--head-type",
            spec.head_type,
            "--output-mode",
            spec.output_mode,
            "--epochs",
            str(self.args.epochs),
            "--batch-size",
            str(self._train_batch_size_for_model(spec.model_name)),
            "--lr",
            str(self.args.lr),
            "--weight-decay",
            str(self.args.weight_decay),
            "--class-weighting",
            spec.class_weighting,
            "--beta",
            str(self.args.beta),
            "--loss-type",
            spec.loss_type,
            "--focal-gamma",
            str(spec.focal_gamma),
            "--lambda-unsafe",
            str(spec.lambda_unsafe),
            "--seed",
            str(seed),
            "--num-workers",
            str(self.args.num_workers),
            "--train-split",
            self.args.train_split,
            "--val-split",
            self.args.val_split,
            "--early-stopping-patience",
            str(self.args.early_stopping_patience),
            "--early-stopping-min-delta",
            str(self.args.early_stopping_min_delta),
            "--overwrite-run",
        ]
        train_cmd.append("--early-stopping" if self.args.early_stopping else "--no-early-stopping")
        train_log = self.logs_dir / f"{run_name}__train.log"
        self._run_with_oom_retry(
            cmd=train_cmd,
            log_path=train_log,
            batch_arg="--batch-size",
            start_batch=self._train_batch_size_for_model(spec.model_name),
            min_batch=self.args.min_train_batch_size,
        )

        for split_name in (self.args.val_split, self.args.test_split):
            eval_cmd = [
                self.python,
                "-m",
                "models.clip.evaluate",
                "--run-dir",
                str(run_dir),
                "--split",
                split_name,
                "--checkpoint",
                "best",
                "--num-workers",
                str(self.args.num_workers),
            ]
            eval_log = self.logs_dir / f"{run_name}__eval_{split_name}.log"
            _run(eval_cmd, dry_run=self.args.dry_run, log_path=eval_log)

        if self.args.dry_run:
            return FinalRunResult(
                finalist_rank=finalist_rank,
                spec_signature=spec.signature(),
                run_name=run_name,
                run_dir=str(run_dir),
                seed=seed,
                val_macro_f1=float("nan"),
                val_weighted_f1=float("nan"),
                test_macro_f1=float("nan"),
                test_weighted_f1=float("nan"),
                val_metrics_path=str(val_metrics_path),
                test_metrics_path=str(test_metrics_path),
            )

        val_payload = _read_json(val_metrics_path)
        test_payload = _read_json(test_metrics_path)
        val_macro, val_weighted, _, _ = self._extract_metrics(spec, val_payload)
        test_macro, test_weighted, _, _ = self._extract_metrics(spec, test_payload)
        return FinalRunResult(
            finalist_rank=finalist_rank,
            spec_signature=spec.signature(),
            run_name=run_name,
            run_dir=str(run_dir),
            seed=seed,
            val_macro_f1=val_macro,
            val_weighted_f1=val_weighted,
            test_macro_f1=test_macro,
            test_weighted_f1=test_weighted,
            val_metrics_path=str(val_metrics_path),
            test_metrics_path=str(test_metrics_path),
        )

    def _calibrate_finalist(self, run_dir: Path) -> Path:
        out_dir = self.calibration_root / run_dir.name
        summary_path = out_dir / "calibration_summary.json"
        if summary_path.exists() and not self.args.rerun_completed:
            return summary_path

        cmd = [
            self.python,
            "-m",
            "models.clip.calibrate",
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            "best",
            "--out-dir",
            str(out_dir),
            "--batch-size",
            str(self.args.calibration_batch_size),
            "--num-workers",
            str(self.args.num_workers),
            "--fit-split",
            self.args.val_split,
            "--eval-splits",
            self.args.val_split,
            self.args.test_split,
        ]
        log_path = self.logs_dir / f"{run_dir.name}__calibrate.log"
        _run(cmd, dry_run=self.args.dry_run, log_path=log_path)
        return summary_path

    def _finalist_summary(self, finalist_rank: int, node: BeamNode) -> dict[str, Any]:
        runs = [
            self._train_eval_standard(finalist_rank=finalist_rank, spec=node.spec, seed=seed)
            for seed in self.final_fit_seed_values
        ]
        best_run = sorted(runs, key=lambda r: (r.val_macro_f1, r.val_weighted_f1), reverse=True)[0]
        calibration_path = self._calibrate_finalist(Path(best_run.run_dir))
        calibration_payload = _read_json(calibration_path) if calibration_path.exists() and not self.args.dry_run else {}

        payload = {
            "finalist_rank": finalist_rank,
            "spec_signature": node.spec.signature(),
            "spec_description": node.spec.describe(),
            "lineage": list(node.lineage),
            "seed_values": list(self.final_fit_seed_values),
            "runs": [asdict(r) for r in runs],
            "aggregate": {
                "val_macro_f1_mean": _nan_safe_mean(r.val_macro_f1 for r in runs),
                "val_macro_f1_std": _nan_safe_pstdev(r.val_macro_f1 for r in runs),
                "test_macro_f1_mean": _nan_safe_mean(r.test_macro_f1 for r in runs),
                "test_macro_f1_std": _nan_safe_pstdev(r.test_macro_f1 for r in runs),
            },
            "best_seed_run": asdict(best_run),
            "calibration_summary_path": str(calibration_path),
            "calibration_summary": calibration_payload,
        }
        self.finalists.append(payload)
        return payload

    def run(self) -> None:
        self.clear_output()
        self.ensure_splits()
        data_profile = self._write_data_profile()

        print(
            "runtime_settings="
            f"cv_folds={self.args.cv_folds}, "
            f"search_seed_values={self.search_seed_values}, "
            f"robust_seed_values={self.robust_seed_values}, "
            f"final_fit_seed_values={self.final_fit_seed_values}, "
            f"beam_width={self.args.beam_width}"
        )

        beam: list[BeamNode] = []
        for blueprint in self._make_stage_blueprints():
            beam = self._run_stage(blueprint=blueprint, beam=beam)

        finalists = self._run_robustness_stage(beam=beam)
        finalist_payloads = [
            self._finalist_summary(finalist_rank=rank, node=node)
            for rank, node in enumerate(finalists, start=1)
        ]

        summary = {
            "created_at_utc": _utc_now_token(),
            "plan_tag": self.plan_tag,
            "clear_scope": self.args.clear_scope,
            "selection_rule": (
                "beam-search ranking: max mean(cv_macro_f1), tie-break mean(cv_weighted_f1), "
                "then lower macro std and lower weighted std"
            ),
            "runtime": {
                "python_executable": self.python,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cwd": str(PROJECT_ROOT),
                "argv": sys.argv,
            },
            "cv": {
                "folds": self.args.cv_folds,
                "cv_seed": self.args.cv_seed,
                "beam_width": self.args.beam_width,
                "beam_tolerance": self.args.beam_tolerance,
                "beam_max_keep": self.args.beam_max_keep,
            },
            "search_seed_list": self.search_seed_values,
            "robust_seed_list": self.robust_seed_values,
            "final_fit_seed_list": self.final_fit_seed_values,
            "data_profile": data_profile,
            "stage_summaries": self.stage_summaries,
            "finalists": finalist_payloads,
        }

        if not self.args.dry_run:
            write_json(self.summary_path, summary)

        print("experiment_plan_complete=ok")
        print(f"summary={self.summary_path}")
        print(f"logs_dir={self.logs_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a staged beam-search CV experiment plan for frozen-CLIP fusion heads."
    )
    p.add_argument("--plan-tag", type=str, default=None)
    p.add_argument("--clear-scope", choices=["none", "plan"], default="none")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rerun-completed", action="store_true")

    p.add_argument("--bootstrap-split", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bootstrap-cache", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--cache-dir", type=Path, default=OUTPUT_DIR / "caches")
    p.add_argument("--data-csv", type=Path, default=PROJECT_ROOT / "data" / "vlsu_mod.csv")
    p.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "data" / "vlsu_images")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--min-class-count", type=int, default=3)
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="val")
    p.add_argument("--test-split", type=str, default="test")

    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batch-size-l14", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.999)
    p.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--early-stopping-patience", type=int, default=2)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--cache-batch-size-l14", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--oom-retry", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-train-batch-size", type=int, default=16)
    p.add_argument("--min-cache-batch-size", type=int, default=1)
    p.add_argument("--calibration-batch-size", type=int, default=256)

    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-seed", type=int, default=42)
    p.add_argument("--seed-list", type=str, default="42")
    p.add_argument("--robust-seed-list", type=str, default="40,41,42")
    p.add_argument("--final-fit-seed-list", type=str, default="42,43,44")
    p.add_argument("--beam-width", type=int, default=3)
    p.add_argument("--beam-tolerance", type=float, default=0.005)
    p.add_argument("--beam-max-keep", type=int, default=5)
    p.add_argument("--finalist-count", type=int, default=2)
    return p


def main() -> None:
    args = build_parser().parse_args()
    runner = ExperimentRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
