from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import ANALYSIS_DIR, OUTPUT_DIR, PROJECT_ROOT, SPLITS_DIR
from .utils import ensure_dir


def _utc_now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
            raise ValueError(f"Invalid seed in --seed-list: '{item}'") from exc

    if len(values) == 0:
        raise ValueError("--seed-list produced no seeds. Provide at least one integer seed.")
    return values


def _validate_seed_count(seed_values: list[int], arg_name: str) -> None:
    if len(seed_values) < 3 or len(seed_values) > 5:
        raise ValueError(
            f"{arg_name} must contain 3 to 5 seeds for robust selection, got {len(seed_values)}"
        )


@dataclass(frozen=True)
class CandidateSpec:
    model_name: str
    pretrained: str
    fusion: str
    head_type: str
    output_mode: str


@dataclass(frozen=True)
class RunResult:
    stage: str
    run_name: str
    run_dir: str
    model_name: str
    pretrained: str
    fusion: str
    head_type: str
    output_mode: str
    seed: int
    val_macro_f1: float
    val_weighted_f1: float
    val_metrics_path: str

@dataclass(frozen=True)
class CandidateAggregate:
    stage: str
    model_name: str
    pretrained: str
    fusion: str
    head_type: str
    output_mode: str
    seed_count: int
    seed_values: list[int]
    val_macro_f1_mean: float
    val_macro_f1_std: float
    val_weighted_f1_mean: float
    val_weighted_f1_std: float
    best_seed: int
    best_run_name: str
    best_val_macro_f1: float
    best_val_weighted_f1: float


class ExperimentRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.python = sys.executable
        self.timestamp = _utc_now_token()
        self.plan_tag = args.plan_tag or self.timestamp

        self.output_dir = OUTPUT_DIR
        self.runs_dir = self.output_dir / "runs"
        self.analysis_dir = ANALYSIS_DIR / "model_search"
        self.logs_dir = self.analysis_dir / "logs" / self.plan_tag
        ensure_dir(self.analysis_dir)
        ensure_dir(self.logs_dir)

        self.seed_values = _parse_seed_list(self.args.seed_list)
        _validate_seed_count(self.seed_values, "--seed-list")

        stage5_seed_raw = self.args.stage5_seed_list or self.args.seed_list
        self.stage5_seed_values = _parse_seed_list(stage5_seed_raw)
        _validate_seed_count(self.stage5_seed_values, "--stage5-seed-list")

        self.results: list[RunResult] = []
        self.aggregates: list[CandidateAggregate] = []
        self.prepared_caches: set[tuple[str, str]] = set()

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

    def _num_workers(self) -> int:
        return int(self.args.num_workers)

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
                is_oom = _contains_oom(log_path)
                if (not self.args.oom_retry) or (not is_oom):
                    raise
                next_batch = batch // 2
                if next_batch < min_batch:
                    raise
                print(
                    f"oom_retry=attempt{attempt} arg={batch_arg} batch={batch} -> retry_with={next_batch}"
                )
                batch = next_batch
                attempt += 1

    def _needed_splits(self) -> list[str]:
        names = [self.args.train_split, self.args.val_split, self.args.test_split]
        seen: set[str] = set()
        out: list[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def ensure_splits(self) -> None:
        if not self.args.bootstrap_split:
            return

        required = [SPLITS_DIR / f"{name}.csv" for name in self._needed_splits()]
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
        if key in self.prepared_caches:
            return

        safe_model = model_name.replace("/", "-").replace("\\", "-").replace(":", "-").replace(" ", "_")
        safe_pretrained = pretrained.replace("/", "-").replace("\\", "-").replace(":", "-").replace(" ", "_")
        cache_root = Path(self.args.cache_dir) / f"{safe_model}__{safe_pretrained}"

        need_splits = self._needed_splits()
        all_present = True
        for split_name in need_splits:
            if not (cache_root / split_name / "text_features.pt").exists() or not (
                cache_root / split_name / "image_features.pt"
            ).exists():
                all_present = False
                break

        if all_present:
            print(f"bootstrap_cache=skip ({model_name}, {pretrained}) already cached")
            self.prepared_caches.add(key)
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
            *need_splits,
        ]
        cache_log = self.logs_dir / f"bootstrap__cache__{safe_model}__{safe_pretrained}.log"
        self._run_with_oom_retry(
            cmd=cmd,
            log_path=cache_log,
            batch_arg="--batch-size",
            start_batch=self._cache_batch_size_for_model(model_name),
            min_batch=self.args.min_cache_batch_size,
        )
        self.prepared_caches.add(key)

    def clear_output(self) -> None:
        if self.args.clear_scope == "none":
            return

        if self.args.clear_scope == "runs":
            target = self.runs_dir
        else:
            target = self.output_dir

        if not target.exists():
            print(f"clear-scope={self.args.clear_scope}: nothing to remove at {target}")
            return

        print(f"clear-scope={self.args.clear_scope}: removing {target}")
        if not self.args.dry_run:
            shutil.rmtree(target)

        # Recreate base output folders that downstream modules expect.
        if not self.args.dry_run:
            ensure_dir(self.output_dir)
            ensure_dir(self.runs_dir)

    def train_and_eval_val(self, stage: str, spec: CandidateSpec, seed: int) -> RunResult:
        self.ensure_cache_for(spec.model_name, spec.pretrained)

        run_name = (
            f"{self.plan_tag}__{stage}__{spec.model_name}__{spec.pretrained}"
            f"__{spec.fusion}__{spec.head_type}__{spec.output_mode}__s{seed}"
        )
        run_name = run_name.replace("/", "-").replace("\\", "-").replace(":", "-").replace(" ", "_")
        run_dir = self.runs_dir / run_name

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
            self.args.class_weighting,
            "--beta",
            str(self.args.beta),
            "--loss-type",
            self.args.loss_type,
            "--focal-gamma",
            str(self.args.focal_gamma),
            "--lambda-unsafe",
            str(self.args.lambda_unsafe),
            "--seed",
            str(seed),
            "--num-workers",
            str(self._num_workers()),
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

        eval_cmd = [
            self.python,
            "-m",
            "models.clip.evaluate",
            "--run-dir",
            str(run_dir),
            "--split",
            self.args.val_split,
            "--checkpoint",
            "best",
            "--num-workers",
            str(self._num_workers()),
        ]
        val_eval_log = self.logs_dir / f"{run_name}__eval_val.log"
        _run(eval_cmd, dry_run=self.args.dry_run, log_path=val_eval_log)

        metrics_path = run_dir / "evaluation" / f"metrics_{self.args.val_split}_best.json"

        if self.args.dry_run:
            return RunResult(
                stage=stage,
                run_name=run_name,
                run_dir=str(run_dir),
                model_name=spec.model_name,
                pretrained=spec.pretrained,
                fusion=spec.fusion,
                head_type=spec.head_type,
                output_mode=spec.output_mode,
                seed=seed,
                val_macro_f1=float("nan"),
                val_weighted_f1=float("nan"),
                val_metrics_path=str(metrics_path),
            )

        metrics_payload = _read_json(metrics_path)
        if spec.output_mode == "flat":
            key = "flat9_metrics"
        else:
            key = "derived_flat9_metrics_from_hierarchical"

        block = metrics_payload.get(key, {})
        macro_f1 = float(block.get("macro_f1", float("nan")))
        weighted_f1 = float(block.get("weighted_f1", float("nan")))

        return RunResult(
            stage=stage,
            run_name=run_name,
            run_dir=str(run_dir),
            model_name=spec.model_name,
            pretrained=spec.pretrained,
            fusion=spec.fusion,
            head_type=spec.head_type,
            output_mode=spec.output_mode,
            seed=seed,
            val_macro_f1=macro_f1,
            val_weighted_f1=weighted_f1,
            val_metrics_path=str(metrics_path),
        )

    def train_and_eval_val_many(self, stage: str, spec: CandidateSpec, seeds: list[int]) -> list[RunResult]:
        return [self.train_and_eval_val(stage=stage, spec=spec, seed=seed) for seed in seeds]

    @staticmethod
    def aggregate(stage: str, spec: CandidateSpec, runs: list[RunResult]) -> CandidateAggregate:
        if len(runs) == 0:
            raise ValueError("Cannot aggregate empty run list")

        macro_vals = [r.val_macro_f1 for r in runs]
        weighted_vals = [r.val_weighted_f1 for r in runs]
        best_run = sorted(runs, key=lambda r: (r.val_macro_f1, r.val_weighted_f1), reverse=True)[0]

        macro_std = statistics.pstdev(macro_vals) if len(macro_vals) > 1 else 0.0
        weighted_std = statistics.pstdev(weighted_vals) if len(weighted_vals) > 1 else 0.0

        return CandidateAggregate(
            stage=stage,
            model_name=spec.model_name,
            pretrained=spec.pretrained,
            fusion=spec.fusion,
            head_type=spec.head_type,
            output_mode=spec.output_mode,
            seed_count=len(runs),
            seed_values=[r.seed for r in runs],
            val_macro_f1_mean=float(statistics.fmean(macro_vals)),
            val_macro_f1_std=float(macro_std),
            val_weighted_f1_mean=float(statistics.fmean(weighted_vals)),
            val_weighted_f1_std=float(weighted_std),
            best_seed=best_run.seed,
            best_run_name=best_run.run_name,
            best_val_macro_f1=best_run.val_macro_f1,
            best_val_weighted_f1=best_run.val_weighted_f1,
        )

    @staticmethod
    def best_aggregate(aggregates: list[CandidateAggregate]) -> CandidateAggregate:
        if len(aggregates) == 0:
            raise ValueError("No aggregate results to rank")
        return sorted(
            aggregates,
            key=lambda a: (
                a.val_macro_f1_mean,
                a.val_weighted_f1_mean,
                -a.val_macro_f1_std,
                -a.val_weighted_f1_std,
            ),
            reverse=True,
        )[0]

    @staticmethod
    def to_spec(agg: CandidateAggregate) -> CandidateSpec:
        return CandidateSpec(
            model_name=agg.model_name,
            pretrained=agg.pretrained,
            fusion=agg.fusion,
            head_type=agg.head_type,
            output_mode=agg.output_mode,
        )

    @staticmethod
    def best(results: list[RunResult]) -> RunResult:
        if len(results) == 0:
            raise ValueError("No results to rank")
        return sorted(
            results,
            key=lambda r: (r.val_macro_f1, r.val_weighted_f1),
            reverse=True,
        )[0]

    def evaluate_test(self, winner: RunResult) -> Path:
        run_dir = Path(winner.run_dir)
        cmd = [
            self.python,
            "-m",
            "models.clip.evaluate",
            "--run-dir",
            str(run_dir),
            "--split",
            self.args.test_split,
            "--checkpoint",
            "best",
            "--num-workers",
            str(self._num_workers()),
        ]
        test_log = self.logs_dir / f"{Path(winner.run_dir).name}__eval_test.log"
        _run(cmd, dry_run=self.args.dry_run, log_path=test_log)
        return run_dir / "evaluation" / f"metrics_{self.args.test_split}_best.json"

    def run(self) -> None:
        self.clear_output()
        self.ensure_splits()

        print(
            "runtime_settings="
            f"num_workers={self._num_workers()}, "
            f"cache_batch_B16={self._cache_batch_size_for_model('ViT-B-16')}, "
            f"cache_batch_L14={self._cache_batch_size_for_model('ViT-L-14')}, "
            f"train_batch_B16={self._train_batch_size_for_model('ViT-B-16')}, "
            f"train_batch_L14={self._train_batch_size_for_model('ViT-L-14')}, "
            f"stage_seed_values={self.seed_values}, "
            f"stage5_seed_values={self.stage5_seed_values}"
        )

        # Stage 1: five flat models on ViT-B-16 + laion2b_s34b_b88k.
        stage1_specs = [
            CandidateSpec("ViT-B-16", "laion2b_s34b_b88k", "concat", "linear", "flat"),
            CandidateSpec("ViT-B-16", "laion2b_s34b_b88k", "concat", "small", "flat"),
            CandidateSpec("ViT-B-16", "laion2b_s34b_b88k", "interaction", "linear", "flat"),
            CandidateSpec("ViT-B-16", "laion2b_s34b_b88k", "interaction", "small", "flat"),
            CandidateSpec("ViT-B-16", "laion2b_s34b_b88k", "interaction", "medium", "flat"),
        ]
        stage1_aggregates: list[CandidateAggregate] = []
        for spec in stage1_specs:
            runs = self.train_and_eval_val_many("stage1", spec, seeds=self.seed_values)
            self.results.extend(runs)
            stage1_aggregates.append(self.aggregate("stage1", spec, runs))
        self.aggregates.extend(stage1_aggregates)
        winner1_agg = self.best_aggregate(stage1_aggregates)
        winner1_spec = self.to_spec(winner1_agg)

        # Stage 2: switch winner architecture to hierarchical, same backbone/pretrained.
        stage2_spec = CandidateSpec(
            model_name=winner1_spec.model_name,
            pretrained=winner1_spec.pretrained,
            fusion=winner1_spec.fusion,
            head_type=winner1_spec.head_type,
            output_mode="hierarchical",
        )
        stage2_runs = self.train_and_eval_val_many("stage2", stage2_spec, seeds=self.seed_values)
        self.results.extend(stage2_runs)
        stage2_agg = self.aggregate("stage2", stage2_spec, stage2_runs)
        self.aggregates.append(stage2_agg)

        # Stage 3: compare stage2 winner against same architecture on ViT-L-14 + laion2b_s32b_b82k.
        stage3_baseline_spec = stage2_spec
        stage3_alt_spec = CandidateSpec(
            model_name="ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            fusion=stage2_spec.fusion,
            head_type=stage2_spec.head_type,
            output_mode=stage2_spec.output_mode,
        )
        stage3_baseline_runs = self.train_and_eval_val_many("stage3_baseline", stage3_baseline_spec, seeds=self.seed_values)
        stage3_alt_runs = self.train_and_eval_val_many("stage3", stage3_alt_spec, seeds=self.seed_values)
        self.results.extend(stage3_baseline_runs)
        self.results.extend(stage3_alt_runs)
        stage3_baseline_agg = self.aggregate("stage3_baseline", stage3_baseline_spec, stage3_baseline_runs)
        stage3_alt_agg = self.aggregate("stage3", stage3_alt_spec, stage3_alt_runs)
        self.aggregates.extend([stage3_baseline_agg, stage3_alt_agg])
        winner3_agg = self.best_aggregate([stage3_baseline_agg, stage3_alt_agg])
        winner3_spec = self.to_spec(winner3_agg)

        # Stage 4: swap winner to datacomp_xl_s13b_b90k, keep same model_name/architecture.
        stage4_spec = CandidateSpec(
            model_name=winner3_spec.model_name,
            pretrained="datacomp_xl_s13b_b90k",
            fusion=winner3_spec.fusion,
            head_type=winner3_spec.head_type,
            output_mode=winner3_spec.output_mode,
        )
        stage4_runs = self.train_and_eval_val_many("stage4", stage4_spec, seeds=self.seed_values)
        self.results.extend(stage4_runs)
        stage4_agg = self.aggregate("stage4", stage4_spec, stage4_runs)
        self.aggregates.append(stage4_agg)

        # Stage 5: rerun top-2 robust configs from stages 2-4 with expanded seeds.
        prior_aggs = [stage2_agg, stage3_alt_agg, stage4_agg]
        top2_aggs = sorted(
            prior_aggs,
            key=lambda a: (
                a.val_macro_f1_mean,
                a.val_weighted_f1_mean,
                -a.val_macro_f1_std,
                -a.val_weighted_f1_std,
            ),
            reverse=True,
        )[:2]

        stage5_results: list[RunResult] = []
        stage5_aggs: list[CandidateAggregate] = []
        for i, agg in enumerate(top2_aggs, start=1):
            spec = self.to_spec(agg)
            runs = self.train_and_eval_val_many(f"stage5_top{i}", spec, seeds=self.stage5_seed_values)
            stage5_results.extend(runs)
            stage5_aggs.append(self.aggregate(f"stage5_top{i}", spec, runs))
        self.results.extend(stage5_results)
        self.aggregates.extend(stage5_aggs)

        # Final winner by robust aggregate; test the best-seed run within that config.
        winner_final_agg = self.best_aggregate(stage5_aggs)
        winner_final_runs = [
            r
            for r in stage5_results
            if (
                r.model_name == winner_final_agg.model_name
                and r.pretrained == winner_final_agg.pretrained
                and r.fusion == winner_final_agg.fusion
                and r.head_type == winner_final_agg.head_type
                and r.output_mode == winner_final_agg.output_mode
            )
        ]
        winner_final = self.best(winner_final_runs)
        test_metrics_path = self.evaluate_test(winner_final)

        summary = {
            "created_at_utc": _utc_now_token(),
            "plan_tag": self.plan_tag,
            "clear_scope": self.args.clear_scope,
            "selection_rule": (
                "robust aggregate: max mean(val_macro_f1), tie-break mean(val_weighted_f1), "
                "then lower macro std"
            ),
            "runtime": {
                "python_executable": self.python,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cwd": str(PROJECT_ROOT),
                "argv": sys.argv,
            },
            "base_seed": self.args.base_seed,
            "seed_list": self.seed_values,
            "stage5_seed_list": self.stage5_seed_values,
            "results": [asdict(r) for r in self.results],
            "aggregate_results": [asdict(a) for a in self.aggregates],
            "winner_aggregate": asdict(winner_final_agg),
            "winner": asdict(winner_final),
            "winner_test_metrics_path": str(test_metrics_path),
        }

        if (not self.args.dry_run) and test_metrics_path.exists():
            summary["winner_test_metrics"] = _read_json(test_metrics_path)

        summary_path = self.analysis_dir / f"summary_{self.plan_tag}.json"
        ensure_dir(summary_path.parent)
        if not self.args.dry_run:
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            # Copy winner checkpoint/config for easy handoff.
            winner_dir = Path(winner_final.run_dir)
            export_dir = self.analysis_dir / f"winner_{self.plan_tag}"
            ensure_dir(export_dir)
            shutil.copy2(winner_dir / "best.ckpt", export_dir / "best.ckpt")
            shutil.copy2(winner_dir / "run_config.json", export_dir / "run_config.json")

        print("experiment_plan_complete=ok")
        print(f"summary={summary_path}")
        print(f"winner_run={winner_final.run_dir}")
        print(f"logs_dir={self.logs_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run end-to-end CLIP fusion-head model selection plan.")
    p.add_argument("--plan-tag", type=str, default=None)
    p.add_argument("--clear-scope", choices=["none", "runs", "all"], default="all")
    p.add_argument("--dry-run", action="store_true")

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
    # g4dn.xlarge defaults: conservative and stable first-run profile.
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batch-size-l14", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--class-weighting", choices=["none", "inverse", "effective"], default="effective")
    p.add_argument("--beta", type=float, default=0.999)
    p.add_argument("--loss-type", choices=["ce", "focal"], default="focal")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--lambda-unsafe", type=float, default=1.0)
    p.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--early-stopping-patience", type=int, default=2)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--cache-batch-size-l14", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--oom-retry", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-train-batch-size", type=int, default=16)
    p.add_argument("--min-cache-batch-size", type=int, default=1)

    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--seed-list", type=str, default="42,43,44")
    p.add_argument("--stage5-seed-list", type=str, default="40,41,42,43,44")
    return p


def main() -> None:
    args = build_parser().parse_args()
    runner = ExperimentRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
