from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import DEFAULT_POSITIVE_THRESHOLD, DEFAULT_SEED, PROTOCOL_NAME, leaderboard_row, write_leaderboard
from evaluation.leaderboard import sort_leaderboard
from scripts.audit_data_quality import build_report, write_reports as write_data_quality_reports
from scripts.benchmark_recbole import prepare_recbole_dataset
from scripts.build_leaderboard import rows_from_manifest, rows_from_recbole_report
from scripts.train_baseline import evaluate as evaluate_baseline
from scripts.train_content_baseline import evaluate as evaluate_content_baseline
from scripts.train_lightgcn import train as train_lightgcn
from scripts.train_svd import train as train_svd
from scripts.train_two_tower import train as train_two_tower
from scripts.tune_hybrid import tune as tune_hybrid


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset
    data_dir = Path(args.data_dir)
    scoped_root = Path(args.artifact_root) / dataset
    checkpoint_root = Path(args.checkpoint_root) / dataset
    leaderboard_prefix = Path(args.leaderboard_root) / dataset
    scoped_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    completed: list[str] = []
    skipped: dict[str, str] = {}

    if not args.skip_audit:
        report = build_report(str(data_dir), example_limit=10)
        write_data_quality_reports(report, Path(args.data_quality_root) / dataset)
        completed.append("data_audit")

    prepare_recbole_dataset(data_dir, args.recbole_root, dataset, args.positive_threshold)
    completed.append("recbole_atomic_split")

    if should_run(args, "baseline"):
        artifact_dir = scoped_root / "baseline"
        metrics = evaluate_baseline(str(data_dir), args.top_k, str(artifact_dir), dataset)
        rows.append(
            leaderboard_row(
                dataset=dataset,
                model="hybrid-baseline",
                model_family="hybrid",
                source="runtime_fit",
                metrics=metrics,
                artifact_dir=str(artifact_dir),
                command=f"scripts/train_baseline.py --data-dir {data_dir}",
            )
        )
        completed.append("baseline")

    if should_run(args, "content"):
        artifact_dir = scoped_root / "content"
        metrics = evaluate_content_baseline(str(data_dir), args.top_k, str(artifact_dir), dataset)
        rows.append(
            leaderboard_row(
                dataset=dataset,
                model="content-tfidf",
                model_family="content",
                source="content_baseline",
                metrics=metrics,
                artifact_dir=str(artifact_dir),
                command=f"scripts/train_content_baseline.py --data-dir {data_dir}",
            )
        )
        completed.append("content")

    if should_run(args, "svd"):
        try:
            artifact_dir = scoped_root / "svd"
            result = train_svd(default_svd_args(args, data_dir, checkpoint_root / "svd.pt", artifact_dir))
            rows.append(
                leaderboard_row(
                    dataset=dataset,
                    model="svd-pytorch",
                    model_family="svd",
                    source="pytorch_svd",
                    metrics=asdict(result),
                    artifact_dir=str(artifact_dir),
                    command=f"scripts/train_svd.py --data-dir {data_dir}",
                )
            )
            completed.append("svd")
        except SystemExit as exc:
            skipped["svd"] = str(exc)

    if should_run(args, "lightgcn"):
        try:
            artifact_dir = scoped_root / "lightgcn"
            result = train_lightgcn(default_lightgcn_args(args, data_dir, checkpoint_root / "lightgcn.pt", artifact_dir))
            rows.append(
                leaderboard_row(
                    dataset=dataset,
                    model="lightgcn-pytorch",
                    model_family="lightgcn",
                    source="pytorch_lightgcn",
                    metrics=asdict(result),
                    artifact_dir=str(artifact_dir),
                    command=f"scripts/train_lightgcn.py --data-dir {data_dir}",
                )
            )
            completed.append("lightgcn")
        except SystemExit as exc:
            skipped["lightgcn"] = str(exc)

    if should_run(args, "two_tower"):
        try:
            artifact_dir = scoped_root / "two-tower"
            result = train_two_tower(default_two_tower_args(args, data_dir, checkpoint_root / "two_tower.pt", artifact_dir))
            rows.append(
                leaderboard_row(
                    dataset=dataset,
                    model=f"two-tower-{args.content_backend}",
                    model_family="two_tower",
                    source="pytorch_two_tower",
                    metrics=asdict(result),
                    artifact_dir=str(artifact_dir),
                    command=f"scripts/train_two_tower.py --data-dir {data_dir} --content-backend {args.content_backend}",
                )
            )
            completed.append("two_tower")
        except SystemExit as exc:
            skipped["two_tower"] = str(exc)

    if should_run(args, "recbole"):
        recbole_report = Path(args.recbole_report or Path("artifacts/benchmarks") / f"{dataset}.json")
        if has_matching_recbole_protocol(recbole_report):
            rows.extend(rows_from_recbole_report(recbole_report, dataset, scoped_root))
            completed.append("recbole_report")
        else:
            skipped["recbole_report"] = f"missing or stale report: {recbole_report}"

    if should_run(args, "tune"):
        tuned_rows = tune_available_artifacts(args, data_dir, scoped_root)
        rows.extend(tuned_rows)
        if tuned_rows:
            completed.append("tune")
        else:
            skipped["tune"] = "no compatible collaborative artifacts"

    rows = merge_rows(existing_artifact_rows(scoped_root), rows)
    if rows:
        write_leaderboard(rows, leaderboard_prefix)
        best_artifact = first_existing_artifact(sort_leaderboard(rows))
        if best_artifact:
            copy_artifact(best_artifact, scoped_root / "latest")
            sync_global_latest = args.sync_global_latest
            if sync_global_latest is None:
                sync_global_latest = dataset == "ml-latest-small"
            if sync_global_latest:
                copy_artifact(best_artifact, Path(args.artifact_root) / "latest")
    update_overall_leaderboard(Path(args.leaderboard_root), dataset, rows)

    result = {
        "dataset": dataset,
        "data_dir": str(data_dir),
        "protocol": PROTOCOL_NAME,
        "profile": args.profile,
        "completed": completed,
        "skipped": skipped,
        "leaderboard": str(leaderboard_prefix.with_suffix(".csv")),
    }
    (leaderboard_prefix.parent / f"{dataset}.pipeline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def existing_artifact_rows(scoped_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(scoped_root.glob("*/manifest.json")):
        if manifest_path.parent.name == "latest":
            continue
        rows.extend(rows_from_manifest(manifest_path))
    return rows


def merge_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for group in groups:
        for row in group:
            key = (str(row.get("dataset", "")), str(row.get("model", "")), str(row.get("artifact_dir", "")))
            merged[key] = row
    return list(merged.values())


def tune_available_artifacts(args: argparse.Namespace, data_dir: Path, scoped_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = {
        "SVD": scoped_root / "svd",
        "LightGCN": scoped_root / "lightgcn",
        "TwoTower": scoped_root / "two-tower",
        "BPR": scoped_root / "recbole-bpr",
        "RecBoleLightGCN": scoped_root / "recbole-lightgcn",
    }
    for name, artifact_dir in candidates.items():
        if not (artifact_dir / "manifest.json").exists():
            continue
        output_dir = scoped_root / f"hybrid-{name.lower()}-{args.content_backend}"
        tune_args = SimpleNamespace(
            data_dir=str(data_dir),
            dataset_name=args.dataset,
            cf_model=name,
            cf_artifact_dir=str(artifact_dir),
            artifact_root=str(scoped_root),
            content_backend=args.content_backend,
            sbert_model_name=args.sbert_model_name,
            output_dir=str(output_dir),
            top_k=args.top_k,
            positive_threshold=args.positive_threshold,
            eval_user_limit=args.eval_user_limit,
            seed=args.seed,
        )
        result = tune_hybrid(tune_args)
        rows.append(
            leaderboard_row(
                dataset=args.dataset,
                model=f"hybrid-{name.lower()}-{result.get('content_backend', args.content_backend)}",
                model_family="hybrid",
                source="tuned_hybrid",
                metrics={
                    "test": result.get("test", {}),
                    "test_segments": result.get("test_segments", {}),
                },
                artifact_dir=str(output_dir),
                tuned_weights=result.get("weights", {}),
                command=f"scripts/tune_hybrid.py --data-dir {data_dir} --cf-artifact-dir {artifact_dir} --content-backend {args.content_backend}",
            )
        )
    return rows


def should_run(args: argparse.Namespace, name: str) -> bool:
    return name in args.models or "all" in args.models


def default_svd_args(args: argparse.Namespace, data_dir: Path, artifact_path: Path, recommender_artifact_dir: Path) -> SimpleNamespace:
    smoke = args.profile == "smoke"
    return SimpleNamespace(
        data_dir=str(data_dir),
        artifact_path=str(artifact_path),
        recommender_artifact_dir=str(recommender_artifact_dir),
        dataset_name=args.dataset,
        factors=12 if smoke else 24,
        epochs=2 if smoke else 80,
        batch_size=1024 if smoke else 2048,
        lr=0.01,
        optimizer="adamw",
        momentum=0.0,
        weight_decay=0.0,
        embedding_reg=0.02,
        bias_reg=0.005,
        bias_shrinkage=5.0,
        init_std=0.05,
        max_grad_norm=5.0,
        lr_decay_factor=0.5,
        lr_patience=2,
        min_lr=1e-5,
        patience=2 if smoke else 8,
        top_k=args.top_k,
        positive_threshold=args.positive_threshold,
        max_train_pairs=args.max_train_pairs,
        eval_user_limit=args.eval_user_limit,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        cf_weight=0.55,
        content_weight=0.35,
        popularity_weight=0.10,
    )


def default_lightgcn_args(args: argparse.Namespace, data_dir: Path, artifact_path: Path, recommender_artifact_dir: Path) -> SimpleNamespace:
    smoke = args.profile == "smoke"
    return SimpleNamespace(
        data_dir=str(data_dir),
        artifact_path=str(artifact_path),
        recommender_artifact_dir=str(recommender_artifact_dir),
        dataset_name=args.dataset,
        factors=16 if smoke else 64,
        layers=1 if smoke else 3,
        epochs=2 if smoke else 50,
        batch_size=512 if smoke else 1024,
        lr=0.001,
        loss=args.lightgcn_loss,
        warp_margin=args.warp_margin,
        warp_max_trials=args.warp_max_trials,
        weight_decay=1e-5,
        l2_reg=0.0,
        max_grad_norm=5.0,
        patience=2 if smoke else 10,
        top_k=args.top_k,
        positive_threshold=args.positive_threshold,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        cf_weight=0.55,
        content_weight=0.35,
        popularity_weight=0.10,
    )


def default_two_tower_args(args: argparse.Namespace, data_dir: Path, artifact_path: Path, recommender_artifact_dir: Path) -> SimpleNamespace:
    smoke = args.profile == "smoke"
    return SimpleNamespace(
        data_dir=str(data_dir),
        artifact_path=str(artifact_path),
        recommender_artifact_dir=str(recommender_artifact_dir),
        dataset_name=args.dataset,
        content_backend=args.content_backend,
        sbert_model_name=args.sbert_model_name,
        max_feature_dim=32 if smoke else 256,
        hidden_dim=32 if smoke else 128,
        output_dim=16 if smoke else 64,
        dropout=0.1,
        epochs=2 if smoke else 50,
        batch_size=512 if smoke else 1024,
        lr=0.001,
        weight_decay=1e-5,
        max_grad_norm=5.0,
        patience=2 if smoke else 8,
        top_k=args.top_k,
        positive_threshold=args.positive_threshold,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        cf_weight=0.55,
        content_weight=0.35,
        popularity_weight=0.10,
    )


def has_matching_recbole_protocol(report_path: Path) -> bool:
    if not report_path.exists():
        return False
    try:
        results = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(results) and all(result.get("protocol") == PROTOCOL_NAME for result in results if isinstance(result, dict))


def first_existing_artifact(rows: list[dict[str, Any]]) -> Path | None:
    for row in rows:
        path = Path(str(row.get("artifact_dir", "")))
        if (path / "manifest.json").exists() and (path / "collaborative.npz").exists():
            return path
    return None


def copy_artifact(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def update_overall_leaderboard(root: Path, dataset: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    overall_path = root / "overall.csv"
    current = pd.read_csv(overall_path) if overall_path.exists() else pd.DataFrame()
    new_rows = pd.DataFrame(sort_leaderboard(rows))
    if not current.empty and "dataset" in current.columns:
        current = current.loc[current["dataset"].astype(str) != dataset]
    pd.concat([current, new_rows], ignore_index=True).to_csv(overall_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local benchmark pipeline and build a leaderboard.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--models", default="baseline,content,svd,lightgcn,two_tower,tune,recbole")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--positive-threshold", type=float, default=DEFAULT_POSITIVE_THRESHOLD)
    parser.add_argument("--content-backend", choices=["tfidf", "sbert", "auto"], default="tfidf")
    parser.add_argument("--sbert-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--lightgcn-loss", choices=["bpr", "warp"], default="bpr")
    parser.add_argument("--warp-margin", type=float, default=1.0)
    parser.add_argument("--warp-max-trials", type=int, default=20)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--eval-user-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="")
    parser.add_argument("--artifact-root", default="artifacts/recommender")
    parser.add_argument("--checkpoint-root", default="artifacts/checkpoints")
    parser.add_argument("--leaderboard-root", default="artifacts/leaderboards")
    parser.add_argument("--data-quality-root", default="artifacts/data_quality")
    parser.add_argument("--recbole-root", default="artifacts/recbole-splits")
    parser.add_argument("--recbole-report", default="")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--sync-global-latest", dest="sync_global_latest", action="store_true")
    parser.add_argument("--no-sync-global-latest", dest="sync_global_latest", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(sync_global_latest=None)
    args = parser.parse_args()
    args.models = [model.strip() for model in str(args.models).split(",") if model.strip()]
    return args


def main() -> None:
    print(json.dumps(run_pipeline(parse_args()), indent=2))


if __name__ == "__main__":
    main()
