from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import MovieLensDataLoader


DEFAULT_MODELS = ["Pop", "ItemKNN", "BPR", "LightGCN"]


def dataset_name_from_dir(data_dir: str | Path) -> str:
    return Path(data_dir).resolve().name.replace("_", "-")


def prepare_recbole_dataset(
    data_dir: str | Path,
    output_root: str | Path,
    dataset_name: str | None = None,
    positive_threshold: float = 4.0,
) -> Path:
    dataset = dataset_name or dataset_name_from_dir(data_dir)
    output_dir = Path(output_root) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = MovieLensDataLoader(data_dir).load()
    interactions = bundle.ratings.loc[bundle.ratings["rating"] >= positive_threshold].copy()
    interactions = interactions.sort_values(["userId", "timestamp", "movieId"])
    interactions = interactions.rename(
        columns={
            "userId": "user_id:token",
            "movieId": "item_id:token",
            "rating": "rating:float",
            "timestamp": "timestamp:float",
        }
    )
    interactions = interactions[["user_id:token", "item_id:token", "rating:float", "timestamp:float"]]
    interactions.to_csv(output_dir / f"{dataset}.inter", sep="\t", index=False)
    return output_dir


def build_recbole_config(args: argparse.Namespace, model_name: str, dataset_name: str, data_root: Path) -> dict[str, Any]:
    checkpoint_dir = Path(args.checkpoint_dir) / model_name.lower()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "data_path": str(data_root),
        "field_separator": "\t",
        "seq_separator": " ",
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": "rating",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
        "eval_args": {
            "group_by": "user",
            "order": "TO",
            "split": {"RS": [0.8, 0.1, 0.1]},
            "mode": "full",
        },
        "metrics": ["Precision", "Recall", "NDCG", "MRR"],
        "topk": args.top_k,
        "valid_metric": f"NDCG@{args.top_k[0]}",
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "stopping_step": args.stopping_step,
        "checkpoint_dir": str(checkpoint_dir),
        "seed": args.seed,
        "reproducibility": True,
        "show_progress": args.verbose,
        "device": args.device,
    }
    if model_name in {"BPR", "LightGCN", "SGL", "NCL"}:
        config.update(
            {
                "embedding_size": args.embedding_size,
                "learning_rate": args.learning_rate,
                "reg_weight": args.reg_weight,
            }
        )
    if model_name in {"LightGCN", "SGL", "NCL"}:
        config["n_layers"] = args.n_layers
    return config


def normalise_result(model_name: str, result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        payload = dict(result)
    elif isinstance(result, tuple):
        payload = {"raw_result": [str(value) for value in result]}
        for value in result:
            if isinstance(value, dict):
                payload.update(value)
    else:
        payload = {"raw_result": str(result)}

    test_result = payload.get("test_result")
    if not isinstance(test_result, dict):
        test_result = {key: value for key, value in payload.items() if "@" in str(key)}
    return {
        "model": model_name,
        "best_valid_score": payload.get("best_valid_score"),
        "best_valid_result": payload.get("best_valid_result"),
        "test_result": test_result,
    }


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(checkpoint_dir.rglob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def export_recbole_embeddings(
    checkpoint_dir: Path,
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    metrics: dict[str, Any],
    positive_threshold: float,
) -> Path | None:
    checkpoint = find_latest_checkpoint(checkpoint_dir / model_name.lower())
    if checkpoint is None:
        return None

    try:
        import torch
        from recbole.quick_start import load_data_and_model
    except ImportError:
        return None

    try:
        config, model, dataset, *_ = load_data_and_model(model_file=str(checkpoint))
    except Exception:
        return None

    model.eval()
    with torch.no_grad():
        try:
            user_embeddings, item_embeddings = model.forward()
        except Exception:
            user_embeddings = model.user_embedding.weight
            item_embeddings = model.item_embedding.weight

    user_embeddings = user_embeddings.detach().cpu().numpy().astype(np.float32)
    item_embeddings = item_embeddings.detach().cpu().numpy().astype(np.float32)
    user_ids = dataset.id2token(config["USER_ID_FIELD"], np.arange(user_embeddings.shape[0]))
    item_ids = dataset.id2token(config["ITEM_ID_FIELD"], np.arange(item_embeddings.shape[0]))

    keep_users = [idx for idx, token in enumerate(user_ids) if str(token) != "[PAD]"]
    keep_items = [idx for idx, token in enumerate(item_ids) if str(token) != "[PAD]"]
    raw_user_ids = np.asarray([int(user_ids[idx]) for idx in keep_users], dtype=np.int64)
    raw_item_ids = np.asarray([int(item_ids[idx]) for idx in keep_items], dtype=np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "collaborative.npz",
        user_ids=raw_user_ids,
        movie_ids=raw_item_ids,
        user_embeddings=user_embeddings[keep_users],
        item_embeddings=item_embeddings[keep_items],
        global_mean=np.asarray([0.0], dtype=np.float32),
    )
    manifest = {
        "artifact_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "model_name": f"recbole-{model_name}",
        "model_source": "recbole",
        "positive_threshold": positive_threshold,
        "weights": {"collaborative": 0.55, "content": 0.35, "popularity": 0.10},
        "collaborative": {"mode": "embedding", "engine": "recbole", "checkpoint": str(checkpoint)},
        "content": {"backend": "tfidf"},
        "files": {"collaborative": "collaborative.npz"},
        "metrics": metrics.get("test_result", {}),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def write_reports(results: list[dict[str, Any]], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for result in results:
        row = {"model": result["model"], "best_valid_score": result.get("best_valid_score")}
        test_result = result.get("test_result") or {}
        if isinstance(test_result, dict):
            row.update(test_result)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_prefix.with_suffix(".csv"), index=False)

    lines = ["| model | best_valid_score | metrics |", "| --- | ---: | --- |"]
    for result in results:
        metrics = result.get("test_result") or {}
        metric_text = ", ".join(f"{key}: {value}" for key, value in metrics.items()) if isinstance(metrics, dict) else str(metrics)
        lines.append(f"| {result['model']} | {result.get('best_valid_score', '')} | {metric_text} |")
    output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        from recbole.quick_start import run_recbole
    except ImportError as exc:
        raise SystemExit(
            "RecBole is required for this script. Use the trainer Docker profile or install requirements-train-recbole.txt."
        ) from exc

    dataset_name = args.dataset_name or dataset_name_from_dir(args.data_dir)
    recbole_root = Path(args.recbole_root)
    prepare_recbole_dataset(args.data_dir, recbole_root, dataset_name, args.positive_threshold)

    results: list[dict[str, Any]] = []
    for model_name in args.models:
        config = build_recbole_config(args, model_name, dataset_name, recbole_root)
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]]
        try:
            raw_result = run_recbole(model=model_name, dataset=dataset_name, config_dict=config)
        finally:
            sys.argv = original_argv
        result = normalise_result(model_name, raw_result)
        results.append(result)
        if model_name in {"BPR", "LightGCN", "SGL", "NCL"}:
            export_dir = Path(args.artifact_root) / f"recbole-{model_name.lower()}"
            result["artifact_dir"] = str(
                export_recbole_embeddings(Path(args.checkpoint_dir), export_dir, dataset_name, model_name, result, args.positive_threshold)
                or ""
            )

    write_reports(results, Path(args.output_prefix))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MovieLens data with RecBole general recommenders.")
    parser.add_argument("--data-dir", default="data/ml-latest-small")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--recbole-root", default="artifacts/recbole")
    parser.add_argument("--checkpoint-dir", default="artifacts/recbole/checkpoints")
    parser.add_argument("--artifact-root", default="artifacts/recommender")
    parser.add_argument("--output-prefix", default="artifacts/benchmarks/ml-latest-small")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--stopping-step", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--reg-weight", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.models = [model.strip() for model in str(args.models).split(",") if model.strip()]
    return args


def main() -> None:
    results = run_benchmark(parse_args())
    for result in results:
        print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
