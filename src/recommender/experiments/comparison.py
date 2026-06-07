"""Shared orchestration for offline recommender model comparison."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from recommender.data.movielens import (
    build_sparse_interaction_matrix,
    build_user_item_sets,
    filter_catalog_to_items,
    prepare_interactions,
    read_movielens,
)
from recommender.data.tmdb import ENRICHED_FIELDS, ENRICHED_TEXT_FIELDS
from recommender.eval.metrics import evaluate_score_fn
from recommender.models.base import ModelSkip
from recommender.models.baselines import (
    ContentAverageRecommender,
    EASERecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
    SLIMElasticNetRecommender,
    SVDRankingRecommender,
    UserKNNRecommender,
)
from recommender.models.learned_two_tower import LearnedTwoTowerRecommender
from recommender.models.matrix_factorization import (
    BPRMFRecommender,
    ImplicitALSRecommender,
    LightFMWARPRecommender,
    LightGCNRecommender,
    NeuMFRecommender,
)
from recommender.models.rankers import SGDRankHybridRecommender, WeightedHybridRecommender
from recommender.models.two_tower import EmbeddingBackend, encode_item_texts


@dataclass
class ExperimentDataset:
    """A split dataset plus reusable matrices and content embeddings."""

    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    catalog: pd.DataFrame
    train_matrix: sparse.csr_matrix
    train_user_items: dict[int, set[int]]
    val_user_items: dict[int, set[int]]
    test_user_items: dict[int, set[int]]
    user_mapping: dict[int, int]
    item_mapping: dict[int, int]
    content_embeddings: np.ndarray | None = None
    content_embeddings_no_tmdb: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_users(self) -> int:
        return len(self.user_mapping)

    @property
    def num_items(self) -> int:
        return len(self.item_mapping)


@dataclass
class ComparisonConfig:
    k: int = 10
    batch_size: int = 512
    models: str = "core"
    content_backend: EmbeddingBackend = "tfidf"
    sbert_model: str = "sentence-transformers/all-mpnet-base-v2"
    min_rating: float = 4.0
    epochs: int = 5
    device: str = "cpu"
    knn_top_k: int = 100
    svd_components: int = 64
    mf_dim: int = 64
    max_ease_items: int = 8000
    max_slim_items: int = 1000
    max_ranker_samples: int = 200_000
    seed: int = 42


def load_experiment_dataset(
    name: str,
    data_dir: str | Path,
    enriched_catalog_path: str | Path | None,
    config: ComparisonConfig,
) -> ExperimentDataset:
    """Load a MovieLens-compatible dataset and prepare shared experiment tensors."""
    data_dir = Path(data_dir)
    enriched_catalog = Path(enriched_catalog_path) if enriched_catalog_path else None
    raw = read_movielens(data_dir)
    prepared = prepare_interactions(raw.ratings, min_rating=config.min_rating)
    catalog = _ordered_catalog(raw.movies, enriched_catalog, prepared.item_mapping)
    content_embeddings = encode_item_texts(catalog, backend=config.content_backend, model_name=config.sbert_model)
    content_embeddings_no_tmdb = encode_item_texts(
        _catalog_without_tmdb_text(catalog),
        backend=config.content_backend,
        model_name=config.sbert_model,
    )
    train_matrix = build_sparse_interaction_matrix(prepared.train, prepared.num_users, prepared.num_items)
    split_strategy = "random_per_user_synthetic_timestamp" if name == "letterboxd" else "timestamp_per_user"
    return ExperimentDataset(
        name=name,
        train=prepared.train,
        val=prepared.val,
        test=prepared.test,
        catalog=catalog,
        train_matrix=train_matrix,
        train_user_items=build_user_item_sets(prepared.train),
        val_user_items=build_user_item_sets(prepared.val),
        test_user_items=build_user_item_sets(prepared.test),
        user_mapping=prepared.user_mapping,
        item_mapping=prepared.item_mapping,
        content_embeddings=content_embeddings,
        content_embeddings_no_tmdb=content_embeddings_no_tmdb,
        metadata={
            "data_dir": str(data_dir),
            "enriched_catalog": str(enriched_catalog) if enriched_catalog else "",
            "split_strategy": split_strategy,
            "min_rating": config.min_rating,
            "content_backend": config.content_backend,
        },
    )


def run_comparison_for_dataset(dataset: ExperimentDataset, config: ComparisonConfig) -> list[dict[str, Any]]:
    """Fit/evaluate all requested models for one dataset."""
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    base_models: list[Any] = [
        RandomRecommender(seed=config.seed),
        PopularityRecommender(name="popularity_only"),
        ItemKNNRecommender(top_k=config.knn_top_k),
        UserKNNRecommender(top_k=config.knn_top_k),
        SVDRankingRecommender(n_components=config.svd_components, random_state=config.seed),
        EASERecommender(max_items=config.max_ease_items),
        ContentAverageRecommender(name=f"{config.content_backend}_only", embedding_attr="content_embeddings"),
        BPRMFRecommender(embedding_dim=config.mf_dim, epochs=config.epochs, device=config.device, seed=config.seed),
        LightGCNRecommender(embedding_dim=config.mf_dim, epochs=config.epochs, device=config.device, seed=config.seed),
        LearnedTwoTowerRecommender(embedding_dim=config.mf_dim, epochs=config.epochs, device=config.device, seed=config.seed),
    ]
    if config.models == "full":
        base_models.extend(
            [
                SLIMElasticNetRecommender(max_items=config.max_slim_items),
                ImplicitALSRecommender(factors=config.mf_dim, iterations=max(1, config.epochs)),
                LightFMWARPRecommender(no_components=config.mf_dim, epochs=max(1, config.epochs), random_state=config.seed),
                NeuMFRecommender(embedding_dim=max(8, config.mf_dim // 2), epochs=config.epochs, device=config.device, seed=config.seed),
            ]
        )

    for model in base_models:
        model, row = fit_and_evaluate_model(dataset, model, config)
        rows.append(row)
        if row["status"] == "ok":
            fitted[row["model"]] = model

    no_tmdb_content = ContentAverageRecommender(name=f"{config.content_backend}_no_tmdb_only", embedding_attr="content_embeddings_no_tmdb")
    no_tmdb_content, no_tmdb_row = fit_and_evaluate_model(dataset, no_tmdb_content, config, include_in_summary=False)
    if no_tmdb_row["status"] == "ok":
        fitted[no_tmdb_row["model"]] = no_tmdb_content

    hybrid_components = ["lightgcn_only", "learned_two_tower", f"{config.content_backend}_only"]
    no_tmdb_components = ["lightgcn_only", f"{config.content_backend}_no_tmdb_only"]
    hybrid_specs = [
        ("hybrid_weighted_no_popularity", "weighted", hybrid_components, False),
        ("hybrid_weighted_full", "weighted", hybrid_components, True),
        ("hybrid_ranker_no_popularity", "ranker", hybrid_components, False),
        ("hybrid_ranker_full", "ranker", hybrid_components, True),
        ("hybrid_no_tmdb", "weighted", no_tmdb_components, True),
    ]
    for name, kind, component_names, include_popularity in hybrid_specs:
        components, missing = _collect_components(fitted, component_names)
        if missing:
            rows.append(skipped_row(dataset, name, "skipped_dependency", f"missing fitted components: {', '.join(missing)}"))
            continue
        if kind == "weighted":
            model = WeightedHybridRecommender(
                components=components,
                include_popularity=include_popularity,
                tune=True,
                k=config.k,
                name=name,
            )
        else:
            model = SGDRankHybridRecommender(
                components=components,
                include_popularity=include_popularity,
                max_train_samples=config.max_ranker_samples,
                seed=config.seed,
                name=name,
            )
        model, row = fit_and_evaluate_model(dataset, model, config)
        rows.append(row)
        if row["status"] == "ok":
            fitted[row["model"]] = model

    return rows


def run_comparison(
    datasets: list[tuple[str, Path, Path | None]],
    config: ComparisonConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, data_dir, enriched_catalog in datasets:
        try:
            dataset = load_experiment_dataset(name, data_dir, enriched_catalog, config)
        except Exception as exc:
            rows.append(
                {
                    "dataset": name,
                    "model": "__dataset_load__",
                    "status": "failed",
                    "error": str(exc),
                    "seconds": 0.0,
                    "metadata": {"data_dir": str(data_dir), "enriched_catalog": str(enriched_catalog) if enriched_catalog else ""},
                }
            )
            continue
        rows.extend(run_comparison_for_dataset(dataset, config))
    return rows


def fit_and_evaluate_model(
    dataset: ExperimentDataset,
    model: Any,
    config: ComparisonConfig,
    include_in_summary: bool = True,
) -> tuple[Any | None, dict[str, Any]]:
    start = time.perf_counter()
    try:
        fitted = model.fit(dataset)
        metrics = evaluate_score_fn(
            dataset.num_users,
            dataset.num_items,
            lambda users: fitted.score_users(users),
            dataset.train_user_items,
            dataset.test_user_items,
            k=config.k,
            batch_size=config.batch_size,
        )
        return fitted, {
            "dataset": dataset.name,
            "model": fitted.name,
            "status": "ok",
            "metrics": metrics,
            "seconds": round(time.perf_counter() - start, 4),
            "metadata": _jsonable({**dataset.metadata, **getattr(fitted, "metadata", {})}),
            "include_in_summary": include_in_summary,
        }
    except ModelSkip as exc:
        status = _skip_status(str(exc))
        return None, skipped_row(dataset, model.name, status, str(exc), time.perf_counter() - start, include_in_summary)
    except Exception as exc:
        return None, skipped_row(dataset, model.name, "failed", str(exc), time.perf_counter() - start, include_in_summary)


def skipped_row(
    dataset: ExperimentDataset,
    model_name: str,
    status: str,
    error: str,
    seconds: float = 0.0,
    include_in_summary: bool = True,
) -> dict[str, Any]:
    return {
        "dataset": dataset.name,
        "model": model_name,
        "status": status,
        "error": error,
        "seconds": round(seconds, 4),
        "metadata": _jsonable(dataset.metadata),
        "include_in_summary": include_in_summary,
    }


def write_comparison_outputs(rows: list[dict[str, Any]], output_dir: str | Path, k: int = 10) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison_results.csv"
    json_path = output_dir / "comparison_results.json"
    markdown_path = output_dir / "comparison_summary.md"

    flattened = [_flatten_row(row, k) for row in rows]
    pd.DataFrame(flattened).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(rows), fh, ensure_ascii=False, indent=2)
    markdown_path.write_text(_summary_markdown(rows, k), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": markdown_path}


def _ordered_catalog(raw_movies: pd.DataFrame, enriched_path: Path | None, item_mapping: dict[int, int]) -> pd.DataFrame:
    if enriched_path and enriched_path.exists():
        source = pd.read_parquet(enriched_path)
        if "movieId" not in source.columns:
            source = raw_movies
    else:
        source = raw_movies
    catalog = filter_catalog_to_items(source, item_mapping)
    for field in ENRICHED_FIELDS:
        if field not in catalog.columns:
            catalog[field] = ""
    return catalog


def _catalog_without_tmdb_text(catalog: pd.DataFrame) -> pd.DataFrame:
    stripped = catalog.copy()
    for field in ENRICHED_TEXT_FIELDS:
        if field in stripped.columns:
            stripped[field] = ""
    return stripped


def _collect_components(fitted: dict[str, Any], names: list[str]) -> tuple[list[Any], list[str]]:
    components: list[Any] = []
    missing: list[str] = []
    for name in names:
        component = fitted.get(name)
        if component is None:
            missing.append(name)
        else:
            components.append(component)
    return components, missing


def _skip_status(message: str) -> str:
    lower = message.lower()
    if "not installed" in lower or "requires" in lower or "missing fitted components" in lower:
        return "skipped_dependency"
    if "skipped because" in lower or ">" in lower:
        return "skipped_limit"
    return "skipped"


def _flatten_row(row: dict[str, Any], k: int) -> dict[str, Any]:
    metrics = row.get("metrics", {}) or {}
    metadata = row.get("metadata", {}) or {}
    return {
        "dataset": row.get("dataset", ""),
        "model": row.get("model", ""),
        "status": row.get("status", ""),
        f"precision@{k}": metrics.get(f"precision@{k}", ""),
        f"recall@{k}": metrics.get(f"recall@{k}", ""),
        f"ndcg@{k}": metrics.get(f"ndcg@{k}", ""),
        "mrr": metrics.get("mrr", ""),
        "seconds": row.get("seconds", ""),
        "error": row.get("error", ""),
        "metadata_json": json.dumps(_jsonable(metadata), ensure_ascii=False),
    }


def _summary_markdown(rows: list[dict[str, Any]], k: int) -> str:
    lines = [
        "# Comparison Summary",
        "",
        f"Ranking metrics use train-item masking and Top-{k} evaluation on the test split.",
    ]
    datasets = sorted({row.get("dataset", "") for row in rows if row.get("include_in_summary", True)})
    for dataset in datasets:
        dataset_rows = [row for row in rows if row.get("dataset") == dataset and row.get("include_in_summary", True)]
        ok_rows = [row for row in dataset_rows if row.get("status") == "ok"]
        ok_rows.sort(key=lambda row: row.get("metrics", {}).get(f"ndcg@{k}", 0.0), reverse=True)
        lines.extend(["", f"## {dataset}", "", f"| Model | Precision@{k} | Recall@{k} | NDCG@{k} | MRR | Seconds |", "|---|---:|---:|---:|---:|---:|"])
        if not ok_rows:
            lines.append("| none |  |  |  |  |  |")
        for row in ok_rows:
            metrics = row.get("metrics", {})
            lines.append(
                "| {model} | {precision:.4f} | {recall:.4f} | {ndcg:.4f} | {mrr:.4f} | {seconds:.2f} |".format(
                    model=row.get("model", ""),
                    precision=float(metrics.get(f"precision@{k}", 0.0)),
                    recall=float(metrics.get(f"recall@{k}", 0.0)),
                    ndcg=float(metrics.get(f"ndcg@{k}", 0.0)),
                    mrr=float(metrics.get("mrr", 0.0)),
                    seconds=float(row.get("seconds", 0.0)),
                )
            )
        skipped = [row for row in dataset_rows if row.get("status") != "ok"]
        if skipped:
            lines.extend(["", "Skipped/failed:", ""])
            for row in skipped:
                error = row.get("error", "")
                lines.append(f"- `{row.get('model', '')}`: `{row.get('status', '')}` {error}")
    lines.append("")
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value

