"""Shared orchestration for offline recommender model comparison."""

from __future__ import annotations

import json
import hashlib
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
from recommender.models.rankers import SGDRankHybridRecommender, StrongHybridRankerRecommender, WeightedHybridRecommender
from recommender.models.two_tower import EmbeddingBackend, build_item_text, encode_item_texts


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
    split_stats: dict[str, Any] = field(default_factory=dict)

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
    preset: str = "none"
    hybrid_grid_step: float = 0.25
    embedding_cache_dir: Path | None = Path(".cache/recommender/content_embeddings")
    use_content_cache: bool = True
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
    content_embeddings = encode_item_texts_cached(
        catalog,
        dataset_name=name,
        variant="full",
        config=config,
    )
    content_embeddings_no_tmdb = encode_item_texts_cached(
        _catalog_without_tmdb_text(catalog),
        dataset_name=name,
        variant="no_tmdb",
        config=config,
    )
    train_matrix = build_sparse_interaction_matrix(prepared.train, prepared.num_users, prepared.num_items)
    split_stats = _split_stats(prepared.train, prepared.val, prepared.test, train_matrix)
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
        split_stats=split_stats,
        metadata={
            "data_dir": str(data_dir),
            "enriched_catalog": str(enriched_catalog) if enriched_catalog else "",
            "split_strategy": split_strategy,
            "min_rating": config.min_rating,
            "content_backend": config.content_backend,
            "split_stats": split_stats,
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
        BPRMFRecommender(embedding_dim=config.mf_dim, epochs=config.epochs, batch_size=config.batch_size, device=config.device, seed=config.seed),
        LightGCNRecommender(embedding_dim=config.mf_dim, epochs=config.epochs, batch_size=config.batch_size, device=config.device, seed=config.seed),
        LearnedTwoTowerRecommender(embedding_dim=config.mf_dim, epochs=config.epochs, batch_size=config.batch_size, device=config.device, seed=config.seed),
    ]
    if config.models == "full":
        base_models.extend(
            [
                SLIMElasticNetRecommender(max_items=config.max_slim_items),
                ImplicitALSRecommender(factors=config.mf_dim, iterations=max(1, config.epochs)),
                LightFMWARPRecommender(no_components=config.mf_dim, epochs=max(1, config.epochs), random_state=config.seed),
                NeuMFRecommender(embedding_dim=max(8, config.mf_dim // 2), epochs=config.epochs, batch_size=config.batch_size, device=config.device, seed=config.seed),
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
    if config.preset in {"letterboxd-pdf-clean", "letterboxd-strong"}:
        hybrid_specs.insert(0, ("hybrid_pdf_clean", "weighted", hybrid_components, True))
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
                grid_step=config.hybrid_grid_step if name == "hybrid_pdf_clean" else 0.25,
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

    if config.preset == "letterboxd-strong":
        strong_names = [
            "lightgcn_only",
            "learned_two_tower",
            f"{config.content_backend}_only",
            "ease",
            "item_knn_cosine",
            "user_knn_cosine",
            "svd_ranking",
            "lightfm_warp",
            "implicit_als",
            "popularity_only",
        ]
        components, missing = _collect_available_components(fitted, strong_names)
        if len(components) < 2:
            rows.append(skipped_row(dataset, "hybrid_strong_ranker", "skipped_dependency", f"not enough fitted components; missing: {', '.join(missing)}"))
        else:
            model = StrongHybridRankerRecommender(
                components=components,
                include_popularity=True,
                max_train_samples=config.max_ranker_samples,
                seed=config.seed,
                ranker="auto",
            )
            model, row = fit_and_evaluate_model(dataset, model, config)
            row["metadata"]["missing_optional_components"] = missing
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
        metrics.update(evaluate_slice_metrics(dataset, lambda users: fitted.score_users(users), config))
        return fitted, {
            "dataset": dataset.name,
            "model": fitted.name,
            "group": _model_group(fitted.name),
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
        "group": _model_group(model_name),
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


def encode_item_texts_cached(
    catalog: pd.DataFrame,
    dataset_name: str,
    variant: str,
    config: ComparisonConfig,
) -> np.ndarray:
    if not config.use_content_cache or config.embedding_cache_dir is None:
        return encode_item_texts(catalog, backend=config.content_backend, model_name=config.sbert_model)
    cache_dir = Path(config.embedding_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    docs = build_item_text(catalog)
    digest = hashlib.sha256()
    digest.update(config.content_backend.encode("utf-8"))
    digest.update(config.sbert_model.encode("utf-8"))
    digest.update("\n".join(docs).encode("utf-8", errors="ignore"))
    key = digest.hexdigest()[:20]
    model_key = config.sbert_model.split("/")[-1].replace(":", "_")
    cache_path = cache_dir / f"{dataset_name}_{variant}_{config.content_backend}_{model_key}_{key}.npy"
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32)
    embeddings = encode_item_texts(catalog, backend=config.content_backend, model_name=config.sbert_model)
    np.save(cache_path, embeddings.astype(np.float32))
    return embeddings.astype(np.float32)


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


def _collect_available_components(fitted: dict[str, Any], names: list[str]) -> tuple[list[Any], list[str]]:
    components: list[Any] = []
    missing: list[str] = []
    for name in names:
        component = fitted.get(name)
        if component is None:
            missing.append(name)
        else:
            components.append(component)
    return components, missing


def evaluate_slice_metrics(dataset: ExperimentDataset, score_fn, config: ComparisonConfig) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for prefix, ground_truth in _slice_ground_truths(dataset).items():
        if not ground_truth:
            continue
        values = evaluate_score_fn(
            dataset.num_users,
            dataset.num_items,
            score_fn,
            dataset.train_user_items,
            ground_truth,
            k=config.k,
            batch_size=config.batch_size,
        )
        for key, value in values.items():
            metrics[f"{prefix}_{key}"] = value
    return metrics


def _slice_ground_truths(dataset: ExperimentDataset) -> dict[str, dict[int, set[int]]]:
    train_counts = np.asarray([len(dataset.train_user_items.get(user, set())) for user in range(dataset.num_users)], dtype=np.float32)
    active_counts = train_counts[train_counts > 0]
    sparse_threshold = float(np.percentile(active_counts, 25)) if active_counts.size else 0.0
    item_counts = np.asarray(dataset.train_matrix.sum(axis=0)).ravel().astype(np.float32)
    nonzero_item_counts = item_counts[item_counts > 0]
    tail_threshold = float(np.percentile(nonzero_item_counts, 50)) if nonzero_item_counts.size else 0.0
    head_threshold = float(np.percentile(nonzero_item_counts, 75)) if nonzero_item_counts.size else 0.0
    tail_items = {int(idx) for idx, count in enumerate(item_counts) if 0 < count <= tail_threshold}
    head_items = {int(idx) for idx, count in enumerate(item_counts) if count >= head_threshold and count > 0}
    result = {
        "sparse_user": {},
        "warm_user": {},
        "long_tail": {},
        "head_item": {},
    }
    for user, truth in dataset.test_user_items.items():
        if not truth:
            continue
        if train_counts[int(user)] <= sparse_threshold:
            result["sparse_user"][int(user)] = set(truth)
        else:
            result["warm_user"][int(user)] = set(truth)
        tail_truth = set(truth) & tail_items
        if tail_truth:
            result["long_tail"][int(user)] = tail_truth
        head_truth = set(truth) & head_items
        if head_truth:
            result["head_item"][int(user)] = head_truth
    return result


def _split_stats(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, train_matrix: sparse.csr_matrix) -> dict[str, Any]:
    train_counts = train.groupby("user_idx").size() if not train.empty else pd.Series(dtype=np.int64)
    item_counts = np.asarray(train_matrix.sum(axis=0)).ravel().astype(np.float32)
    nonzero_item_counts = item_counts[item_counts > 0]
    sparse_threshold = float(train_counts.quantile(0.25)) if not train_counts.empty else 0.0
    tail_threshold = float(np.percentile(nonzero_item_counts, 50)) if nonzero_item_counts.size else 0.0
    head_threshold = float(np.percentile(nonzero_item_counts, 75)) if nonzero_item_counts.size else 0.0
    return {
        "users": int(train_matrix.shape[0]),
        "items": int(train_matrix.shape[1]),
        "train_interactions": int(len(train)),
        "val_interactions": int(len(val)),
        "test_interactions": int(len(test)),
        "train_users": int(train_counts.size),
        "train_items_nonzero": int(np.count_nonzero(item_counts)),
        "train_interactions_per_user_mean": float(train_counts.mean()) if not train_counts.empty else 0.0,
        "train_interactions_per_user_median": float(train_counts.median()) if not train_counts.empty else 0.0,
        "train_interactions_per_user_q25": sparse_threshold,
        "train_interactions_per_user_q75": float(train_counts.quantile(0.75)) if not train_counts.empty else 0.0,
        "sparse_users": int((train_counts <= sparse_threshold).sum()) if not train_counts.empty else 0,
        "warm_users": int((train_counts > sparse_threshold).sum()) if not train_counts.empty else 0,
        "sparse_user_threshold_q25": sparse_threshold,
        "long_tail_items": int(np.sum((item_counts > 0) & (item_counts <= tail_threshold))),
        "head_items": int(np.sum((item_counts > 0) & (item_counts >= head_threshold))),
        "tail_item_threshold_q50": tail_threshold,
        "head_item_threshold_q75": head_threshold,
    }


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
    flattened = {
        "dataset": row.get("dataset", ""),
        "model": row.get("model", ""),
        "group": row.get("group", _model_group(row.get("model", ""))),
        "status": row.get("status", ""),
        f"precision@{k}": metrics.get(f"precision@{k}", ""),
        f"recall@{k}": metrics.get(f"recall@{k}", ""),
        f"ndcg@{k}": metrics.get(f"ndcg@{k}", ""),
        "mrr": metrics.get("mrr", ""),
        "seconds": row.get("seconds", ""),
        "error": row.get("error", ""),
        "metadata_json": json.dumps(_jsonable(metadata), ensure_ascii=False),
    }
    for key, value in metrics.items():
        flattened.setdefault(key, value)
    return flattened


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
        lines.extend(["", f"## {dataset}"])
        stats = _first_split_stats(dataset_rows)
        if stats:
            lines.extend(
                [
                    "",
                    "Split stats:",
                    "",
                    (
                        f"- users={int(stats.get('users', 0))}, items={int(stats.get('items', 0))}, "
                        f"train={int(stats.get('train_interactions', 0))}, val={int(stats.get('val_interactions', 0))}, "
                        f"test={int(stats.get('test_interactions', 0))}"
                    ),
                    (
                        f"- sparse_users={int(stats.get('sparse_users', 0))}, warm_users={int(stats.get('warm_users', 0))}, "
                        f"long_tail_items={int(stats.get('long_tail_items', 0))}, head_items={int(stats.get('head_items', 0))}"
                    ),
                ]
            )
        lines.extend(["", f"| Group | Model | Precision@{k} | Recall@{k} | NDCG@{k} | MRR | Sparse NDCG@{k} | Tail NDCG@{k} | Seconds |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
        if not ok_rows:
            lines.append("| none |  |  |  |  |  |")
        for row in ok_rows:
            metrics = row.get("metrics", {})
            lines.append(
                "| {group} | {model} | {precision:.4f} | {recall:.4f} | {ndcg:.4f} | {mrr:.4f} | {sparse:.4f} | {tail:.4f} | {seconds:.2f} |".format(
                    group=row.get("group", _model_group(row.get("model", ""))),
                    model=row.get("model", ""),
                    precision=float(metrics.get(f"precision@{k}", 0.0)),
                    recall=float(metrics.get(f"recall@{k}", 0.0)),
                    ndcg=float(metrics.get(f"ndcg@{k}", 0.0)),
                    mrr=float(metrics.get("mrr", 0.0)),
                    sparse=float(metrics.get(f"sparse_user_ndcg@{k}", 0.0)),
                    tail=float(metrics.get(f"long_tail_ndcg@{k}", 0.0)),
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


def _first_split_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        stats = (row.get("metadata", {}) or {}).get("split_stats", {})
        if stats:
            return stats
    return {}


def _model_group(model_name: str) -> str:
    if model_name == "hybrid_strong_ranker":
        return "strongest_ranker"
    if model_name in {
        "random",
        "popularity_only",
        "item_knn_cosine",
        "user_knn_cosine",
        "svd_ranking",
        "bpr_mf",
        "ease",
        "slim_elasticnet",
        "implicit_als",
        "lightfm_warp",
        "neumf",
    }:
        return "baselines"
    if model_name in {
        "hybrid_pdf_clean",
        "lightgcn_only",
        "learned_two_tower",
        "hybrid_weighted_no_popularity",
        "hybrid_weighted_full",
        "hybrid_ranker_no_popularity",
        "hybrid_ranker_full",
        "hybrid_no_tmdb",
    } or model_name.endswith("_only") or model_name.endswith("_no_tmdb_only"):
        return "pdf_clean_and_ablation"
    return "baselines"


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
