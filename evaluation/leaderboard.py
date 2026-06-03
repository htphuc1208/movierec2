from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .protocol import DEFAULT_SEED, PROTOCOL_NAME


LEADERBOARD_COLUMNS = [
    "dataset",
    "protocol",
    "model",
    "model_family",
    "source",
    "tuned_weights",
    "precision@10",
    "recall@10",
    "ndcg@10",
    "mrr@10",
    "warm_precision@10",
    "warm_recall@10",
    "warm_ndcg@10",
    "warm_mrr@10",
    "cold_precision@10",
    "cold_recall@10",
    "cold_ndcg@10",
    "cold_mrr@10",
    "precision@20",
    "recall@20",
    "ndcg@20",
    "mrr@20",
    "rmse",
    "mae",
    "best_epoch",
    "warm_interactions",
    "cold_interactions",
    "artifact_dir",
    "command",
    "seed",
]


def leaderboard_row(
    *,
    dataset: str,
    model: str,
    model_family: str,
    source: str,
    metrics: dict[str, Any],
    artifact_dir: str = "",
    command: str = "",
    tuned_weights: dict[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
    protocol: str = PROTOCOL_NAME,
) -> dict[str, Any]:
    normalised = normalise_metrics(metrics)
    row = {column: "" for column in LEADERBOARD_COLUMNS}
    row.update(
        {
            "dataset": dataset,
            "protocol": protocol,
            "model": model,
            "model_family": model_family,
            "source": source,
            "tuned_weights": json.dumps(tuned_weights or {}, sort_keys=True),
            "artifact_dir": artifact_dir,
            "command": command,
            "seed": seed,
        }
    )
    for key in [
        "precision@10",
        "recall@10",
        "ndcg@10",
        "mrr@10",
        "warm_precision@10",
        "warm_recall@10",
        "warm_ndcg@10",
        "warm_mrr@10",
        "cold_precision@10",
        "cold_recall@10",
        "cold_ndcg@10",
        "cold_mrr@10",
        "precision@20",
        "recall@20",
        "ndcg@20",
        "mrr@20",
        "rmse",
        "mae",
        "best_epoch",
        "warm_interactions",
        "cold_interactions",
    ]:
        if key in normalised:
            row[key] = normalised[key]
    return row


def normalise_metrics(metrics: dict[str, Any], top_k: int = 10) -> dict[str, Any]:
    """Normalize metric variants from local trainers, RecBole, manifests, and tuning results."""

    root_metrics = dict(metrics or {})
    segments = _extract_segments(root_metrics)
    metrics = _unwrap_metrics(root_metrics)
    normalised: dict[str, Any] = {}

    for key, value in metrics.items():
        canonical = _canonical_key(str(key), top_k)
        if canonical in _metric_keys():
            normalised[canonical] = _json_scalar(value)

    for metric in ["precision", "recall", "ndcg", "mrr"]:
        raw_key = f"{metric}_at_k"
        all_key = f"all_{metric}_at_k"
        warm_key = f"warm_{metric}@{top_k}"
        if raw_key in metrics and all_key in metrics and warm_key in _metric_keys():
            normalised[warm_key] = _json_scalar(metrics[raw_key])

    for prefix, segment_metrics in segments.items():
        if prefix in {"warm", "cold"} and isinstance(segment_metrics, dict):
            for key, value in segment_metrics.items():
                canonical = _canonical_key(f"{prefix}_{key}", top_k)
                if canonical in _metric_keys():
                    normalised[canonical] = _json_scalar(value)
        elif prefix in {"warm_interactions", "cold_interactions"}:
            normalised[prefix] = _json_scalar(segment_metrics)

    if "test_rmse" in normalised and "rmse" not in normalised:
        normalised["rmse"] = normalised["test_rmse"]
    if "rating_rmse" in normalised and "rmse" not in normalised:
        normalised["rmse"] = normalised["rating_rmse"]
    if "test_mae" in normalised and "mae" not in normalised:
        normalised["mae"] = normalised["test_mae"]

    # Some local trainers only run one top-k value but expose it as all_*_at_k.
    for metric in ["precision", "recall", "ndcg", "mrr"]:
        source_key = f"{metric}@{top_k}"
        if source_key in normalised and f"{metric}@10" not in normalised and top_k == 10:
            normalised[f"{metric}@10"] = normalised[source_key]
    return normalised


def write_leaderboard(rows: list[dict[str, Any]], output_prefix: str | Path) -> None:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = [_ordered_row(row) for row in sort_leaderboard(rows)]
    prefix.with_suffix(".json").write_text(json.dumps(ordered_rows, indent=2), encoding="utf-8")
    pd.DataFrame(ordered_rows, columns=LEADERBOARD_COLUMNS).to_csv(prefix.with_suffix(".csv"), index=False)
    prefix.with_suffix(".md").write_text(_markdown_table(ordered_rows), encoding="utf-8")


def sort_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            _as_float(row.get("ndcg@10")),
            _as_float(row.get("mrr@10")),
            _as_float(row.get("recall@10")),
            _as_float(row.get("precision@10")),
        )

    return sorted(rows, key=key, reverse=True)


def _unwrap_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    current = dict(metrics or {})
    for key in ["test", "test_result"]:
        value = current.get(key)
        if isinstance(value, dict):
            current = dict(value)
    return current


def _extract_segments(metrics: dict[str, Any]) -> dict[str, Any]:
    segments: dict[str, Any] = {}
    for container in [metrics, metrics.get("test", {}) if isinstance(metrics.get("test"), dict) else {}]:
        if not isinstance(container, dict):
            continue
        nested = container.get("test_segments", {})
        if isinstance(nested, dict):
            for key in ["warm", "cold"]:
                if isinstance(nested.get(key), dict):
                    segments[key] = nested[key]
            if "warm_interactions" in nested:
                segments["warm_interactions"] = nested["warm_interactions"]
            if "cold_interactions" in nested:
                segments["cold_interactions"] = nested["cold_interactions"]
        for key in ["warm_interactions", "cold_interactions", "warm_test_interactions", "cold_test_interactions"]:
            if key in container:
                segments[key] = container[key]

    if "warm_interactions" not in metrics and "warm_test_interactions" in segments:
        segments["warm_interactions"] = segments["warm_test_interactions"]
    if "cold_interactions" not in metrics and "cold_test_interactions" in segments:
        segments["cold_interactions"] = segments["cold_test_interactions"]
    return segments


def _canonical_key(key: str, top_k: int) -> str:
    canonical = key.lower().replace("_at_k", f"@{top_k}")
    if canonical.startswith("all_"):
        canonical = canonical.removeprefix("all_")
    if canonical == "warm_test_interactions":
        canonical = "warm_interactions"
    if canonical == "cold_test_interactions":
        canonical = "cold_interactions"
    return canonical


def _metric_keys() -> set[str]:
    return {
        "precision@10",
        "recall@10",
        "ndcg@10",
        "mrr@10",
        "warm_precision@10",
        "warm_recall@10",
        "warm_ndcg@10",
        "warm_mrr@10",
        "cold_precision@10",
        "cold_recall@10",
        "cold_ndcg@10",
        "cold_mrr@10",
        "precision@20",
        "recall@20",
        "ndcg@20",
        "mrr@20",
        "rmse",
        "mae",
        "test_rmse",
        "rating_rmse",
        "test_mae",
        "best_epoch",
        "warm_interactions",
        "cold_interactions",
    }


def _ordered_row(row: dict[str, Any]) -> dict[str, Any]:
    return {column: row.get(column, "") for column in LEADERBOARD_COLUMNS}


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    visible = [
        "model",
        "source",
        "ndcg@10",
        "warm_ndcg@10",
        "cold_ndcg@10",
        "mrr@10",
        "recall@10",
        "precision@10",
        "rmse",
        "artifact_dir",
    ]
    lines = ["| " + " | ".join(visible) + " |", "| " + " | ".join("---" for _ in visible) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in visible) + " |")
    return "\n".join(lines) + "\n"


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _as_float(value: Any) -> float:
    try:
        if value == "":
            return float("-inf")
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")
