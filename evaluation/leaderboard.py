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
    "precision@20",
    "recall@20",
    "ndcg@20",
    "mrr@20",
    "rmse",
    "mae",
    "best_epoch",
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
        "precision@20",
        "recall@20",
        "ndcg@20",
        "mrr@20",
        "rmse",
        "mae",
        "best_epoch",
    ]:
        if key in normalised:
            row[key] = normalised[key]
    return row


def normalise_metrics(metrics: dict[str, Any], top_k: int = 10) -> dict[str, Any]:
    """Normalize metric variants from local trainers, RecBole, manifests, and tuning results."""

    metrics = _unwrap_metrics(metrics)
    normalised: dict[str, Any] = {}

    for key, value in metrics.items():
        lowered = str(key).lower()
        canonical = lowered.replace("_at_k", f"@{top_k}")
        if canonical.startswith("all_"):
            canonical = canonical.removeprefix("all_")
        elif canonical.startswith("warm_"):
            continue
        elif canonical.startswith("cold_"):
            continue
        if canonical in {
            "precision@10",
            "recall@10",
            "ndcg@10",
            "mrr@10",
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
        }:
            normalised[canonical] = _json_scalar(value)

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


def _ordered_row(row: dict[str, Any]) -> dict[str, Any]:
    return {column: row.get(column, "") for column in LEADERBOARD_COLUMNS}


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    visible = ["model", "source", "ndcg@10", "mrr@10", "recall@10", "precision@10", "rmse", "artifact_dir"]
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
