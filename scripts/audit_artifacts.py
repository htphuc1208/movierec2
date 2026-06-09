#!/usr/bin/env python3
"""Audit exported recommender artifact bundles.

The API can run with the required artifact bundle only. The project-level
"full" target is stricter: SBERT content, LightGCN, learned Two-Tower and a
persisted strong ranker when using the strong artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from recommender.inference.artifacts import artifact_status, load_artifact_bundle


DEFAULT_TARGETS = {
    "movielens_current": "artifacts",
    "letterboxd_current": "artifacts/letterboxd",
    "movielens_pdf_clean": "artifacts/movielens_pdf_clean",
    "letterboxd_pdf_clean": "artifacts/letterboxd_pdf_clean",
    "movielens_strong": "artifacts/movielens_strong",
    "letterboxd_strong": "artifacts/letterboxd_strong",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exported artifact directories")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Extra target in name=path format. Can be provided multiple times.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def audit_dir(name: str, path: str | Path) -> dict[str, Any]:
    artifact_dir = Path(path)
    status = artifact_status(artifact_dir)
    row: dict[str, Any] = {
        "name": name,
        "path": str(artifact_dir),
        "api_ready": bool(status["ready"]),
        "missing": status["missing"],
        "model_type": "",
        "content_backend": "",
        "users": 0,
        "movies": 0,
        "content_dim": 0,
        "has_lightgcn": False,
        "has_two_tower": False,
        "has_ranker": False,
        "has_component_score_config": (artifact_dir / "component_score_config.json").exists(),
        "shape_ok": False,
        "pdf_clean_ready": False,
        "sbert_ready": False,
        "strong_ready": False,
        "errors": [],
    }
    if not status["ready"]:
        return row

    try:
        bundle = load_artifact_bundle(artifact_dir)
    except Exception as exc:
        row["api_ready"] = False
        row["errors"].append(f"load_failed: {exc}")
        return row

    config = bundle.hybrid_config or {}
    row["model_type"] = str(config.get("model_type") or "")
    row["content_backend"] = str(config.get("content_backend") or "")
    row["users"] = len(bundle.user_mapping)
    row["movies"] = len(bundle.item_mapping)
    row["content_dim"] = int(bundle.content_embeddings.shape[1]) if bundle.content_embeddings.ndim == 2 else 0
    row["has_lightgcn"] = bundle.lightgcn_user_embeddings is not None and bundle.lightgcn_item_embeddings is not None
    row["has_two_tower"] = bundle.two_tower_user_embeddings is not None and bundle.two_tower_item_embeddings is not None
    row["has_ranker"] = (artifact_dir / str(config.get("ranker_path", "ranker.joblib"))).exists()

    expected_items = len(bundle.item_mapping)
    expected_users = len(bundle.user_mapping)
    checks = [
        ("catalog_items", len(bundle.catalog) == expected_items),
        ("content_items", bundle.content_embeddings.shape[0] == expected_items),
        ("popularity_items", bundle.item_popularity.shape[0] == expected_items),
        ("profile_users", bundle.user_profiles.shape[0] == expected_users),
    ]
    if row["has_lightgcn"]:
        checks.extend(
            [
                ("lightgcn_users", bundle.lightgcn_user_embeddings.shape[0] == expected_users),
                ("lightgcn_items", bundle.lightgcn_item_embeddings.shape[0] == expected_items),
            ]
        )
    if row["has_two_tower"]:
        checks.extend(
            [
                ("two_tower_users", bundle.two_tower_user_embeddings.shape[0] == expected_users),
                ("two_tower_items", bundle.two_tower_item_embeddings.shape[0] == expected_items),
            ]
        )
    row["shape_ok"] = all(ok for _, ok in checks)
    row["errors"].extend(name for name, ok in checks if not ok)

    # This is the research/demo-ready weighted hybrid artifact.
    row["pdf_clean_ready"] = bool(row["api_ready"] and row["shape_ok"] and row["has_lightgcn"] and row["has_two_tower"])
    row["sbert_ready"] = row["content_backend"] == "sbert"
    row["strong_ready"] = bool(
        row["api_ready"]
        and row["shape_ok"]
        and row["model_type"] == "strong_ranker"
        and row["has_ranker"]
        and row["has_component_score_config"]
    )
    return row


def format_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "name",
        "path",
        "api",
        "pdf",
        "strong",
        "backend",
        "users",
        "movies",
        "lgcn",
        "2tower",
        "ranker",
        "missing/errors",
    ]
    widths = {header: len(header) for header in headers}
    formatted_rows = []
    for row in rows:
        missing = ",".join(row["missing"]) if row["missing"] else ";".join(row["errors"])
        values = {
            "name": row["name"],
            "path": row["path"],
            "api": yes(row["api_ready"] and row["shape_ok"]),
            "pdf": yes(row["pdf_clean_ready"]),
            "strong": yes(row["strong_ready"]),
            "backend": row["content_backend"] or "-",
            "users": str(row["users"]),
            "movies": str(row["movies"]),
            "lgcn": yes(row["has_lightgcn"]),
            "2tower": yes(row["has_two_tower"]),
            "ranker": yes(row["has_ranker"]),
            "missing/errors": missing or "-",
        }
        formatted_rows.append(values)
        for header, value in values.items():
            widths[header] = max(widths[header], len(value))

    lines = []
    lines.append(" | ".join(header.ljust(widths[header]) for header in headers))
    lines.append("-+-".join("-" * widths[header] for header in headers))
    for values in formatted_rows:
        lines.append(" | ".join(values[header].ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def yes(value: Any) -> str:
    return "yes" if bool(value) else "no"


def targets_from_args(args: argparse.Namespace) -> dict[str, str]:
    targets = dict(DEFAULT_TARGETS)
    for item in args.target:
        if "=" not in item:
            raise SystemExit(f"--target must use name=path format: {item}")
        name, path = item.split("=", 1)
        targets[name.strip()] = path.strip()
    return targets


def main() -> None:
    args = parse_args()
    rows = [audit_dir(name, path) for name, path in targets_from_args(args).items()]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(format_table(rows))
        print()
        print("api = required bundle can serve API/UI")
        print("pdf = API-ready + LightGCN + learned Two-Tower")
        print("strong = persisted strong ranker artifact")


if __name__ == "__main__":
    main()
