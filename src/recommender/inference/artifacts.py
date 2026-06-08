"""Artifact bundle serialization for API and UI inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArtifactBundle:
    catalog: pd.DataFrame
    user_mapping: dict[int, int]
    item_mapping: dict[int, int]
    content_embeddings: np.ndarray
    user_profiles: np.ndarray
    item_popularity: np.ndarray
    metrics: dict[str, Any]
    hybrid_config: dict[str, Any]
    lightgcn_user_embeddings: np.ndarray | None = None
    lightgcn_item_embeddings: np.ndarray | None = None
    two_tower_user_embeddings: np.ndarray | None = None
    two_tower_item_embeddings: np.ndarray | None = None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_artifact_bundle(
    artifacts_dir: str | Path,
    catalog: pd.DataFrame,
    user_mapping: dict[int, int],
    item_mapping: dict[int, int],
    content_embeddings: np.ndarray,
    user_profiles: np.ndarray,
    item_popularity: np.ndarray,
    metrics: dict[str, Any] | None = None,
    hybrid_config: dict[str, Any] | None = None,
    lightgcn_user_embeddings: np.ndarray | None = None,
    lightgcn_item_embeddings: np.ndarray | None = None,
    two_tower_user_embeddings: np.ndarray | None = None,
    two_tower_item_embeddings: np.ndarray | None = None,
) -> None:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(artifacts_dir / "movie_catalog.parquet", index=False)
    _write_json(artifacts_dir / "user_mapping.json", {str(k): int(v) for k, v in user_mapping.items()})
    _write_json(artifacts_dir / "item_mapping.json", {str(k): int(v) for k, v in item_mapping.items()})
    np.save(artifacts_dir / "content_embeddings.npy", content_embeddings.astype(np.float32))
    np.save(artifacts_dir / "user_profiles.npy", user_profiles.astype(np.float32))
    np.save(artifacts_dir / "item_popularity.npy", item_popularity.astype(np.float32))
    _write_json(artifacts_dir / "metrics.json", metrics or {})
    _write_json(artifacts_dir / "hybrid_config.json", hybrid_config or {})
    if lightgcn_user_embeddings is not None and lightgcn_item_embeddings is not None:
        np.save(artifacts_dir / "lightgcn_user_embeddings.npy", lightgcn_user_embeddings.astype(np.float32))
        np.save(artifacts_dir / "lightgcn_item_embeddings.npy", lightgcn_item_embeddings.astype(np.float32))
    if two_tower_user_embeddings is not None and two_tower_item_embeddings is not None:
        np.save(artifacts_dir / "two_tower_user_embeddings.npy", two_tower_user_embeddings.astype(np.float32))
        np.save(artifacts_dir / "two_tower_item_embeddings.npy", two_tower_item_embeddings.astype(np.float32))


def artifact_status(artifacts_dir: str | Path) -> dict[str, Any]:
    artifacts_dir = Path(artifacts_dir)
    required = [
        "movie_catalog.parquet",
        "user_mapping.json",
        "item_mapping.json",
        "content_embeddings.npy",
        "user_profiles.npy",
        "item_popularity.npy",
        "hybrid_config.json",
        "metrics.json",
    ]
    missing = [name for name in required if not (artifacts_dir / name).exists()]
    return {"artifacts_dir": str(artifacts_dir), "ready": not missing, "missing": missing}


def load_artifact_bundle(artifacts_dir: str | Path) -> ArtifactBundle:
    artifacts_dir = Path(artifacts_dir)
    status = artifact_status(artifacts_dir)
    if not status["ready"]:
        raise FileNotFoundError(f"Artifact bundle is incomplete: missing {status['missing']}")

    lightgcn_user_path = artifacts_dir / "lightgcn_user_embeddings.npy"
    lightgcn_item_path = artifacts_dir / "lightgcn_item_embeddings.npy"
    two_tower_user_path = artifacts_dir / "two_tower_user_embeddings.npy"
    two_tower_item_path = artifacts_dir / "two_tower_item_embeddings.npy"
    return ArtifactBundle(
        catalog=pd.read_parquet(artifacts_dir / "movie_catalog.parquet"),
        user_mapping={int(k): int(v) for k, v in _read_json(artifacts_dir / "user_mapping.json").items()},
        item_mapping={int(k): int(v) for k, v in _read_json(artifacts_dir / "item_mapping.json").items()},
        content_embeddings=np.load(artifacts_dir / "content_embeddings.npy"),
        user_profiles=np.load(artifacts_dir / "user_profiles.npy"),
        item_popularity=np.load(artifacts_dir / "item_popularity.npy"),
        metrics=_read_json(artifacts_dir / "metrics.json"),
        hybrid_config=_read_json(artifacts_dir / "hybrid_config.json"),
        lightgcn_user_embeddings=np.load(lightgcn_user_path) if lightgcn_user_path.exists() else None,
        lightgcn_item_embeddings=np.load(lightgcn_item_path) if lightgcn_item_path.exists() else None,
        two_tower_user_embeddings=np.load(two_tower_user_path) if two_tower_user_path.exists() else None,
        two_tower_item_embeddings=np.load(two_tower_item_path) if two_tower_item_path.exists() else None,
    )
