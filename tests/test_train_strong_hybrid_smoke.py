from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_train_strong_hybrid_exports_artifacts_on_tiny_dataset(tmp_path: Path) -> None:
    data_dir = tmp_path / "ml"
    data_dir.mkdir()
    pd.DataFrame(
        [
            (1, 1, 5.0, 1),
            (1, 2, 5.0, 2),
            (1, 3, 5.0, 3),
            (1, 4, 5.0, 4),
            (2, 2, 5.0, 1),
            (2, 3, 5.0, 2),
            (2, 5, 5.0, 3),
            (2, 6, 5.0, 4),
            (3, 1, 5.0, 1),
            (3, 4, 5.0, 2),
            (3, 5, 5.0, 3),
            (3, 6, 5.0, 4),
            (4, 1, 5.0, 1),
            (4, 2, 5.0, 2),
            (4, 5, 5.0, 3),
            (4, 6, 5.0, 4),
        ],
        columns=["userId", "movieId", "rating", "timestamp"],
    ).to_csv(data_dir / "ratings.csv", index=False)
    pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5, 6],
            "title": ["A", "B", "C", "D", "E", "F"],
            "genres": ["Drama", "Drama", "Action", "Action", "Comedy", "Comedy"],
        }
    ).to_csv(data_dir / "movies.csv", index=False)

    artifacts_dir = tmp_path / "artifacts"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd() / 'src'}:{Path.cwd()}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_strong_hybrid.py",
            "--dataset",
            "movielens",
            "--raw-dir",
            str(data_dir),
            "--enriched-catalog",
            str(tmp_path / "missing.parquet"),
            "--artifacts-dir",
            str(artifacts_dir),
            "--content-backend",
            "tfidf",
            "--ranker",
            "sgd",
            "--lightgcn-epochs",
            "1",
            "--lightgcn-dim",
            "8",
            "--batch-size",
            "8",
            "--device",
            "cpu",
            "--max-ease-items",
            "100",
            "--max-ranker-samples",
            "100",
            "--embedding-cache-dir",
            str(tmp_path / "cache"),
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (artifacts_dir / "movie_catalog.parquet").exists()
    assert (artifacts_dir / "hybrid_config.json").exists()
    assert (artifacts_dir / "component_score_config.json").exists()
    assert (artifacts_dir / "metrics.json").exists()
