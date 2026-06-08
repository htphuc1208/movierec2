from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_compare_models_script_smoke(tmp_path: Path) -> None:
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
    pd.DataFrame({"movieId": [1, 2, 3, 4, 5, 6], "tmdbId": [101, 102, 103, 104, 105, 106]}).to_csv(
        data_dir / "links.csv",
        index=False,
    )

    output_dir = tmp_path / "out"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd() / 'src'}:{Path.cwd()}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_models.py",
            "--dataset",
            "movielens",
            "--movielens-dir",
            str(data_dir),
            "--movielens-enriched-catalog",
            str(tmp_path / "missing.parquet"),
            "--output-dir",
            str(output_dir),
            "--content-backend",
            "tfidf",
            "--models",
            "core",
            "--preset",
            "letterboxd-pdf-clean",
            "--k",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--mf-dim",
            "8",
            "--max-ease-items",
            "100",
            "--max-ranker-samples",
            "100",
            "--hybrid-grid-step",
            "0.5",
            "--no-content-cache",
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "comparison_results.csv").exists()
    assert (output_dir / "comparison_results.json").exists()
    assert (output_dir / "comparison_summary.md").exists()
    rows = json.loads((output_dir / "comparison_results.json").read_text(encoding="utf-8"))
    ok_models = {row["model"] for row in rows if row["status"] == "ok"}
    assert {"popularity_only", "item_knn_cosine", "user_knn_cosine", "tfidf_only"}.issubset(ok_models)
    assert "hybrid_pdf_clean" in {row["model"] for row in rows}
