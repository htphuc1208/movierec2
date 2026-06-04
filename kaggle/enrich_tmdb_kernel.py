"""Kaggle script kernel for TMDb enrichment.

This file is staged and submitted by scripts/submit_kaggle_enrich.py.
It expects a Kaggle Secret named TMDB_API_KEY unless kernel_config.json
overrides the secret name.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


WORKDIR = Path("/kaggle/working")
PROJECT_DIR = Path.cwd()
CONFIG_PATH = PROJECT_DIR / "kernel_config.json"


def load_config() -> dict:
    defaults = {
        "dataset": "ml-latest-small",
        "limit": None,
        "sleep_seconds": 0.5,
        "timeout": 60.0,
        "max_retries": 8,
        "retry_backoff": 3.0,
        "tmdb_secret_name": "TMDB_API_KEY",
        "tmdb_base_url": "https://api.themoviedb.org/3",
        "tmdb_language": "en-US",
    }
    if CONFIG_PATH.exists():
        defaults.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return defaults


def get_tmdb_api_key(secret_name: str) -> str:
    if os.getenv("TMDB_API_KEY"):
        return os.environ["TMDB_API_KEY"]
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as exc:
        raise RuntimeError("kaggle_secrets is unavailable and TMDB_API_KEY env var is missing") from exc
    key = UserSecretsClient().get_secret(secret_name)
    if not key:
        raise RuntimeError(f"Kaggle Secret {secret_name!r} is empty or missing")
    return key


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def zip_outputs(processed_dir: Path) -> None:
    output_zip = WORKDIR / "tmdb_enrichment_outputs.zip"
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in processed_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(WORKDIR))
    print(f"Zipped outputs: {output_zip}")


def main() -> None:
    config = load_config()
    os.environ["PYTHONPATH"] = f"{PROJECT_DIR / 'src'}:{PROJECT_DIR}"
    os.environ["TMDB_API_KEY"] = get_tmdb_api_key(config["tmdb_secret_name"])
    os.environ["TMDB_BASE_URL"] = config["tmdb_base_url"]
    os.environ["TMDB_LANGUAGE"] = config["tmdb_language"]

    raw_root = WORKDIR / "data" / "raw"
    processed_dir = WORKDIR / "data" / "processed"
    raw_dir = raw_root / config["dataset"]
    output_path = processed_dir / "movie_catalog_enriched.parquet"
    cache_path = processed_dir / "tmdb_cache.json"

    run(
        [
            sys.executable,
            "scripts/download_movielens.py",
            "--dataset",
            config["dataset"],
            "--output-dir",
            str(raw_root),
        ]
    )

    enrich_command = [
        sys.executable,
        "scripts/enrich_tmdb.py",
        "--raw-dir",
        str(raw_dir),
        "--output",
        str(output_path),
        "--cache",
        str(cache_path),
        "--sleep-seconds",
        str(config["sleep_seconds"]),
        "--timeout",
        str(config["timeout"]),
        "--max-retries",
        str(config["max_retries"]),
        "--retry-backoff",
        str(config["retry_backoff"]),
        "--base-url",
        config["tmdb_base_url"],
    ]
    if config["limit"] is not None:
        enrich_command.extend(["--limit", str(config["limit"])])
    run(enrich_command)
    zip_outputs(processed_dir)


if __name__ == "__main__":
    main()
