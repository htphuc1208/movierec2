#!/usr/bin/env python3
"""Stage and submit a Kaggle kernel that runs TMDb enrichment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from recommender.config import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit TMDb enrichment as a Kaggle script kernel")
    parser.add_argument("--username", default=os.getenv("KAGGLE_USERNAME"), help="Kaggle username/owner for the kernel")
    parser.add_argument("--slug", default="movierec3-tmdb-enrich-v2")
    parser.add_argument("--title", default="MovieRec3 TMDb Enrichment V2")
    parser.add_argument("--dataset", choices=["ml-latest-small", "ml-20m"], default="ml-latest-small")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-backoff", type=float, default=3.0)
    parser.add_argument("--tmdb-secret-name", default="TMDB_API_KEY")
    parser.add_argument("--tmdb-base-url", default="https://api.themoviedb.org/3")
    parser.add_argument("--tmdb-language", default="en-US")
    parser.add_argument("--no-submit", action="store_true", help="Only create the staging directory")
    parser.add_argument("--keep-staging", action="store_true")
    return parser.parse_args()


def kaggle_command() -> list[str]:
    if shutil.which("kaggle"):
        return ["kaggle"]
    try:
        import kaggle  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle CLI is not installed. Install it with: pip install kaggle"
        ) from exc
    return [sys.executable, "-m", "kaggle"]


def check_credentials() -> None:
    has_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    has_env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    if not has_file and not has_env:
        raise RuntimeError(
            "Missing Kaggle credentials. Create ~/.kaggle/kaggle.json or set KAGGLE_USERNAME and KAGGLE_KEY."
        )


def copy_tree(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    shutil.copytree(src, dst, ignore=ignore)


def stage_kernel(args: argparse.Namespace, staging_dir: Path) -> None:
    if not args.username:
        raise RuntimeError("Missing --username or KAGGLE_USERNAME")

    copy_tree(PROJECT_ROOT / "src", staging_dir / "src")
    (staging_dir / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / "scripts" / "download_movielens.py", staging_dir / "scripts" / "download_movielens.py")
    shutil.copy2(PROJECT_ROOT / "scripts" / "enrich_tmdb.py", staging_dir / "scripts" / "enrich_tmdb.py")
    shutil.copy2(PROJECT_ROOT / "kaggle" / "enrich_tmdb_kernel.py", staging_dir / "enrich_tmdb_kernel.py")

    metadata = {
        "id": f"{args.username}/{args.slug}",
        "title": args.title,
        "code_file": "enrich_tmdb_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    config = {
        "dataset": args.dataset,
        "limit": args.limit,
        "sleep_seconds": args.sleep_seconds,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "retry_backoff": args.retry_backoff,
        "tmdb_secret_name": args.tmdb_secret_name,
        "tmdb_base_url": args.tmdb_base_url,
        "tmdb_language": args.tmdb_language,
    }
    (staging_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (staging_dir / "kernel_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    staging_parent = PROJECT_ROOT / "build" if args.keep_staging or args.no_submit else None
    if staging_parent:
        staging_parent.mkdir(exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix="kaggle_enrich_", dir=staging_parent))
    else:
        staging_dir = Path(tempfile.mkdtemp(prefix="kaggle_enrich_"))

    try:
        stage_kernel(args, staging_dir)
        print(f"Staged Kaggle kernel at: {staging_dir}")
        if args.no_submit:
            return
        check_credentials()
        command = [*kaggle_command(), "kernels", "push", "-p", str(staging_dir)]
        print("+", " ".join(command))
        subprocess.run(command, check=True)
        print(f"Submitted kernel: {args.username}/{args.slug}")
        print(f"Download outputs after it finishes with: kaggle kernels output {args.username}/{args.slug} -p kaggle_outputs")
    finally:
        if not args.keep_staging and not args.no_submit:
            shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
