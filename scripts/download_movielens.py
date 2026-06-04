#!/usr/bin/env python3
"""Download and extract MovieLens datasets."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from recommender.config import PROJECT_ROOT
from recommender.data.movielens import MOVIELENS_URLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MovieLens data")
    parser.add_argument("--dataset", choices=sorted(MOVIELENS_URLS), default="ml-latest-small")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    url = MOVIELENS_URLS[args.dataset]
    zip_path = args.output_dir / f"{args.dataset}.zip"
    dataset_dir = args.output_dir / args.dataset

    if dataset_dir.exists():
        print(f"{dataset_dir} already exists")
        return

    print(f"Downloading {url} -> {zip_path}")
    urlretrieve(url, zip_path)
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(args.output_dir)
    print(f"Ready: {dataset_dir}")


if __name__ == "__main__":
    main()
