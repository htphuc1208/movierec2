from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


URLS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-latest": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
}


def download(url: str, target: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with target.open("wb") as handle, tqdm(total=total, unit="B", unit_scale=True) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            progress.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a MovieLens dataset.")
    parser.add_argument("--variant", choices=sorted(URLS), default="ml-latest-small")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{args.variant}.zip"
    download(URLS[args.variant], zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    print(f"Extracted {args.variant} to {output_dir}")


if __name__ == "__main__":
    main()
