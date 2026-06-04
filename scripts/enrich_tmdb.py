#!/usr/bin/env python3
"""Enrich MovieLens movies with TMDb metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from recommender.config import PROJECT_ROOT, get_settings
from recommender.data.movielens import read_movielens
from recommender.data.tmdb import TMDBClient, enrich_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich MovieLens catalog through TMDb")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "ml-latest-small")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "movie_catalog_enriched.parquet")
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "data" / "processed" / "tmdb_cache.json")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests")
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--base-url", default=None, help="Override TMDb base URL, useful when routing through a proxy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    data = read_movielens(args.raw_dir)
    client = TMDBClient(
        api_key=settings.tmdb_api_key or "",
        language=settings.tmdb_language,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        base_url=args.base_url or settings.tmdb_base_url,
    )
    enriched = enrich_catalog(
        data.movies,
        data.links,
        client=client,
        cache_path=args.cache,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.output, index=False)
    print(f"Saved enriched catalog: {args.output} ({len(enriched)} movies)")


if __name__ == "__main__":
    main()
