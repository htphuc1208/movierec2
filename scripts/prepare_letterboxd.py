#!/usr/bin/env python3
"""Prepare Letterboxd crawler data for the existing training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from recommender.config import PROJECT_ROOT, get_settings
from recommender.data.letterboxd import (
    build_base_letterboxd_catalog,
    enrich_letterboxd_catalog,
    materialize_letterboxd,
)
from recommender.data.tmdb import TMDBClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Letterboxd data into MovieLens-compatible files")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "letterboxd" / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "letterboxd")
    parser.add_argument("--split", choices=["cf", "raw"], default="cf")
    parser.add_argument("--rating-policy", choices=["implicit", "explicit"], default="implicit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enrich-tmdb", action="store_true")
    parser.add_argument("--tmdb-cache", type=Path, default=None)
    parser.add_argument("--enrich-limit", type=int, default=None)
    parser.add_argument("--min-match-score", type=int, default=75)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-backoff", type=float, default=3.0)
    parser.add_argument("--base-url", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = materialize_letterboxd(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        split=args.split,
        rating_policy=args.rating_policy,
        seed=args.seed,
    )
    print(
        "Prepared Letterboxd "
        f"split={args.split} rating_policy={args.rating_policy}: "
        f"{len(prepared.ratings)} ratings, "
        f"{prepared.ratings['userId'].nunique()} users, "
        f"{prepared.ratings['movieId'].nunique()} movies"
    )

    if args.enrich_tmdb:
        settings = get_settings()
        cache_path = args.tmdb_cache or (args.output_dir / "letterboxd_tmdb_cache.json")
        client = TMDBClient(
            api_key=settings.tmdb_api_key or "",
            language=settings.tmdb_language,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            base_url=args.base_url or settings.tmdb_base_url,
        )
        catalog = enrich_letterboxd_catalog(
            output_dir=args.output_dir,
            client=client,
            cache_path=cache_path,
            limit=args.enrich_limit,
            sleep_seconds=args.sleep_seconds,
            min_match_score=args.min_match_score,
        )
        print(f"Saved enriched Letterboxd catalog: {args.output_dir / 'movie_catalog_enriched.parquet'} ({len(catalog)} movies)")
    else:
        catalog = build_base_letterboxd_catalog(args.output_dir)
        output_path = args.output_dir / "movie_catalog_enriched.parquet"
        catalog.to_parquet(output_path, index=False)
        print(f"Saved base Letterboxd catalog without TMDb enrichment: {output_path} ({len(catalog)} movies)")


if __name__ == "__main__":
    main()
