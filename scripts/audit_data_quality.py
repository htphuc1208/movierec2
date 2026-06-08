from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import MovieLensDataLoader


CORE_METADATA_COLUMNS = ["overview", "director", "cast", "poster_url"]
NUMERIC_METADATA_COLUMNS = {
    "budget",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count",
    "popularity",
    "collection_id",
}
ENRICHED_COLUMNS = [
    "overview",
    "tagline",
    "director",
    "cast",
    "poster_url",
    "budget",
    "revenue",
    "movieId",
    "genres",
    "release_date",
    "runtime",
    "original_language",
    "production_companies",
    "production_countries",
    "keywords",
    "vote_average",
    "vote_count",
    "popularity",
    "collection_id",
    "collection_name",
    "certification",
    "imdb_id",
]
OPTIONAL_ENRICHED_COLUMNS = ["enrichment_status"]


@dataclass(frozen=True)
class DataQualityReport:
    generated_at: str
    data_dir: str
    shapes: dict[str, list[int]]
    ids: dict[str, int | float]
    duplicates: dict[str, int]
    orphans: dict[str, int]
    ratings: dict[str, Any]
    split: dict[str, int | float]
    content_coverage: dict[str, dict[str, int | float | str]]
    content_issues: dict[str, int]
    examples: dict[str, list[dict[str, Any]]]
    recommendations: list[str]


def read_optional_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


def as_int(value: Any) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def pct(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) * 100.0 / float(denominator), 4)


def nonempty_count(series: pd.Series) -> int:
    return int(series.fillna("").astype(str).str.strip().ne("").sum())


def positive_numeric_count(series: pd.Series) -> int:
    return int((pd.to_numeric(series, errors="coerce").fillna(0) > 0).sum())


def records(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    clean = frame.head(limit).replace({np.nan: ""})
    return clean.to_dict(orient="records")


def year_missing_mask(movies: pd.DataFrame) -> pd.Series:
    extracted = movies["title"].astype(str).str.extract(r"\((\d{4})(?:[–-]\d{4})?\)")[0]
    return extracted.fillna("").str.strip().eq("")


def build_report(data_dir: str | Path, example_limit: int = 10) -> DataQualityReport:
    data_path = Path(data_dir)
    raw_movies = pd.read_csv(data_path / "movies.csv")
    raw_ratings = pd.read_csv(data_path / "ratings.csv")
    raw_links = read_optional_csv(data_path / "links.csv", ["movieId", "imdbId", "tmdbId"])
    raw_tags = read_optional_csv(data_path / "tags.csv", ["userId", "movieId", "tag", "timestamp"])
    enriched = read_optional_csv(data_path / "enriched_movies.csv", ENRICHED_COLUMNS)

    loader = MovieLensDataLoader(data_path)
    bundle = loader.load()
    train, val, test = loader.train_val_test_split(bundle.ratings)
    warm_val, cold_val = loader.split_warm_cold_items(train, val)
    warm_test, cold_test = loader.split_warm_cold_items(train, test)

    movie_ids = set(raw_movies["movieId"].astype(int))
    rating_movie_ids = set(raw_ratings["movieId"].astype(int))
    tag_movie_ids = set(raw_tags["movieId"].astype(int)) if "movieId" in raw_tags else set()
    link_movie_ids = set(raw_links["movieId"].astype(int)) if "movieId" in raw_links else set()
    enriched_movie_ids = set(enriched["movieId"].astype(int)) if "movieId" in enriched else set()

    user_counts = bundle.ratings.groupby("userId").size()
    item_counts = bundle.ratings.groupby("movieId").size()

    content_coverage: dict[str, dict[str, int | float | str]] = {}
    for column in [
        "overview",
        "tagline",
        "director",
        "cast",
        "poster_url",
        "budget",
        "revenue",
        "tmdbId",
        "imdbId",
        "genres",
        "year",
        "release_date",
        "runtime",
        "original_language",
        "production_companies",
        "production_countries",
        "keywords",
        "tag_genome_tags",
        "vote_average",
        "vote_count",
        "popularity",
        "collection_name",
        "certification",
    ]:
        if column not in bundle.movies.columns:
            continue
        if column in NUMERIC_METADATA_COLUMNS:
            covered = positive_numeric_count(bundle.movies[column])
            mode = ">0"
        else:
            covered = nonempty_count(bundle.movies[column])
            mode = "nonempty"
        content_coverage[column] = {
            "mode": mode,
            "covered": covered,
            "total": int(len(bundle.movies)),
            "coverage_pct": pct(covered, len(bundle.movies)),
        }

    rating_distribution = {
        str(float(rating)): int(count)
        for rating, count in bundle.ratings["rating"].value_counts().sort_index().items()
    }

    missing_core = bundle.movies[CORE_METADATA_COLUMNS].isna().any(axis=1)
    all_core_missing = bundle.movies[CORE_METADATA_COLUMNS].isna().all(axis=1)
    if not bundle.movies.empty:
        empty_core = pd.Series(False, index=bundle.movies.index)
        all_empty_core = pd.Series(True, index=bundle.movies.index)
        for column in CORE_METADATA_COLUMNS:
            column_empty = bundle.movies[column].fillna("").astype(str).str.strip().eq("")
            empty_core = empty_core | column_empty
            all_empty_core = all_empty_core & column_empty
    else:
        empty_core = pd.Series(dtype=bool)
        all_empty_core = pd.Series(dtype=bool)

    missing_enriched_movies = raw_movies.loc[
        raw_movies["movieId"].astype(int).isin(sorted(movie_ids - enriched_movie_ids)),
        ["movieId", "title", "genres"],
    ]
    movies_without_ratings = raw_movies.loc[
        raw_movies["movieId"].astype(int).isin(sorted(movie_ids - rating_movie_ids)),
        ["movieId", "title", "genres"],
    ]
    missing_year = raw_movies.loc[year_missing_mask(raw_movies), ["movieId", "title", "genres"]]
    no_genres = raw_movies.loc[raw_movies["genres"].astype(str).eq("(no genres listed)"), ["movieId", "title", "genres"]]

    recommendations: list[str] = []
    if len(missing_enriched_movies):
        recommendations.append("Run scripts/audit_data_quality.py --fix-enriched to add placeholder rows, then rerun TMDb enrichment for true metadata.")
    if len(movie_ids - rating_movie_ids):
        recommendations.append("Keep unrated movies for content/cold-start demos; filter them out for pure collaborative training if needed.")
    if len(set(test["movieId"].astype(int)) - set(train["movieId"].astype(int))):
        recommendations.append("Use warm-item ranking as the primary pure-CF metric, and report cold-item metrics separately for hybrid/content fallback.")
    if int((item_counts < 5).sum()):
        recommendations.append("Use minimum-interaction filters or segment metrics by item popularity for stable offline comparisons.")
    if "budget" in bundle.movies and "revenue" in bundle.movies:
        no_finance = int((pd.to_numeric(bundle.movies["budget"], errors="coerce").fillna(0).eq(0) & pd.to_numeric(bundle.movies["revenue"], errors="coerce").fillna(0).eq(0)).sum())
        if no_finance:
            recommendations.append("Treat zero budget/revenue as missing values; do not use finance metadata as a primary feature.")

    return DataQualityReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        data_dir=str(data_path),
        shapes={
            "movies": list(raw_movies.shape),
            "ratings": list(raw_ratings.shape),
            "tags": list(raw_tags.shape),
            "links": list(raw_links.shape),
            "enriched_movies": list(enriched.shape),
            "loaded_movies": list(bundle.movies.shape),
            "loaded_ratings": list(bundle.ratings.shape),
        },
        ids={
            "users": int(bundle.ratings["userId"].nunique()),
            "movies": int(bundle.movies["movieId"].nunique()),
            "rated_movies": int(bundle.ratings["movieId"].nunique()),
            "sparsity_pct": pct(1 - (len(bundle.ratings) / max(bundle.ratings["userId"].nunique() * bundle.movies["movieId"].nunique(), 1)), 1),
        },
        duplicates={
            "movies_movieId": int(raw_movies["movieId"].duplicated().sum()),
            "ratings_userId_movieId": int(raw_ratings.duplicated(["userId", "movieId"]).sum()),
            "links_movieId": int(raw_links["movieId"].duplicated().sum()) if "movieId" in raw_links else 0,
            "enriched_movieId": int(enriched["movieId"].duplicated().sum()) if "movieId" in enriched else 0,
            "tags_full_rows": int(raw_tags.duplicated().sum()) if not raw_tags.empty else 0,
        },
        orphans={
            "rating_movieIds_not_in_movies": len(rating_movie_ids - movie_ids),
            "tag_movieIds_not_in_movies": len(tag_movie_ids - movie_ids),
            "link_movieIds_not_in_movies": len(link_movie_ids - movie_ids),
            "enriched_movieIds_not_in_movies": len(enriched_movie_ids - movie_ids),
            "movies_without_ratings": len(movie_ids - rating_movie_ids),
            "movies_without_links": len(movie_ids - link_movie_ids) if link_movie_ids else len(movie_ids),
            "movies_without_enriched": len(movie_ids - enriched_movie_ids),
        },
        ratings={
            "min": float(bundle.ratings["rating"].min()),
            "max": float(bundle.ratings["rating"].max()),
            "mean": round(float(bundle.ratings["rating"].mean()), 4),
            "median": float(bundle.ratings["rating"].median()),
            "distribution": rating_distribution,
            "positive_threshold": 4.0,
            "positive_count": int((bundle.ratings["rating"] >= 4.0).sum()),
            "positive_pct": pct(int((bundle.ratings["rating"] >= 4.0).sum()), len(bundle.ratings)),
            "user_interactions_min": as_int(user_counts.min()),
            "user_interactions_median": float(user_counts.median()),
            "user_interactions_mean": round(float(user_counts.mean()), 4),
            "user_interactions_max": as_int(user_counts.max()),
            "item_interactions_min": as_int(item_counts.min()),
            "item_interactions_median": float(item_counts.median()),
            "item_interactions_mean": round(float(item_counts.mean()), 4),
            "item_interactions_max": as_int(item_counts.max()),
            "items_with_lt_5_ratings": int((item_counts < 5).sum()),
            "items_with_lt_10_ratings": int((item_counts < 10).sum()),
        },
        split={
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "train_pct": pct(len(train), len(bundle.ratings)),
            "val_pct": pct(len(val), len(bundle.ratings)),
            "test_pct": pct(len(test), len(bundle.ratings)),
            "train_users": int(train["userId"].nunique()),
            "val_users": int(val["userId"].nunique()),
            "test_users": int(test["userId"].nunique()),
            "test_users_not_in_train": len(set(test["userId"].astype(int)) - set(train["userId"].astype(int))),
            "val_movies_not_in_train": len(set(val["movieId"].astype(int)) - set(train["movieId"].astype(int))),
            "test_movies_not_in_train": len(set(test["movieId"].astype(int)) - set(train["movieId"].astype(int))),
            "val_warm_interactions": len(warm_val),
            "val_cold_interactions": len(cold_val),
            "test_warm_interactions": len(warm_test),
            "test_cold_interactions": len(cold_test),
            "test_cold_positive_interactions": int((cold_test["rating"] >= 4.0).sum()),
        },
        content_coverage=content_coverage,
        content_issues={
            "any_core_metadata_missing": int(empty_core.sum() or missing_core.sum()),
            "all_core_metadata_missing": int(all_empty_core.sum() or all_core_missing.sum()),
            "no_genres_listed": int(raw_movies["genres"].astype(str).eq("(no genres listed)").sum()),
            "missing_year": int(year_missing_mask(raw_movies).sum()),
            "tagged_movies": int(raw_tags["movieId"].nunique()) if "movieId" in raw_tags else 0,
            "tagged_movies_pct": pct(int(raw_tags["movieId"].nunique()) if "movieId" in raw_tags else 0, len(raw_movies)),
            "no_budget_no_revenue": int((pd.to_numeric(bundle.movies.get("budget", pd.Series(dtype=float)), errors="coerce").fillna(0).eq(0) & pd.to_numeric(bundle.movies.get("revenue", pd.Series(dtype=float)), errors="coerce").fillna(0).eq(0)).sum()),
        },
        examples={
            "missing_enriched_movies": records(missing_enriched_movies, example_limit),
            "movies_without_ratings": records(movies_without_ratings, example_limit),
            "missing_year": records(missing_year, example_limit),
            "no_genres_listed": records(no_genres, example_limit),
        },
        recommendations=recommendations,
    )


def fix_enriched_coverage(data_dir: str | Path) -> dict[str, int]:
    data_path = Path(data_dir)
    movies = pd.read_csv(data_path / "movies.csv")
    enriched_path = data_path / "enriched_movies.csv"
    if enriched_path.exists():
        enriched = pd.read_csv(enriched_path)
    else:
        enriched = pd.DataFrame(columns=ENRICHED_COLUMNS)

    for column in ENRICHED_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = 0 if column in NUMERIC_METADATA_COLUMNS else ""
    if "enrichment_status" not in enriched.columns:
        enriched["enrichment_status"] = "enriched"
        missing_core = enriched[CORE_METADATA_COLUMNS].fillna("").astype(str).apply(lambda col: col.str.strip().eq(""))
        enriched.loc[missing_core.all(axis=1), "enrichment_status"] = "empty_metadata"

    enriched["movieId"] = enriched["movieId"].astype(int)
    before = len(enriched)
    movie_ids = set(movies["movieId"].astype(int))
    enriched_ids = set(enriched["movieId"].astype(int))
    missing_ids = sorted(movie_ids - enriched_ids)

    if missing_ids:
        placeholder_base = {
            column: (0 if column in NUMERIC_METADATA_COLUMNS else "")
            for column in ENRICHED_COLUMNS
            if column != "movieId"
        }
        placeholders = pd.DataFrame(
            [
                {
                    **placeholder_base,
                    "movieId": int(movie_id),
                    "enrichment_status": "missing_enrichment_placeholder",
                }
                for movie_id in missing_ids
            ]
        )
        enriched = pd.concat([enriched, placeholders], ignore_index=True)

    enriched = enriched.loc[enriched["movieId"].astype(int).isin(movie_ids)].copy()
    enriched = enriched.drop_duplicates("movieId", keep="first").sort_values("movieId")
    ordered_columns = ENRICHED_COLUMNS + OPTIONAL_ENRICHED_COLUMNS
    extra_columns = [column for column in enriched.columns if column not in ordered_columns]
    enriched = enriched[ordered_columns + extra_columns]
    enriched.to_csv(enriched_path, index=False)

    return {
        "rows_before": before,
        "missing_added": len(missing_ids),
        "rows_after": len(enriched),
        "target_movies": len(movie_ids),
    }


def write_reports(report: DataQualityReport, output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    report_dict = asdict(report)
    (path / "data_quality.json").write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    coverage_rows = []
    for column, values in report.content_coverage.items():
        row = {"column": column}
        row.update(values)
        coverage_rows.append(row)
    pd.DataFrame(coverage_rows).to_csv(path / "content_coverage.csv", index=False)

    issue_rows = [{"issue": key, "value": value} for key, value in report.content_issues.items()]
    pd.DataFrame(issue_rows).to_csv(path / "content_issues.csv", index=False)

    lines = [
        "# Data Quality Report",
        "",
        f"- Data dir: `{report.data_dir}`",
        f"- Generated at: `{report.generated_at}`",
        "",
        "## Core Counts",
        "",
        f"- Users: {report.ids['users']}",
        f"- Movies: {report.ids['movies']}",
        f"- Ratings: {report.shapes['ratings'][0]}",
        f"- Rated movies: {report.ids['rated_movies']}",
        f"- Sparsity: {report.ids['sparsity_pct']}%",
        "",
        "## Main Issues",
        "",
    ]
    for key, value in report.orphans.items():
        lines.append(f"- {key}: {value}")
    for key, value in report.content_issues.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Split Diagnostics", ""])
    for key, value in report.split.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Recommendations", ""])
    for recommendation in report.recommendations:
        lines.append(f"- {recommendation}")
    if not report.recommendations:
        lines.append("- No critical data quality issues detected.")
    (path / "data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MovieLens-style data quality and optional enrichment coverage.")
    parser.add_argument("--data-dir", default="data/ml-latest-small")
    parser.add_argument("--output-dir", default="artifacts/data_quality/ml-latest-small")
    parser.add_argument("--example-limit", type=int, default=10)
    parser.add_argument("--fix-enriched", action="store_true", help="Add placeholder enriched rows for movieIds missing from enriched_movies.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fix_enriched:
        result = fix_enriched_coverage(args.data_dir)
        print("fixed enriched coverage:")
        for key, value in result.items():
            print(f"{key}: {value}")

    report = build_report(args.data_dir, example_limit=args.example_limit)
    write_reports(report, args.output_dir)
    print(f"wrote data quality reports to {args.output_dir}")
    print(f"movies_without_enriched: {report.orphans['movies_without_enriched']}")
    print(f"all_core_metadata_missing: {report.content_issues['all_core_metadata_missing']}")
    print(f"test_movies_not_in_train: {report.split['test_movies_not_in_train']}")


if __name__ == "__main__":
    main()
