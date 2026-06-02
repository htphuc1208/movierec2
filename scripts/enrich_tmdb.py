from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm


TMDB_API = "https://api.themoviedb.org/3/movie/{tmdb_id}"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_APPEND = "credits,keywords,release_dates"
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


def pipe_names(values: list[dict[str, Any]], key: str = "name", limit: int | None = None) -> str:
    selected = values[:limit] if limit else values
    return "|".join(str(value.get(key, "")).strip() for value in selected if str(value.get(key, "")).strip())


def certification_from_release_dates(payload: dict[str, Any], country: str = "US") -> str:
    results = payload.get("release_dates", {}).get("results", [])
    country_release = next((item for item in results if item.get("iso_3166_1") == country), None)
    if not country_release:
        return ""
    for release in country_release.get("release_dates", []):
        certification = str(release.get("certification", "")).strip()
        if certification:
            return certification
    return ""


def fetch_movie(
    tmdb_id: str,
    api_key: str,
    session: requests.Session,
    max_retries: int = 3,
) -> dict[str, Any]:
    url = TMDB_API.format(tmdb_id=tmdb_id)

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                url,
                params={
                    "api_key": api_key,
                    "append_to_response": TMDB_APPEND,
                },
                timeout=(10, 60),  # connect timeout, read timeout
            )

            response.raise_for_status()
            payload = response.json()

            crew = payload.get("credits", {}).get("crew", [])
            cast = payload.get("credits", {}).get("cast", [])
            genres = payload.get("genres", [])
            keywords = payload.get("keywords", {}).get("keywords", [])
            production_countries = payload.get("production_countries", [])
            collection = payload.get("belongs_to_collection") or {}

            directors = [
                person["name"]
                for person in crew
                if person.get("job") == "Director"
            ]

            top_cast = [person["name"] for person in cast[:5]]
            poster_path = payload.get("poster_path") or ""

            return {
                "overview": payload.get("overview", ""),
                "tagline": payload.get("tagline", ""),
                "director": "|".join(directors),
                "cast": "|".join(top_cast),
                "poster_url": f"{IMAGE_BASE}{poster_path}" if poster_path else "",
                "budget": payload.get("budget", 0),
                "revenue": payload.get("revenue", 0),
                "genres": pipe_names(genres),
                "release_date": payload.get("release_date", ""),
                "runtime": payload.get("runtime", 0) or 0,
                "original_language": payload.get("original_language", ""),
                "production_countries": pipe_names(production_countries, key="iso_3166_1"),
                "keywords": pipe_names(keywords),
                "vote_average": payload.get("vote_average", 0) or 0,
                "vote_count": payload.get("vote_count", 0) or 0,
                "popularity": payload.get("popularity", 0) or 0,
                "collection_id": collection.get("id", 0) or 0,
                "collection_name": collection.get("name", "") or "",
                "certification": certification_from_release_dates(payload),
                "imdb_id": payload.get("imdb_id", "") or "",
            }

        except requests.exceptions.Timeout:
            print(f"timeout tmdb_id={tmdb_id}, retry {attempt}/{max_retries}")

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None

            if status == 429:
                print(f"rate limited tmdb_id={tmdb_id}, retry {attempt}/{max_retries}")
            else:
                raise

        except requests.RequestException:
            raise

        time.sleep(2 * attempt)

    raise requests.exceptions.Timeout(f"Failed after {max_retries} retries: tmdb_id={tmdb_id}")


def load_existing_enriched(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "enriched_movies.csv"
    if path.exists():
        enriched = pd.read_csv(path)
    else:
        enriched = pd.DataFrame(columns=ENRICHED_COLUMNS)

    for column in ENRICHED_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = 0 if column in NUMERIC_METADATA_COLUMNS else ""
    if "enrichment_status" not in enriched.columns:
        enriched["enrichment_status"] = "enriched"
    if not enriched.empty:
        enriched["movieId"] = enriched["movieId"].astype(int)
    return enriched


def select_links(
    links: pd.DataFrame,
    existing: pd.DataFrame,
    *,
    only_missing: bool,
    retry_empty: bool,
    refresh_empty_columns: list[str],
) -> pd.DataFrame:
    links = links.copy()
    links["movieId"] = links["movieId"].astype(int)
    if existing.empty:
        return links
    existing = existing.copy()
    existing["movieId"] = existing["movieId"].astype(int)

    if only_missing:
        known_ids = set(existing["movieId"].tolist())
        return links.loc[~links["movieId"].isin(known_ids)]

    if retry_empty:
        merged = links.merge(existing, on="movieId", how="left")
        status = merged.get("enrichment_status", pd.Series("", index=merged.index)).fillna("").astype(str)
        placeholder = status.isin(["missing_enrichment_placeholder", "empty_metadata", "error"])
        core_empty = pd.Series(True, index=merged.index)
        for column in CORE_METADATA_COLUMNS:
            if column in merged:
                core_empty &= merged[column].fillna("").astype(str).str.strip().eq("")
        return links.loc[placeholder | core_empty]

    if refresh_empty_columns:
        merged = links.merge(existing, on="movieId", how="left")
        needs_refresh = pd.Series(False, index=merged.index)
        for column in refresh_empty_columns:
            if column not in merged.columns:
                needs_refresh = pd.Series(True, index=merged.index)
                continue
            if column in NUMERIC_METADATA_COLUMNS:
                needs_refresh |= pd.to_numeric(merged[column], errors="coerce").fillna(0).eq(0)
            else:
                needs_refresh |= merged[column].fillna("").astype(str).str.strip().eq("")
        return links.loc[needs_refresh]

    return links


def merge_enriched(existing: pd.DataFrame, new_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not new_rows:
        return existing

    updates = pd.DataFrame(new_rows)
    updates["movieId"] = updates["movieId"].astype(int)
    updates["enrichment_status"] = "enriched"

    existing = existing.copy()
    existing["movieId"] = existing["movieId"].astype(int)
    merged = pd.concat(
        [
            existing.loc[~existing["movieId"].isin(set(updates["movieId"].tolist()))],
            updates,
        ],
        ignore_index=True,
    )
    ordered = [column for column in ENRICHED_COLUMNS + ["enrichment_status"] if column in merged.columns]
    extra = [column for column in merged.columns if column not in ordered]
    return merged[ordered + extra].drop_duplicates("movieId", keep="last").sort_values("movieId")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich MovieLens links.csv with TMDb metadata.")
    parser.add_argument("--data-dir", default="data/sample")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument(
        "--retry-empty",
        action="store_true",
        help="Only fetch rows with placeholder/empty core metadata, then merge into enriched_movies.csv.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only fetch movieIds absent from enriched_movies.csv, then merge into enriched_movies.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace enriched_movies.csv with fetched rows. Without this flag, fetched rows are merged safely.",
    )
    parser.add_argument(
        "--refresh-empty-columns",
        default="",
        help="Comma-separated metadata columns to refresh when empty, for example: keywords,release_date,runtime.",
    )
    args = parser.parse_args()

    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise SystemExit("TMDB_API_KEY is required.")

    data_dir = Path(args.data_dir)
    links = pd.read_csv(data_dir / "links.csv")
    existing = load_existing_enriched(data_dir)

    rows = []
    refresh_empty_columns = [column.strip() for column in args.refresh_empty_columns.split(",") if column.strip()]
    selected = select_links(
        links,
        existing,
        only_missing=args.only_missing,
        retry_empty=args.retry_empty,
        refresh_empty_columns=refresh_empty_columns,
    )
    selected = selected.head(args.limit) if args.limit else selected

    session = requests.Session()

    for row in tqdm(selected.itertuples(), total=len(selected)):
        tmdb_id = str(row.tmdbId).split(".")[0]

        if not tmdb_id or tmdb_id == "nan":
            continue

        try:
            metadata = fetch_movie(tmdb_id, api_key, session)
            metadata["movieId"] = int(row.movieId)
            rows.append(metadata)
            time.sleep(args.sleep)

        except requests.RequestException as exc:
            print(f"skip movieId={row.movieId}, tmdbId={tmdb_id}: {exc}")

    if args.overwrite:
        output = pd.DataFrame(rows)
    else:
        output = merge_enriched(existing, rows)
    output.to_csv(data_dir / "enriched_movies.csv", index=False)
    print(f"Wrote {len(output)} rows to {data_dir / 'enriched_movies.csv'}")


if __name__ == "__main__":
    main()
