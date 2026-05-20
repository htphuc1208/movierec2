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


def fetch_movie(tmdb_id: str, api_key: str) -> dict[str, Any]:
    url = TMDB_API.format(tmdb_id=tmdb_id)
    response = requests.get(
        url,
        params={"api_key": api_key, "append_to_response": "credits"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    crew = payload.get("credits", {}).get("crew", [])
    cast = payload.get("credits", {}).get("cast", [])
    directors = [person["name"] for person in crew if person.get("job") == "Director"]
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich MovieLens links.csv with TMDb metadata.")
    parser.add_argument("--data-dir", default="data/sample")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.25)
    args = parser.parse_args()

    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise SystemExit("TMDB_API_KEY is required.")

    data_dir = Path(args.data_dir)
    links = pd.read_csv(data_dir / "links.csv")
    rows = []
    selected = links.head(args.limit) if args.limit else links
    for row in tqdm(selected.itertuples(), total=len(selected)):
        tmdb_id = str(row.tmdbId).split(".")[0]
        if not tmdb_id or tmdb_id == "nan":
            continue
        try:
            metadata = fetch_movie(tmdb_id, api_key)
            metadata["movieId"] = int(row.movieId)
            rows.append(metadata)
            time.sleep(args.sleep)
        except requests.RequestException as exc:
            print(f"skip movieId={row.movieId}: {exc}")

    output = pd.DataFrame(rows)
    output.to_csv(data_dir / "enriched_movies.csv", index=False)
    print(f"Wrote {len(output)} rows to {data_dir / 'enriched_movies.csv'}")


if __name__ == "__main__":
    main()
