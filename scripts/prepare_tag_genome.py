from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def prepare_tag_genome(
    data_dir: str | Path,
    genome_dir: str | Path,
    output_path: str | Path = "",
    top_n: int = 20,
    min_relevance: float = 0.35,
) -> pd.DataFrame:
    data_path = Path(data_dir)
    genome_path = Path(genome_dir)
    output = Path(output_path) if output_path else data_path / "tag_genome.csv"

    movies = pd.read_csv(data_path / "movies.csv", usecols=["movieId"])
    movie_ids = set(movies["movieId"].astype(int).tolist())

    scores = pd.read_csv(genome_path / "genome-scores.csv")
    tags = pd.read_csv(genome_path / "genome-tags.csv")
    required_scores = {"movieId", "tagId", "relevance"}
    required_tags = {"tagId", "tag"}
    if not required_scores.issubset(scores.columns):
        raise ValueError(f"genome-scores.csv must contain: {sorted(required_scores)}")
    if not required_tags.issubset(tags.columns):
        raise ValueError(f"genome-tags.csv must contain: {sorted(required_tags)}")

    scores["movieId"] = scores["movieId"].astype(int)
    scores["relevance"] = pd.to_numeric(scores["relevance"], errors="coerce").fillna(0)
    filtered = scores.loc[
        scores["movieId"].isin(movie_ids) & scores["relevance"].ge(float(min_relevance)),
        ["movieId", "tagId", "relevance"],
    ]
    merged = filtered.merge(tags[["tagId", "tag"]], on="tagId", how="left")
    merged["tag"] = merged["tag"].fillna("").astype(str)
    top_tags = (
        merged.sort_values(["movieId", "relevance"], ascending=[True, False])
        .groupby("movieId")
        .head(int(top_n))
        .groupby("movieId")["tag"]
        .apply(lambda values: " ".join(value for value in values if value.strip()))
        .reset_index(name="tag_genome_tags")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    top_tags.to_csv(output, index=False)
    return top_tags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact per-movie Tag Genome features for MovieLens data.")
    parser.add_argument("--data-dir", default="data/ml-latest-small")
    parser.add_argument("--genome-dir", required=True, help="Directory containing genome-scores.csv and genome-tags.csv.")
    parser.add_argument("--output-path", default="")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-relevance", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_tags = prepare_tag_genome(
        data_dir=args.data_dir,
        genome_dir=args.genome_dir,
        output_path=args.output_path,
        top_n=args.top_n,
        min_relevance=args.min_relevance,
    )
    target = args.output_path or str(Path(args.data_dir) / "tag_genome.csv")
    print(f"wrote {len(top_tags)} movie tag rows to {target}")


if __name__ == "__main__":
    main()
