from __future__ import annotations

import pandas as pd

from recommender.analysis.eda import (
    genre_counts,
    infer_interaction_columns,
    overview_stats,
    rating_distribution,
    top_interacted_movies,
    user_segmentation,
)


def test_eda_helpers_handle_movielens_schema() -> None:
    interactions = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [10, 11, 10, 12, 11, 12],
            "rating": [5.0, 4.0, 3.5, 5.0, 4.5, 4.0],
            "timestamp": [1, 2, 1, 2, 1, 2],
        }
    )
    catalog = pd.DataFrame(
        {
            "movieId": [10, 11, 12],
            "title": ["A", "B", "C"],
            "genres": ["Drama", "Action", "Drama|Action"],
            "tmdb_genres": ["Drama", "Action", "Drama|Action"],
            "vote_average": [7.0, 6.0, 8.0],
            "popularity": [10.0, 5.0, 7.0],
            "release_year": [2000, 2001, 2002],
        }
    )
    columns = infer_interaction_columns(interactions)

    assert overview_stats(interactions, columns)["users"] == 3
    assert rating_distribution(interactions, columns)["count"].sum() == 6
    assert genre_counts(catalog).iloc[0]["genre"] in {"Drama", "Action"}
    assert top_interacted_movies(interactions, catalog, columns).iloc[0]["title"] in {"A", "B", "C"}

    clusters, genres = user_segmentation(interactions, catalog, columns, n_clusters=2)
    assert not clusters.empty
    assert genres
    assert {"PCA1", "PCA2", "cluster_name"}.issubset(clusters.columns)
