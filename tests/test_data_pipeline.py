from __future__ import annotations

import pandas as pd

from recommender.data.movielens import build_user_item_sets, filter_catalog_to_items, prepare_interactions


def test_prepare_interactions_encodes_and_splits_by_user() -> None:
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 2, 2, 2],
            "movieId": [10, 11, 12, 13, 10, 14, 15],
            "rating": [5, 4, 4, 5, 5, 4, 3],
            "timestamp": [1, 2, 3, 4, 1, 2, 3],
        }
    )

    prepared = prepare_interactions(ratings, min_rating=4.0, val_ratio=0.25, test_ratio=0.25)

    assert prepared.num_users == 2
    assert prepared.num_items == 5
    assert {"user_idx", "item_idx"}.issubset(prepared.train.columns)
    assert len(prepared.test) >= 1

    train_pairs = set(zip(prepared.train.user_idx, prepared.train.item_idx))
    test_pairs = set(zip(prepared.test.user_idx, prepared.test.item_idx))
    assert train_pairs.isdisjoint(test_pairs)


def test_user_item_sets_and_catalog_ordering() -> None:
    interactions = pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [2, 3, 2]})
    assert build_user_item_sets(interactions) == {0: {2, 3}, 1: {2}}

    movies = pd.DataFrame(
        {
            "movieId": [42, 7],
            "title": ["Later", "First"],
            "genres": ["Drama", "Action"],
        }
    )
    catalog = filter_catalog_to_items(movies, {7: 0, 42: 1})
    assert catalog["movieId"].tolist() == [7, 42]
    assert catalog["title"].tolist() == ["First", "Later"]
