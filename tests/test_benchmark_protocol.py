from __future__ import annotations

import unittest

import pandas as pd

from evaluation import PROTOCOL_NAME, normalise_metrics, temporal_train_val_test_split


class BenchmarkProtocolTest(unittest.TestCase):
    def test_temporal_split_is_per_user_and_ordered(self) -> None:
        ratings = pd.DataFrame(
            {
                "userId": [1, 1, 1, 1, 1, 2, 2],
                "movieId": [10, 11, 12, 13, 14, 20, 21],
                "rating": [5, 4, 3, 5, 4, 5, 4],
                "timestamp": [1, 2, 3, 4, 5, 1, 2],
            }
        )
        split = temporal_train_val_test_split(ratings)
        self.assertEqual(PROTOCOL_NAME, "per_user_temporal_80_10_10_pos4_full")
        self.assertEqual(split.train.loc[split.train["userId"] == 1, "movieId"].tolist(), [10, 11, 12])
        self.assertEqual(split.validation.loc[split.validation["userId"] == 1, "movieId"].tolist(), [13])
        self.assertEqual(split.test.loc[split.test["userId"] == 1, "movieId"].tolist(), [14])
        self.assertEqual(split.train.loc[split.train["userId"] == 2, "movieId"].tolist(), [20, 21])

    def test_leaderboard_metric_normalizer_prefers_all_metrics(self) -> None:
        metrics = {
            "precision_at_k": 0.1,
            "all_precision_at_k": 0.2,
            "all_ndcg_at_k": 0.3,
            "warm_ndcg_at_k": 0.25,
            "cold_ndcg_at_k": 0.05,
            "warm_test_interactions": 12,
            "cold_test_interactions": 3,
            "test_rmse": 0.9,
            "test_mae": 0.7,
        }
        normalised = normalise_metrics(metrics)
        self.assertEqual(normalised["precision@10"], 0.2)
        self.assertEqual(normalised["ndcg@10"], 0.3)
        self.assertEqual(normalised["warm_ndcg@10"], 0.25)
        self.assertEqual(normalised["cold_ndcg@10"], 0.05)
        self.assertEqual(normalised["warm_interactions"], 12)
        self.assertEqual(normalised["cold_interactions"], 3)
        self.assertEqual(normalised["rmse"], 0.9)
        self.assertEqual(normalised["mae"], 0.7)

    def test_leaderboard_metric_normalizer_reads_nested_segments(self) -> None:
        normalised = normalise_metrics(
            {
                "test": {"ndcg@10": 0.4},
                "test_segments": {
                    "warm_interactions": 9,
                    "cold_interactions": 2,
                    "warm": {"ndcg@10": 0.5, "mrr@10": 0.6},
                    "cold": {"ndcg@10": 0.1, "mrr@10": 0.2},
                },
            }
        )
        self.assertEqual(normalised["ndcg@10"], 0.4)
        self.assertEqual(normalised["warm_ndcg@10"], 0.5)
        self.assertEqual(normalised["cold_mrr@10"], 0.2)
        self.assertEqual(normalised["warm_interactions"], 9)
        self.assertEqual(normalised["cold_interactions"], 2)


if __name__ == "__main__":
    unittest.main()
