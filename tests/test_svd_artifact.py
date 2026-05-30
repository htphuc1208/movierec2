from __future__ import annotations

import json
import unittest
from tempfile import TemporaryDirectory

import pandas as pd

from models import HybridMovieRecommender
from scripts.train_svd import export_recommender_artifact, require_torch


class SVDArtifactTest(unittest.TestCase):
    def test_pytorch_svd_exports_lightweight_recommender_artifact(self) -> None:
        try:
            torch, _, _ = require_torch()
            from models.SVD import SVDModel
        except SystemExit as exc:
            self.skipTest(str(exc))

        model = SVDModel(num_users=2, num_items=3, embedding_dim=4, global_mean=3.5)
        with TemporaryDirectory() as temp_dir:
            export_recommender_artifact(
                model=model,
                output_dir=temp_dir,
                idx_to_user={0: 101, 1: 102},
                idx_to_item={0: 1, 1: 2, 2: 3},
                dataset_name="unit",
                config={"epochs": 1},
                metrics={"test_rmse": 1.0},
                positive_threshold=4.0,
            )

            with open(f"{temp_dir}/manifest.json", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["model_name"], "svd-pytorch")
            self.assertEqual(manifest["collaborative"]["mode"], "funk_svd")

            movies = pd.DataFrame(
                {
                    "movieId": [1, 2, 3],
                    "title": ["A (2000)", "B (2001)", "C (2002)"],
                    "genres": ["Drama", "Comedy", "Action"],
                }
            )
            ratings = pd.DataFrame(
                {
                    "userId": [101, 101, 102],
                    "movieId": [1, 2, 3],
                    "rating": [5.0, 4.0, 3.5],
                    "timestamp": [1, 2, 3],
                }
            )
            loaded = HybridMovieRecommender.from_artifact(temp_dir, movies, ratings)
            info = loaded.model_info()
            self.assertEqual(info["model_name"], "svd-pytorch")
            self.assertEqual(info["collaborative_mode"], "funk_svd")
            self.assertGreaterEqual(loaded.predict_rating(101, 3), 0.5)
            self.assertEqual(len(loaded.recommend(user_id=101, top_k=1)), 1)


if __name__ == "__main__":
    unittest.main()
