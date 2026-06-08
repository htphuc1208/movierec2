from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from data import MovieLensDataLoader
from models import HybridMovieRecommender
from scripts.train_two_tower import train


class TwoTowerTrainingSmokeTest(unittest.TestCase):
    def test_two_tower_exports_lightweight_recommender_artifact(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = SimpleNamespace(
                data_dir="data/sample",
                artifact_path=str(root / "two_tower.pt"),
                recommender_artifact_dir=str(root / "recommender"),
                dataset_name="sample",
                content_backend="tfidf",
                max_feature_dim=16,
                hidden_dim=8,
                output_dim=4,
                dropout=0.0,
                epochs=1,
                batch_size=16,
                lr=0.001,
                weight_decay=0.0,
                max_grad_norm=5.0,
                patience=1,
                top_k=3,
                positive_threshold=4.0,
                seed=42,
                device="",
                verbose=False,
            )
            result = train(args)
            self.assertGreaterEqual(result.all_ndcg_at_k, 0.0)
            self.assertTrue((root / "recommender" / "content.npz").exists())
            manifest = json.loads((root / "recommender" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["content"]["backend"], "tfidf")
            self.assertEqual(manifest["files"]["content"], "content.npz")

            bundle = MovieLensDataLoader("data/sample").load()
            loaded = HybridMovieRecommender.from_artifact(root / "recommender", bundle.movies, bundle.ratings, bundle.tags)
            self.assertTrue(loaded.model_info()["content_from_artifact"])
            recs = loaded.recommend(user_id=104, top_k=3)
            self.assertEqual(len(recs), 3)


if __name__ == "__main__":
    unittest.main()
