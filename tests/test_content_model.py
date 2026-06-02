from __future__ import annotations

import unittest

from data import MovieLensDataLoader
from models import ContentRecommender, MetadataEncoder, TFIDFRecommender, TwoTowerModel


class ContentModelSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = MovieLensDataLoader("data/sample").load()

    def test_metadata_encoder_accepts_current_schema(self) -> None:
        encoded = MetadataEncoder(backend="tfidf").fit_transform(self.bundle.movies, self.bundle.tags)
        self.assertEqual(len(encoded.movie_ids), len(self.bundle.movies))
        self.assertEqual(encoded.vectors.shape[0], len(self.bundle.movies))

    def test_similar_movies(self) -> None:
        model = ContentRecommender(backend="tfidf").fit(self.bundle.movies, self.bundle.tags)
        recs = model.recommend_similar_movies(movie_id=1, top_k=3)
        self.assertEqual(len(recs), 3)
        self.assertIn("movie_id", recs[0])
        self.assertNotEqual(recs[0]["movie_id"], 1)

    def test_user_history_accepts_snake_case_schema(self) -> None:
        model = TFIDFRecommender().fit(self.bundle.movies, self.bundle.tags)
        history = self.bundle.ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})
        recs = model.recommend_for_user(user_id=104, user_history=history, top_k=3)
        self.assertEqual(len(recs), 3)

    def test_two_tower_model_scores_pairs_when_torch_available(self) -> None:
        try:
            import torch
        except ImportError as exc:
            self.skipTest(str(exc))

        model = TwoTowerModel(input_dim=8, hidden_dim=4, output_dim=3)
        scores = model(torch.ones(2, 8), torch.ones(2, 8))
        self.assertEqual(tuple(scores.shape), (2,))


if __name__ == "__main__":
    unittest.main()
