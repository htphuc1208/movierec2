from __future__ import annotations

import numpy as np
import pandas as pd

from recommender.inference.artifacts import save_artifact_bundle, load_artifact_bundle
from recommender.rag.chatbot import MovieRAGChatbot
from recommender.rag.retriever import MovieRAGRetriever


def _save_chat_bundle(tmp_path) -> None:
    save_artifact_bundle(
        tmp_path,
        catalog=pd.DataFrame(
            {
                "movieId": [1, 2, 3],
                "title": ["Space Rescue", "Kitchen Story", "Galaxy War"],
                "genres": ["Sci-Fi", "Drama", "Sci-Fi"],
                "tmdb_genres": ["Science Fiction", "Drama", "Science Fiction|Action"],
                "overview": [
                    "Astronauts rescue a stranded crew near Jupiter.",
                    "A family drama set in a small restaurant kitchen.",
                    "Pilots fight a war across distant galaxies.",
                ],
                "poster_url": [None, None, None],
            }
        ),
        user_mapping={1: 0},
        item_mapping={1: 0, 2: 1, 3: 2},
        content_embeddings=np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32),
        user_profiles=np.array([[1.0, 0.0]], dtype=np.float32),
        item_popularity=np.array([0.2, 0.1, 0.3], dtype=np.float32),
        metrics={},
        hybrid_config={"content_backend": "tfidf"},
    )


def test_rag_retriever_uses_lexical_fallback(tmp_path) -> None:
    _save_chat_bundle(tmp_path)
    retriever = MovieRAGRetriever(load_artifact_bundle(tmp_path))

    results = retriever.retrieve("phim về galaxy và phi công", top_k=2)

    assert results
    assert results[0].title == "Galaxy War"
    assert retriever.mode == "lexical"


def test_chatbot_local_answer_without_openai_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _save_chat_bundle(tmp_path)
    chatbot = MovieRAGChatbot(load_artifact_bundle(tmp_path), api_key="")

    result = chatbot.answer("Tôi muốn xem phim không gian", top_k=2)

    assert "answer" in result
    assert result["sources"]
    assert result["retrieval_mode"] == "lexical"
