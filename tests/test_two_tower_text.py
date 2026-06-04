from __future__ import annotations

import pandas as pd

from recommender.models.two_tower import build_item_text


def test_build_item_text_uses_enriched_fields_without_tagline() -> None:
    catalog = pd.DataFrame(
        {
            "title": ["Movie"],
            "genres": ["Drama"],
            "overview": ["A quiet story."],
            "tagline": ["Marketing slogan"],
            "keywords": ["memory|family"],
            "director": ["Director"],
            "writers": ["Writer"],
            "cast": ["Actor"],
            "collection": ["Collection"],
            "production_companies": ["Studio"],
            "production_countries": ["France"],
            "original_language": ["fr"],
            "release_year": ["2001"],
        }
    )

    text = build_item_text(catalog)[0]

    assert "Marketing slogan" not in text
    assert "memory|family" in text
    assert "Writer" in text
    assert "Studio" in text
