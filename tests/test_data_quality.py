from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.audit_data_quality import build_report, fix_enriched_coverage, write_reports
from scripts.enrich_tmdb import merge_enriched, select_links


class DataQualityAuditTest(unittest.TestCase):
    def test_build_report_for_sample_dataset(self) -> None:
        report = build_report("data/sample", example_limit=2)
        self.assertGreater(report.ids["users"], 0)
        self.assertGreater(report.ids["movies"], 0)
        self.assertIn("overview", report.content_coverage)

    def test_fix_enriched_coverage_adds_missing_movie_rows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "sample"
            shutil.copytree("data/sample", target)
            enriched_path = target / "enriched_movies.csv"
            enriched = pd.read_csv(enriched_path)
            enriched = enriched.iloc[:-1]
            enriched.to_csv(enriched_path, index=False)

            result = fix_enriched_coverage(target)
            movies = pd.read_csv(target / "movies.csv")
            fixed = pd.read_csv(enriched_path)

            self.assertEqual(result["rows_after"], len(movies))
            self.assertEqual(len(fixed), len(movies))
            self.assertIn("enrichment_status", fixed.columns)

    def test_write_reports_creates_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report = build_report("data/sample", example_limit=2)
            write_reports(report, temp_dir)
            self.assertTrue((Path(temp_dir) / "data_quality.json").exists())
            self.assertTrue((Path(temp_dir) / "content_coverage.csv").exists())
            self.assertTrue((Path(temp_dir) / "data_quality.md").exists())

    def test_enrichment_refresh_selector_and_merge(self) -> None:
        links = pd.DataFrame({"movieId": [1, 2], "tmdbId": [10, 20]})
        existing = pd.DataFrame(
            {
                "movieId": [1, 2],
                "overview": ["x", ""],
                "director": ["d", ""],
                "cast": ["c", ""],
                "poster_url": ["p", ""],
                "keywords": ["hero", ""],
                "enrichment_status": ["enriched", "missing_enrichment_placeholder"],
            }
        )

        retry_rows = select_links(links, existing, only_missing=False, retry_empty=True, refresh_empty_columns=[])
        self.assertEqual(retry_rows["movieId"].astype(int).tolist(), [2])

        keyword_rows = select_links(links, existing, only_missing=False, retry_empty=False, refresh_empty_columns=["keywords"])
        self.assertEqual(keyword_rows["movieId"].astype(int).tolist(), [2])

        merged = merge_enriched(existing, [{"movieId": 2, "overview": "new", "keywords": "noir"}])
        by_id = merged.set_index("movieId")
        self.assertEqual(by_id.loc[2, "overview"], "new")
        self.assertEqual(by_id.loc[2, "enrichment_status"], "enriched")


if __name__ == "__main__":
    unittest.main()
