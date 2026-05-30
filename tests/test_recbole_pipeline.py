from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.benchmark_recbole import prepare_recbole_dataset


class RecBolePipelineTest(unittest.TestCase):
    def test_prepare_recbole_dataset_writes_atomic_interactions(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = prepare_recbole_dataset("data/sample", temp_dir, "sample", positive_threshold=4.0)
            inter_path = Path(output_dir) / "sample.inter"
            self.assertTrue(inter_path.exists())

            prepared = pd.read_csv(inter_path, sep="\t")
            source = pd.read_csv("data/sample/ratings.csv")
            expected_count = int((source["rating"] >= 4.0).sum())
            self.assertEqual(list(prepared.columns), ["user_id:token", "item_id:token", "rating:float", "timestamp:float"])
            self.assertEqual(len(prepared), expected_count)


if __name__ == "__main__":
    unittest.main()
