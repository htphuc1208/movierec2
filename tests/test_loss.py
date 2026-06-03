from __future__ import annotations

import unittest

from models.Loss import warp_loss


class LossTest(unittest.TestCase):
    def test_warp_loss_penalizes_margin_violations_when_torch_available(self) -> None:
        try:
            import torch
        except ImportError as exc:
            self.skipTest(str(exc))

        pos_scores = torch.tensor([0.2, 2.0])
        neg_scores = torch.tensor([0.8, 0.5])
        rank_weights = torch.tensor([2.0, 3.0])

        loss = warp_loss(pos_scores, neg_scores, rank_weights, margin=1.0)

        self.assertGreater(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
