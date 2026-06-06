"""Тесты сравнения результатов До/После."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval import compare  # noqa: E402


class TestCompare(unittest.TestCase):
    def setUp(self):
        self.before = {
            "stemmer": "identity", "n_passages": 100, "n_queries": 6,
            "overall": {"recall@1": 0.2, "recall@5": 0.4, "mrr@10": 0.3, "ndcg@10": 0.35},
            "by_category": {
                "inflected": {"recall@1": 0.1, "recall@5": 0.2, "mrr@10": 0.13, "ndcg@10": 0.16},
            },
        }
        self.after = {
            "stemmer": "kazakh", "n_passages": 100, "n_queries": 6,
            "overall": {"recall@1": 0.4, "recall@5": 0.6, "mrr@10": 0.5, "ndcg@10": 0.55},
            "by_category": {
                "inflected": {"recall@1": 0.3, "recall@5": 0.5, "mrr@10": 0.39, "ndcg@10": 0.45},
            },
        }

    def test_pct(self):
        self.assertEqual(compare._pct(0.1, 0.3), "+200%")
        self.assertEqual(compare._pct(0.0, 0.0), "—")
        self.assertEqual(compare._pct(0.5, 0.25), "-50%")

    def test_table_contains_delta_and_categories(self):
        table = compare.format_table(self.before, self.after)
        self.assertIn("ВСЕ", table)
        self.assertIn("inflected", table)
        self.assertIn("прирост", table)
        self.assertIn("+200%", table)  # inflected recall@1: 0.1 -> 0.3

    def test_rows_cover_all_metrics(self):
        rows = compare._rows(self.before, self.after)
        metrics = {m for _, m, _, _ in rows}
        self.assertEqual(metrics, set(compare.METRICS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
