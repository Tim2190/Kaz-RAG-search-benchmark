"""Юнит-тесты метрик ранжирования — запускаются stdlib (без numpy)."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval import metrics  # noqa: E402
from src.preprocess.tokenize import tokenize  # noqa: E402


class TestRankingMetrics(unittest.TestCase):
    def test_hit_at_k(self):
        ranked = ["d3", "d1", "d2"]
        rel = {"d1"}
        self.assertEqual(metrics.hit_at_k(ranked, rel, 1), 0.0)
        self.assertEqual(metrics.hit_at_k(ranked, rel, 2), 1.0)

    def test_recall_at_k(self):
        ranked = ["d1", "d5", "d2", "d9"]
        rel = {"d1", "d2"}
        self.assertAlmostEqual(metrics.recall_at_k(ranked, rel, 1), 0.5)
        self.assertAlmostEqual(metrics.recall_at_k(ranked, rel, 3), 1.0)

    def test_reciprocal_rank(self):
        ranked = ["d9", "d1", "d2"]
        self.assertAlmostEqual(metrics.reciprocal_rank(ranked, {"d1"}, 10), 0.5)
        self.assertAlmostEqual(metrics.reciprocal_rank(ranked, {"d9"}, 10), 1.0)
        self.assertAlmostEqual(metrics.reciprocal_rank(ranked, {"dX"}, 10), 0.0)

    def test_ndcg_perfect_and_partial(self):
        # идеальный порядок -> 1.0
        ranked = ["d1", "d2", "d3"]
        self.assertAlmostEqual(metrics.ndcg_at_k(ranked, {"d1", "d2"}, 3), 1.0)
        # один релевантный на 2-й позиции
        single = metrics.ndcg_at_k(["dx", "d1", "dy"], {"d1"}, 3)
        self.assertAlmostEqual(single, 1.0 / math.log2(3))

    def test_evaluate_run_aggregates(self):
        qrels = {"q1": {"a"}, "q2": {"b"}}
        run = {"q1": ["a", "z"], "q2": ["z", "b"]}
        res = metrics.evaluate_run(run, qrels, metrics=("recall", "mrr"), ks=(1, 2))
        self.assertAlmostEqual(res["recall@1"], 0.5)   # q1 попал, q2 нет
        self.assertAlmostEqual(res["recall@2"], 1.0)
        self.assertAlmostEqual(res["mrr@2"], (1.0 + 0.5) / 2)

    def test_bootstrap_detects_improvement(self):
        # B строго лучше A на всех запросах -> дельта>0, p маленькое
        qrels = {f"q{i}": {"gold"} for i in range(40)}
        run_a = {f"q{i}": ["x", "gold"] for i in range(40)}   # gold на 2-й
        run_b = {f"q{i}": ["gold", "x"] for i in range(40)}   # gold на 1-й
        delta, p, _ = metrics.paired_bootstrap(
            run_a, run_b, qrels, metric="mrr", k=10, n_resamples=2000
        )
        self.assertGreater(delta, 0)
        self.assertLess(p, 0.05)


class TestKazakhTokenizer(unittest.TestCase):
    def test_preserves_kazakh_letters(self):
        # Қазақстан Республикасының — должны выжить все спецбуквы
        toks = tokenize("Қазақстан Республикасының Үкіметі ғылым")
        self.assertIn("қазақстан", toks)
        self.assertIn("үкіметі", toks)
        self.assertIn("ғылым", toks)

    def test_splits_and_lowercases(self):
        self.assertEqual(tokenize("Астана, 2026 жыл!"), ["астана", "2026", "жыл"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
