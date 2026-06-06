"""Тест связки прогонщика бенчмарка на крошечном корпусе (identity, без сети)."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval import run_benchmark  # noqa: E402


class TestRunBenchmark(unittest.TestCase):
    def setUp(self):
        self.corpus = tempfile.mktemp(suffix=".jsonl")
        self.queries = tempfile.mktemp(suffix=".jsonl")
        corpus = [
            {"doc_id": "d1", "text": "Астана Қазақстанның астанасы болып табылады"},
            {"doc_id": "d2", "text": "Алматы ең үлкен қала және бұрынғы астана"},
            {"doc_id": "d3", "text": "Норвегия Скандинавия түбегінде орналасқан ел"},
        ]
        queries = [
            {"query_id": "q1", "text": "Қазақстанның астанасы қай қала",
             "category": "natural", "gold_doc_ids": ["d1"]},
            {"query_id": "q2", "text": "Скандинавия түбегіндегі ел",
             "category": "natural", "gold_doc_ids": ["d3"]},
        ]
        with open(self.corpus, "w", encoding="utf-8") as f:
            for d in corpus:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        with open(self.queries, "w", encoding="utf-8") as f:
            for q in queries:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    def tearDown(self):
        for p in (self.corpus, self.queries):
            if os.path.exists(p):
                os.remove(p)

    def test_run_produces_metrics_and_categories(self):
        res = run_benchmark.run(self.corpus, self.queries, "identity", top_k=10)
        self.assertEqual(res["n_passages"], 3)
        self.assertEqual(res["n_queries"], 2)
        # обе цели находятся легко -> recall@1 должен быть высоким
        self.assertGreaterEqual(res["overall"]["recall@1"], 0.5)
        self.assertIn("natural", res["by_category"])
        self.assertIn("recall@1", res["overall"])

    def test_table_formats(self):
        res = run_benchmark.run(self.corpus, self.queries, "identity", top_k=10)
        table = run_benchmark._fmt_table(res)
        self.assertIn("recall@1", table)
        self.assertIn("natural", table)


if __name__ == "__main__":
    unittest.main(verbosity=2)
