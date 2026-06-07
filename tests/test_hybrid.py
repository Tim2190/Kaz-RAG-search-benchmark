"""Юнит-тесты RRF-слияния (hybrid) — stdlib, без сети и GPU."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval import hybrid  # noqa: E402


class TestReciprocalRankFusion(unittest.TestCase):
    def test_rank_correctness_small(self):
        # Канал a и b ранжируют 4 документа; считаем RRF вручную при k=60.
        channels = {
            "a": {"q1": ["d1", "d2", "d3"]},
            "b": {"q1": ["d2", "d1", "d4"]},
        }
        scores = hybrid.fuse_scores(channels, k=60)["q1"]
        # d1: 1/61 (a:1) + 1/62 (b:2); d2: 1/62 (a:2) + 1/61 (b:1) — равны
        self.assertAlmostEqual(scores["d1"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(scores["d2"], 1 / 62 + 1 / 61)
        # d3 только в a (rank 3), d4 только в b (rank 3) — равны
        self.assertAlmostEqual(scores["d3"], 1 / 63)
        self.assertAlmostEqual(scores["d4"], 1 / 63)

    def test_ranking_order_and_tiebreak(self):
        channels = {
            "a": {"q1": ["d1", "d2", "d3"]},
            "b": {"q1": ["d2", "d1", "d4"]},
        }
        fused = hybrid.reciprocal_rank_fusion(channels, k=60)["q1"]
        # d1==d2 по скору -> тай-брейк по doc_id (d1<d2); d3==d4 -> d3<d4
        self.assertEqual(fused, ["d1", "d2", "d3", "d4"])

    def test_missing_doc_in_one_channel(self):
        # Документ, который вернул только один канал, всё равно ранжируется.
        channels = {
            "a": {"q1": ["dx"]},          # dx только в a
            "b": {"q1": ["dy"]},          # dy только в b
        }
        scores = hybrid.fuse_scores(channels, k=60)["q1"]
        # каждый получил вклад от одного канала (rank 1) и 0 от другого
        self.assertAlmostEqual(scores["dx"], 1 / 61)
        self.assertAlmostEqual(scores["dy"], 1 / 61)
        fused = hybrid.reciprocal_rank_fusion(channels, k=60)["q1"]
        self.assertEqual(set(fused), {"dx", "dy"})

    def test_doc_in_both_channels_outranks_single(self):
        # Документ, найденный обоими каналами, должен обойти найденный одним.
        channels = {
            "a": {"q1": ["shared", "only_a"]},
            "b": {"q1": ["shared", "only_b"]},
        }
        fused = hybrid.reciprocal_rank_fusion(channels, k=60)["q1"]
        self.assertEqual(fused[0], "shared")

    def test_determinism(self):
        channels = {
            "a": {"q1": ["d1", "d2", "d3"], "q2": ["d5", "d6"]},
            "b": {"q1": ["d3", "d2", "d1"], "q2": ["d6", "d7"]},
        }
        first = hybrid.reciprocal_rank_fusion(channels, k=60)
        second = hybrid.reciprocal_rank_fusion(channels, k=60)
        self.assertEqual(first, second)

    def test_top_k_truncation(self):
        channels = {
            "a": {"q1": ["d1", "d2", "d3", "d4", "d5"]},
            "b": {"q1": ["d5", "d4", "d3", "d2", "d1"]},
        }
        fused = hybrid.reciprocal_rank_fusion(channels, k=60, top_k=3)["q1"]
        self.assertEqual(len(fused), 3)

    def test_union_of_query_ids(self):
        # Запрос, который есть только в одном канале, не теряется.
        channels = {
            "a": {"q1": ["d1"], "q2": ["d2"]},
            "b": {"q1": ["d1"]},  # q2 отсутствует
        }
        fused = hybrid.reciprocal_rank_fusion(channels, k=60)
        self.assertIn("q2", fused)
        self.assertEqual(fused["q2"], ["d2"])

    def test_k_affects_scores_not_negative(self):
        channels = {"a": {"q1": ["d1", "d2"]}}
        s10 = hybrid.fuse_scores(channels, k=10)["q1"]
        s100 = hybrid.fuse_scores(channels, k=100)["q1"]
        # больший k -> меньший вклад каждого ранга
        self.assertGreater(s10["d1"], s100["d1"])

    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            hybrid.fuse_scores({"a": {"q1": ["d1"]}}, k=0)

    def test_roundtrip_save_load(self):
        import tempfile
        run = {"q1": ["d1", "d2"], "q2": ["d3"]}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.json")
            hybrid.save_run(path, run)
            loaded = hybrid.load_run(path)
        self.assertEqual(loaded, run)


if __name__ == "__main__":
    unittest.main()
