"""Тест dense-поиска через мок-энкодер (bag-of-words), без моделей и сети."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from src.retrieval.dense import DenseIndex  # noqa: E402
from src.preprocess.tokenize import tokenize  # noqa: E402


class BagOfWordsEncoder:
    """Игрушечный энкодер: вектор = частоты слов по фиксированному словарю."""
    def __init__(self, vocab):
        self.vocab = {w: i for i, w in enumerate(vocab)}

    def encode(self, texts, batch_size=64, show_progress_bar=False):
        out = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for r, t in enumerate(texts):
            for tok in tokenize(t):
                j = self.vocab.get(tok)
                if j is not None:
                    out[r, j] += 1.0
        return out


VOCAB = ["астана", "қазақстан", "алматы", "норвегия", "скандинавия", "ел", "қала"]


class TestDenseIndex(unittest.TestCase):
    def setUp(self):
        self.enc = BagOfWordsEncoder(VOCAB)
        self.docs = [
            ("d1", "Астана Қазақстан астанасы"),
            ("d2", "Алматы үлкен қала"),
            ("d3", "Норвегия Скандинавия ел"),
        ]

    def test_ranks_relevant_doc_first(self):
        idx = DenseIndex(encoder=self.enc).index(self.docs, show_progress=False)
        res = idx.search("Норвегия қай ел", top_k=3)
        self.assertEqual(res[0][0], "d3")  # про Норвегию

    def test_run_returns_rankings(self):
        idx = DenseIndex(encoder=self.enc).index(self.docs, show_progress=False)
        run = idx.run({"q1": "Қазақстан астанасы", "q2": "Скандинавия"}, top_k=2)
        self.assertEqual(run["q1"][0], "d1")
        self.assertEqual(run["q2"][0], "d3")

    def test_query_prefix_applied(self):
        # префикс не должен ломать поиск (слова префикса вне словаря игнорятся)
        idx = DenseIndex(encoder=self.enc, query_prefix="query: ",
                         doc_prefix="passage: ").index(self.docs, show_progress=False)
        res = idx.search("Алматы", top_k=1)
        self.assertEqual(res[0][0], "d2")

    def test_set_embeddings_roundtrip(self):
        idx = DenseIndex(encoder=self.enc).index(self.docs, show_progress=False)
        ids, mat = idx.doc_ids, idx.matrix
        idx2 = DenseIndex(encoder=self.enc).set_embeddings(ids, mat)
        res = idx2.search("Норвегия", top_k=1)
        self.assertEqual(res[0][0], "d3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
