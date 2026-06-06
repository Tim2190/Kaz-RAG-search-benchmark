"""
Сквозной тест архитектуры «до/после»: BM25 + стеммер на inflected-запросе.

Доказывает главный механизм бенчмарка на контролируемом примере:
запрос содержит ДРУГУЮ словоформу, чем документ. Без стеммера BM25
не находит совпадения; со стеммером основы совпадают и документ всплывает.
Используется мок-стеммер (без сети), имитирующий обрезку казахских аффиксов.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval.bm25 import BM25Index, default_analyzer  # noqa: E402
from src.preprocess.stemmer import IdentityStemmer  # noqa: E402
from src.eval import metrics  # noqa: E402


class MockKazakhStemmer:
    """Грубая обрезка известных аффиксов — только для теста пайплайна."""
    name = "mock-kazakh"
    _table = {
        "балаларымызда": "бала", "балалар": "бала", "баланың": "бала",
        "кітабым": "кітап", "кітаптар": "кітап", "кітапты": "кітап",
        "мектепте": "мектеп", "мектебі": "мектеп",
    }

    def stem(self, token: str) -> str:
        return self._table.get(token, token)


# мини-корпус: документы написаны в одних словоформах...
CORPUS = [
    ("d1", "Мектепте балалар оқиды"),
    ("d2", "Кітаптар сөреде тұр"),
    ("d3", "Астана қаласы үлкен"),
]
# ...а запросы — в ДРУГИХ словоформах тех же слов
QUERIES = {
    "q1": "балаларымызда",   # релевантен d1 (бала)
    "q2": "кітабым",          # релевантен d2 (кітап)
}
QRELS = {"q1": {"d1"}, "q2": {"d2"}}


class TestStemmerLiftsRecall(unittest.TestCase):
    def _run(self, stemmer):
        idx = BM25Index(analyzer=default_analyzer(stemmer)).index(CORPUS)
        run = idx.run(QUERIES, top_k=3)
        return metrics.evaluate_run(run, QRELS, metrics=("recall",), ks=(1,))["recall@1"]

    def test_baseline_misses_inflected_queries(self):
        # без стеммера словоформы не совпадают -> ничего не найдено
        self.assertEqual(self._run(IdentityStemmer()), 0.0)

    def test_stemmer_recovers_relevant_docs(self):
        # со стеммером основы совпадают -> оба документа найдены на позиции 1
        self.assertEqual(self._run(MockKazakhStemmer()), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
