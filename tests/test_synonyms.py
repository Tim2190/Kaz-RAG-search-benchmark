"""
Юнит-тесты SynonymExpander — stdlib, без сети и GPU.
Мок: переопределяем _post() аналогично тестам стеммера.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess.synonyms import SynonymExpander  # noqa: E402


class FakeSynonymExpander(SynonymExpander):
    """Подменяет сеть: возвращает фиксированные синонимы по словарю."""

    DICT = {
        "абайла": ["аңда", "байқа"],
        "теңіз": ["мұхит"],
        "үй": [],  # found=false → пустой список
    }

    def __init__(self, **kw):
        self.calls = []
        super().__init__(**kw)

    def _post(self, payload: bytes) -> str:
        words = json.loads(payload.decode())["words"]
        self.calls.append(list(words))
        result = []
        for w in words:
            syns = self.DICT.get(w, [])
            result.append({"word": w, "stem": w, "synonyms": syns, "found": bool(syns)})
        return json.dumps(result)


def _tmp_cache():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)
    return path


class TestSynonymExpander(unittest.TestCase):
    def setUp(self):
        self.cache = _tmp_cache()

    def tearDown(self):
        for p in (self.cache, self.cache + ".tmp"):
            if os.path.exists(p):
                os.remove(p)

    def test_warm_fetches_uncached(self):
        exp = FakeSynonymExpander(cache_path=self.cache, max_per_minute=0)
        exp.warm(["абайла", "теңіз"])
        self.assertEqual(len(exp.calls), 1)
        self.assertCountEqual(exp.calls[0], ["абайла", "теңіз"])

    def test_warm_skips_cached(self):
        exp = FakeSynonymExpander(cache_path=self.cache, max_per_minute=0)
        exp.warm(["абайла"])
        exp.calls.clear()
        exp.warm(["абайла", "теңіз"])
        # абайла уже в кэше → только теңіз идёт в сеть
        self.assertEqual(exp.calls, [["теңіз"]])

    def test_cache_persists_to_disk(self):
        exp1 = FakeSynonymExpander(cache_path=self.cache, max_per_minute=0)
        exp1.warm(["абайла"])
        exp2 = FakeSynonymExpander(cache_path=self.cache, max_per_minute=0)
        # кэш загружен с диска → ноль запросов
        self.assertEqual(exp2.calls, [])
        self.assertEqual(exp2._cache["абайла"], ["аңда", "байқа"])

    def test_batches_by_50(self):
        exp = FakeSynonymExpander(cache_path=self.cache, max_per_minute=0)
        words = [f"сөз{i}" for i in range(120)]
        exp.warm(words)
        self.assertEqual([len(c) for c in exp.calls], [50, 50, 20])

    def test_dedup_within_warm(self):
        exp = FakeSynonymExpander(cache_path=self.cache, max_per_minute=0)
        exp.warm(["абайла", "абайла", "абайла"])
        self.assertEqual(len(exp.calls), 1)
        self.assertEqual(exp.calls[0], ["абайла"])

    def test_expand_adds_synonyms(self):
        exp = FakeSynonymExpander(cache_path=self.cache, max_per_minute=0)
        exp.warm(["абайла", "теңіз"])
        result = exp.expand(["абайла", "теңіз"])
        self.assertIn("аңда", result)
        self.assertIn("байқа", result)
        self.assertIn("мұхит", result)
        # оригинальные стемы сохраняются
        self.assertIn("абайла", result)
        self.assertIn("теңіз", result)

    def test_expand_no_duplicates(self):
        exp = FakeSynonymExpander(cache_path=self.cache, max_per_minute=0)
        exp.warm(["абайла"])
        result = exp.expand(["абайла"])
        self.assertEqual(len(result), len(set(result)), "дубликаты в expand()")

    def test_expand_original_first(self):
        exp = FakeSynonymExpander(cache_path=self.cache, max_per_minute=0)
        exp.warm(["абайла"])
        result = exp.expand(["абайла"])
        # оригинальный стем должен быть первым
        self.assertEqual(result[0], "абайла")

    def test_expand_not_found_returns_original_only(self):
        exp = FakeSynonymExpander(cache_path=self.cache, max_per_minute=0)
        exp.warm(["үй"])  # found=false, пустой список
        result = exp.expand(["үй"])
        self.assertEqual(result, ["үй"])

    def test_expand_uncached_term_returns_original(self):
        exp = FakeSynonymExpander(cache_path=self.cache, max_per_minute=0)
        # не прогревали кэш → expand просто вернёт исходные стемы
        result = exp.expand(["неизвестный"])
        self.assertEqual(result, ["неизвестный"])

    def test_parse_response_correct_format(self):
        raw = json.dumps([
            {"word": "абайла", "stem": "абайла", "synonyms": ["аңда", "байқа"], "found": True},
            {"word": "теңіз", "stem": "теңіз", "synonyms": ["мұхит"], "found": True},
        ])
        result = SynonymExpander._parse_response(raw, ["абайла", "теңіз"])
        self.assertEqual(result, [["аңда", "байқа"], ["мұхит"]])

    def test_parse_response_not_found(self):
        raw = json.dumps([
            {"word": "үй", "stem": "үй", "synonyms": [], "found": False},
        ])
        result = SynonymExpander._parse_response(raw, ["үй"])
        self.assertEqual(result, [[]])

    def test_parse_response_missing_synonyms_key(self):
        raw = json.dumps([{"word": "нечто", "found": False}])
        result = SynonymExpander._parse_response(raw, ["нечто"])
        self.assertEqual(result, [[]])

    def test_parse_response_wrong_length_raises(self):
        raw = json.dumps([{"word": "a", "synonyms": []}])
        with self.assertRaises(ValueError):
            SynonymExpander._parse_response(raw, ["a", "b"])

    def test_invalid_cache_file_is_ignored(self):
        with open(self.cache, "w") as f:
            f.write("не-JSON")
        # не должно падать — просто начинается с пустым кэшем
        exp = FakeSynonymExpander(cache_path=self.cache, max_per_minute=0)
        self.assertEqual(exp._cache, {})


class TestBM25SearchTerms(unittest.TestCase):
    """Проверяем, что search_terms() даёт те же результаты, что search()."""

    def test_search_terms_matches_search(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from src.retrieval.bm25 import BM25Index, default_analyzer
        from src.preprocess.stemmer import IdentityStemmer

        stemmer = IdentityStemmer()
        index = BM25Index(analyzer=default_analyzer(stemmer))
        index.index([
            ("d1", "кітап оқу маңызды"),
            ("d2", "мектеп кітапхана"),
            ("d3", "балалар оқиды"),
        ])

        query = "кітап оқу"
        result_search = [doc_id for doc_id, _ in index.search(query, top_k=10)]
        q_terms = index.analyzer(query)
        result_terms = [doc_id for doc_id, _ in index.search_terms(q_terms, top_k=10)]
        self.assertEqual(result_search, result_terms)

    def test_run_terms_matches_run(self):
        from src.retrieval.bm25 import BM25Index, default_analyzer
        from src.preprocess.stemmer import IdentityStemmer

        stemmer = IdentityStemmer()
        index = BM25Index(analyzer=default_analyzer(stemmer))
        index.index([("d1", "үй мектеп"), ("d2", "мектеп кітап"), ("d3", "үй кітап")])

        queries = {"q1": "үй кітап", "q2": "мектеп"}
        run_str = index.run(queries, top_k=3)
        queries_terms = {qid: index.analyzer(text) for qid, text in queries.items()}
        run_terms = index.run_terms(queries_terms, top_k=3)
        self.assertEqual(run_str, run_terms)

    def test_synonym_expansion_improves_recall(self):
        """Документ, содержащий синоним (мұхит), должен ранжироваться выше без синонима."""
        from src.retrieval.bm25 import BM25Index, default_analyzer
        from src.preprocess.stemmer import IdentityStemmer

        stemmer = IdentityStemmer()
        index = BM25Index(analyzer=default_analyzer(stemmer))
        index.index([
            ("d_rel", "мұхит жағалауы"),   # содержит синоним
            ("d_irr", "далалық жер"),
        ])

        # без расширения — "теңіз" не найдёт "мұхит"
        without = [doc_id for doc_id, _ in index.search_terms(["теңіз"], top_k=5)]
        self.assertNotIn("d_rel", without)

        # с расширением ["теңіз", "мұхит"] — d_rel должен найтись
        with_syn = [doc_id for doc_id, _ in index.search_terms(["теңіз", "мұхит"], top_k=5)]
        self.assertIn("d_rel", with_syn)


if __name__ == "__main__":
    unittest.main()
