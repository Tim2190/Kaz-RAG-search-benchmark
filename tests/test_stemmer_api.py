"""Тесты клиента стеммера: батчинг, кэш, парсер ответа — через мок _post()."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess.stemmer_api import KazakhStemmerAPI  # noqa: E402


class FakeRemoteStemmer(KazakhStemmerAPI):
    """Подменяет сеть: возвращает 'основу' = слово без последних 2 символов."""

    def __init__(self, response_shape="results_dicts", **kw):
        self.calls = []  # список батчей, реально ушедших «в сеть»
        self.response_shape = response_shape
        super().__init__(**kw)

    def _post(self, payload: bytes) -> str:
        words = json.loads(payload.decode())["words"]
        self.calls.append(list(words))
        stems = [w[:-2] if len(w) > 2 else w for w in words]
        if self.response_shape == "results_dicts":
            return json.dumps({"results": [{"word": w, "stem": s} for w, s in zip(words, stems)]})
        if self.response_shape == "stems_list":
            return json.dumps({"stems": stems})
        if self.response_shape == "bare_list":
            return json.dumps(stems)
        if self.response_shape == "real":
            # подтверждённый формат сервиса: список объектов с полем stem
            return json.dumps([
                {"word": w, "stem": s, "suffixes": [], "found_in_dictionary": True,
                 "confidence": "dictionary"}
                for w, s in zip(words, stems)
            ])
        raise AssertionError("unknown shape")


def _tmp_cache():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)
    return path


class TestStemmerApiClient(unittest.TestCase):
    def setUp(self):
        self.cache = _tmp_cache()

    def tearDown(self):
        for p in (self.cache, self.cache + ".tmp"):
            if os.path.exists(p):
                os.remove(p)

    def test_batches_by_50(self):
        st = FakeRemoteStemmer(cache_path=self.cache, max_per_minute=0)
        words = [f"сөз{i}аб" for i in range(120)]
        st.warm(words)
        # 120 уникальных -> 3 запроса (50+50+20)
        self.assertEqual([len(c) for c in st.calls], [50, 50, 20])

    def test_cache_prevents_repeat_requests(self):
        st = FakeRemoteStemmer(cache_path=self.cache, max_per_minute=0)
        st.warm(["кітабым", "балам"])
        self.assertEqual(len(st.calls), 1)
        st.calls.clear()
        # повторный прогрев тех же слов не должен бить в сеть
        st.warm(["кітабым", "балам"])
        self.assertEqual(st.calls, [])
        self.assertEqual(st.stem("кітабым"), "кітаб")

    def test_cache_persists_to_disk(self):
        st1 = FakeRemoteStemmer(cache_path=self.cache, max_per_minute=0)
        st1.warm(["қаладан"])
        # новый клиент читает кэш с диска -> ноль запросов
        st2 = FakeRemoteStemmer(cache_path=self.cache, max_per_minute=0)
        self.assertEqual(st2.stem("қаладан"), "қалад")
        self.assertEqual(st2.calls, [])

    def test_dedup_within_warm(self):
        st = FakeRemoteStemmer(cache_path=self.cache, max_per_minute=0)
        st.warm(["сөзаб", "сөзаб", "сөзаб"])
        self.assertEqual(st.calls, [["сөзаб"]])

    def test_parses_real_service_response(self):
        # дословный ответ сервиса для "балаларымызда" -> "бала"
        raw = json.dumps([
            {"word": "балаларымызда", "stem": "бала",
             "suffixes": ["да", "ымыз", "лар"],
             "found_in_dictionary": True, "confidence": "dictionary"}
        ])
        out = KazakhStemmerAPI._parse_response(raw, ["балаларымызда"])
        self.assertEqual(out, ["бала"])

    def test_parser_handles_multiple_shapes(self):
        for shape in ("results_dicts", "stems_list", "bare_list", "real"):
            cache = _tmp_cache()
            st = FakeRemoteStemmer(response_shape=shape, cache_path=cache, max_per_minute=0)
            st.warm(["балалар"])
            self.assertEqual(st.stem("балалар"), "балал")
            os.remove(cache)

    def test_empty_stem_falls_back_to_word(self):
        st = FakeRemoteStemmer(cache_path=self.cache, max_per_minute=0)
        # слово из 2 символов: фейк вернёт его же (len<=2)
        st.warm(["ба"])
        self.assertEqual(st.stem("ба"), "ба")


if __name__ == "__main__":
    unittest.main(verbosity=2)
