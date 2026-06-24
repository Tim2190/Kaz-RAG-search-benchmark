"""
Юнит-тесты слоя input-preprocessing: 4 линии, симметрия, inline-стемминг.
Stdlib, без сети/GPU — стеммер замокан фиктивным словарём.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess.input_preprocess import (  # noqa: E402
    LINES, build_transform, collect_tokens, needs_stemmer, stem_inline,
)
from src.preprocess.translit import transliterate  # noqa: E402


class FakeStemmer:
    """Обрезает окончание -лар/-тар и -да; иначе возвращает слово."""
    name = "fake"
    DICT = {"балалар": "бала", "қалада": "қала", "үйлер": "үй"}

    def stem(self, token: str) -> str:
        return self.DICT.get(token, token)


class TestInputPreprocess(unittest.TestCase):
    def test_baseline_is_identity(self):
        f = build_transform("baseline")
        self.assertEqual(f("Қазақ тілі"), "Қазақ тілі")

    def test_latin_line_equals_translit(self):
        f = build_transform("latin")
        s = "Балалар қалада"
        self.assertEqual(f(s), transliterate(s))

    def test_stem_inline_preserves_punctuation(self):
        st = FakeStemmer()
        # пунктуация и пробелы сохраняются, слова → основы (lowercase)
        self.assertEqual(stem_inline("Балалар, үйлер!", st), "бала, үй!")

    def test_stem_line(self):
        f = build_transform("stem", FakeStemmer())
        self.assertEqual(f("балалар қалада"), "бала қала")

    def test_stem_latin_is_stem_then_translit(self):
        st = FakeStemmer()
        f = build_transform("stem_latin", st)
        # сначала стем (балалар→бала, қалада→қала), потом латиница
        self.assertEqual(f("балалар қалада"), transliterate("бала қала"))
        self.assertEqual(f("балалар қалада"), "bala qala")

    def test_symmetry_same_function_both_sides(self):
        """Одна и та же transform применяется к запросу и корпусу."""
        f = build_transform("latin")
        query = "қала"
        doc = "Бұл қала үлкен"
        # подстрока запроса в документе остаётся подстрокой после преобразования
        self.assertIn(f(query), f(doc))

    def test_needs_stemmer_flags(self):
        self.assertFalse(needs_stemmer("baseline"))
        self.assertFalse(needs_stemmer("latin"))
        self.assertTrue(needs_stemmer("stem"))
        self.assertTrue(needs_stemmer("stem_latin"))

    def test_stem_requires_stemmer(self):
        with self.assertRaises(ValueError):
            build_transform("stem")

    def test_unknown_line_raises(self):
        with self.assertRaises(ValueError):
            build_transform("nonsense")

    def test_collect_tokens_unique_lowercase(self):
        toks = collect_tokens(["Бала бала", "ҮЙ үй", "qala"])
        self.assertEqual(sorted(toks), sorted(["бала", "үй", "qala"]))

    def test_all_lines_constant(self):
        self.assertEqual(LINES, ("baseline", "stem", "latin", "stem_latin"))


if __name__ == "__main__":
    unittest.main()
