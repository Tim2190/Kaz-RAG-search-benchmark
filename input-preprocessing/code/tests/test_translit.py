"""
Юнит-тесты транслитерации кириллица→латиница (стандарт 2021). Stdlib, без сети.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # code/

from translit import transliterate  # noqa: E402


class TestTransliterate(unittest.TestCase):
    def test_standard_anchor_letters(self):
        """Ключевые соответствия стандарта 2021, зафиксированные планом."""
        anchors = {
            "ә": "ä", "ғ": "ğ", "қ": "q", "ң": "ñ", "ө": "ö",
            "ұ": "ū", "ү": "ü", "һ": "h", "і": "i", "ш": "ş", "ж": "j",
        }
        for cyr, lat in anchors.items():
            self.assertEqual(transliterate(cyr), lat, f"{cyr}→{lat}")

    def test_word(self):
        self.assertEqual(transliterate("Қазақстан"), "Qazaqstan")
        self.assertEqual(transliterate("Жамбыл"), "Jambyl")

    def test_uppercase_maps_to_uppercase(self):
        self.assertEqual(transliterate("Әл"), "Äl")
        self.assertEqual(transliterate("Ө"), "Ö")

    def test_digraph_capitalization(self):
        """Заглавная Ц/Ч → диграф с большой первой буквой, не CAPS."""
        self.assertEqual(transliterate("Цирк"), "Tsirk")
        self.assertEqual(transliterate("Чай"), "Chay")

    def test_passthrough_non_cyrillic(self):
        """Латиница, цифры, пунктуация и пробелы не меняются."""
        self.assertEqual(transliterate("ВВП 2021 — рост!"), "VVP 2021 — rost!")

    def test_soft_hard_signs_dropped(self):
        self.assertEqual(transliterate("объект"), "obekt")

    def test_deterministic_and_idempotent_on_latin(self):
        s = "Норвегия қай түбекте орналасқан?"
        once = transliterate(s)
        self.assertEqual(once, transliterate(s))          # детерминизм
        self.assertEqual(transliterate(once), once)       # латиница уже не меняется

    def test_empty(self):
        self.assertEqual(transliterate(""), "")


if __name__ == "__main__":
    unittest.main()
