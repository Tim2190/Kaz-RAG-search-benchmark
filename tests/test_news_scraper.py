"""Тесты чистых функций универсального скрапера (без сети)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scraping.news_scraper import looks_kazakh, site_slug  # noqa: E402


class TestLooksKazakh(unittest.TestCase):
    def test_accepts_kazakh_text(self):
        kz = ("Қазақстан Республикасының Үкіметі ауыл шаруашылығын дамыту "
              "жөніндегі жаңа бағдарламаны бекітті. Бұл шешім өңірлерге әсер етеді.")
        self.assertTrue(looks_kazakh(kz))

    def test_rejects_russian_text(self):
        ru = ("Правительство Республики Казахстан утвердило новую программу "
              "развития сельского хозяйства в регионах страны сегодня утром.")
        self.assertFalse(looks_kazakh(ru))

    def test_rejects_english_and_empty(self):
        self.assertFalse(looks_kazakh("The government approved a new program today."))
        self.assertFalse(looks_kazakh(""))


class TestSiteSlug(unittest.TestCase):
    def test_extracts_domain_core(self):
        self.assertEqual(site_slug("https://baq.kz"), "baq")
        self.assertEqual(site_slug("https://kaz.inform.kz/lenta/"), "inform")
        self.assertEqual(site_slug("https://www.gov.kz/x"), "gov")


if __name__ == "__main__":
    unittest.main(verbosity=2)
