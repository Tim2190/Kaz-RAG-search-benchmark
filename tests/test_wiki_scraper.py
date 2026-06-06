"""Тесты разбора ответа MediaWiki API (без сети)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scraping.wiki_scraper import parse_extracts  # noqa: E402


class TestParseExtracts(unittest.TestCase):
    def test_parses_pages(self):
        data = {
            "query": {
                "pages": {
                    "123": {"pageid": 123, "title": "Алматы",
                            "extract": "Алматы — Қазақстандағы ірі қала."},
                    "456": {"pageid": 456, "title": "Астана",
                            "extract": "Астана — Қазақстан Республикасының астанасы."},
                }
            }
        }
        out = dict((pid, (title, text)) for pid, title, text in parse_extracts(data))
        self.assertEqual(out[123][0], "Алматы")
        self.assertIn("қала", out[123][1])
        self.assertEqual(len(out), 2)

    def test_skips_empty_extracts(self):
        data = {"query": {"pages": {
            "1": {"pageid": 1, "title": "Стаб", "extract": ""},
            "2": {"pageid": 2, "title": "Бар", "extract": "Мәтіні бар мақала."},
        }}}
        out = parse_extracts(data)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], 2)

    def test_empty_response(self):
        self.assertEqual(parse_extracts({}), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
