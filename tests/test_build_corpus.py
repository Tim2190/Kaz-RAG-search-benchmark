"""Тесты чистки и нарезки корпуса на пассажи."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.corpus import build_corpus as bc  # noqa: E402
from src.preprocess.tokenize import tokenize  # noqa: E402


class TestCleanText(unittest.TestCase):
    def test_drops_short_and_boilerplate_lines(self):
        raw = "Қысқа\nҚазақстан Республикасының Үкіметі маңызды шешім қабылдады бүгін\nПодписывайтесь на Telegram"
        out = bc.clean_text(raw)
        self.assertIn("Үкіметі", out)
        self.assertNotIn("Telegram", out)
        self.assertNotIn("Қысқа", out)  # короче 15 символов

    def test_collapses_whitespace(self):
        self.assertEqual(
            bc.clean_text("Астана    қаласында   үлкен жиналыс өтті бүгін ертең"),
            "Астана қаласында үлкен жиналыс өтті бүгін ертең",
        )


class TestKazakhPassageFilter(unittest.TestCase):
    def test_accepts_kazakh(self):
        self.assertTrue(bc.is_kazakh_passage(
            "Қазақстан Республикасының астанасы Астана қаласы болып табылады әрі дамып келеді"))

    def test_rejects_russian_bibliography(self):
        self.assertFalse(bc.is_kazakh_passage(
            "История Казахстана средних веков. — Алматы: Атамура, 2003. — 279 с."))

    def test_rejects_english(self):
        self.assertFalse(bc.is_kazakh_passage(
            "Texas Politics. An online textbook from the College of Liberal Arts."))


class TestChunking(unittest.TestCase):
    def test_chunks_respect_target_size(self):
        sentences = [f"Бұл {i} нөмірлі сөйлем мысалы болып табылады." for i in range(40)]
        text = " ".join(sentences)
        chunks = bc.chunk_text(text, target_words=30, overlap_words=6)
        self.assertGreater(len(chunks), 1)
        # каждый чанк не сильно превышает целевой размер
        for c in chunks:
            self.assertLessEqual(len(tokenize(c)), 30 + 12)

    def test_overlap_between_consecutive_chunks(self):
        sentences = [f"Сөйлем нөмір {i} осында тұр." for i in range(20)]
        chunks = bc.chunk_text(" ".join(sentences), target_words=20, overlap_words=6)
        # хвост предыдущего чанка должен встречаться в начале следующего
        self.assertGreaterEqual(len(chunks), 2)
        prev_tail = chunks[0].split()[-1]
        self.assertIn(prev_tail, chunks[1])

    def test_short_text_single_chunk(self):
        chunks = bc.chunk_text("Бір ғана қысқа сөйлем осында.", target_words=120)
        self.assertEqual(len(chunks), 1)


class TestIterPassages(unittest.TestCase):
    def test_builds_passages_with_metadata(self):
        raw = {
            "id": "moa_42",
            "title": "Тақырып",
            "source_name": "МинСельХоз",
            "url": "https://www.gov.kz/x",
            "text": " ".join(
                f"Қазақстан Республикасының ауыл шаруашылығы саласында {i} маңызды оқиға болды."
                for i in range(30)
            ),
        }
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(raw, ensure_ascii=False) + "\n")
        try:
            passages = list(bc.iter_passages(path, target_words=40, overlap_words=8))
        finally:
            os.remove(path)

        self.assertGreater(len(passages), 1)
        first = passages[0]
        self.assertEqual(first["source_doc_id"], "moa_42")
        self.assertTrue(first["doc_id"].startswith("moa_42_p"))
        self.assertEqual(first["source_name"], "МинСельХоз")
        self.assertEqual(first["lang"], "kz")


class TestBuildCorpusDedup(unittest.TestCase):
    def test_duplicate_passages_dropped(self):
        # два документа с ОДИНАКОВЫМ боилерплейтом -> в корпусе один пассаж
        boiler = (
            "Біздің сайтқа жазылыңыз және барлық соңғы жаңалықтарды үнемі қадағалап "
            "отырыңыз әлеуметтік желілерде бізбен бірге болыңыз әрдайым осында сіз үшін "
            "маңызды оқиғалар туралы бірінші болып біліп отырыңыз достарыңызбен бөлісіңіз."
        )
        docs = [
            {"id": "a", "title": "A", "source_name": "s", "url": "u1", "text": boiler},
            {"id": "b", "title": "B", "source_name": "s", "url": "u2", "text": boiler},
        ]
        fd_in, raw = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd_in, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        fd_out, out = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd_out)
        try:
            n = bc.build_corpus(raw, out, target_words=120, overlap_words=20)
            self.assertEqual(n, 1)  # дубликат отброшен
        finally:
            os.remove(raw)
            os.remove(out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
