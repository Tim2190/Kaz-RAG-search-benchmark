"""Тесты RAG-скоринга и харнесса (через мок-LLM, без моделей/сети)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag import score  # noqa: E402
from src.rag import prompt as rag_prompt  # noqa: E402
from src.eval import run_rag  # noqa: E402


class TestScore(unittest.TestCase):
    def test_correct(self):
        self.assertEqual(score.classify("Жауап: Астана", "Астана"), "correct")

    def test_abstain(self):
        self.assertEqual(score.classify("Ақпарат жоқ", "Астана"), "abstain")

    def test_hallucination(self):
        self.assertEqual(score.classify("Бұл — Париж қаласы", "Астана"), "hallucination")

    def test_correct_accepts_list(self):
        self.assertEqual(score.classify("жауабы — Иран", ["Түркия", "Иран"]), "correct")

    def test_aggregate_rates(self):
        agg = score.aggregate(["correct", "hallucination", "abstain", "correct"])
        self.assertAlmostEqual(agg["accuracy"], 0.5)
        self.assertAlmostEqual(agg["hallucination_rate"], 0.25)
        self.assertAlmostEqual(agg["abstention_rate"], 0.25)


class ContextReaderLLM:
    """Мок-LLM: отвечает целевым словом, если оно есть в контексте; иначе врёт."""
    def __init__(self, target: str):
        self.target = target

    def generate(self, prompt: str) -> str:
        # последний блок КОНТЕКСТ (в промпте есть ещё пример one-shot)
        ctx = prompt.split("КОНТЕКСТ:")[-1].split("СҰРАҚ:")[0]
        if self.target in ctx:
            return f"Жауап: {self.target}"
        return "Бұл — Париж"  # уверенный неверный ответ -> галлюцинация


class TestRunRag(unittest.TestCase):
    def setUp(self):
        # документ с ответом написан в одной форме, запрос — в другой (морфология)
        self.corpus = [
            ("d1", "Астана Қазақстанның астанасы болып табылады"),
            ("d2", "Алматы ең үлкен қала"),
            ("d3", "Норвегия Скандинавия түбегіндегі ел"),
        ]
        self.queries = [
            {"query_id": "q1", "text": "Қазақстанның астанасы қай қала",
             "category": "natural", "gold_doc_ids": ["d1"], "answer": "Астана"},
        ]

    def test_hit_yields_correct(self):
        llm = ContextReaderLLM("Астана")
        res = run_rag.run_system(self.corpus, self.queries, "identity", llm, top_k=3)
        self.assertEqual(res["overall"]["accuracy"], 1.0)
        self.assertEqual(res["overall"]["retrieval_hit_rate"], 1.0)

    def test_miss_yields_hallucination(self):
        # запрос, который BM25 не найдёт (нет общих слов с корпусом) -> промах
        q = [{"query_id": "q9", "text": "Жапонияның императоры кім",
              "category": "natural", "gold_doc_ids": ["dX"], "answer": "Нарухито"}]
        llm = ContextReaderLLM("Нарухито")
        res = run_rag.run_system(self.corpus, q, "identity", llm, top_k=3)
        self.assertEqual(res["overall"]["hallucination_rate"], 1.0)
        self.assertEqual(res["overall"]["retrieval_hit_rate"], 0.0)

    def test_prompt_contains_instruction_and_question(self):
        p = rag_prompt.build_prompt("Сұрақ?", ["мәтін бір", "мәтін екі"])
        self.assertIn("Ақпарат жоқ", p)
        self.assertIn("Сұрақ?", p)
        self.assertIn("[1]", p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
