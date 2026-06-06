"""Сборка RAG-промпта: контекст из найденных пассажей + вопрос + инструкция."""

from __future__ import annotations

from typing import List

INSTRUCTION = (
    "Сен — фактологиялық сұрақ-жауап жүйесісің. Тек төмендегі КОНТЕКСТ негізінде "
    "сұраққа ӨТЕ ҚЫСҚА жауап бер (бір сөз немесе бір сөз тіркесі). "
    "Сұрақты ҚАЙТАЛАМА. Егер жауап КОНТЕКСТте болмаса, тек «Ақпарат жоқ» деп жаз. "
    "Ойдан шығарма."
)

# Пример (one-shot) — фиксирует формат «короткий ответ, без повтора вопроса».
_EXAMPLE = (
    "КОНТЕКСТ:\n"
    "[1] Париж — Францияның астанасы, Сена өзенінің жағасында орналасқан ірі қала.\n"
    "СҰРАҚ: Францияның астанасы қай қала?\n"
    "ЖАУАП: Париж"
)


def build_context(passages: List[str], max_passages: int = 5) -> str:
    """Нумерованный блок контекста из текстов пассажей."""
    chosen = passages[:max_passages]
    return "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(chosen))


def build_prompt(question: str, passages: List[str], max_passages: int = 5) -> str:
    context = build_context(passages, max_passages)
    return (f"{INSTRUCTION}\n\n"
            f"{_EXAMPLE}\n\n"
            f"КОНТЕКСТ:\n{context}\n\n"
            f"СҰРАҚ: {question}\n"
            f"ЖАУАП:")
