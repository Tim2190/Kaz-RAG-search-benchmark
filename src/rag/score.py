"""
Классификация ответа LLM: correct / hallucination / abstain.

Логика (фактологические вопросы с коротким эталонным ответом):
  - correct       — эталонный ответ присутствует в тексте ответа;
  - abstain       — модель честно отказалась («Ақпарат жоқ» и т.п.);
  - hallucination — модель уверенно ответила, но эталона нет (выдумала).

Галлюцинация — самый вредный исход: модель уверенно говорит неправду.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

# Фразы отказа (казахский / русский / английский)
_ABSTAIN = [
    "ақпарат жоқ", "мәлімет жоқ", "ақпарат берілмеген", "контексте жоқ",
    "контекстте жоқ", "білмеймін", "белгісіз", "жауап табылмады",
    "анықтау мүмкін емес", "жауап жоқ",
    "не знаю", "нет информации", "не указано", "недостаточно информации",
    "i don't know", "no information", "not enough", "cannot determine",
]


def _norm(s: str) -> str:
    return " ".join((s or "").casefold().split())


def is_abstain(answer: str) -> bool:
    a = _norm(answer)
    return any(p in a for p in _ABSTAIN)


def is_correct(answer: str, gold: Union[str, Sequence[str]]) -> bool:
    """Эталон (или любой из списка эталонов) встречается в ответе."""
    a = _norm(answer)
    golds = [gold] if isinstance(gold, str) else list(gold)
    return any(_norm(g) and _norm(g) in a for g in golds)


def classify(answer: str, gold: Union[str, Sequence[str]]) -> str:
    if is_correct(answer, gold):
        return "correct"
    if is_abstain(answer):
        return "abstain"
    return "hallucination"


def aggregate(labels: List[str]) -> Dict[str, float]:
    """Доли correct / hallucination / abstain + accuracy."""
    n = len(labels) or 1
    counts = {k: labels.count(k) for k in ("correct", "hallucination", "abstain")}
    return {
        "n": len(labels),
        "accuracy": counts["correct"] / n,
        "hallucination_rate": counts["hallucination"] / n,
        "abstention_rate": counts["abstain"] / n,
        "counts": counts,
    }
