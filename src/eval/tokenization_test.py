"""
Спринт 2 — токенизационный тест (опциональный бонус, без GPU).

Гипотеза: если Granite-97m-R2 рвёт морфологически богатые казахские слова на
большее число sub-word токенов, чем e5/311m — это готовое объяснение возможной
просадки на категории `inflected`.

Прогоняет N морфологически богатых казахских слов через токенизаторы и считает
среднее число sub-word на слово. Меньше — лучше (целее морфология).

Слова берутся из реальных запросов/корпуса бенчмарка (длинные словоформы),
поэтому тест не синтетический.

ЗАПУСК (Colab; нужен transformers, доступ к HF):
    python -m src.eval.tokenization_test
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

TOKENIZERS = {
    "granite-97m-r2":  "ibm-granite/granite-embedding-97m-multilingual-r2",
    "granite-311m-r2": "ibm-granite/granite-embedding-311m-multilingual-r2",
    "e5-base":         "intfloat/multilingual-e5-base",
    "granite-278m-r1": "ibm-granite/granite-embedding-278m-multilingual",
}

# Кириллица казахского (вкл. спецбуквы ә і ң ғ ү ұ қ ө һ)
_WORD = re.compile(r"[А-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]+")


def _harvest_words(n: int = 100, min_len: int = 9) -> List[str]:
    """Собрать длинные (морфологически богатые) казахские словоформы из корпуса."""
    corpus = os.path.join(ROOT, "data/corpus/corpus.jsonl")
    counter: Counter = Counter()
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line).get("text", "")
            except Exception:
                continue
            for w in _WORD.findall(text.lower()):
                if len(w) >= min_len:
                    counter[w] += 1
    # частые длинные слова = реально употребимые словоформы, не опечатки
    return [w for w, _ in counter.most_common(n)]


def main() -> None:
    from transformers import AutoTokenizer

    words = _harvest_words(n=100, min_len=9)
    print(f"Морфологически богатых слов (len≥9): {len(words)}")
    print("Примеры:", ", ".join(words[:8]), "…\n")

    results = {}
    for alias, hf_id in TOKENIZERS.items():
        try:
            tok = AutoTokenizer.from_pretrained(hf_id)
        except Exception as e:
            print(f"[skip] {alias}: {e}")
            continue
        total = 0
        for w in words:
            ids = tok.encode(w, add_special_tokens=False)
            total += len(ids)
        avg = total / max(len(words), 1)
        results[alias] = avg
        print(f"{alias:18s} среднее sub-word/слово: {avg:.2f}")

    if results:
        print("\n--- Вывод ---")
        worst = max(results, key=results.get)
        best = min(results, key=results.get)
        print(f"Сильнее всех дробит казахский: {worst} ({results[worst]:.2f})")
        print(f"Бережнее всех:                {best} ({results[best]:.2f})")
        print("Чем больше sub-word на слово, тем сильнее модель рвёт морфологию — "
              "что коррелирует с просадкой на inflected.")


if __name__ == "__main__":
    main()
