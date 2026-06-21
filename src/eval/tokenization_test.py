"""
Токенизационный тест: насколько сильно каждый токенизатор дробит казахские
словоформы на sub-word-токены. Меньше — целее морфология.

Что считаем и почему так:

  * Слова берём из корпуса бенчмарка (частые длинные словоформы, len≥9) —
    не синтетика. Частотный отбор СКОРЕЕ ЗАНИЖАЕТ разрыв (частые слова
    токенизатор знает лучше), то есть это безопасная, консервативная сторона.
    Список слов кэшируется в data/words_tokenization_100.json, чтобы все
    5 токенизаторов прогонялись на ОДНОМ и том же наборе слов.

  * Считаем ДВА варианта на каждое слово:
      (1) слово в изоляции:         "сөзжасам"
      (2) слово с ведущим пробелом: " сөзжасам"
    Это важно, потому что R2 (ModernBERT) использует byte-level BPE, где
    ведущий пробел — часть токена, а R1/e5 (SentencePiece) к нему почти
    нечувствительны. Показываем ОБА числа, чтобы сравнение нельзя было
    оспорить как «несправедливое из-за отсутствия пробела».

  * Печатаем примеры разбиения (слово → куски у каждого токенизатора), чтобы
    результат можно было проверить глазами за секунды.

Интерпретация (осторожно): более сильная фрагментация казахского — ОДИН ИЗ
ФАКТОРОВ общей просадки R2 на казахском, не единственная причина и не
объяснение конкретно морфологической категории. По данным бенчмарка R2-311m
на `inflected` не просел (≈ R1), просел в агрегате и на семантике; везде просел
только маленький R2-97m. Это дескриптивный замер, не контролируемая абляция.

ЗАПУСК (нужен transformers + доступ к HF; модели не качаются, только токенизаторы):
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
    "granite-97m-r2":    "ibm-granite/granite-embedding-97m-multilingual-r2",
    "granite-311m-r2":   "ibm-granite/granite-embedding-311m-multilingual-r2",
    "e5-base":           "intfloat/multilingual-e5-base",
    "granite-278m-r1":   "ibm-granite/granite-embedding-278m-multilingual",
    "jina-v3":           "jinaai/jina-embeddings-v3",
    "tilcore_morph256k": "stukenov/sozkz-morphbpe-256k-kk-v1",
}

# несколько репрезентативных слов для наглядных примеров (морфологически богатые)
EXAMPLE_WORDS = ["сөзжасам", "мүмкіндіктерін", "ұйымдастырушылық", "халықаралық"]

# Кириллица казахского (вкл. спецбуквы ә і ң ғ ү ұ қ ө һ)
_WORD = re.compile(r"[А-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]+")

# Кэш списка слов — гарантирует одинаковые 100 слов при любом числе прогонов
_WORDS_CACHE = os.path.join(ROOT, "data", "words_tokenization_100.json")


def _harvest_words(n: int = 100, min_len: int = 9) -> List[str]:
    """Собрать длинные (морфологически богатые) казахские словоформы из корпуса.
    При первом вызове сохраняет список в data/words_tokenization_100.json."""
    if os.path.exists(_WORDS_CACHE):
        with open(_WORDS_CACHE, encoding="utf-8") as f:
            return json.load(f)
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
    words = [w for w, _ in counter.most_common(n)]
    os.makedirs(os.path.dirname(_WORDS_CACHE), exist_ok=True)
    with open(_WORDS_CACHE, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)
    return words


def _avg_subwords(tok, words: List[str], lead_space: bool) -> float:
    total = 0
    for w in words:
        s = (" " + w) if lead_space else w
        total += len(tok.encode(s, add_special_tokens=False))
    return total / max(len(words), 1)


def main() -> None:
    from transformers import AutoTokenizer

    words = _harvest_words(n=100, min_len=9)
    print(f"Морфологически богатых слов (len≥9): {len(words)}")
    print("Примеры:", ", ".join(words[:8]), "…\n")

    loaded = {}
    for alias, hf_id in TOKENIZERS.items():
        try:
            loaded[alias] = AutoTokenizer.from_pretrained(hf_id)
        except Exception:
            try:
                loaded[alias] = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            except Exception as e:
                print(f"[skip] {alias}: {e}")

    # --- словари ---
    print(f"{'токенизатор':22s} {'vocab_size':>12s}")
    print("-" * 36)
    for alias, tok in loaded.items():
        print(f"{alias:22s} {tok.vocab_size:12d}")

    # --- средние: два варианта (без пробела / с ведущим пробелом) ---
    print(f"\n{'токенизатор':22s} {'без пробела':>12s} {'с пробелом':>12s}")
    print("-" * 48)
    results = {}
    for alias, tok in loaded.items():
        a_bare = _avg_subwords(tok, words, lead_space=False)
        a_lead = _avg_subwords(tok, words, lead_space=True)
        results[alias] = (a_bare, a_lead)
        print(f"{alias:22s} {a_bare:12.2f} {a_lead:12.2f}")

    # --- наглядные примеры разбиения ---
    print("\n--- Примеры разбиения (слово → куски; ▁/Ġ = ведущий пробел) ---")
    for w in EXAMPLE_WORDS:
        print(f"\n«{w}»")
        for alias, tok in loaded.items():
            bare = tok.tokenize(w)
            lead = tok.tokenize(" " + w)
            print(f"  {alias:18s} {len(bare)}: {bare}")
            print(f"  {'':18s} +пробел {len(lead)}: {lead}")

    if results:
        print("\n--- Вывод (осторожно) ---")
        worst = max(results, key=lambda k: results[k][0])
        best = min(results, key=lambda k: results[k][0])
        print(f"Сильнее всех дробит казахский: {worst} "
              f"(без пробела {results[worst][0]:.2f}, с пробелом {results[worst][1]:.2f})")
        print(f"Бережнее всех:                {best} "
              f"(без пробела {results[best][0]:.2f}, с пробелом {results[best][1]:.2f})")
        print("Более сильная фрагментация — ОДИН ИЗ факторов общей просадки R2 на "
              "казахском (см. докстринг), а не единственная причина и не объяснение "
              "конкретно категории inflected.")


if __name__ == "__main__":
    main()
