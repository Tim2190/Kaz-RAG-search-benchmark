"""
Сравнение sub-word fertility на ДВУХ доменах: Казахская Вики vs Akorda.

Вопрос: фрагментирует ли формальный политический казахский (Akorda) сильнее,
чем энциклопедический (Вики), — на ОДНИХ И ТЕХ ЖЕ токенизаторах? Если да, это
candidate mechanism (не доказательство) для общей просадки ВСЕХ dense-моделей на
Akorda: чем мельче дробятся словоформы, тем хуже эмбеддинг держит смысл.

ЧЕСТНАЯ ОГОВОРКА ПРО shyngys-e5:
  shyngys879/kazakh-e5-rag-embedding — это ДООБУЧЕНИЕ multilingual-e5; дообучение
  НЕ меняет токенизатор. Значит fertility(shyngys) ≡ fertility(e5) тождественно.
  Поэтому fertility НЕ может объяснить, почему shyngys просел СИЛЬНЕЕ базового e5
  (Finding 5) — у них один токенизатор. Fertility — это свойство КОРПУСА на данном
  токенизаторе, и объясняет (как кандидат) общий доменный сдвиг, а не дифференциал
  между моделями с общим токенизатором. Подаём именно так.

Метод (тот же, что src/eval/tokenization_test.py):
  * Берём 100 частых длинных (len≥9) словоформ из КАЖДОГО корпуса. Частотный
    отбор скорее ЗАНИЖАЕТ разрыв (частые слова токенизатор знает лучше).
  * Считаем средние sub-words на слово в двух режимах (без пробела / с ведущим).
  * Списки слов кэшируются (data/words_tokenization_100.json для Вики,
    data/words_akorda_100.json для Akorda) — воспроизводимо.

Слова харвестятся БЕЗ СЕТИ (корпуса локальны). Токенизаторы качаются с HF —
этот шаг требует интернета (Kaggle/Colab или хост huggingface.co в egress-allowlist).

ЗАПУСК:
    python -m src.eval.fertility_compare
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# те же 5 токенизаторов, что в tokenization_test.py
TOKENIZERS = {
    "granite-97m-r2":    "ibm-granite/granite-embedding-97m-multilingual-r2",
    "granite-311m-r2":   "ibm-granite/granite-embedding-311m-multilingual-r2",
    "e5-base":           "intfloat/multilingual-e5-base",
    "granite-278m-r1":   "ibm-granite/granite-embedding-278m-multilingual",
    "jina-v3":           "jinaai/jina-embeddings-v3",
    "tilcore_morph256k": "stukenov/sozkz-morphbpe-256k-kk-v1",
}

_WORD = re.compile(r"[А-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]+")

CORPORA = {
    # alias: (jsonl-путь, поле текста, кэш-файл словника)
    "wiki":   (os.path.join(ROOT, "data/corpus/corpus.jsonl"), "text",
               os.path.join(ROOT, "data/words_tokenization_100.json")),
    "akorda": (os.path.join(ROOT, "data/akorda/passages.jsonl"), "text",
               os.path.join(ROOT, "data/words_akorda_100.json")),
}


def harvest_words(corpus_path: str, field: str, cache_path: str,
                  n: int = 100, min_len: int = 9) -> List[str]:
    """Частые длинные словоформы из корпуса; кэшируется при первом вызове.
    Сети не требует."""
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    counter: Counter = Counter()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line).get(field, "")
            except Exception:
                continue
            for w in _WORD.findall(text.lower()):
                if len(w) >= min_len:
                    counter[w] += 1
    words = [w for w, _ in counter.most_common(n)]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
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

    word_sets = {alias: harvest_words(*cfg) for alias, cfg in CORPORA.items()}
    for alias, words in word_sets.items():
        print(f"{alias}: {len(words)} слов (len≥9). Примеры: {', '.join(words[:6])} …")
    print()

    loaded = {}
    for alias, hf_id in TOKENIZERS.items():
        try:
            loaded[alias] = AutoTokenizer.from_pretrained(hf_id)
        except Exception:
            try:
                loaded[alias] = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            except Exception as e:
                print(f"[skip] {alias}: {e}")

    # --- таблица: fertility Вики vs Akorda, оба режима, и дельта ---
    print(f"{'токенизатор':20s} "
          f"{'wiki(bare)':>11s} {'akorda(bare)':>13s} {'Δbare':>7s}   "
          f"{'wiki(+sp)':>10s} {'akorda(+sp)':>12s} {'Δ+sp':>6s}")
    print("-" * 92)
    rows = {}
    for alias, tok in loaded.items():
        wb = _avg_subwords(tok, word_sets["wiki"],   lead_space=False)
        ab = _avg_subwords(tok, word_sets["akorda"], lead_space=False)
        ws = _avg_subwords(tok, word_sets["wiki"],   lead_space=True)
        as_ = _avg_subwords(tok, word_sets["akorda"], lead_space=True)
        rows[alias] = (wb, ab, ws, as_)
        print(f"{alias:20s} {wb:11.2f} {ab:13.2f} {ab-wb:>+7.2f}   "
              f"{ws:10.2f} {as_:12.2f} {as_-ws:>+6.2f}")

    print("\nИнтерпретация (осторожно):")
    print("  Δ>0 => Akorda фрагментируется СИЛЬНЕЕ Вики на этом токенизаторе —")
    print("  candidate mechanism для общей просадки dense-моделей на Akorda.")
    print("  e5 и shyngys-e5 делят ОДИН токенизатор: их Δ тождественны, поэтому")
    print("  fertility НЕ объясняет дифференциал shyngys vs e5 (см. докстринг).")


if __name__ == "__main__":
    main()
