"""
Sub-word fertility для Cohere embed-v4.0 — через Cohere tokenize API.

Зачем отдельный скрипт: у Cohere НЕТ публичного HF-токенизатора, поэтому он не
прогоняется в src/eval/tokenization_test.py и fertility_compare.py (там
AutoTokenizer.from_pretrained по HF id). Токенизация Cohere доступна только через
API — co.tokenize(text=..., model=...).

Метод ТОТ ЖЕ, что в fertility_compare.py (сравнимость гарантирована):
  * берём ТЕ ЖЕ кэшированные 100 частых длинных (len≥9) словоформ из обоих
    корпусов: data/words_tokenization_100.json (Вики), data/words_akorda_100.json
    (Akorda);
  * считаем среднее число sub-word-токенов на слово (без ведущего пробела).

Требования: pip install cohere>=5 + COHERE_API_KEY.
ВНИМАНИЕ: tokenize — лёгкий endpoint, но на trial-ключе тоже есть rate-limit.
Скрипт делает 200 вызовов (100 wiki + 100 akorda); при 429 ждёт 60 с (как embed).

ЗАПУСК:
    python -m src.eval.cohere_tokenization --model embed-v4.0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

WORD_SETS = {
    "wiki":   os.path.join(ROOT, "data", "words_tokenization_100.json"),
    "akorda": os.path.join(ROOT, "data", "words_akorda_100.json"),
}

EXAMPLE_WORDS = ["сөзжасам", "мүмкіндіктерін", "ұйымдастырушылық", "халықаралық"]

_RETRY_SLEEP = 60
_MAX_RETRIES = 5


def _tokenize_with_retry(co, text: str, model: str):
    """co.tokenize с backoff при 429 (как в cohere_index._embed_with_retry)."""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = co.tokenize(text=text, model=model)
            return resp.tokens, getattr(resp, "token_strings", None)
        except Exception as exc:
            is_rl = (getattr(exc, "status_code", None) == 429
                     or "429" in str(exc) or "rate" in str(exc).lower()
                     or "too many" in str(exc).lower())
            if not is_rl or attempt == _MAX_RETRIES:
                raise
            print(f"  [rate-limit] tokenize заблокирован; ожидание {_RETRY_SLEEP} с "
                  f"(попытка {attempt + 1}/{_MAX_RETRIES})…")
            time.sleep(_RETRY_SLEEP)
    raise RuntimeError("unreachable")


def _avg_fertility(co, words: List[str], model: str) -> float:
    total = 0
    for w in words:
        toks, _ = _tokenize_with_retry(co, w, model)
        total += len(toks)
    return total / max(len(words), 1)


def main() -> None:
    import cohere

    ap = argparse.ArgumentParser(description="Cohere fertility (через tokenize API)")
    ap.add_argument("--model", default="embed-v4.0")
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    key = args.api_key or os.environ.get("COHERE_API_KEY", "")
    if not key:
        raise SystemExit("COHERE_API_KEY не задан.")
    co = cohere.ClientV2(api_key=key)

    word_sets = {a: json.load(open(p, encoding="utf-8")) for a, p in WORD_SETS.items()}
    for a, ws in word_sets.items():
        print(f"{a}: {len(ws)} слов (len≥9). Примеры: {', '.join(ws[:6])} …")
    print()

    # наглядные примеры разбиения
    print("--- Примеры разбиения (слово → token_strings) ---")
    for w in EXAMPLE_WORDS:
        toks, strs = _tokenize_with_retry(co, w, args.model)
        shown = strs if strs is not None else toks
        print(f"  «{w}» {len(toks)}: {shown}")
    print()

    # fertility по обоим корпусам
    print(f"{'модель':18s} {'wiki':>8s} {'akorda':>8s} {'Δ':>7s}")
    print("-" * 44)
    wiki_f = _avg_fertility(co, word_sets["wiki"], args.model)
    akorda_f = _avg_fertility(co, word_sets["akorda"], args.model)
    print(f"{'cohere-' + args.model:18s} {wiki_f:8.2f} {akorda_f:8.2f} "
          f"{akorda_f - wiki_f:>+7.2f}")

    print("\nДля сравнения (HF-токенизаторы, из fertility_compare.py):")
    print("  XLM-R cluster (BGE-M3/Jina/E5): 1.81 / 1.82")
    print("  Granite R2-311M:                4.20 / 4.43")
    print("  Qwen3-0.6B:                     6.20 / 6.27")
    print("  Nomic v1.5:                     3.49 / 4.01  (артефакт [UNK])")


if __name__ == "__main__":
    main()
