"""
Прямой замер: рвётся ли казахская ЛАТИНИЦА токенизатором так же, как кириллица?

Контекст
--------
Слой input-preprocessing показал: транслитерация кириллица→латиница НЕ чинит
byte-fallback модель (Qwen3) — латиница значимо ХУЖЕ кириллического baseline.
Объяснение-гипотеза: токенизатор не видел казахскую латиницу в претрейне, поэтому
дробит её так же (или сильнее), как кириллицу. Этот скрипт ИЗМЕРЯЕТ гипотезу
напрямую, а не выводит из результата поиска.

Метод (тот же fertility, что src/eval/tokenization_test.py)
  * Берём те же кэшированные 100 длинных словоформ (wiki + akorda).
  * Для каждого слова считаем sub-words в кириллице и в её транслитерации
    (translit.transliterate, стандарт 2021), в двух режимах (bare / +ведущий пробел).
  * Сравниваем средние. Если fertility(латиница) ≈ или > fertility(кириллица) —
    латиница НЕ помогает токенизатору, гипотеза подтверждается ДАННЫМИ.

Слова локальны (без сети). Токенизаторы качаются с HF — нужен интернет
(Kaggle/Colab; в egress-allowlist huggingface.co обычно закрыт).

ЗАПУСК:
    python input-preprocessing/code/translit_fertility.py
"""

from __future__ import annotations

import os
import sys

# Соседний модуль translit + пакет репозитория (src.eval.fertility_compare).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from src.eval.fertility_compare import CORPORA, harvest_words, _avg_subwords
from translit import transliterate

# Сравниваем «сломанную» (byte-fallback) и «нормальную» (XLM-R) модели.
TOKENIZERS = {
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "e5-base":    "intfloat/multilingual-e5-base",
}


def main() -> None:
    from transformers import AutoTokenizer

    # те же кэшированные словники, что у остальных fertility-замеров
    word_sets = {alias: harvest_words(*cfg) for alias, cfg in CORPORA.items()}
    latin_sets = {alias: [transliterate(w) for w in ws]
                  for alias, ws in word_sets.items()}

    for alias, ws in word_sets.items():
        print(f"{alias}: {len(ws)} слов. Пример кир→лат: "
              f"{ws[0]} → {transliterate(ws[0])}")
    print()

    loaded = {}
    for alias, hf_id in TOKENIZERS.items():
        try:
            loaded[alias] = AutoTokenizer.from_pretrained(hf_id)
        except Exception as e:
            print(f"[skip] {alias}: {e}")

    # таблица: для каждой (модель, домен) — кириллица vs латиница, оба режима
    print(f"{'токенизатор':12s} {'домен':7s} "
          f"{'cyr(bare)':>10s} {'lat(bare)':>10s} {'Δbare':>7s}   "
          f"{'cyr(+sp)':>9s} {'lat(+sp)':>9s} {'Δ+sp':>6s}")
    print("-" * 78)
    for alias, tok in loaded.items():
        for dom in word_sets:
            cb = _avg_subwords(tok, word_sets[dom],  lead_space=False)
            lb = _avg_subwords(tok, latin_sets[dom], lead_space=False)
            cs = _avg_subwords(tok, word_sets[dom],  lead_space=True)
            ls = _avg_subwords(tok, latin_sets[dom], lead_space=True)
            print(f"{alias:12s} {dom:7s} "
                  f"{cb:10.2f} {lb:10.2f} {lb-cb:>+7.2f}   "
                  f"{cs:9.2f} {ls:9.2f} {ls-cs:>+6.2f}")

    print("\nИнтерпретация:")
    print("  Δ ≥ 0  => латиница дробится НЕ лучше (часто хуже) кириллицы:")
    print("           токенизатор не знает казахскую латиницу → транслитерация")
    print("           не может починить byte-fallback. Это подтверждает ДАННЫМИ")
    print("           вывод слоя input-preprocessing (проблема байтовая, не письма).")
    print("  Δ < 0  => латиница ЦЕЛЕЕ кириллицы — тогда провал поиска на латинице")
    print("           объяснялся бы не токенизацией, а смысловым сдвигом.")


if __name__ == "__main__":
    main()
