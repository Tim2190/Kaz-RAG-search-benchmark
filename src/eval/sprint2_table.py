"""
Спринт 2 — сравнительная таблица новых embedding-моделей на бенчмарке.

Собирает 7 систем × {recall@k, MRR@10, nDCG@10} × {inflected, natural,
vocab-gap, ALL} в один markdown-отчёт results/SPRINT2_NEW_MODELS.md.

Критический артефакт — разбивка ПО КАТЕГОРИЯМ (не только общий nDCG):
именно она показывает, в какой категории какая модель проседает.

ЗАПУСК (после того как 3 новых прогона из Colab лежат в results/):
    python -m src.eval.sprint2_table

Файлы, которых ещё нет, помечаются как «— (pending)» — отчёт собирается
и без них, чтобы видеть прогресс.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "SPRINT2_NEW_MODELS.md")

CATS = ["inflected", "natural", "vocabulary-gap"]
CAT_SHORT = {"inflected": "inflected", "natural": "natural", "vocabulary-gap": "vocab-gap"}

# Порядок строк в таблице: (отображаемое имя, файл результата, происхождение, kind)
#  kind="dense"  -> JSON c полями overall/by_category (run_dense)
#  kind="hybrid" -> JSON от run_hybrid; берём metrics.hybrid.{overall,by_category}
#  — 4 системы из Статьи 1 (готовы, dense/lexical)
#  — 3 новые dense-системы Спринта 2 (без стеммера)
#  — 3 гибрида Спринта 2 (BM25+стеммер ⊕ новая модель, RRF)
SYSTEMS: List[Tuple[str, str, str, str]] = [
    ("BM25 + stemmer",            "bm25_kazakh.json",            "Статья 1", "dense"),
    ("multilingual-e5-base",      "dense_e5_300.json",           "Статья 1", "dense"),
    ("LaBSE",                     "dense_labse_300.json",        "Статья 1", "dense"),
    ("Granite-278m (R1)",         "dense_granite_300.json",      "Статья 1", "dense"),
    ("Granite-97m (R2)",          "dense_granite_r2_97m.json",   "Спринт 2", "dense"),
    ("Granite-311m (R2)",         "dense_granite_r2_311m.json",  "Спринт 2", "dense"),
    ("kazakh-e5 (shyngys879)",    "dense_shyngys.json",          "Спринт 2", "dense"),
    ("Granite-97m R2 ⊕ BM25+st",  "hybrid_granite_r2_97m.json",  "Спринт 2", "hybrid"),
    ("Granite-311m R2 ⊕ BM25+st", "hybrid_granite_r2_311m.json", "Спринт 2", "hybrid"),
    ("kazakh-e5 ⊕ BM25+st",       "hybrid_shyngys.json",         "Спринт 2", "hybrid"),
]


def _load(fname: str, kind: str = "dense") -> Optional[Dict]:
    """Вернуть нормализованный блок {overall, by_category} независимо от kind."""
    path = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    if kind == "hybrid":
        return d.get("metrics", {}).get("hybrid")  # run_hybrid формат
    return d  # run_dense формат: overall/by_category на верхнем уровне


def _cell(block: Optional[Dict], cat: Optional[str], metric: str) -> str:
    """Достать метрику. cat=None -> overall. Нет данных -> прочерк."""
    if block is None:
        return "—"
    src = block["overall"] if cat is None else block.get("by_category", {}).get(cat)
    if not src or metric not in src:
        return "—"
    return f"{src[metric]:.3f}"


def _bold_max(values: List[str]) -> List[str]:
    """Выделить максимум жирным среди числовых значений столбца-строки."""
    nums = [(i, float(v)) for i, v in enumerate(values) if v not in ("—",)]
    if not nums:
        return values
    best = max(nums, key=lambda t: t[1])[1]
    out = []
    for v in values:
        if v != "—" and abs(float(v) - best) < 1e-9:
            out.append(f"**{v}**")
        else:
            out.append(v)
    return out


def build() -> str:
    rows = [(name, _load(fname, kind), origin) for name, fname, origin, kind in SYSTEMS]
    n_ready = sum(1 for _, d, _ in rows if d is not None)

    lines: List[str] = []
    lines.append("# Спринт 2 — новые embedding-модели на бенчмарке\n")
    lines.append(
        "Сравнение 3 новых embedding-моделей с 4 системами из Статьи 1 на том же "
        "бенчмарке (`Tim2190/kaz-rag-search-benchmark`: 300 запросов, 3 категории, "
        "8 370 пассажей). README и выводы Статьи 1 не меняются — это отдельный "
        "артефакт.\n"
    )
    lines.append(f"**Статус прогонов:** {n_ready}/{len(SYSTEMS)} систем готово "
                 "(4 baseline + 3 dense + 3 hybrid).\n")

    # --- Главная таблица: nDCG@10 по категориям (критический артефакт) ---
    lines.append("## Главное: nDCG@10 по категориям\n")
    header = "| Система | происхождение | inflected | natural | vocab-gap | **ALL** |"
    sep    = "|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for name, d, origin in rows:
        vals = [_cell(d, c, "ndcg@10") for c in CATS] + [_cell(d, None, "ndcg@10")]
        lines.append(f"| {name} | {origin} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append(
        "> Жирным в детальных таблицах ниже — лучшая модель в каждой ячейке. "
        "Общее среднее (ALL) скрывает категориальные дыры; смотри разбивку.\n"
    )

    # --- Детальные таблицы по каждой метрике, с выделением максимума по столбцу ---
    metrics_spec = [
        ("nDCG@10", "ndcg@10"),
        ("MRR@10",  "mrr@10"),
        ("Recall@1", "recall@1"),
        ("Recall@5", "recall@5"),
        ("Recall@10", "recall@10"),
    ]
    cols = CATS + [None]  # None = ALL
    col_names = [CAT_SHORT[c] for c in CATS] + ["ALL"]

    for title, metric in metrics_spec:
        lines.append(f"## {title}\n")
        lines.append("| Система | " + " | ".join(col_names) + " |")
        lines.append("|---|" + "---|" * len(col_names))
        # собрать матрицу значений, затем выделить максимум по столбцу
        raw = [[_cell(d, c, metric) for c in cols] for _, d, _ in rows]
        for j in range(len(cols)):
            col_vals = [raw[i][j] for i in range(len(rows))]
            col_vals = _bold_max(col_vals)
            for i in range(len(rows)):
                raw[i][j] = col_vals[i]
        for (name, _, _), vals in zip(rows, raw):
            lines.append(f"| {name} | " + " | ".join(vals) + " |")
        lines.append("")

    # --- Токенизация ---
    lines.append("## Токенизация: sub-word на казахское слово\n")
    lines.append(
        "100 частых морфологически богатых казахских словоформ (len≥9) из корпуса, "
        "среднее число sub-word токенов на слово (меньше = бережнее к морфологии). "
        "Два варианта: слово в изоляции и с ведущим пробелом (R2 — byte-level BPE, "
        "чувствителен к пробелу; R1/e5 — SentencePiece, нет):\n"
    )
    lines.append("| Токенизатор | в изоляции | с пробелом |")
    lines.append("|---|---|---|")
    lines.append("| multilingual-e5-base | **1.81** | **1.81** |")
    lines.append("| Granite-278m (R1) | **1.81** | **1.81** |")
    lines.append("| Granite-97m (R2) | 4.00 | 3.57 |")
    lines.append("| Granite-311m (R2) | 4.20 | 3.82 |")
    lines.append("")
    lines.append(
        "> Токенизатор R2 (ModernBERT) дробит казахские слова в ~2.3× сильнее, "
        "чем R1/e5 (замер «в изоляции»; вариант с ведущим пробелом и примеры — в "
        "`src/eval/tokenization_test.py`). Это **один из факторов** общей просадки "
        "R2 на казахском, а не единственная причина: R2-311M на морфологической "
        "категории (inflected) не просел (0.791 = R1), сильнее всего фрагментация "
        "бьёт по маленькой R2-97M. Подробнее — `results/GRANITE_R2_REVIEW.md`.\n"
    )

    # --- Вопрос спринта + вывод ---
    lines.append("## Вопрос спринта\n")
    lines.append(
        "**Есть ли у Granite-R2 (97m или 311m) категория, где он на казахском "
        "проседает — несмотря на enhanced-support — и которую можно закрыть "
        "fine-tune'ом?**\n"
    )
    lines.append("### ⚠️ Важная оговорка про категорию `vocabulary-gap`\n")
    lines.append(
        "Категория задумывалась как «вопрос без слов текста, через синонимы/"
        "описание» (true vocabulary gap). По факту валидация показала обратное:\n"
    )
    lines.append("| категория | overlap запрос∩gold | ответ лежит в gold |")
    lines.append("|---|---|---|")
    lines.append("| natural | 0.61 | 36% |")
    lines.append("| inflected | 0.63 | 43% |")
    lines.append("| **vocabulary-gap** | **0.70** (самый высокий) | **93%** |")
    lines.append("")
    lines.append(
        "У `vocabulary-gap` лексическое пересечение с gold-пассажем **самое "
        "большое**, а не самое маленькое. Это не синонимная подмена, а длинные "
        "**описательные** вопросы, насыщенные редкими ключевыми словами "
        "(`фьорд`, `Скандинавия`, топонимы), которые LLM-генератор не смог "
        "перефразировать. Поэтому **категория НЕ измеряет vocabulary gap.**\n"
    )
    lines.append("### Вывод (с поправкой)\n")
    lines.append(
        "**BM25+стеммер выигрывает на `vocabulary-gap` (0.764) не потому, что "
        "морфология «понимает синонимы», а потому что цепляется за совпадающие "
        "редкие ключевые слова.** Плотные модели там проседают по другой причине: "
        "на длинном описательном запросе эмбеддинг уплывает к семантически "
        "близким, но неверным сущностям (Швеция/Дания вместо Норвегии), а "
        "лексический якорь (`фьорд`) пришпиливает точный пассаж. Это «dense путает "
        "близкие сущности», а **не** «dense слеп к синонимам».\n"
    )
    lines.append("### Что остаётся валидным\n")
    lines.append("1. **Granite-R2 ≠ улучшение** над R1 на казахском: токенизатор "
                 "ModernBERT дробит казахские слова в **2.3×** сильнее (4.0–4.2 "
                 "против 1.81 sub-word/слово), и по всем категориям R2 не выше R1. "
                 "Маркетинговый «enhanced multilingual support» на казахском не "
                 "работает. — **чистый результат.**")
    lines.append("2. **Наивный казахский fine-tune ломает модель:** kazakh-e5 "
                 "(shyngys879) **хуже** базового e5 по ВСЕМ категориям (ALL 0.747 "
                 "против 0.785). — **чистый результат, важное предупреждение.**")
    lines.append("3. **Гибрид BM25+стеммер ⊕ kazakh-e5 = 0.808 nDCG@10 (ALL)** — "
                 "лучшее число бенчмарка, выше базового e5 (0.785), без обучения. "
                 "Слияние лексики и плотного канала надёжно поднимает retrieval. "
                 "(Оговорка: часть прироста ALL идёт через `vocabulary-gap`, чья "
                 "интерпретация ограничена; на `inflected` гибрид тоже значимо "
                 "лучше обоих каналов — этот прирост чистый.)\n")
    lines.append("### Решение\n")
    lines.append(
        "- Заявленной «дыры на синонимах» бенчмарк **не доказывает** — категория "
        "`vocabulary-gap` для этого непригодна. Слепую зону к синонимам надо "
        "измерять отдельным, валидированным synonym-тестом (низкое лексическое "
        "пересечение запрос↔gold).\n"
        "- Что доказано надёжно: (а) Granite-R2 — регресс на казахском; "
        "(б) наивный fine-tune вредит; (в) гибрид лексика⊕dense — лучший "
        "результат.\n\n"
        "**Следующий шаг:** построить честный synonym-тест и перемерить — только "
        "тогда вопрос «есть ли дыра под fine-tune» получит корректный ответ. "
        "До тех пор гибрид — единственный надёжно-публикуемый вывод спринта.\n"
    )

    return "\n".join(lines)


def main() -> None:
    report = build()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n→ Отчёт сохранён: {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
