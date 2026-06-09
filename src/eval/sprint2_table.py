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

# Порядок строк в таблице: (отображаемое имя, файл результата, происхождение)
#  — 4 системы из Статьи 1 (готовы)
#  — 3 новые системы Спринта 2 (привозятся из Colab)
SYSTEMS: List[Tuple[str, str, str]] = [
    ("BM25 + stemmer",          "bm25_kazakh.json",            "Статья 1"),
    ("multilingual-e5-base",    "dense_e5_300.json",           "Статья 1"),
    ("LaBSE",                   "dense_labse_300.json",        "Статья 1"),
    ("Granite-278m (R1)",       "dense_granite_300.json",      "Статья 1"),
    ("Granite-97m (R2)",        "dense_granite_r2_97m.json",   "Спринт 2"),
    ("Granite-311m (R2)",       "dense_granite_r2_311m.json",  "Спринт 2"),
    ("kazakh-e5 (shyngys879)",  "dense_shyngys.json",          "Спринт 2"),
]


def _load(fname: str) -> Optional[Dict]:
    path = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


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
    rows = [(name, _load(fname), origin) for name, fname, origin in SYSTEMS]
    n_ready = sum(1 for _, d, _ in rows if d is not None)

    lines: List[str] = []
    lines.append("# Спринт 2 — новые embedding-модели на бенчмарке\n")
    lines.append(
        "Сравнение 3 новых embedding-моделей с 4 системами из Статьи 1 на том же "
        "бенчмарке (`Tim2190/kaz-rag-search-benchmark`: 300 запросов, 3 категории, "
        "8 370 пассажей). README и выводы Статьи 1 не меняются — это отдельный "
        "артефакт.\n"
    )
    lines.append(f"**Статус прогонов:** {n_ready}/7 систем готово.\n")

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

    # --- Вопрос спринта ---
    lines.append("## Вопрос спринта\n")
    lines.append(
        "**Есть ли у Granite-R2 (97m или 311m) категория, где он на казахском "
        "проседает — несмотря на enhanced-support — и которую можно закрыть "
        "fine-tune'ом?**\n"
    )
    lines.append("- **Да, есть дыра** → fine-tune под неё (Репо 2, Статья 2). Проект-строитель.")
    lines.append("- **Нет дыры, R2 хорош везде** → разворот на независимую верификацию + "
                 "гибрид / сравнительное тюркское исследование. Проект-оценщик.\n")
    lines.append("_Вывод заполняется после прогона всех 3 моделей._\n")

    return "\n".join(lines)


def main() -> None:
    report = build()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n→ Отчёт сохранён: {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
