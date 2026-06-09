"""
Спринт 2 — быстрый synonym-тест (без GPU/сети).

Контекст: категория `vocabulary-gap` по факту НЕ создаёт vocabulary gap —
у неё самое высокое лексическое пересечение запрос↔gold (0.70) и ответ лежит
в gold-тексте в 93% случаев. Значит победа BM25 там объясняется совпадением
редких ключевых слов, а не «пониманием синонимов».

Этот скрипт честно отбирает ПОДМНОЖЕСТВО запросов с реально НИЗКИМ лексическим
пересечением (настоящий vocabulary gap) и перемеряет на нём системы — на уже
сохранённых per-query ранжировках (`results/runs_*.json`), без повторного
прогона моделей.

Вопрос: на честных low-overlap запросах кто выигрывает — лексика (BM25) или
плотные эмбеддинги? Если dense обгоняет — «морфология бьёт семантику»
опровергается; если BM25 держится — подтверждается уже чисто.

Метод пересечения: префиксный стемминг (корень ≈ первые 5 букв), offline и
воспроизводимо. overlap = доля содержательных слов запроса (len≥3), чей корень
встречается в gold-пассаже.

ЗАПУСК:
    python -m src.eval.synonym_subset
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Set, Tuple

from ..queries import dataset
from ..retrieval.hybrid import load_run
from ..eval import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "SPRINT2_SYNONYM_SUBSET.md")

CORPUS = os.path.join(ROOT, "data/corpus/corpus.jsonl")
QUERIES = os.path.join(ROOT, "data/queries/queries.jsonl")

# Системы, для которых есть сохранённые ранжировки
SYSTEMS: List[Tuple[str, str]] = [
    ("BM25 + stemmer",        "runs_bm25_kazakh.json"),
    ("Granite-278m (R1)",     "runs_dense_granite.json"),
    ("Granite-97m (R2)",      "runs_dense_granite_r2_97m.json"),
    ("Granite-311m (R2)",     "runs_dense_granite_r2_311m.json"),
    ("kazakh-e5 (shyngys879)", "runs_dense_shyngys.json"),
]

THRESHOLDS = [0.30, 0.20]  # «настоящий gap» = пересечение ниже порога

_WORD = re.compile(r"[А-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]+")


def _stem(w: str) -> str:
    return w[:5]


def _toks(s: str) -> Set[str]:
    return set(_WORD.findall(s.lower()))


def _overlap(qtext: str, gtext: str) -> float:
    q = {_stem(w) for w in _toks(qtext) if len(w) >= 3}
    g = {_stem(w) for w in _toks(gtext)}
    if not q:
        return 1.0
    return len(q & g) / len(q)


def _fmt_metrics(name: str, m: Dict[str, float]) -> str:
    return (f"| {name} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | "
            f"{m['recall@10']:.3f} | {m['mrr@10']:.3f} | {m['ndcg@10']:.3f} |")


def _bold_best_ndcg(rows: List[Tuple[str, Dict[str, float]]]) -> Dict[str, bool]:
    best = max(rows, key=lambda r: r[1]["ndcg@10"])[1]["ndcg@10"]
    return {name: abs(m["ndcg@10"] - best) < 1e-9 for name, m in rows}


def main() -> None:
    corpus = dict(dataset.load_corpus(CORPUS))
    queries = dataset.load_queries(QUERIES)
    qrels_all = dataset.qrels_from_queries(queries)
    cats = dataset.categories_from_queries(queries)

    # overlap по каждому запросу
    overlaps: Dict[str, float] = {}
    for q in queries:
        gid = q["gold_doc_ids"][0]
        overlaps[q["query_id"]] = _overlap(q["text"], corpus.get(gid, ""))

    runs = {name: load_run(os.path.join(RESULTS, f)) for name, f in SYSTEMS}

    lines: List[str] = []
    lines.append("# Спринт 2 — быстрый synonym-тест (low-overlap подмножество)\n")
    lines.append(
        "Категория `vocabulary-gap` оказалась непригодной как synonym-тест "
        "(самое высокое лексическое пересечение запрос↔gold, 0.70; ответ в gold "
        "93%). Здесь отбираем **настоящие low-overlap запросы** из всех 300 и "
        "перемеряем системы на готовых ранжировках — без повторного прогона.\n"
    )
    lines.append(
        "**Метод:** overlap = доля содержательных слов запроса (len≥3, корень ≈ "
        "первые 5 букв), чей корень встречается в gold-пассаже. Низкий overlap = "
        "слова запроса не совпадают с текстом = настоящий vocabulary gap.\n"
    )

    # распределение overlap по исходным категориям
    lines.append("## Распределение overlap по исходным категориям\n")
    lines.append("| категория | n | средний overlap |")
    lines.append("|---|---|---|")
    for c in ["natural", "inflected", "vocabulary-gap"]:
        vals = [overlaps[q["query_id"]] for q in queries if q["category"] == c]
        lines.append(f"| {c} | {len(vals)} | {sum(vals)/len(vals):.2f} |")
    lines.append("")

    # для каждого порога — состав подмножества и метрики систем
    for thr in THRESHOLDS:
        subset = {qid for qid, ov in overlaps.items() if ov < thr}
        lines.append(f"## Подмножество: overlap < {thr:.2f}  →  {len(subset)} запросов\n")
        if not subset:
            lines.append("_Пусто — нет запросов ниже порога._\n")
            continue

        # из каких исходных категорий пришли честные запросы
        by_cat = {}
        for q in queries:
            if q["query_id"] in subset:
                by_cat[q["category"]] = by_cat.get(q["category"], 0) + 1
        comp = ", ".join(f"{k}: {v}" for k, v in sorted(by_cat.items()))
        lines.append(f"Состав по исходным категориям: {comp}\n")

        qrels_sub = {q: r for q, r in qrels_all.items() if q in subset}
        rows = []
        for name, _ in SYSTEMS:
            run_sub = {q: docs for q, docs in runs[name].items() if q in subset}
            m = metrics.evaluate_run(run_sub, qrels_sub,
                                     metrics=("recall", "mrr", "ndcg"), ks=(1, 5, 10))
            rows.append((name, m))

        bold = _bold_best_ndcg(rows)
        lines.append("| Система | recall@1 | recall@5 | recall@10 | mrr@10 | nDCG@10 |")
        lines.append("|---|---|---|---|---|---|")
        for name, m in rows:
            line = _fmt_metrics(name, m)
            if bold[name]:  # выделить лучший nDCG@10
                line = line.rsplit("|", 2)[0] + f"| **{m['ndcg@10']:.3f}** |"
            lines.append(line)
        lines.append("")

        winner = max(rows, key=lambda r: r[1]["ndcg@10"])
        bm25 = next(m for n, m in rows if n.startswith("BM25"))
        verdict = (
            "плотная модель обгоняет BM25 → на честных синонимах лексика НЕ лучшая"
            if not winner[0].startswith("BM25")
            else "BM25+стеммер всё равно лучший → лексика держится даже на low-overlap"
        )
        lines.append(f"**Лидер:** {winner[0]} (nDCG@10 {winner[1]['ndcg@10']:.3f}); "
                     f"BM25 {bm25['ndcg@10']:.3f}. {verdict}\n")

    lines.append("## Ограничения\n")
    lines.append("- Базовый multilingual-e5 и LaBSE не включены: их per-query "
                 "ранжировки не сохранялись (`--runs-out` не гонялся). Чтобы "
                 "добавить — один дешёвый прогон в Kaggle с `--runs-out` "
                 "(эмбеддинги кэшированы).")
    lines.append("- Подмножество маленькое → метрики шумнее, чем на n=300; "
                 "это разведочный сигнал, а не финальный вывод.")
    lines.append("- Префиксный стемминг (5 букв) — грубая оценка корня; для "
                 "Статьи 2 нужен валидированный synonym-набор с ручной проверкой.\n")

    report = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n→ Отчёт сохранён: {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
