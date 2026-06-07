"""
Диагностика: кривая hit@k по готовой ранжировке (без сети/GPU/LLM).

hit@k = доля запросов, у которых хотя бы один gold-документ попал в top-k.
Это ровно тот сигнал, который определяет, увидит ли RAG правильный пассаж
в окне контекста размера k. Полезно, чтобы понять, СТОИТ ли гонять дорогой
LLM-прогон при большем top-k, не запуская сам LLM.

ЗАПУСК:
    python -m src.eval.hit_at_k --runs results/runs_bm25_kazakh.json
    python -m src.eval.hit_at_k --runs results/runs_bm25_synonym.json
    # сравнить два канала бок о бок:
    python -m src.eval.hit_at_k \
        --runs results/runs_bm25_kazakh.json results/runs_bm25_synonym.json \
        --labels kazakh synonym
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Set

from ..queries import dataset

KS = (1, 3, 5, 10)


def hit_at_k(run: Dict[str, List[str]], qrels: Dict[str, Set[str]],
             qids: Set[str], k: int) -> float:
    """Доля запросов из qids, у которых gold попал в top-k ранжировки."""
    qids = [q for q in qids if q in run and qrels.get(q)]
    if not qids:
        return 0.0
    hits = 0
    for q in qids:
        gold = qrels[q]
        if gold & set(run[q][:k]):
            hits += 1
    return hits / len(qids)


def curve(run, qrels, cats, ks=KS) -> Dict:
    """hit@k для ALL и по категориям."""
    groups = {"ALL": set(qrels)}
    for c in sorted(set(cats.values())):
        groups[c] = {q for q, cc in cats.items() if cc == c}
    return {g: {k: hit_at_k(run, qrels, qids, k) for k in ks}
            for g, qids in groups.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="hit@k по готовой ранжировке")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="один или несколько JSON {query_id: [doc_id,...]}")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="подписи для каждого --runs (по умолчанию имена файлов)")
    ap.add_argument("--queries", default="data/queries/queries.jsonl")
    args = ap.parse_args()

    queries = dataset.load_queries(args.queries)
    qrels = dataset.qrels_from_queries(queries)
    cats = dataset.categories_from_queries(queries)

    labels = args.labels or [p.split("/")[-1] for p in args.runs]
    curves = {}
    for path, label in zip(args.runs, labels):
        run = json.load(open(path, encoding="utf-8"))
        # ограничиваем оценку запросами, реально присутствующими в ранжировке
        present = set(run)
        sub_qrels = {q: r for q, r in qrels.items() if q in present}
        sub_cats = {q: c for q, c in cats.items() if q in present}
        curves[label] = curve(run, sub_qrels, sub_cats)
        print(f"[{label}] n={len(sub_qrels)} запросов в ранжировке")

    groups = ["inflected", "natural", "vocabulary-gap", "ALL"]
    print(f"\nhit@k (gold в top-k):")
    header = f"{'срез':<16}{'k':>4}" + "".join(f"{lab:>12}" for lab in labels)
    print(header)
    print("-" * len(header))
    for g in groups:
        for k in KS:
            row = f"{g:<16}{k:>4}"
            for label in labels:
                v = curves[label].get(g, {}).get(k, 0.0)
                row += f"{v:>12.3f}"
            print(row)
        print()


if __name__ == "__main__":
    main()
