"""
Статистическая значимость прироста «После vs До» (paired bootstrap).

Прогоняет BM25 с identity- и kazakh-стеммером (kazakh берёт стемы из кэша),
затем для каждой категории и в целом считает дельту ndcg@10 / recall@5 и
p-value парного бутстрэпа. Нужен полный стем-кэш (data/resources/stem_cache.json).

ЗАПУСК:
    python -m src.eval.significance
"""

from __future__ import annotations

import argparse
from typing import Dict, Set

from ..eval import run_benchmark, metrics
from ..queries import dataset


def _subset(qrels: Dict[str, Set[str]], qids: Set[str]) -> Dict[str, Set[str]]:
    return {q: rel for q, rel in qrels.items() if q in qids}


def main() -> None:
    ap = argparse.ArgumentParser(description="Значимость прироста стеммера (bootstrap)")
    ap.add_argument("--corpus", default="data/corpus/corpus.jsonl")
    ap.add_argument("--queries", default="data/queries/queries.jsonl")
    ap.add_argument("--metric", default="ndcg", choices=["ndcg", "recall", "mrr"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--resamples", type=int, default=10000)
    args = ap.parse_args()

    res_id = run_benchmark.run(args.corpus, args.queries, "identity")
    res_kz = run_benchmark.run(args.corpus, args.queries, "kazakh")
    run_a, run_b = res_id["run"], res_kz["run"]

    queries = dataset.load_queries(args.queries)
    qrels = dataset.qrels_from_queries(queries)
    cats = dataset.categories_from_queries(queries)

    print(f"\nПарный bootstrap по {args.metric}@{args.k} "
          f"({args.resamples} ресэмплов), n={len(queries)} запросов")
    print(f"{'категория':<16} {'Δ':>8} {'p-value':>9} {'значимо?':>10}")
    print("-" * 46)

    groups = {"ВСЕ": set(qrels)}
    for c in sorted(set(cats.values())):
        groups[c] = {q for q, cc in cats.items() if cc == c}

    for name, qids in groups.items():
        sub = _subset(qrels, qids)
        delta, p, _ = metrics.paired_bootstrap(
            run_a, run_b, sub, metric=args.metric, k=args.k, n_resamples=args.resamples)
        mark = "да (p<0.05)" if p < 0.05 else "нет"
        print(f"{name:<16} {delta:>+8.3f} {p:>9.4f} {mark:>10}")


if __name__ == "__main__":
    main()
