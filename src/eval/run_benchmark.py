"""
Прогон бенчмарка: BM25 поверх корпуса с заданным стеммером, метрики по
запросам — всего и в разбивке по категориям (inflected / vocabulary-gap /
natural). Это инструмент для замера «до/после» вклада стеммера.

ЗАПУСК:
    # baseline без стеммера
    python -m src.eval.run_benchmark --stemmer identity --out results/bm25_identity.json
    # со стеммером (нужен доступ к API стеммера — например, локально/Colab)
    python -m src.eval.run_benchmark --stemmer kazakh --out results/bm25_kazakh.json
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Set

from ..preprocess.tokenize import tokenize
from ..preprocess.stemmer import get_stemmer
from ..retrieval.bm25 import BM25Index, default_analyzer
from ..eval import metrics
from ..queries import dataset


def _subset(qrels: Dict[str, Set[str]], qids: Set[str]) -> Dict[str, Set[str]]:
    return {q: rel for q, rel in qrels.items() if q in qids}


def warm_stemmer_if_needed(stemmer, corpus_pairs, query_texts: List[str]) -> None:
    """Если у стеммера есть пакетный прогрев кэша — собираем все токены и греем."""
    warm = getattr(stemmer, "warm", None)
    if not callable(warm):
        return
    tokens = set()
    for _, text in corpus_pairs:
        tokens.update(tokenize(text))
    for text in query_texts:
        tokens.update(tokenize(text))
    print(f"Прогрев стеммера: {len(tokens)} уникальных токенов…")
    warm(tokens)


def run(corpus_path: str, queries_path: str, stemmer_name: str,
        top_k: int = 10) -> Dict:
    corpus_pairs = dataset.load_corpus(corpus_path)
    queries = dataset.load_queries(queries_path)
    qmap = dataset.queries_as_map(queries)
    qrels = dataset.qrels_from_queries(queries)
    cats = dataset.categories_from_queries(queries)

    stemmer = get_stemmer(stemmer_name)
    warm_stemmer_if_needed(stemmer, corpus_pairs, list(qmap.values()))

    print(f"Индексация {len(corpus_pairs)} пассажей (стеммер={stemmer_name})…")
    index = BM25Index(analyzer=default_analyzer(stemmer)).index(corpus_pairs)

    run_result = index.run(qmap, top_k=top_k)

    overall = metrics.evaluate_run(run_result, qrels,
                                   metrics=("recall", "mrr", "ndcg"), ks=(1, 5, 10))

    # разбивка по категориям
    by_cat: Dict[str, Dict] = {}
    cat_names = sorted(set(cats.values()))
    for cat in cat_names:
        qids = {q for q, c in cats.items() if c == cat}
        by_cat[cat] = metrics.evaluate_run(run_result, _subset(qrels, qids),
                                           metrics=("recall", "mrr", "ndcg"), ks=(1, 5, 10))

    return {
        "stemmer": stemmer_name,
        "n_passages": len(corpus_pairs),
        "n_queries": len(queries),
        "overall": overall,
        "by_category": by_cat,
        "run": run_result,
    }


def _fmt_table(result: Dict) -> str:
    lines = [f"\n=== BM25 (стеммер={result['stemmer']}) | "
             f"{result['n_passages']} пассажей, {result['n_queries']} запросов ===",
             f"{'категория':<16} {'recall@1':>9} {'recall@5':>9} {'mrr@10':>8} {'ndcg@10':>8}"]
    def row(name, m):
        return (f"{name:<16} {m['recall@1']:>9.3f} {m['recall@5']:>9.3f} "
                f"{m['mrr@10']:>8.3f} {m['ndcg@10']:>8.3f}")
    lines.append(row("ВСЕ", result["overall"]))
    for cat, m in result["by_category"].items():
        lines.append(row(cat, m))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Прогон BM25-бенчмарка с метриками")
    ap.add_argument("--corpus", default="data/corpus/corpus.jsonl")
    ap.add_argument("--queries", default="data/queries/queries.jsonl")
    ap.add_argument("--stemmer", choices=["identity", "kazakh"], default="identity")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default=None, help="куда сохранить JSON с метриками")
    ap.add_argument("--runs-out", default=None,
                    help="куда сохранить per-query ранжировки {query_id: [doc_id,...]} "
                         "(нужно для RRF-гибрида; используй --top-k 100)")
    args = ap.parse_args()

    result = run(args.corpus, args.queries, args.stemmer, args.top_k)
    print(_fmt_table(result))

    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        # run сохраняем отдельно, чтобы JSON метрик был компактным
        compact = {k: v for k, v in result.items() if k != "run"}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(compact, f, ensure_ascii=False, indent=2)
        print(f"\nМетрики сохранены → {args.out}")

    if args.runs_out:
        from ..retrieval.hybrid import save_run
        save_run(args.runs_out, result["run"])
        print(f"Ранжировки сохранены → {args.runs_out}")


if __name__ == "__main__":
    main()
