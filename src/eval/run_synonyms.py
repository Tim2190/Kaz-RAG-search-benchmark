"""
BM25 + казахский стеммер + synonym query expansion.

Гипотеза: расширение запроса синонимами (из ~/10k-слов словаря) улучшает
nDCG@10 прежде всего на vocabulary-gap, при этом не ухудшает inflected и natural.

Логика:
  1. Корпус индексируется ТАК ЖЕ, как в bm25_kazakh: токены + стеммер.
  2. Запросы расширяются: токены запроса стеммятся, затем для каждого стема
     из словаря добавляются уже-стеммленные синонимы. Корпус НЕ трогается.
  3. Расширенные токены передаются напрямую в BM25 (минуя анализатор),
     т.к. они уже стеммлены и находятся в том же пространстве, что корпус.
  4. Синонимы кэшируются на диск (data/resources/synonym_cache.json), поэтому
     повторные запуски не бьют в API.

Pre-registered критерий:
  nDCG@10(BM25+syn) >= nDCG@10(BM25+stemmer, bm25_kazakh.json) на ALL.

ЗАПУСК (нужна сеть для synonym API, GPU не нужен):
    python -m src.eval.run_synonyms \\
        --out results/bm25_synonym_300.json \\
        [--runs-out results/runs_bm25_synonym.json]

Первый запуск: ~5 мин на прогрев кэша синонимов (~1000-2000 уникальных стемов).
Повторный запуск: секунды (всё из кэша).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Set

from ..preprocess.tokenize import tokenize
from ..preprocess.stemmer import get_stemmer
from ..preprocess.synonyms import SynonymExpander
from ..retrieval.bm25 import BM25Index, default_analyzer
from ..eval import metrics
from ..queries import dataset


def _subset(qrels: Dict[str, Set[str]], qids: Set[str]) -> Dict[str, Set[str]]:
    return {q: rel for q, rel in qrels.items() if q in qids}


def _stem_query(text: str, stemmer) -> List[str]:
    """Токенизирует и стеммит один текст; возвращает список стемов."""
    from ..preprocess.stemmer import stem_tokens
    return stem_tokens(tokenize(text), stemmer)


def run(corpus_path: str, queries_path: str,
        synonym_cache: str = "data/resources/synonym_cache.json",
        top_k: int = 10) -> Dict:
    corpus_pairs = dataset.load_corpus(corpus_path)
    queries = dataset.load_queries(queries_path)
    qmap = dataset.queries_as_map(queries)
    qrels = dataset.qrels_from_queries(queries)
    cats = dataset.categories_from_queries(queries)

    stemmer = get_stemmer("kazakh")

    # прогрев стеммера (все токены корпуса + запросов)
    warm = getattr(stemmer, "warm", None)
    if callable(warm):
        tokens = set()
        for _, text in corpus_pairs:
            tokens.update(tokenize(text))
        for text in qmap.values():
            tokens.update(tokenize(text))
        print(f"Прогрев стеммера: {len(tokens)} уникальных токенов…")
        warm(tokens)

    # --- BM25 индекс корпуса (тот же, что bm25_kazakh) ---
    print(f"Индексация {len(corpus_pairs)} пассажей (стеммер=kazakh)…")
    index = BM25Index(analyzer=default_analyzer(stemmer)).index(corpus_pairs)

    # --- стемминг запросов ---
    query_stems: Dict[str, List[str]] = {
        qid: _stem_query(text, stemmer)
        for qid, text in qmap.items()
    }

    # --- прогрев кэша синонимов ---
    all_stems: Set[str] = set()
    for stems in query_stems.values():
        all_stems.update(stems)

    expander = SynonymExpander(cache_path=synonym_cache)
    print(f"Прогрев кэша синонимов: {len(all_stems)} уникальных стемов запросов…")
    expander.warm(all_stems)

    # --- расширение запросов + подсчёт статистики ---
    expanded_queries: Dict[str, List[str]] = {}
    total_original = 0
    total_expanded = 0
    for qid, stems in query_stems.items():
        exp = expander.expand(stems)
        expanded_queries[qid] = exp
        total_original += len(stems)
        total_expanded += len(exp)
    avg_orig = total_original / len(query_stems) if query_stems else 0
    avg_exp = total_expanded / len(query_stems) if query_stems else 0
    print(f"Расширение: {avg_orig:.1f} → {avg_exp:.1f} токенов/запрос в среднем")

    # --- BM25 поиск с расширенными запросами ---
    run_result = index.run_terms(expanded_queries, top_k=top_k)

    # --- метрики ---
    overall = metrics.evaluate_run(run_result, qrels,
                                   metrics=("recall", "mrr", "ndcg"), ks=(1, 5, 10))

    by_cat: Dict[str, Dict] = {}
    for cat in sorted(set(cats.values())):
        qids = {q for q, c in cats.items() if c == cat}
        by_cat[cat] = metrics.evaluate_run(run_result, _subset(qrels, qids),
                                           metrics=("recall", "mrr", "ndcg"), ks=(1, 5, 10))

    return {
        "stemmer": "kazakh",
        "query_expansion": "synonym",
        "n_passages": len(corpus_pairs),
        "n_queries": len(queries),
        "avg_query_len_original": round(avg_orig, 2),
        "avg_query_len_expanded": round(avg_exp, 2),
        "overall": overall,
        "by_category": by_cat,
        "run": run_result,
    }


def _fmt_table(result: Dict) -> str:
    lines = [
        f"\n=== BM25 + стеммер + synonym-expansion | "
        f"{result['n_passages']} пассажей, {result['n_queries']} запросов ===",
        f"  средний размер запроса: {result['avg_query_len_original']} → "
        f"{result['avg_query_len_expanded']} токенов после расширения",
        f"{'категория':<20} {'recall@1':>9} {'recall@5':>9} {'mrr@10':>8} {'ndcg@10':>8}",
    ]

    def row(name, m):
        return (f"{name:<20} {m['recall@1']:>9.3f} {m['recall@5']:>9.3f} "
                f"{m['mrr@10']:>8.3f} {m['ndcg@10']:>8.3f}")

    lines.append(row("ВСЕ", result["overall"]))
    for cat, m in result["by_category"].items():
        lines.append(row(cat, m))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="BM25 + казахский стеммер + synonym query expansion"
    )
    ap.add_argument("--corpus", default="data/corpus/corpus.jsonl")
    ap.add_argument("--queries", default="data/queries/queries.jsonl")
    ap.add_argument("--synonym-cache", default="data/resources/synonym_cache.json")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default="results/bm25_synonym_300.json")
    ap.add_argument("--runs-out", default=None,
                    help="куда сохранить per-query ранжировки для RRF (--top-k 100)")
    args = ap.parse_args()

    result = run(args.corpus, args.queries, args.synonym_cache, args.top_k)
    print(_fmt_table(result))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    compact = {k: v for k, v in result.items() if k != "run"}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)
    print(f"\nМетрики сохранены → {args.out}")

    if args.runs_out:
        os.makedirs(os.path.dirname(args.runs_out) or ".", exist_ok=True)
        with open(args.runs_out, "w", encoding="utf-8") as f:
            json.dump(result["run"], f, ensure_ascii=False)
        print(f"Ранжировки сохранены → {args.runs_out}")


if __name__ == "__main__":
    main()
