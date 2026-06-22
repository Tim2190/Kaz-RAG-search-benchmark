"""
Прогон Cohere Embed на бенчмарке (Wikipedia n=300 + опционально Akorda n=244).
Формат результата совпадает с run_dense → совместим с compare.py.

Требования:
  pip install cohere>=5
  export COHERE_API_KEY=<ваш ключ>   # или --api-key

Поддерживаемые модели (--model):
  embed-v4.0              — флагман 2025, 100+ языков, 128K ctx (default)
  embed-multilingual-v3.0 — предыдущий флагман, 1024-dim, 512 токенов

Эмбеддинги корпуса кэшируются (.npy), повторный прогон не делает API-вызовов.

ЗАПУСК (нужен COHERE_API_KEY; корпус локальный):
  python -m src.eval.run_cohere \
      --model embed-v4.0 \
      --out results/dense_cohere_v4_300.json \
      --runs-out results/runs_dense_cohere_v4.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Set

import numpy as np

from ..retrieval.cohere_index import CohereIndex
from ..eval import metrics
from ..queries import dataset


def _subset(qrels: Dict[str, Set[str]], qids: Set[str]) -> Dict[str, Set[str]]:
    return {q: rel for q, rel in qrels.items() if q in qids}


def _metrics_block(run_result, qrels, cats) -> tuple:
    overall = metrics.evaluate_run(
        run_result, qrels, metrics=("recall", "mrr", "ndcg"), ks=(1, 5, 10)
    )
    by_cat = {}
    for cat in sorted(set(cats.values())):
        qids = {q for q, c in cats.items() if c == cat}
        by_cat[cat] = metrics.evaluate_run(
            run_result, _subset(qrels, qids),
            metrics=("recall", "mrr", "ndcg"), ks=(1, 5, 10),
        )
    return overall, by_cat


def _load_emb_cache(path: str, expected_ids):
    if (
        not path
        or not os.path.exists(path + ".npy")
        or not os.path.exists(path + ".ids.json")
    ):
        return None
    ids = json.load(open(path + ".ids.json", encoding="utf-8"))
    if ids != list(expected_ids):
        return None
    return np.load(path + ".npy")


def _save_emb_cache(path: str, doc_ids, matrix) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path + ".npy", matrix)
    json.dump(list(doc_ids), open(path + ".ids.json", "w", encoding="utf-8"))


def run(
    corpus_path: str,
    queries_path: str,
    model_name: str = "embed-v4.0",
    api_key: str = None,
    emb_cache: str = None,
    top_k: int = 10,
    batch_size: int = 96,
) -> Dict:
    corpus_pairs = dataset.load_corpus(corpus_path)
    queries = dataset.load_queries(queries_path)
    qmap = dataset.queries_as_map(queries)
    qrels = dataset.qrels_from_queries(queries)
    cats = dataset.categories_from_queries(queries)

    index = CohereIndex(model_name=model_name, api_key=api_key, batch_size=batch_size)
    doc_ids = [d for d, _ in corpus_pairs]

    cached = _load_emb_cache(emb_cache, doc_ids)
    if cached is not None:
        print(f"Эмбеддинги из кэша: {emb_cache}.npy")
        index.set_embeddings(doc_ids, cached)
    else:
        index.index(corpus_pairs)
        _save_emb_cache(emb_cache, index.doc_ids, index.matrix)

    run_result = index.run(qmap, top_k=top_k)
    overall, by_cat = _metrics_block(run_result, qrels, cats)

    short_key = model_name.replace("/", "-")
    return {
        "stemmer": f"dense:cohere-{short_key}",
        "system": "cohere",
        "model": f"cohere/{model_name}",
        "n_passages": len(corpus_pairs),
        "n_queries": len(queries),
        "overall": overall,
        "by_category": by_cat,
        "run": run_result,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Cohere Embed — dense benchmark")
    ap.add_argument("--model", default="embed-v4.0",
                    help="embed-v4.0 | embed-multilingual-v3.0")
    ap.add_argument("--api-key", default=None,
                    help="Cohere API key (иначе из COHERE_API_KEY env)")
    ap.add_argument("--corpus",  default="data/corpus/corpus.jsonl")
    ap.add_argument("--queries", default="data/queries/queries.jsonl")
    ap.add_argument("--emb-cache", default=None,
                    help="префикс кэша эмбеддингов (напр. results/emb_cohere_v4)")
    ap.add_argument("--top-k",     type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--out",       default=None)
    ap.add_argument("--runs-out",  default=None,
                    help="куда сохранить per-query ранжировки (для RRF-гибрида; --top-k 100)")
    args = ap.parse_args()

    if not args.emb_cache:
        safe = args.model.replace("/", "-").replace(".", "_")
        args.emb_cache = f"results/emb_cohere_{safe}"

    result = run(
        args.corpus, args.queries, args.model, args.api_key,
        args.emb_cache, args.top_k, args.batch_size,
    )

    from .run_benchmark import _fmt_table
    print(_fmt_table(result))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        compact = {k: v for k, v in result.items() if k != "run"}
        json.dump(compact, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"\nМетрики сохранены → {args.out}")

    if args.runs_out:
        from ..retrieval.hybrid import save_run
        save_run(args.runs_out, result["run"])
        print(f"Ранжировки сохранены → {args.runs_out}")


if __name__ == "__main__":
    main()
