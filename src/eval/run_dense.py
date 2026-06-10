"""
Прогон dense-бенчмарка: мультиязычные эмбеддинги (IBM Granite / Google LaBSE /
multilingual-e5) поверх корпуса, метрики всего и по категориям. Формат
результата совпадает с run_benchmark -> работает в compare.py.

ЗАПУСК (Colab/GPU; нужны sentence-transformers + torch):
    python -m src.eval.run_dense --model labse   --out results/dense_labse_300.json
    python -m src.eval.run_dense --model e5      --out results/dense_e5_300.json
    python -m src.eval.run_dense --model granite --out results/dense_granite_300.json

Эмбеддинги корпуса кэшируются (.npy), повторный прогон не пересчитывает.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Set

import numpy as np

from ..retrieval.dense import DenseIndex
from ..eval import metrics
from ..queries import dataset

# короткое имя -> (HF id, префикс запроса, префикс документа)
MODELS = {
    "labse":   ("sentence-transformers/LaBSE", "", ""),
    "e5":      ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    "granite": ("ibm-granite/granite-embedding-278m-multilingual", "", ""),
    # --- Спринт 2: новые модели на том же бенчмарке ---
    "granite-r2-97m":  ("ibm-granite/granite-embedding-97m-multilingual-r2", "", ""),
    "granite-r2-311m": ("ibm-granite/granite-embedding-311m-multilingual-r2", "", ""),
    # дообучена от multilingual-e5 -> нужны e5-префиксы query:/passage:
    "shyngys-e5":      ("shyngys879/kazakh-e5-rag-embedding", "query: ", "passage: "),
}


def _subset(qrels: Dict[str, Set[str]], qids: Set[str]) -> Dict[str, Set[str]]:
    return {q: rel for q, rel in qrels.items() if q in qids}


def _metrics_block(run_result, qrels, cats) -> Dict:
    overall = metrics.evaluate_run(run_result, qrels,
                                   metrics=("recall", "mrr", "ndcg"), ks=(1, 5, 10))
    by_cat = {}
    for cat in sorted(set(cats.values())):
        qids = {q for q, c in cats.items() if c == cat}
        by_cat[cat] = metrics.evaluate_run(run_result, _subset(qrels, qids),
                                           metrics=("recall", "mrr", "ndcg"), ks=(1, 5, 10))
    return overall, by_cat


def _load_emb_cache(path: str, expected_ids):
    if not path or not os.path.exists(path + ".npy") or not os.path.exists(path + ".ids.json"):
        return None
    ids = json.load(open(path + ".ids.json", encoding="utf-8"))
    if ids != list(expected_ids):
        return None  # корпус изменился — пересчитать
    return np.load(path + ".npy")


def _save_emb_cache(path: str, doc_ids, matrix) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path + ".npy", matrix)
    json.dump(list(doc_ids), open(path + ".ids.json", "w", encoding="utf-8"))


def run(corpus_path: str, queries_path: str, model_key: str,
        emb_cache: str = None, top_k: int = 10, batch_size: int = 64,
        max_seq_len: int = 512) -> Dict:
    hf_id, qpref, dpref = MODELS.get(model_key, (model_key, "", ""))
    corpus_pairs = dataset.load_corpus(corpus_path)
    queries = dataset.load_queries(queries_path)
    qmap = dataset.queries_as_map(queries)
    qrels = dataset.qrels_from_queries(queries)
    cats = dataset.categories_from_queries(queries)

    index = DenseIndex(model_name=hf_id, query_prefix=qpref, doc_prefix=dpref,
                       batch_size=batch_size, max_seq_len=max_seq_len)
    doc_ids = [d for d, _ in corpus_pairs]
    cached = _load_emb_cache(emb_cache, doc_ids)
    if cached is not None:
        print(f"Эмбеддинги из кэша: {emb_cache}.npy")
        index.set_embeddings(doc_ids, cached)
    else:
        print(f"Кодирование {len(corpus_pairs)} пассажей моделью {hf_id}…")
        index.index(corpus_pairs)
        _save_emb_cache(emb_cache, index.doc_ids, index.matrix)

    run_result = index.run(qmap, top_k=top_k)
    overall, by_cat = _metrics_block(run_result, qrels, cats)
    return {
        "stemmer": f"dense:{model_key}",  # метка для compare.py
        "system": "dense", "model": hf_id,
        "n_passages": len(corpus_pairs), "n_queries": len(queries),
        "overall": overall, "by_category": by_cat, "run": run_result,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Dense-бенчмарк (мультиязычные эмбеддинги)")
    ap.add_argument("--model", default="labse",
                    help="labse | e5 | granite | или любой HF id")
    ap.add_argument("--corpus", default="data/corpus/corpus.jsonl")
    ap.add_argument("--queries", default="data/queries/queries.jsonl")
    ap.add_argument("--emb-cache", default=None,
                    help="префикс пути кэша эмбеддингов (напр. results/emb_labse)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=512,
                    help="лимит длины контекста (ModernBERT/Granite-R2 иначе OOM)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--runs-out", default=None,
                    help="куда сохранить per-query ранжировки {query_id: [doc_id,...]} "
                         "(нужно для RRF-гибрида; используй --top-k 100)")
    args = ap.parse_args()

    if not args.emb_cache:
        args.emb_cache = f"results/emb_{args.model}"
    result = run(args.corpus, args.queries, args.model, args.emb_cache,
                 args.top_k, args.batch_size, args.max_seq_len)

    # таблица
    from .run_benchmark import _fmt_table
    print(_fmt_table(result))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        compact = {k: v for k, v in result.items() if k != "run"}
        json.dump(compact, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\nМетрики сохранены → {args.out}")

    if args.runs_out:
        from ..retrieval.hybrid import save_run
        save_run(args.runs_out, result["run"])
        print(f"Ранжировки сохранены → {args.runs_out}")


if __name__ == "__main__":
    main()
