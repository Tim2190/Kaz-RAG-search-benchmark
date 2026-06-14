"""
Прогон всех систем бенчмарка на датасете Akorda (OOD confirmatory set).

Тот же протокол, что на основном бенчмарке (n=300, Казахская Вики):
  - те же системы: BM25 ±стеммер, Granite R1-278M, R2-97M, R2-311M, e5, shyngys-e5
  - та же метрика: nDCG@10 / MRR@10 / recall@{1,5,10}
  - те же входные форматы: e5/shyngys с префиксами query:/passage:, Granite без
  - paired bootstrap 10 000 resamples для сравнения систем

Категории датасета:
  factoid     — прямые вопросы, высокий overlap (≈0.79): аналог natural
  paraphrase  — перефразированные вопросы (≈0.36)
  low_overlap — низкий лексический overlap (≈0.32): аналог vocab-gap/synonym

Данные: data/akorda/{passages,queries,qrels}.jsonl
Источник: akorda.kz (официальные выступления Президента Казахстана)

ЗАПУСК:

  # BM25 — CPU, без GPU:
  python -m src.eval.run_akorda --system bm25         --out results/akorda/bm25_identity.json
  python -m src.eval.run_akorda --system bm25-stemmer --out results/akorda/bm25_kazakh.json

  # Dense — GPU (Kaggle/Colab):
  python -m src.eval.run_akorda --system e5          --out results/akorda/dense_e5.json
  python -m src.eval.run_akorda --system granite-r1  --out results/akorda/dense_granite_r1.json
  python -m src.eval.run_akorda --system granite-r2-97m  --out results/akorda/dense_granite_r2_97m.json
  python -m src.eval.run_akorda --system granite-r2-311m --out results/akorda/dense_granite_r2_311m.json
  python -m src.eval.run_akorda --system shyngys-e5  --out results/akorda/dense_shyngys.json

  # Hybrid RRF (нужны готовые runs двух каналов):
  python -m src.eval.run_akorda --system hybrid-e5 \\
      --bm25-runs results/akorda/runs_bm25_kazakh.json \\
      --dense-runs results/akorda/runs_dense_e5.json \\
      --out results/akorda/hybrid_e5.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

DATA_DIR   = os.path.join(ROOT, "data", "akorda")
PASSAGES   = os.path.join(DATA_DIR, "passages.jsonl")
QUERIES    = os.path.join(DATA_DIR, "queries.jsonl")
QRELS_FILE = os.path.join(DATA_DIR, "qrels.jsonl")

# (HF model id, query prefix, passage prefix)
DENSE_MODELS: Dict[str, Tuple[str, str, str]] = {
    "e5":             ("intfloat/multilingual-e5-base",
                       "query: ", "passage: "),
    "granite-r1":     ("ibm-granite/granite-embedding-278m-multilingual",
                       "", ""),
    "granite-r2-97m": ("ibm-granite/granite-embedding-97m-multilingual-r2",
                       "", ""),
    "granite-r2-311m":("ibm-granite/granite-embedding-311m-multilingual-r2",
                       "", ""),
    "shyngys-e5":     ("shyngys879/kazakh-e5-rag-embedding",
                       "query: ", "passage: "),
}

RRF_K = 60  # совпадает с основным бенчмарком


# ── загрузка данных ──────────────────────────────────────────────────────────

def _load_corpus() -> List[Tuple[str, str]]:
    """passages.jsonl → [(doc_id, text)]"""
    out = []
    with open(PASSAGES, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append((d["id"], d["text"]))
    return out


def _load_qrels() -> Dict[str, Set[str]]:
    """{query_id: {passage_id, ...}}"""
    qrels: Dict[str, Set[str]] = {}
    with open(QRELS_FILE, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["query_id"] not in qrels:
                qrels[d["query_id"]] = set()
            qrels[d["query_id"]].add(d["passage_id"])
    return qrels


def _load_queries() -> List[Dict]:
    """queries.jsonl → unified list (поля: query_id, text, category, overlap)"""
    out = []
    with open(QUERIES, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append({
                "query_id": d["query_id"],
                "text":     d["query"],       # 'query' → 'text' для pipeline
                "category": d["type"],        # 'type'  → 'category'
                "overlap":  d.get("overlap"),
            })
    return out


def _subset_qrels(qrels: Dict[str, Set[str]],
                  qids: Set[str]) -> Dict[str, Set[str]]:
    return {q: r for q, r in qrels.items() if q in qids}


# ── метрики и bootstrap ──────────────────────────────────────────────────────

def _metrics_block(run_result, qrels, queries):
    from ..eval import metrics
    cats = {q["query_id"]: q["category"] for q in queries}
    overall = metrics.evaluate_run(run_result, qrels,
                                   metrics=("recall", "mrr", "ndcg"), ks=(1, 5, 10))
    by_cat = {}
    for cat in sorted(set(cats.values())):
        qids = {q for q, c in cats.items() if c == cat}
        by_cat[cat] = metrics.evaluate_run(run_result, _subset_qrels(qrels, qids),
                                           metrics=("recall", "mrr", "ndcg"), ks=(1, 5, 10))
    return overall, by_cat


def _cache_only_stemmer(stemmer):
    """Обёртка: использует только кэш стеммера, без API-вызовов.
    Для некэшированных токенов возвращает оригинальный токен (identity).
    Используется, когда API стеммера недоступен (нет сети/403).
    Покрытие кэша на Akorda: ~78% токенов."""
    class _Wrapper:
        name = stemmer.name + "-cached"

        def stem(self, token: str) -> str:
            return stemmer._cache.get(token, token)

    return _Wrapper()


# ── BM25 ─────────────────────────────────────────────────────────────────────

def run_bm25(stemmer_name: str, top_k: int = 100) -> Dict:
    from ..preprocess.stemmer import get_stemmer
    from ..retrieval.bm25 import BM25Index, default_analyzer
    from ..eval import metrics

    corpus = _load_corpus()
    queries = _load_queries()
    qrels = _load_qrels()
    qmap = {q["query_id"]: q["text"] for q in queries}

    stemmer = get_stemmer(stemmer_name)
    if stemmer_name != "identity":
        warm = getattr(stemmer, "warm", None)
        if callable(warm):
            from ..preprocess.tokenize import tokenize
            toks: set = set()
            for _, text in corpus:
                toks.update(tokenize(text))
            for text in qmap.values():
                toks.update(tokenize(text))
            print(f"Прогрев стеммера: {len(toks)} токенов…")
            try:
                warm(toks)
            except Exception as e:
                print(f"  API недоступен ({type(e).__name__}): переключаемся на cache-only mode.")
                print(f"  Кэш покрывает ~78% токенов Akorda; остальные → identity.")
                stemmer = _cache_only_stemmer(stemmer)

    print(f"Индексация {len(corpus)} пассажей (stemmer={stemmer_name})…")
    index = BM25Index(analyzer=default_analyzer(stemmer)).index(corpus)
    run_result = index.run(qmap, top_k=top_k)

    overall, by_cat = _metrics_block(run_result, qrels, queries)
    return {
        "system": f"bm25-{stemmer_name}",
        "dataset": "akorda",
        "n_passages": len(corpus),
        "n_queries": len(queries),
        "overall": overall,
        "by_category": by_cat,
        "run": run_result,
    }


# ── Dense ─────────────────────────────────────────────────────────────────────

def run_dense(model_key: str, top_k: int = 100) -> Dict:
    import numpy as np
    from ..retrieval.dense import DenseIndex
    from ..eval import metrics

    hf_id, q_prefix, p_prefix = DENSE_MODELS[model_key]
    corpus = _load_corpus()
    queries = _load_queries()
    qrels = _load_qrels()
    qmap = {q["query_id"]: q_prefix + q["text"] for q in queries}
    corpus_prefixed = [(doc_id, p_prefix + text) for doc_id, text in corpus]

    cache_path = os.path.join(DATA_DIR, f".cache_emb_{model_key}.npy")
    print(f"Dense encode ({hf_id})…")
    index = DenseIndex(hf_id, cache_path=cache_path)
    index.fit(corpus_prefixed)
    run_result = index.run(qmap, top_k=top_k)

    overall, by_cat = _metrics_block(run_result, qrels, queries)
    return {
        "system": model_key,
        "dataset": "akorda",
        "n_passages": len(corpus),
        "n_queries": len(queries),
        "overall": overall,
        "by_category": by_cat,
        "run": run_result,
    }


# ── Hybrid RRF ────────────────────────────────────────────────────────────────

def run_hybrid(bm25_runs_path: str, dense_runs_path: str,
               k: int = RRF_K) -> Dict:
    from ..eval import metrics

    with open(bm25_runs_path) as f:
        bm25_run = json.load(f)["run"]
    with open(dense_runs_path) as f:
        dense_run = json.load(f)["run"]

    qids = set(bm25_run) | set(dense_run)
    run_result: Dict[str, List[str]] = {}
    for qid in qids:
        scores: Dict[str, float] = {}
        for rank, doc_id in enumerate(bm25_run.get(qid, []), 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        for rank, doc_id in enumerate(dense_run.get(qid, []), 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        run_result[qid] = sorted(scores, key=scores.get, reverse=True)[:100]

    queries = _load_queries()
    qrels = _load_qrels()
    overall, by_cat = _metrics_block(run_result, qrels, queries)
    return {
        "system": "hybrid-rrf",
        "dataset": "akorda",
        "n_passages": _load_corpus().__len__(),
        "n_queries": len(queries),
        "overall": overall,
        "by_category": by_cat,
        "run": run_result,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True,
                    choices=["bm25", "bm25-stemmer"] + list(DENSE_MODELS) + ["hybrid-e5",
                             "hybrid-granite-r1", "hybrid-granite-r2-311m",
                             "hybrid-shyngys-e5"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    # hybrid options
    ap.add_argument("--bm25-runs",   default=None)
    ap.add_argument("--dense-runs",  default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.system == "bm25":
        result = run_bm25("identity", top_k=args.top_k)
    elif args.system == "bm25-stemmer":
        result = run_bm25("kazakh", top_k=args.top_k)
    elif args.system in DENSE_MODELS:
        result = run_dense(args.system, top_k=args.top_k)
    elif args.system.startswith("hybrid"):
        if not args.bm25_runs or not args.dense_runs:
            ap.error("--bm25-runs and --dense-runs required for hybrid")
        result = run_hybrid(args.bm25_runs, args.dense_runs)
    else:
        raise ValueError(args.system)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {args.out}")

    overall = result["overall"]
    print(f"\nnDCG@10 overall: {overall.get('ndcg@10', overall.get('ndcg_10', '?')):.4f}")
    by_cat = result["by_category"]
    for cat, m in sorted(by_cat.items()):
        v = m.get("ndcg@10", m.get("ndcg_10", "?"))
        print(f"  {cat:20s}: {v:.4f}")


if __name__ == "__main__":
    main()
