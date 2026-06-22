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
    # Jina v3: task-based encoding, no string prefixes
    "jina-v3":        ("jinaai/jina-embeddings-v3", "", ""),
    # Nomic v1.5: string prefixes (e5-style) + trust_remote_code
    "nomic-v1.5":     ("nomic-ai/nomic-embed-text-v1.5",
                       "search_query: ", "search_document: "),
    # BGE-M3: no instruction prefix required (BAAI documentation)
    "bge-m3":         ("BAAI/bge-m3", "", ""),
    # Qwen3-Embedding-0.6B: LLM-based embedder, instruction prefix for queries
    "qwen3-0.6b":     ("Qwen/Qwen3-Embedding-0.6B",
                       "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
                       ""),
}

# API-based models handled outside DENSE_MODELS (different loader)
# key → Cohere model name
COHERE_MODELS: Dict[str, str] = {
    "cohere-v4": "embed-v4.0",
}

# Extra encode params: query_task/doc_task → task= in encode() (Jina);
# trust_remote_code → passed to loader (Jina: AutoModel; Nomic: SentenceTransformer).
DENSE_MODEL_EXTRA: dict = {
    "jina-v3": {
        "query_task":        "retrieval.query",
        "doc_task":          "retrieval.passage",
        "trust_remote_code": True,
    },
    "nomic-v1.5": {
        "trust_remote_code": True,
    },
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
    qmap = {q["query_id"]: q["text"] for q in queries}

    extra = DENSE_MODEL_EXTRA.get(model_key, {})
    print(f"Dense encode ({hf_id})…")
    index = DenseIndex(
        model_name=hf_id,
        query_prefix=q_prefix,
        doc_prefix=p_prefix,
        query_task=extra.get("query_task"),
        doc_task=extra.get("doc_task"),
        trust_remote_code=extra.get("trust_remote_code", False),
    )
    index.index(corpus, show_progress=True)
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


# ── Cohere API dense ──────────────────────────────────────────────────────────

def run_cohere_system(model_key: str, top_k: int = 100) -> Dict:
    import numpy as np, json as _json, os as _os
    from ..retrieval.cohere_index import CohereIndex

    cohere_model = COHERE_MODELS[model_key]
    corpus  = _load_corpus()
    queries = _load_queries()
    qrels   = _load_qrels()
    qmap    = {q["query_id"]: q["text"] for q in queries}

    cache_prefix = _os.path.join(ROOT, f"results/akorda/emb_{model_key}")
    cache_npy  = cache_prefix + ".npy"
    cache_ids  = cache_prefix + ".ids.json"

    index = CohereIndex(model_name=cohere_model)
    if _os.path.exists(cache_npy) and _os.path.exists(cache_ids):
        ids = _json.load(open(cache_ids, encoding="utf-8"))
        if ids == [d for d, _ in corpus]:
            print(f"Эмбеддинги из кэша: {cache_npy}")
            index.set_embeddings(ids, np.load(cache_npy))
    if index.matrix is None:
        index.index(corpus)
        _os.makedirs(_os.path.dirname(cache_prefix) or ".", exist_ok=True)
        np.save(cache_npy, index.matrix)
        _json.dump(index.doc_ids, open(cache_ids, "w", encoding="utf-8"))

    run_result = index.run(qmap, top_k=top_k)
    overall, by_cat = _metrics_block(run_result, qrels, queries)
    return {
        "system": model_key,
        "dataset": "akorda",
        "model": f"cohere/{cohere_model}",
        "n_passages": len(corpus),
        "n_queries": len(queries),
        "overall": overall,
        "by_category": by_cat,
        "run": run_result,
    }


# ── Hybrid RRF ────────────────────────────────────────────────────────────────
#
# Зеркалит протокол Wiki-гибрида (src/eval/run_hybrid.py): сливает BM25+стеммер ⊕
# dense через каноническую reciprocal_rank_fusion, считает метрики обоих каналов +
# гибрида, значимость (paired bootstrap), sensitivity по k и pre-registered
# критерии успеха. Единственная адаптация под Akorda — семантический срез
# называется low_overlap (на Wiki это была vocabulary-gap).

_SEMANTIC_CAT = "low_overlap"  # аналог vocabulary-gap на Wiki


def _metrics_block_full(run_result, qrels, cats) -> Dict:
    from ..eval import metrics
    overall = metrics.evaluate_run(run_result, qrels,
                                   metrics=("recall", "mrr", "ndcg"), ks=(1, 5, 10))
    by_cat = {}
    for cat in sorted(set(cats.values())):
        qids = {q for q, c in cats.items() if c == cat}
        by_cat[cat] = metrics.evaluate_run(run_result, _subset_qrels(qrels, qids),
                                           metrics=("recall", "mrr", "ndcg"), ks=(1, 5, 10))
    return {"overall": overall, "by_category": by_cat}


def _ndcg10(block: Dict, group: str) -> float:
    if group == "ALL":
        return block["overall"]["ndcg@10"]
    return block["by_category"].get(group, {}).get("ndcg@10", 0.0)


def run_hybrid(bm25_runs_path: str, dense_runs_path: str,
               dense_label: str = "dense",
               k: int = RRF_K, resamples: int = 10000) -> Dict:
    from ..retrieval.hybrid import reciprocal_rank_fusion
    from ..eval import metrics

    with open(bm25_runs_path) as f:
        bm25_run = json.load(f)["run"]
    with open(dense_runs_path) as f:
        dense_run = json.load(f)["run"]

    queries = _load_queries()
    qrels = _load_qrels()
    cats = {q["query_id"]: q["category"] for q in queries}

    # PRE-FLIGHT: оба канала на одном наборе запросов
    qb, qd = set(bm25_run), set(dense_run)
    if qb != qd:
        only_b, only_d = qb - qd, qd - qb
        raise SystemExit(
            "PRE-FLIGHT FAILED: наборы запросов в каналах не совпадают.\n"
            f"  BM25: {len(qb)}, dense: {len(qd)}.\n"
            f"  только в BM25: {sorted(only_b)[:5]}\n"
            f"  только в dense: {sorted(only_d)[:5]}")

    # оцениваем только по запросам, реально присутствующим в каналах
    qrels = {q: rel for q, rel in qrels.items() if q in qb}
    cats = {q: c for q, c in cats.items() if q in qb}

    channels = {"bm25_stemmer": bm25_run, dense_label: dense_run}
    hybrid_run = reciprocal_rank_fusion(channels, k=k)

    blocks = {
        "bm25_stemmer": _metrics_block_full(bm25_run, qrels, cats),
        dense_label:    _metrics_block_full(dense_run, qrels, cats),
        "hybrid":       _metrics_block_full(hybrid_run, qrels, cats),
    }

    # значимость: Hybrid vs каждый канал, по срезам, ndcg@10
    groups = {"ALL": set(qrels)}
    for c in sorted(set(cats.values())):
        groups[c] = {q for q, cc in cats.items() if cc == c}
    sig = {"hybrid_vs_bm25": {}, "hybrid_vs_dense": {}}
    for gname, qids in groups.items():
        sub = _subset_qrels(qrels, qids)
        d_b, p_b, _ = metrics.paired_bootstrap(
            bm25_run, hybrid_run, sub, metric="ndcg", k=10, n_resamples=resamples)
        d_d, p_d, _ = metrics.paired_bootstrap(
            dense_run, hybrid_run, sub, metric="ndcg", k=10, n_resamples=resamples)
        sig["hybrid_vs_bm25"][gname]  = {"delta": d_b, "p": p_b}
        sig["hybrid_vs_dense"][gname] = {"delta": d_d, "p": p_d}

    # sensitivity по k (для честности, не для подбора лучшего)
    sweep_table = {}
    for kk in (10, 30, 60, 100):
        run_kk = reciprocal_rank_fusion(channels, k=kk)
        blk = _metrics_block_full(run_kk, qrels, cats)
        sweep_table[kk] = {g: _ndcg10(blk, g) for g in groups}

    # pre-registered критерии (vocab-gap → low_overlap)
    all_hyb = _ndcg10(blocks["hybrid"], "ALL")
    all_max = max(_ndcg10(blocks["bm25_stemmer"], "ALL"),
                  _ndcg10(blocks[dense_label], "ALL"))
    sem_hyb  = _ndcg10(blocks["hybrid"], _SEMANTIC_CAT)
    sem_bm25 = _ndcg10(blocks["bm25_stemmer"], _SEMANTIC_CAT)
    criteria = {
        "all_hybrid_ge_max_channel": all_hyb >= all_max,
        "low_overlap_hybrid_ge_bm25": sem_hyb >= sem_bm25 - 1e-9,
        "all_hybrid_ndcg10": all_hyb,
        "all_max_channel_ndcg10": all_max,
        "low_overlap_hybrid_ndcg10": sem_hyb,
        "low_overlap_bm25_ndcg10": sem_bm25,
    }

    return {
        "system": "hybrid:rrf",
        "dataset": "akorda",
        "channels": ["bm25_stemmer", dense_label],
        "rrf_k": k,
        "n_queries": len(qrels),
        "metrics": blocks,
        "significance": sig,
        "k_sweep": sweep_table,
        "criteria": criteria,
        "run": hybrid_run,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True,
                    choices=["bm25", "bm25-stemmer"] + list(DENSE_MODELS)
                            + list(COHERE_MODELS) + ["hybrid-e5",
                             "hybrid-granite-r1", "hybrid-granite-r2-97m",
                             "hybrid-granite-r2-311m", "hybrid-shyngys-e5",
                             "hybrid-jina-v3", "hybrid-nomic-v1.5",
                             "hybrid-bge-m3", "hybrid-qwen3-0.6b",
                             "hybrid-cohere-v4"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    # hybrid options
    ap.add_argument("--bm25-runs",   default=None)
    ap.add_argument("--dense-runs",  default=None)
    ap.add_argument("--dense-label", default="dense",
                    help="имя dense-канала в выводе гибрида (напр. e5, granite-r1)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.system == "bm25":
        result = run_bm25("identity", top_k=args.top_k)
    elif args.system == "bm25-stemmer":
        result = run_bm25("kazakh", top_k=args.top_k)
    elif args.system in DENSE_MODELS:
        result = run_dense(args.system, top_k=args.top_k)
    elif args.system in COHERE_MODELS:
        result = run_cohere_system(args.system, top_k=args.top_k)
    elif args.system.startswith("hybrid"):
        if not args.bm25_runs or not args.dense_runs:
            ap.error("--bm25-runs and --dense-runs required for hybrid")
        result = run_hybrid(args.bm25_runs, args.dense_runs,
                            dense_label=args.dense_label)
    else:
        raise ValueError(args.system)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {args.out}")

    if result.get("system") == "hybrid:rrf":
        # богатый формат: печатаем nDCG@10 по 3 системам + критерии
        dense_label = result["channels"][1]
        b = result["metrics"]
        groups = ["factoid", "paraphrase", _SEMANTIC_CAT, "ALL"]
        print(f"\nГибрид BM25+стеммер ⊕ {dense_label} (RRF k={result['rrf_k']}), "
              f"n={result['n_queries']}")
        print(f"{'система':<16}" + "".join(f"{g[:11]:>13}" for g in groups))
        for sysname in ("bm25_stemmer", dense_label, "hybrid"):
            row = f"{sysname:<16}" + "".join(
                f"{_ndcg10(b[sysname], g):>13.4f}" for g in groups)
            print(row)
        c = result["criteria"]
        print(f"\nКритерии: ALL hybrid {c['all_hybrid_ndcg10']:.4f} ≥ "
              f"max(каналов) {c['all_max_channel_ndcg10']:.4f} → "
              f"{'ДА' if c['all_hybrid_ge_max_channel'] else 'НЕТ'}")
        print(f"          low_overlap hybrid {c['low_overlap_hybrid_ndcg10']:.4f} ≥ "
              f"BM25 {c['low_overlap_bm25_ndcg10']:.4f} → "
              f"{'ДА' if c['low_overlap_hybrid_ge_bm25'] else 'НЕТ'}")
    else:
        overall = result["overall"]
        print(f"\nnDCG@10 overall: {overall.get('ndcg@10', overall.get('ndcg_10', '?')):.4f}")
        by_cat = result["by_category"]
        for cat, m in sorted(by_cat.items()):
            v = m.get("ndcg@10", m.get("ndcg_10", "?"))
            print(f"  {cat:20s}: {v:.4f}")


if __name__ == "__main__":
    main()
