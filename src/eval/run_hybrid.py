"""
Гибридный ретривер: BM25+стеммер ⊕ Granite-278M через Reciprocal Rank Fusion.

Гипотеза (falsifiable): гибрид даёт более устойчивый retrieval по ВСЕМ трём
категориям, чем любой канал по отдельности. Логика — лексический канал
закрывает провал Granite на синонимах (vocabulary-gap), Granite вытягивает
там, где dense силён.

Критерий успеха (pre-registered):
  1) nDCG@10(Hybrid) ≥ max(BM25+стеммер, Granite) на срезе ALL;
  2) на vocabulary-gap гибрид не хуже BM25 (Granite не утягивает вниз).

Берёт УЖЕ ПОСЧИТАННЫЕ per-query ранжировки обоих каналов (одинаковый набор
запросов!) и только сливает их — retrieval не перепрогоняется. Метрики и
значимость считаются существующим src/eval (ничего не переписываем).

ЗАПУСК (CPU, без сети/GPU):
    python -m src.eval.run_hybrid \
        --bm25-runs    results/runs_bm25_kazakh.json \
        --granite-runs results/runs_dense_granite.json \
        --out results/hybrid_kazakh.json

PRE-FLIGHT: оба дампа ранжировок должны быть на ОДНОМ наборе запросов
(целевой — все 300). Скрипт проверяет совпадение query_id и падает с ошибкой,
если наборы разные, — чтобы не смешать n=300 с n=60.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Set

from ..retrieval.hybrid import reciprocal_rank_fusion, load_run, DEFAULT_K
from ..eval import metrics
from ..queries import dataset

CATS = ["inflected", "natural", "vocabulary-gap"]
METRIC_SET = ("recall", "mrr", "ndcg")
KS = (1, 5, 10)


def _subset(qrels: Dict[str, Set[str]], qids: Set[str]) -> Dict[str, Set[str]]:
    return {q: rel for q, rel in qrels.items() if q in qids}


def _metrics_block(run: Dict[str, List[str]], qrels, cats) -> Dict:
    overall = metrics.evaluate_run(run, qrels, metrics=METRIC_SET, ks=KS)
    by_cat = {}
    for cat in sorted(set(cats.values())):
        qids = {q for q, c in cats.items() if c == cat}
        by_cat[cat] = metrics.evaluate_run(run, _subset(qrels, qids),
                                           metrics=METRIC_SET, ks=KS)
    return {"overall": overall, "by_category": by_cat}


def _groups(qrels, cats) -> Dict[str, Set[str]]:
    g = {"ALL": set(qrels)}
    for c in sorted(set(cats.values())):
        g[c] = {q for q, cc in cats.items() if cc == c}
    return g


def _ndcg10(block: Dict, group: str) -> float:
    if group == "ALL":
        return block["overall"]["ndcg@10"]
    return block["by_category"].get(group, {}).get("ndcg@10", 0.0)


def run(bm25_runs: str, granite_runs: str, queries_path: str,
        k: int = DEFAULT_K, sweep: List[int] = None,
        resamples: int = 10000) -> Dict:
    bm25 = load_run(bm25_runs)
    granite = load_run(granite_runs)

    # --- PRE-FLIGHT: один и тот же набор запросов в обоих каналах ---
    qb, qg = set(bm25), set(granite)
    if qb != qg:
        only_b, only_g = qb - qg, qg - qb
        raise SystemExit(
            "PRE-FLIGHT FAILED: наборы запросов в каналах не совпадают.\n"
            f"  BM25: {len(qb)} запросов, Granite: {len(qg)} запросов.\n"
            f"  только в BM25: {sorted(only_b)[:5]}{'…' if len(only_b) > 5 else ''}\n"
            f"  только в Granite: {sorted(only_g)[:5]}{'…' if len(only_g) > 5 else ''}\n"
            "Выровняй оба канала на единый набор (целевой — все 300) и не "
            "смешивай n=300 с n=60."
        )

    queries = dataset.load_queries(queries_path)
    qrels = dataset.qrels_from_queries(queries)
    cats = dataset.categories_from_queries(queries)
    # оцениваем только по запросам, реально присутствующим в каналах
    qrels = {q: rel for q, rel in qrels.items() if q in qb}
    cats = {q: c for q, c in cats.items() if q in qb}

    channels = {"bm25_stemmer": bm25, "granite": granite}

    # --- основной гибрид при pre-registered k ---
    hybrid_run = reciprocal_rank_fusion(channels, k=k)

    blocks = {
        "bm25_stemmer": _metrics_block(bm25, qrels, cats),
        "granite": _metrics_block(granite, qrels, cats),
        "hybrid": _metrics_block(hybrid_run, qrels, cats),
    }

    # --- значимость: Hybrid vs каждый канал, по срезам, ndcg@10 ---
    groups = _groups(qrels, cats)
    sig = {"hybrid_vs_bm25": {}, "hybrid_vs_granite": {}}
    for gname, qids in groups.items():
        sub = _subset(qrels, qids)
        d_b, p_b, _ = metrics.paired_bootstrap(
            bm25, hybrid_run, sub, metric="ndcg", k=10, n_resamples=resamples)
        d_g, p_g, _ = metrics.paired_bootstrap(
            granite, hybrid_run, sub, metric="ndcg", k=10, n_resamples=resamples)
        sig["hybrid_vs_bm25"][gname] = {"delta": d_b, "p": p_b}
        sig["hybrid_vs_granite"][gname] = {"delta": d_g, "p": p_g}

    # --- sensitivity-анализ по k (НЕ для выбора лучшего, а для честности) ---
    sweep = sweep or [10, 30, 60, 100]
    sweep_table = {}
    for kk in sweep:
        run_kk = reciprocal_rank_fusion(channels, k=kk)
        blk = _metrics_block(run_kk, qrels, cats)
        sweep_table[kk] = {g: _ndcg10(blk, g) for g in groups}

    # --- проверка pre-registered критериев успеха ---
    all_hyb = _ndcg10(blocks["hybrid"], "ALL")
    all_max = max(_ndcg10(blocks["bm25_stemmer"], "ALL"),
                  _ndcg10(blocks["granite"], "ALL"))
    vg_hyb = _ndcg10(blocks["hybrid"], "vocabulary-gap")
    vg_bm25 = _ndcg10(blocks["bm25_stemmer"], "vocabulary-gap")
    criteria = {
        "all_hybrid_ge_max_channel": all_hyb >= all_max,
        "vocab_gap_hybrid_ge_bm25": vg_hyb >= vg_bm25 - 1e-9,
        "all_hybrid_ndcg10": all_hyb,
        "all_max_channel_ndcg10": all_max,
        "vocab_gap_hybrid_ndcg10": vg_hyb,
        "vocab_gap_bm25_ndcg10": vg_bm25,
    }

    return {
        "system": "hybrid:rrf",
        "channels": ["bm25_stemmer", "granite"],
        "rrf_k": k,
        "n_queries": len(qrels),
        "metrics": blocks,
        "significance": sig,
        "k_sweep": sweep_table,
        "criteria": criteria,
        "run": hybrid_run,
    }


# ---------- печать ----------

def _fmt(result: Dict) -> str:
    groups = ["inflected", "natural", "vocabulary-gap", "ALL"]
    b = result["metrics"]
    lines = [
        f"\n=== Гибрид BM25+стеммер ⊕ Granite (RRF k={result['rrf_k']}), "
        f"n={result['n_queries']} ===",
        f"\nnDCG@10 — 3 системы × срезы:",
        f"{'система':<16}" + "".join(f"{g[:12]:>14}" for g in groups),
    ]
    for sysname, label in [("bm25_stemmer", "BM25+стеммер"),
                           ("granite", "Granite-278M"),
                           ("hybrid", "Hybrid (RRF)")]:
        row = f"{label:<16}"
        for g in groups:
            row += f"{_ndcg10(b[sysname], g):>14.3f}"
        lines.append(row)

    lines.append("\nЗначимость (paired bootstrap, ndcg@10):")
    lines.append(f"{'срез':<16}{'Δ vs BM25':>12}{'p':>9}{'Δ vs Granite':>14}{'p':>9}")
    for g in groups:
        sb = result["significance"]["hybrid_vs_bm25"][g]
        sg = result["significance"]["hybrid_vs_granite"][g]
        lines.append(f"{g:<16}{sb['delta']:>+12.3f}{sb['p']:>9.4f}"
                     f"{sg['delta']:>+14.3f}{sg['p']:>9.4f}")

    lines.append("\nSensitivity по k (ndcg@10, ALL не должен скакать):")
    lines.append(f"{'k':<6}" + "".join(f"{g[:12]:>14}" for g in groups))
    for kk, vals in result["k_sweep"].items():
        lines.append(f"{kk:<6}" + "".join(f"{vals[g]:>14.3f}" for g in groups))

    c = result["criteria"]
    lines.append("\nPre-registered критерии успеха:")
    lines.append(f"  ALL: hybrid {c['all_hybrid_ndcg10']:.3f} ≥ "
                 f"max(каналов) {c['all_max_channel_ndcg10']:.3f}  → "
                 f"{'✅ ДА' if c['all_hybrid_ge_max_channel'] else '❌ НЕТ'}")
    lines.append(f"  vocab-gap: hybrid {c['vocab_gap_hybrid_ndcg10']:.3f} ≥ "
                 f"BM25 {c['vocab_gap_bm25_ndcg10']:.3f}  → "
                 f"{'✅ ДА' if c['vocab_gap_hybrid_ge_bm25'] else '❌ НЕТ'}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="RRF-гибрид BM25+стеммер ⊕ Granite")
    ap.add_argument("--bm25-runs", default="results/runs_bm25_kazakh.json")
    ap.add_argument("--granite-runs", default="results/runs_dense_granite.json")
    ap.add_argument("--queries", default="data/queries/queries.jsonl")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="константа RRF (pre-registered=60)")
    ap.add_argument("--sweep", default="10,30,60,100",
                    help="значения k для sensitivity-анализа")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--out", default="results/hybrid_kazakh.json")
    args = ap.parse_args()

    sweep = [int(x) for x in args.sweep.split(",") if x.strip()]
    result = run(args.bm25_runs, args.granite_runs, args.queries,
                 k=args.k, sweep=sweep, resamples=args.resamples)
    print(_fmt(result))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nРезультат сохранён → {args.out}")


if __name__ == "__main__":
    main()
