"""
Sprint 3 — финальный пересчёт метрик с устранением слабостей single-gold.

Считает для каждой системы три варианта метрик:
  1. passage  — классический single-gold (как в первых прогонах);
  2. article  — попадание в ЛЮБОЙ пассаж той же статьи, что gold
                (автоматическая верхняя граница поверх single-gold);
  3. pooled   — multi-gold по ручной разметке пула (если файл размечен).

Также считает 95% доверительные интервалы (bootstrap) для Hit@10.

ЗАПУСК:
    python -m src.eval.sprint3_rescore \
        --runs results/sprint3_runs.json \
        [--pool annotation/pool_relevance.tsv] \
        --out results/sprint3_final.json
"""
from __future__ import annotations
import argparse, csv, json, os, random

from . import metrics
from .sprint3_pool import load_queries

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def article(doc_id: str) -> str:
    return doc_id.rsplit("_p", 1)[0]


def load_pool_qrels(path: str) -> dict[str, set[str]]:
    """qid -> {doc_id, ...} из размеченного пула (RELEVANT == 1)."""
    qrels: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            val = str(row.get("RELEVANT", "")).strip()
            if val == "1":
                qrels.setdefault(row["qid"], set()).add(row["doc_id"])
    return qrels


def hit10_ci(run: dict, qrels: dict, n_boot: int = 2000, seed: int = 13):
    """Точечная оценка и 95% bootstrap-CI для Hit@10."""
    scores = [metrics.hit_at_k(run[q], qrels[q], 10)
              for q in run if q in qrels]
    if not scores:
        return 0.0, (0.0, 0.0)
    rng = random.Random(seed)
    boots = []
    n = len(scores)
    for _ in range(n_boot):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    return (sum(scores) / n,
            (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]))


def evaluate_variant(runs: dict, qrels: dict) -> dict:
    out = {}
    for system, run in runs.items():
        m = metrics.evaluate_run(run, qrels,
                                 metrics=("hit", "mrr", "ndcg"), ks=(1, 5, 10))
        point, (lo, hi) = hit10_ci(run, qrels)
        m["hit@10_ci95"] = [round(lo, 3), round(hi, 3)]
        out[system] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--pool", default=None,
                    help="размеченный annotation/pool_relevance.tsv")
    ap.add_argument("--out", default="results/sprint3_final.json")
    args = ap.parse_args()

    with open(args.runs, encoding="utf-8") as f:
        runs = json.load(f)

    queries = load_queries()
    qrels_passage = {qid: {q["gold_doc_id"]} for qid, q in queries.items()}

    result = {
        "n_queries": len(queries),
        "systems": sorted(runs.keys()),
        "variants": {},
    }

    # --- 1. passage-level (single gold) ---
    result["variants"]["passage"] = evaluate_variant(runs, qrels_passage)

    # --- 2. article-level: выдача и qrels схлопываются до статей ---
    runs_art = {
        s: {qid: list(dict.fromkeys(article(d) for d in docs))
            for qid, docs in run.items()}
        for s, run in runs.items()
    }
    qrels_art = {qid: {article(g) for g in rel}
                 for qid, rel in qrels_passage.items()}
    result["variants"]["article"] = evaluate_variant(runs_art, qrels_art)

    # --- 3. pooled multi-gold (если разметка готова) ---
    if args.pool and os.path.exists(args.pool):
        qrels_pool = load_pool_qrels(args.pool)
        n_marked = sum(len(v) for v in qrels_pool.values())
        if qrels_pool:
            result["variants"]["pooled"] = evaluate_variant(runs, qrels_pool)
            result["pooled_relevant_total"] = n_marked

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # --- печать ---
    for variant, table in result["variants"].items():
        print(f"\n=== {variant} (n={len(queries)}) ===")
        hdr = f"{'система':<22} {'Hit@1':>6} {'Hit@10':>7} {'CI95':>16} {'nDCG@10':>8}"
        print(hdr)
        for system in sorted(table):
            m = table[system]
            ci = m["hit@10_ci95"]
            print(f"{system:<22} {m['hit@1']:>6.3f} {m['hit@10']:>7.3f} "
                  f"[{ci[0]:.3f},{ci[1]:.3f}]  {m['ndcg@10']:>8.3f}")
    print(f"\nСохранено → {args.out}")


if __name__ == "__main__":
    main()
