"""
Парный бутстрэп для двух строк в таблице значимости PREPRINT2.md:

  1. BM25+stemmer → e5             (Wikipedia, n=300)
  2. e5 → kazakh-e5 (shyngys-e5)  (Wikipedia, n=300)

Требует: results/runs_dense_e5.json (получить прогоном через notebooks/e5_rerun_wiki.py)
Уже есть: results/runs_bm25_kazakh.json, results/runs_dense_shyngys.json

ЗАПУСК:
    python -m src.eval.compute_e5_pvalues
"""

from __future__ import annotations
import json
from ..eval import metrics
from ..queries import dataset

QRELS_PATH  = "data/queries/queries.jsonl"
BM25_RUNS   = "results/runs_bm25_kazakh.json"
E5_RUNS     = "results/runs_dense_e5.json"
SHYNGYS_RUNS = "results/runs_dense_shyngys.json"


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    queries = dataset.load_queries(QRELS_PATH)
    qrels   = dataset.qrels_from_queries(queries)
    cats    = dataset.categories_from_queries(queries)

    try:
        bm25_run    = load(BM25_RUNS)
        e5_run      = load(E5_RUNS)
        shyngys_run = load(SHYNGYS_RUNS)
    except FileNotFoundError as exc:
        print(f"Файл не найден: {exc}")
        print("Запусти notebooks/e5_rerun_wiki.py на Kaggle и спушь results/runs_dense_e5.json")
        return

    print(f"Загружено запросов: bm25={len(bm25_run)}, e5={len(e5_run)}, shyngys={len(shyngys_run)}")
    print(f"qrels: {len(qrels)} запросов\n")

    pairs = [
        ("BM25+stemmer → e5",      bm25_run,    e5_run),
        ("e5 → kazakh-e5",         e5_run,      shyngys_run),
    ]

    print(f"{'Сравнение':<30} {'Δ nDCG@10':>10} {'p-value':>9} {'Sig?':>6}")
    print("-" * 58)
    for name, run_a, run_b in pairs:
        delta, p, _ = metrics.paired_bootstrap(run_a, run_b, qrels,
                                               metric="ndcg", k=10, n_resamples=10_000)
        sig = "✓" if p < 0.05 else "—"
        print(f"{name:<30} {delta:>+10.4f} {p:>9.4f} {sig:>6}")

    print("\n--- По категориям ---")
    for name, run_a, run_b in pairs:
        print(f"\n{name}:")
        for cat in sorted(set(cats.values())):
            qids = {q for q, c in cats.items() if c == cat}
            sub  = {q: v for q, v in qrels.items() if q in qids}
            delta, p, _ = metrics.paired_bootstrap(run_a, run_b, sub,
                                                   metric="ndcg", k=10, n_resamples=10_000)
            sig = "✓" if p < 0.05 else "—"
            print(f"  {cat:<20} Δ={delta:>+7.4f}  p={p:.4f}  {sig}")

    print("\nВставь эти p-value в PREPRINT2.md, таблицу 4.3 (Wikipedia significance).")


if __name__ == "__main__":
    main()
