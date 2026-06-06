"""
Сравнение результатов «До/После»: два JSON от run_benchmark -> таблица с
дельтой и приростом по категориям + (опционально) столбчатый график.

ЗАПУСК:
    python -m src.eval.compare \
        --before results/bm25_identity.json \
        --after  results/bm25_kazakh.json \
        --chart results/before_after.png   # график, если есть matplotlib
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

METRICS = ["recall@1", "recall@5", "mrr@10", "ndcg@10"]


def _load(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rows(before: Dict, after: Dict) -> List[Tuple[str, str, float, float]]:
    """[(категория, метрика, до, после), ...] по всем категориям и ВСЕ."""
    rows = []
    cats = ["ВСЕ"] + sorted(set(before.get("by_category", {})) | set(after.get("by_category", {})))
    for cat in cats:
        b = before["overall"] if cat == "ВСЕ" else before.get("by_category", {}).get(cat, {})
        a = after["overall"] if cat == "ВСЕ" else after.get("by_category", {}).get(cat, {})
        for m in METRICS:
            rows.append((cat, m, b.get(m, 0.0), a.get(m, 0.0)))
    return rows


def _pct(before: float, after: float) -> str:
    if before == 0:
        return "—" if after == 0 else "+∞"
    return f"{(after - before) / before * 100:+.0f}%"


def format_table(before: Dict, after: Dict) -> str:
    lines = []
    lines.append(f"Сравнение: До (стеммер={before.get('stemmer')}) → "
                 f"После (стеммер={after.get('stemmer')})")
    lines.append(f"Корпус: {after.get('n_passages')} пассажей, "
                 f"{after.get('n_queries')} запросов\n")
    header = f"{'категория':<16} {'метрика':<10} {'До':>7} {'После':>7} {'Δ':>7} {'прирост':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    last_cat = None
    for cat, m, b, a in _rows(before, after):
        cat_label = cat if cat != last_cat else ""
        last_cat = cat
        lines.append(f"{cat_label:<16} {m:<10} {b:>7.3f} {a:>7.3f} {a-b:>+7.3f} {_pct(b, a):>8}")
    return "\n".join(lines)


def make_chart(before: Dict, after: Dict, path: str, metric: str = "ndcg@10") -> bool:
    """Столбчатый график До/После по категориям для одной метрики."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен — график пропущен (pip install matplotlib)")
        return False

    cats = ["ВСЕ"] + sorted(set(before.get("by_category", {})) | set(after.get("by_category", {})))
    bvals, avals = [], []
    for cat in cats:
        b = before["overall"] if cat == "ВСЕ" else before["by_category"].get(cat, {})
        a = after["overall"] if cat == "ВСЕ" else after["by_category"].get(cat, {})
        bvals.append(b.get(metric, 0.0))
        avals.append(a.get(metric, 0.0))

    x = range(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w / 2 for i in x], bvals, w, label="До (BM25)", color="#bbbbbb")
    ax.bar([i + w / 2 for i in x], avals, w, label="После (BM25 + стеммер)", color="#2b8cbe")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats)
    ax.set_ylabel(metric)
    ax.set_title(f"Казахский поиск: {metric} До/После стеммера")
    ax.legend()
    ax.set_ylim(0, 1)
    for i, (bv, av) in enumerate(zip(bvals, avals)):
        ax.text(i - w / 2, bv + 0.02, f"{bv:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, av + 0.02, f"{av:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"График сохранён → {path}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Сравнение результатов До/После")
    ap.add_argument("--before", default="results/bm25_identity.json")
    ap.add_argument("--after", default="results/bm25_kazakh.json")
    ap.add_argument("--chart", default=None, help="путь для PNG-графика (опционально)")
    ap.add_argument("--metric", default="ndcg@10", help="метрика для графика")
    args = ap.parse_args()

    before, after = _load(args.before), _load(args.after)
    print(format_table(before, after))
    if args.chart:
        make_chart(before, after, args.chart, args.metric)


if __name__ == "__main__":
    main()
