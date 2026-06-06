"""
График сравнения 3 систем: BM25+стеммер / Granite-278M / Hybrid (RRF).
nDCG@10 по 4 срезам, с отметкой значимости гибрида над лучшим каналом.

Читает results/hybrid_kazakh.json (его пишет run_hybrid).

Usage:
    python -m src.eval.chart_hybrid --out results/systems_hybrid_ndcg.png
"""
from __future__ import annotations

import argparse
import json

CATS = ["inflected", "natural", "vocabulary-gap", "ALL"]
SERIES = [
    ("bm25_stemmer", "BM25+Stemmer", "#1565c0"),
    ("granite", "Dense Granite-278M", "#e65100"),
    ("hybrid", "Hybrid (RRF)", "#2e7d32"),
]


def _ndcg10(block: dict, cat: str) -> float:
    if cat == "ALL":
        return block["overall"]["ndcg@10"]
    return block["by_category"].get(cat, {}).get("ndcg@10", 0.0)


def build_chart(data: dict, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    blocks = data["metrics"]
    sig = data.get("significance", {}).get("hybrid_vs_bm25", {})
    k = data.get("rrf_k", 60)
    n = data.get("n_queries", "?")

    x = np.arange(len(CATS))
    width = 0.26
    offsets = np.linspace(-1, 1, len(SERIES)) * width

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for (key, label, color), off in zip(SERIES, offsets):
        vals = [_ndcg10(blocks[key], c) for c in CATS]
        bars = ax.bar(x + off, vals, width, label=label, color=color, alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # отметка значимости гибрида (vs BM25) над группой ALL/категориями
    for i, cat in enumerate(CATS):
        s = sig.get(cat if cat != "ALL" else "ALL")
        if not s:
            continue
        p = s["p"]
        mark = "n.s." if p >= 0.05 else (f"p<0.001" if p < 0.001 else f"p={p:.3f}")
        ax.text(x[i], 1.0, mark, ha="center", va="bottom", fontsize=7.5,
                color="#2e7d32" if p < 0.05 else "#777")

    cat_labels = ["inflected\n(morphology)", "natural\n(standard)",
                  "vocab-gap\n(synonyms)", "ALL"]
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel("nDCG@10", fontsize=11)
    ax.set_title(
        f"Hybrid retrieval (RRF, k={k}): BM25+Stemmer ⊕ Granite-278M\n"
        f"Kazakh · 8 370 passages · n={n} queries · marks = Hybrid vs BM25 significance",
        fontsize=11,
    )
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/hybrid_kazakh.json")
    ap.add_argument("--out", default="results/systems_hybrid_ndcg.png")
    args = ap.parse_args()
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    build_chart(data, args.out)


if __name__ == "__main__":
    main()
