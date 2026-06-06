"""
Generate a multi-system comparison chart (ndcg@10 by category).

Usage:
    python -m src.eval.chart_all --out results/systems_ndcg.png
"""
from __future__ import annotations

import argparse
import json
import os


SYSTEMS = [
    ("results/bm25_identity.json",      "BM25",           "#9e9e9e"),
    ("results/bm25_kazakh.json",         "BM25+Stemmer",   "#1565c0"),
    ("results/dense_labse_300.json",     "Dense LaBSE",    "#6a1b9a"),
    ("results/dense_granite_300.json",   "Dense Granite",  "#e65100"),
    ("results/dense_e5_300.json",        "Dense E5",       "#2e7d32"),
]

CATS = ["inflected", "natural", "vocabulary-gap", "ALL"]


def _ndcg10(data: dict, cat: str) -> float:
    if cat == "ALL":
        return data["overall"]["ndcg@10"]
    return data.get("by_category", {}).get(cat, {}).get("ndcg@10", 0.0)


def build_chart(out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    records = []
    for path, label, color in SYSTEMS:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        records.append((label, color, [_ndcg10(d, c) for c in CATS]))

    n_cats = len(CATS)
    n_sys = len(records)
    x = np.arange(n_cats)
    width = 0.14
    offsets = np.linspace(-(n_sys - 1) / 2, (n_sys - 1) / 2, n_sys) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for (label, color, vals), offset in zip(records, offsets):
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    cat_labels = ["inflected\n(morphology)", "natural\n(standard)", "vocab-gap\n(synonyms)", "ALL"]
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel("nDCG@10", fontsize=11)
    ax.set_title(
        "Kazakh Information Retrieval: 5-system comparison (nDCG@10)\n"
        "Corpus: 8 370 passages · n=300 queries (consistent across all systems)",
        fontsize=11,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=9,
              frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/systems_ndcg.png")
    args = ap.parse_args()
    build_chart(args.out)


if __name__ == "__main__":
    main()
