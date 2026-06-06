"""
RAG end-to-end chart: stemmer effect on accuracy across 3 LLMs.

Usage:
    python -m src.eval.chart_rag --out results/rag_models.png
"""
from __future__ import annotations

import argparse
import json


MODELS = [
    ("results/rag_granite.json",   "Granite-2B",  "#9e9e9e"),
    ("results/rag_granite8b.json", "Granite-8B",  "#e65100"),
    ("results/rag_qwen7b.json",    "Qwen2.5-7B",  "#2e7d32"),
]


def build_chart(out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels, ident, kazakh = [], [], []
    for path, label, _ in MODELS:
        d = json.load(open(path, encoding="utf-8"))
        labels.append(label)
        ident.append(d["identity"]["overall"]["accuracy"])
        kazakh.append(d["kazakh"]["overall"]["accuracy"])

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - width / 2, ident, width, label="No stemmer (retrieval hit 0.467)",
                color="#bdbdbd")
    b2 = ax.bar(x + width / 2, kazakh, width, label="Kazakh stemmer (retrieval hit 0.667)",
                color="#1565c0")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    # draw the delta arrows
    for i, (a, b) in enumerate(zip(ident, kazakh)):
        ax.annotate(f"+{b - a:.2f}", xy=(i, max(a, b) + 0.05),
                    ha="center", fontsize=9, color="#1565c0", weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("End-to-end accuracy (60 questions)", fontsize=11)
    ax.set_title(
        "RAG: Kazakh stemmer improves end-to-end accuracy\n"
        "Effect grows with model competence (same retrieval hit-rate gain 0.47→0.67)",
        fontsize=11,
    )
    ax.set_ylim(0, 0.7)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/rag_models.png")
    build_chart(ap.parse_args().out)


if __name__ == "__main__":
    main()
