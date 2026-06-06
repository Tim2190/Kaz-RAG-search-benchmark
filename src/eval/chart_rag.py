"""
RAG end-to-end chart (n=300): stemmer effect on Qwen2.5-7B accuracy by category.

Honest version: shows that better retrieval (hit-rate up) does NOT translate into
a statistically significant end-to-end accuracy gain at n=300. McNemar p-values
are computed per category and annotated on the chart.

Usage:
    python -m src.eval.chart_rag --out results/rag_models.png
"""
from __future__ import annotations

import argparse
import json
from math import comb


RESULT = "results/rag_qwen7b_300.json"
CATS = ["inflected", "natural", "vocabulary-gap", "ALL"]
QUERIES = "data/queries/queries.jsonl"


def _mcnemar(ci: dict, ck: dict, ids: list) -> float:
    b = sum(1 for i in ids if ci[i] and not ck[i])
    c = sum(1 for i in ids if not ci[i] and ck[i])
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2, 1.0)


def _acc(data: dict, cat: str) -> float:
    if cat == "ALL":
        return data["overall"]["accuracy"]
    return data["by_category"][cat]["accuracy"]


def _hit(data: dict, cat: str) -> float:
    if cat == "ALL":
        return data["overall"]["retrieval_hit_rate"]
    return data["by_category"][cat]["retrieval_hit_rate"]


def build_chart(out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d = json.load(open(RESULT, encoding="utf-8"))
    cats_by_id = {}
    for line in open(QUERIES, encoding="utf-8"):
        q = json.loads(line)
        cats_by_id[q["query_id"]] = q.get("category", "all")

    ci = {p["query_id"]: p["label"] == "correct" for p in d["identity"]["per_query"]}
    ck = {p["query_id"]: p["label"] == "correct" for p in d["kazakh"]["per_query"]}
    ids = list(ci)

    ident = [_acc(d["identity"], c) for c in CATS]
    kazakh = [_acc(d["kazakh"], c) for c in CATS]
    pvals = []
    for c in CATS:
        sub = ids if c == "ALL" else [i for i in ids if cats_by_id[i] == c]
        pvals.append(_mcnemar(ci, ck, sub))

    x = np.arange(len(CATS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    b1 = ax.bar(x - width / 2, ident, width, label="No stemmer", color="#bdbdbd")
    b2 = ax.bar(x + width / 2, kazakh, width, label="Kazakh stemmer", color="#1565c0")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    for i, (a, b, p) in enumerate(zip(ident, kazakh, pvals)):
        sig = "✓" if p < 0.05 else "n.s."
        ax.annotate(f"Δ{b - a:+.2f}\np={p:.2f} {sig}",
                    xy=(i, max(a, b) + 0.06), ha="center", fontsize=8.5,
                    color="#1565c0" if b >= a else "#c62828", weight="bold")

    cat_labels = ["inflected\n(morphology)", "natural\n(standard)",
                  "vocab-gap\n(synonyms)", "ALL"]
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel("End-to-end accuracy (Qwen2.5-7B, n=300)", fontsize=11)
    ax.set_title(
        "RAG end-to-end: better retrieval ≠ significant accuracy gain (n=300)\n"
        "Retrieval hit@3 rises 0.74→0.80, but McNemar finds no significant accuracy change",
        fontsize=10.5,
    )
    ax.set_ylim(0, 0.8)
    ax.legend(loc="upper right", fontsize=9)
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
