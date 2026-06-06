"""
Генератор канонических markdown-таблиц RESULTS.md из n=300 JSON-артефактов.

Единый источник правды: все таблицы по 5 системам + детальные per-model
печатаются ровно из закоммиченных результатов, чтобы исключить расхождения
от ручной правки. Запуск без сети/GPU.

    python -m src.eval.gen_results_tables
"""
from __future__ import annotations

import json

# (путь, отображаемое имя) — порядок как в таблицах
SYSTEMS = [
    ("results/bm25_identity.json",    "BM25"),
    ("results/bm25_kazakh.json",      "BM25 + Stemmer"),
    ("results/dense_labse_300.json",  "Dense LaBSE"),
    ("results/dense_granite_300.json","Dense Granite"),
    ("results/dense_e5_300.json",     "Dense E5"),
]
CATS = ["inflected", "natural", "vocabulary-gap"]


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cell(block, cat, metric):
    src = block["overall"] if cat == "ALL" else block["by_category"][cat]
    return src[metric]


def main_result_table(data):
    """5 систем × 4 среза, nDCG@10. Жирным — лучшее в колонке."""
    cols = ["inflected", "natural", "vocabulary-gap", "ALL"]
    # лучшее значение в каждой колонке
    best = {c: max(_cell(d, c, "ndcg@10") for _, _, d in data) for c in cols}
    lines = [
        "| System | inflected | natural | vocab-gap | **ALL** |",
        "|--------|----------:|--------:|----------:|--------:|",
    ]
    for _, name, d in data:
        cells = []
        for c in cols:
            v = _cell(d, c, "ndcg@10")
            s = f"{v:.3f}"
            if abs(v - best[c]) < 1e-9:
                s = f"**{s}**"
            cells.append(s)
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    return "\n".join(lines)


def detailed_table(name, d):
    """recall@{1,5,10}, mrr@10, ndcg@10 по категориям + ALL для одной системы."""
    lines = [
        f"### {name} — n={d['n_queries']}",
        "",
        "| category | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 |",
        "|----------|--------:|---------:|----------:|-------:|--------:|",
    ]
    for cat in CATS:
        b = d["by_category"][cat]
        lines.append(
            f"| {cat} | {b['recall@1']:.2f} | {b['recall@5']:.2f} | "
            f"{b['recall@10']:.2f} | {b['mrr@10']:.3f} | {b['ndcg@10']:.3f} |"
        )
    o = d["overall"]
    lines.append(
        f"| **ALL** | {o['recall@1']:.2f} | {o['recall@5']:.2f} | "
        f"{o['recall@10']:.2f} | {o['mrr@10']:.3f} | {o['ndcg@10']:.3f} |"
    )
    return "\n".join(lines)


def main():
    data = [(path, name, _load(path)) for path, name in SYSTEMS]
    # все на одном наборе?
    ns = {name: d["n_queries"] for _, name, d in data}
    assert len(set(ns.values())) == 1, f"Разные n_queries: {ns}"

    print("## Main Result: nDCG@10 (n=300)\n")
    print(main_result_table(data))
    print("\n\n## Full Metrics Tables (n=300)\n")
    for _, name, d in data:
        print(detailed_table(name, d))
        print()


if __name__ == "__main__":
    main()
