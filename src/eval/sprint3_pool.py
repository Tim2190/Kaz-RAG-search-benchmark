"""
Sprint 3 — pooling для multi-gold разметки.

Берёт полные выдачи всех систем (sprint3_runs.json из Kaggle-ноутбука),
объединяет топ-10 каждой системы по каждому запросу в пул кандидатов и
выгружает TSV для ручной разметки релевантности (стандартный pooling из IR).

Назначенный gold помечается автоматически (relevant=1, менять не нужно).
Разметчику остаётся пройтись по остальным кандидатам и поставить 1/0.

ЗАПУСК:
    python -m src.eval.sprint3_pool \
        --runs results/sprint3_runs.json \
        --out annotation/pool_relevance.tsv [--depth 10]
"""
from __future__ import annotations
import argparse, csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def load_queries() -> dict:
    """qid -> {query, gold_doc_id, topic}; носительски валидированный набор."""
    out = {}
    path = os.path.join(ROOT, "data/queries/synonym_queries_final.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            out[item["qid"]] = item
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="JSON {system: {qid: [doc_id, ...]}} из Kaggle")
    ap.add_argument("--out", default="annotation/pool_relevance.tsv")
    ap.add_argument("--depth", type=int, default=10)
    args = ap.parse_args()

    with open(args.runs, encoding="utf-8") as f:
        runs = json.load(f)

    corpus = {}
    with open(os.path.join(ROOT, "data/corpus/corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["doc_id"]] = d["text"]

    queries = load_queries()

    rows = []
    for qid in sorted(queries):
        q = queries[qid]
        gold = q["gold_doc_id"]
        pool: dict[str, None] = {gold: None}   # gold всегда первый
        for system, run in runs.items():
            for did in run.get(qid, [])[: args.depth]:
                pool.setdefault(did, None)
        for did in pool:
            rows.append({
                "qid": qid,
                "query": q["query"],
                "doc_id": did,
                "is_assigned_gold": 1 if did == gold else 0,
                "passage_excerpt": corpus.get(did, "")[:300]
                                   .replace("\t", " ").replace("\n", " "),
                "RELEVANT": 1 if did == gold else "",
            })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    n_q = len({r["qid"] for r in rows})
    n_todo = sum(1 for r in rows if r["RELEVANT"] == "")
    print(f"{args.out}: {len(rows)} пар (запрос, пассаж), "
          f"{n_q} запросов, к разметке: {n_todo}")


if __name__ == "__main__":
    main()
