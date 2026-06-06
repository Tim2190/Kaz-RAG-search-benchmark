"""
Export the benchmark to BEIR-compatible format for Hugging Face Datasets.

Produces:
  hf_dataset/
    corpus.jsonl      {"_id": ..., "title": ..., "text": ...}
    queries.jsonl     {"_id": ..., "text": ..., "metadata": {category, answer}}
    qrels/test.tsv    query-id \\t corpus-id \\t score

Usage:
    python -m src.eval.export_hf --out hf_dataset/
    # Then push:
    # huggingface-cli upload <your-username>/kaz-rag-search-benchmark hf_dataset/ --repo-type dataset
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def export_corpus(src: str, dst: Path) -> int:
    out = dst / "corpus.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for line in open(src, encoding="utf-8"):
            p = json.loads(line)
            record = {
                "_id": p["doc_id"],
                "title": p.get("title", ""),
                "text": p["text"],
                "metadata": {
                    "source_doc_id": p.get("source_doc_id", ""),
                    "url": p.get("url", ""),
                    "lang": p.get("lang", "kk"),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def export_queries(src: str, qrels_dir: Path, queries_out: Path) -> tuple[int, int]:
    qrels_dir.mkdir(exist_ok=True)
    tsv = qrels_dir / "test.tsv"
    nq = nr = 0
    with open(queries_out, "w", encoding="utf-8") as fq, \
         open(tsv, "w", encoding="utf-8") as fr:
        fr.write("query-id\tcorpus-id\tscore\n")
        for line in open(src, encoding="utf-8"):
            q = json.loads(line)
            record = {
                "_id": q["query_id"],
                "text": q["text"],
                "metadata": {
                    "category": q.get("category", ""),
                    "answer": q.get("answer", ""),
                },
            }
            fq.write(json.dumps(record, ensure_ascii=False) + "\n")
            nq += 1
            for doc_id in q.get("gold_doc_ids", []):
                fr.write(f"{q['query_id']}\t{doc_id}\t1\n")
                nr += 1
    return nq, nr


def write_readme(dst: Path, n_passages: int, n_queries: int) -> None:
    text = f"""---
language:
  - kk
license: cc-by-sa-4.0
task_categories:
  - question-answering
  - text-retrieval
tags:
  - kazakh
  - information-retrieval
  - RAG
  - morphology
  - BEIR
pretty_name: Kaz-RAG-Search-Benchmark
size_categories:
  - 1K<n<10K
configs:
  - config_name: corpus
    default: true
    data_files:
      - split: corpus
        path: corpus.jsonl
  - config_name: queries
    data_files:
      - split: queries
        path: queries.jsonl
  - config_name: qrels
    data_files:
      - split: test
        path: qrels/test.tsv
---

# Kaz-RAG-Search-Benchmark

Evidence-based benchmark for Kazakh information retrieval — the independent proof base
for the [**Kazakh Stemmer**](https://qaz-api.vercel.app/).

**Corpus:** {n_passages:,} passages from Kazakh Wikipedia
**Queries:** {n_queries} queries × 3 categories (natural / inflected / vocabulary-gap)
**Format:** BEIR-compatible — three subsets: `corpus`, `queries`, `qrels`

> Browse the data: use the **subset switcher** at the top of the Data Studio viewer to
> move between `corpus` (Kazakh passages), `queries` (the 300 questions), and `qrels`
> (relevance judgements). The default view is the corpus.

## Key result

A Kazakh morphological stemmer significantly improves lexical search: **+16% nDCG@10 on
inflected queries** (p=0.0017) and **+9% overall** (p=0.0001), on 300 queries with
paired-bootstrap significance. BM25+stemmer also outperforms Dense LaBSE overall
(nDCG@10 0.599 vs 0.412).

End-to-end RAG (Qwen2.5-7B, n=300) is reported honestly: the stemmer raises retrieval
hit@3 (0.737 → 0.803), but the end-to-end accuracy gain is **not** statistically
significant (McNemar p=0.63) — the bottleneck is the Kazakh-language generator, not the
retriever. The stemmer's value is proven *at the retrieval level*.

This benchmark is the evidence base for the [Kazakh Stemmer](https://qaz-api.vercel.app/).
See [github.com/Tim2190/Kaz-RAG-search-benchmark](https://github.com/Tim2190/Kaz-RAG-search-benchmark)
for full results, code, and reproduction instructions.

## Source & license

The corpus is derived from Kazakh Wikipedia (the `wikimedia/wikipedia` dump on
Hugging Face), licensed under **CC BY-SA 4.0**. This dataset inherits the same license:
attribute the source and share derivatives alike.

## Usage

```python
from datasets import load_dataset

corpus  = load_dataset("tim2190/kaz-rag-search-benchmark", "corpus",  split="corpus")
queries = load_dataset("tim2190/kaz-rag-search-benchmark", "queries", split="queries")
qrels   = load_dataset("tim2190/kaz-rag-search-benchmark", "qrels",   split="test")
```
"""
    (dst / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",  default="data/corpus/corpus.jsonl")
    ap.add_argument("--queries", default="data/queries/queries.jsonl")
    ap.add_argument("--out",     default="hf_dataset")
    args = ap.parse_args()

    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "qrels").mkdir(exist_ok=True)

    n_passages = export_corpus(args.corpus, dst)
    n_queries, n_qrels = export_queries(
        args.queries,
        dst / "qrels",
        dst / "queries.jsonl",
    )
    write_readme(dst, n_passages, n_queries)

    print(f"Exported to {dst}/")
    print(f"  corpus.jsonl   : {n_passages:,} passages")
    print(f"  queries.jsonl  : {n_queries} queries")
    print(f"  qrels/test.tsv : {n_qrels} relevance judgements")
    print()
    print("Push to HF Hub:")
    print("  pip install huggingface_hub")
    print("  huggingface-cli login")
    print(f"  huggingface-cli upload <username>/kaz-rag-search-benchmark {dst}/ --repo-type dataset")


if __name__ == "__main__":
    main()
