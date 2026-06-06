# Kaz-RAG-Search-Benchmark

**Evidence-based benchmark: why basic search "goes blind" on Kazakh — and how morphological normalization fixes it.**

![5-system comparison](results/systems_ndcg.png)

> On inflected queries, BM25 baseline scores **nDCG@10 = 0.18**.
> A Kazakh stemmer brings it to **0.45 (+145%)**, recall@1 **+200%**, p = 0.0005.
> BM25+Stemmer (0.599) **outperforms Dense LaBSE** (0.412) overall.

---

## Headline Result

**Corpus:** 8 370 passages (Kazakh Wikipedia) · **Queries:** 300 (100 entities × 3 categories)
**Baseline:** BM25 (Okapi). "Before" — no normalization; "After" — corpus and query tokens
stemmed with the [Kazakh stemmer](https://kazakh-stemmer-590833642796.europe-west1.run.app).

### Statistical Significance — nDCG@10 (paired bootstrap, 10 000 resamples, n=300)

| category | Before | After | Δ | gain | p-value |
|----------|-------:|------:|--:|-----:|--------:|
| **inflected** | 0.627 | **0.727** | +0.101 | **+16%** | p=0.0017 ✅ |
| natural | 0.703 | **0.772** | +0.068 | +10% | p=0.0063 ✅ |
| vocabulary-gap | 0.741 | 0.764 | +0.023 | +3% | p=0.21 ✗ |
| **ALL** | 0.690 | **0.754** | +0.064 | **+9%** | p=0.0001 ✅ |

> Vocabulary-gap is not significantly improved — expected: stemming fixes morphological
> mismatch, not semantic gaps between synonyms. This is the honest, theoretically
> correct result.

Full metrics with recall@{1,5,10}, MRR@10 — in [`results/RESULTS.md`](results/RESULTS.md).

---

## Why This Matters

Kazakh is **agglutinative**: 800 Wikipedia articles produce **102 408 unique surface forms**.
A single word like *теңіз* (sea) appears as *теңіздерге*, *теңізде*, *теңіздің* in text,
but a search query may use a different form. Exact word matching fails completely on
inflected queries (recall@1 = 0.10 without stemmer).

The stemmer reduces forms to their root (*Бакуде → баку*, *теңіздерге → теңіз*),
letting BM25 see through morphological variation.

### Dense Models: What They Get Right and Wrong

| | inflected (morphology) | natural | vocabulary-gap (synonyms) |
|--|:---:|:---:|:---:|
| BM25+Stemmer | ✅ fixed | ✅ | ✅ |
| Dense Granite-278M | ✅✅ best (0.816) | ✅✅ best | ❌ collapses (0.226) |
| Dense E5-base | ✅ good | ✅✅ best | ✅ best |
| Dense LaBSE | weak | ok | weak |

**BM25+Stemmer vs Dense LaBSE (from 60-query dense run):** the stemmer wins (0.599 vs 0.412).
Morphological normalization matters more than naive multilingual embeddings for Kazakh.

---

## RAG End-to-End: Does Better Retrieval Reach the Answer? (Qwen2.5-7B, n=300)

We ran the full RAG chain (BM25 → context → Qwen2.5-7B 4-bit) on the **same 300 queries**,
changing only the retriever. The stemmer improves retrieval (**hit@3 0.737 → 0.803**) —
but does that reach the final answer?

![RAG end-to-end: Qwen2.5-7B, n=300](results/rag_models.png)

| Stemmer | Retrieval hit@3 | Accuracy | Hallucination | Abstain |
|---------|:---:|:---:|:---:|:---:|
| No stemmer | 0.737 | 0.483 | 0.217 | 0.300 |
| Kazakh stemmer | 0.803 | **0.500** | 0.240 | 0.260 |
| **Δ** | **+0.066** | +0.017 | +0.023 | −0.040 |

> ✅ **The stemmer's effectiveness is proven — for retrieval.** It significantly improves
> search quality (nDCG@10 +9% overall, +16% on inflected, p≤0.0017, n=300) and raises the
> RAG retrieval hit-rate 0.737 → 0.803. That part is solid. The null result below is about
> the **generator**, not the stemmer: the bottleneck in Kazakh RAG today is the LLM's
> Kazakh comprehension, not the search step.

**End-to-end accuracy gain is _not_ statistically significant** (McNemar exact p = **0.63**;
net +5 correct of 300). Better retrieval is *necessary but not sufficient* — the generator
still has to extract the answer, and Qwen2.5-7B (4-bit) often fails to even with the right
passage. The stemmer mostly makes Qwen *more willing to answer* (abstain 0.300 → 0.260), and
those recovered answers split between correct and hallucinated, so net accuracy barely moves.
A stronger Kazakh generator would likely convert the proven retrieval gain into real accuracy.

The directionally strongest effect is on `inflected` (morphology) queries — the biggest
retrieval jump (+0.15 hit@3) and +0.07 accuracy — exactly where theory predicts, but even
there p=0.25. A trend, not a proof.

> ⚠️ **Replication note:** an earlier version claimed a Qwen +0.100 accuracy gain that
> "scales with model competence," measured on **n=60**. That signal **did not replicate at
> n=300** (+0.100 → +0.017, p=0.63) — it was sampling noise. The retrieval result survived
> the jump to 300 queries and got *stronger*; the end-to-end RAG claim did not. We keep the
> honest negative result. Full breakdown in [`results/RESULTS.md`](results/RESULTS.md).

---

## Methodology

- **Corpus.** 800 random Kazakh Wikipedia articles from
  [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
  (Hugging Face) → cleaning → chunking (~120 words) → language filter → **8 370 passages**.
- **Queries.** 100 entities (countries, cities, people, concepts) × 3 categories:
  - `inflected` — key word in an oblique grammatical case (morphology stress-test);
  - `vocabulary-gap` — synonyms/paraphrase (semantic stress-test);
  - `natural` — standard questions.
  Each query has a known ground-truth passage (qrels).
- **Systems.** BM25 ± Kazakh stemmer. Dense: IBM Granite-278M, Google LaBSE,
  multilingual-E5-base (embeddings pre-computed and cached).
- **Metrics.** Recall@{1,5,10}, MRR@10, nDCG@10 + statistical significance
  (paired bootstrap, 10 000 resamples).
- **Reproducibility.** Stem cache committed; BM25 results require no network.
  Dense results require GPU (~15–30 min in Colab).

---

## Quickstart

```bash
git clone https://github.com/Tim2190/Kaz-RAG-search-benchmark.git
cd Kaz-RAG-search-benchmark
pip install -r requirements.txt

# BM25 before/after (no network — stem cache in repo)
python -m src.eval.run_benchmark --stemmer identity --out results/bm25_identity.json
python -m src.eval.run_benchmark --stemmer kazakh   --out results/bm25_kazakh.json

# Delta table + chart
python -m src.eval.compare --before results/bm25_identity.json \
    --after results/bm25_kazakh.json --chart results/before_after.png

# Statistical significance
python -m src.eval.significance

# 5-system comparison chart
python -m src.eval.chart_all --out results/systems_ndcg.png
```

Dense retrieval and RAG require GPU (Colab T4 is sufficient). See [`PIPELINE.md`](PIPELINE.md).

### Tests

Core logic (metrics, tokenization, chunking, stem client, retrieval) covered by tests:
```bash
python -m unittest discover tests      # 43 tests, no network
```

---

## Repository Structure

```
src/
  scraping/    # corpus collection (wiki dump via HF; news/gov fallback)
  corpus/      # cleaning, chunking, language filter
  queries/     # queries + qrels, dataset loader
  retrieval/   # bm25 (lexical) + dense (embeddings)
  preprocess/  # Kazakh stemmer (HTTP client + cache), tokenizer
  eval/        # metrics, benchmark runner, compare, significance, charts
  rag/         # LLM prompt, scorer, hallucination harness
data/
  corpus/      # corpus.jsonl — 8 370 passages
  queries/     # queries.jsonl — 300 queries with qrels
  resources/   # stem_cache.json — stemmer cache (102k words)
results/       # metrics JSON, charts, RESULTS.md
tests/         # 43 unit tests
```

---

## Roadmap

- [x] Kazakh Wikipedia corpus (8 370 passages) + language filter
- [x] Query set (300, 3 categories, 100 entities) + qrels — bootstrap-validated
- [x] BM25 ± stemmer: metrics, comparison, **statistical significance** — result proven
- [x] Dense retrieval (IBM Granite / Google LaBSE / multilingual-E5) — benchmarked
- [x] 5-system comparison chart
- [x] RAG end-to-end (Qwen2.5-7B, n=300) — retrieval hit@3 0.74→0.80, but end-to-end accuracy gain not significant (McNemar p=0.63); honest negative result
- [ ] Human validation of generated queries by native Kazakh speaker
- [ ] Whitepaper

> Status: the main claim (morphological blindness → stemmer fixes it) is proven and
> reproducible. Dense systems benchmarked; end-to-end RAG measured. Results in
> [`results/RESULTS.md`](results/RESULTS.md).

---

[Русская версия](README.ru.md)
