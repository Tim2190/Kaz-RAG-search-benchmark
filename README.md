# Kaz-RAG-Search-Benchmark

**Evidence-based benchmark: why basic search "goes blind" on Kazakh — and how morphological normalization fixes it.**

![5-system comparison](results/systems_ndcg.png)

> On inflected queries, BM25 baseline scores **nDCG@10 = 0.18**.
> A Kazakh stemmer brings it to **0.45 (+145%)**, recall@1 **+200%**, p = 0.0005.
> BM25+Stemmer (0.599) **outperforms Dense LaBSE** (0.412) overall.

---

## Headline Result

**Corpus:** 8 370 passages (Kazakh Wikipedia) · **Queries:** 60 (20 entities × 3 categories)
**Baseline:** BM25 (Okapi). "Before" — no normalization; "After" — corpus and query tokens
stemmed with the [Kazakh stemmer](https://kazakh-stemmer-590833642796.europe-west1.run.app).

### nDCG@10 — 5-System Comparison

| System | inflected | natural | vocab-gap | **ALL** |
|--------|----------:|--------:|----------:|--------:|
| BM25 | 0.184 | 0.537 | 0.538 | 0.420 |
| **BM25 + Stemmer** | 0.451 | 0.675 | 0.672 | **0.599** |
| Dense LaBSE | 0.366 | 0.561 | 0.310 | 0.412 |
| Dense Granite-278M | **0.816** | **0.872** | 0.226 | 0.638 |
| Dense E5-base | 0.695 | 0.889 | **0.519** | **0.701** |

### Statistical Significance (BM25 Before → After Stemmer)

| category | Before | After | gain | p-value |
|----------|-------:|------:|-----:|--------:|
| **inflected** | 0.184 | **0.451** | **+145%** | p=0.0005 ✅ |
| natural | 0.537 | 0.675 | +26% | p=0.025 ✅ |
| vocabulary-gap | 0.538 | 0.672 | +25% | p=0.039 ✅ |
| **ALL** | 0.420 | **0.599** | **+43%** | p<0.0001 ✅ |

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

**BM25+Stemmer vs Dense LaBSE:** the stemmer wins (0.599 vs 0.412).
Morphological normalization matters more than naive multilingual embeddings for Kazakh.

---

## RAG End-to-End: Stemmer Effect Across 3 LLMs

The stemmer raises the **retrieval hit rate 0.467 → 0.667** (+43%) — the correct passage
reaches the LLM context 1.4× more often. Tested on three models of increasing capability:

![RAG: stemmer effect across models](results/rag_models.png)

| Model | Accuracy (no stem → stem) | Δ acc | Hallucination Δ |
|-------|:---:|:---:|:---:|
| Granite-2B | 0.250 → 0.267 | +0.017 | −0.033 |
| Granite-8B (4-bit) | 0.350 → 0.417 | +0.067 | −0.067 |
| **Qwen2.5-7B** (4-bit) | **0.400 → 0.500** | **+0.100** | −0.017 |

**Key pattern:** the accuracy gain from the stemmer **grows with model competence**
(`+0.017 → +0.067 → +0.100`). The retrieval gain is identical for all three; a more
capable generator converts more of that extra correct context into correct answers.
Qwen2.5-7B is the best end-to-end system — it abstains (`Ақпарат жоқ`) instead of
inventing, and the stemmer turns those abstentions into correct answers.

> ⚠️ **Honesty note:** the end-to-end deltas are measured on n=60, single run, **not**
> bootstrap-validated (unlike the retrieval result). The direction is consistent and the
> mechanism is clear, but a rigorous statistical claim would need a larger query set. We
> report a **demonstrated trend**; the retrieval gain that drives it (0.467 → 0.667) is
> directly measured and solid. Full breakdown in [`results/RESULTS.md`](results/RESULTS.md).

---

## Methodology

- **Corpus.** 800 random Kazakh Wikipedia articles from
  [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
  (Hugging Face) → cleaning → chunking (~120 words) → language filter → **8 370 passages**.
- **Queries.** 20 entities (countries, cities, people, concepts) × 3 categories:
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
  queries/     # queries.jsonl — 60 queries with qrels
  resources/   # stem_cache.json — stemmer cache (102k words)
results/       # metrics JSON, charts, RESULTS.md
tests/         # 43 unit tests
```

---

## Roadmap

- [x] Kazakh Wikipedia corpus (8 370 passages) + language filter
- [x] Query set (60, 3 categories) + qrels
- [x] BM25 ± stemmer: metrics, comparison, **statistical significance** — result proven
- [x] Dense retrieval (IBM Granite / Google LaBSE / multilingual-E5) — benchmarked
- [x] 5-system comparison chart
- [x] RAG end-to-end on 3 LLMs (Granite-2B/8B, Qwen2.5-7B) — accuracy gain scales with model competence (+0.017 → +0.067 → +0.100)
- [ ] Larger query set for stronger statistics + Kazakh validation
- [ ] Whitepaper

> Status: the main claim (morphological blindness → stemmer fixes it) is proven and
> reproducible. Dense systems benchmarked; end-to-end RAG measured. Results in
> [`results/RESULTS.md`](results/RESULTS.md).

---

[Русская версия](README.ru.md)
