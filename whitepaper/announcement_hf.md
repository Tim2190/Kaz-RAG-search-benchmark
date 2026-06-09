# Morphology beats multilingual embeddings for Kazakh retrieval — a 300-query benchmark

**TL;DR.** I built an open, reproducible retrieval benchmark for Kazakh (300 queries × 3
categories, 8,370 Kazakh Wikipedia passages, BEIR format). The headline: a simple
**BM25 + morphological stemmer** — no GPU, no embedding model — beats two of three
multilingual dense models (Google LaBSE, IBM Granite) outright, and is the **most robust
system on the hardest query category**, where the dense models collapse. Dataset, code,
and a preprint with DOI are all public.

---

## Why Kazakh breaks out-of-the-box retrieval

Kazakh is agglutinative: one word appears in dozens of inflected forms.

```
бала (child) → балалар, баланың, балама, балалардың, балаларымызға …
```

A lexical index treats each surface form as a different token, so a query in one form
misses documents written in another. Multilingual embeddings are supposed to paper over
this — but how well do they actually do it on Kazakh? Nobody had measured it on an open,
reproducible benchmark. So I did.

## The benchmark

- **Corpus:** 8,370 passages from Kazakh Wikipedia (CC BY-SA 4.0)
- **Queries:** 300, split into three categories:
  - `natural` — everyday phrasing
  - `inflected` — query and answer use different grammatical forms (the morphology stress test)
  - `vocabulary-gap` — query uses a synonym/paraphrase not present in the gold passage (the semantic stress test)
- **Metrics:** Recall@k, MRR@10, nDCG@10, with paired-bootstrap significance
- **Format:** BEIR-compatible (`corpus`, `queries`, `qrels`)

## Results — nDCG@10 (n=300)

| System | inflected | natural | vocab-gap | **ALL** |
|---|---|---|---|---|
| BM25 (baseline) | 0.627 | 0.703 | 0.741 | 0.690 |
| **BM25 + Stemmer** | 0.727 | 0.772 | **0.764** | **0.754** |
| Dense LaBSE (Google) | 0.477 | 0.546 | 0.419 | 0.481 |
| Dense Granite (IBM) | 0.791 | 0.923 | 0.303 | 0.672 |
| Dense E5 (multilingual-e5-base) | **0.845** | **0.947** | 0.562 | 0.785 |

Three things jump out:

**1. A stemmer alone closes most of the gap.** BM25 → BM25+Stemmer lifts overall nDCG@10
from 0.690 to **0.754 (+9%, p=0.0001)**, and on the morphology-heavy `inflected` queries
from 0.627 to 0.727 (**+16%, p=0.0017**). No neural network, no GPU — just morphological
normalization of corpus and queries.

**2. BM25+Stemmer beats two of three multilingual dense models.** It outscores LaBSE
(0.481) by a wide margin and IBM Granite (0.672) overall — while running on a CPU in
milliseconds. The "just use embeddings" reflex is not free, and on Kazakh it isn't even
clearly better.

**3. On the hardest category, dense models collapse — the stemmer doesn't.** On
`vocabulary-gap`, IBM Granite drops to **0.303** and E5 to 0.562, while BM25+Stemmer holds
at **0.764** — the most robust system on the queries that are supposed to be the dense
models' home turf. (I reported this collapse to the Granite team.)

To be clear and honest: **E5 wins overall (0.785 vs 0.754).** A strong multilingual
embedder does edge out lexical+morphology on average. But it does so at the cost of a GPU,
a heavier pipeline, and a hard collapse on vocabulary-gap — where the cheap, transparent,
CPU-only stemmer is the safest bet.

## A negative result I'm keeping in the open

I also tried the obvious "fix" for vocabulary-gap: **synonym query expansion**. It made
things **worse across the board** — overall nDCG@10 fell from 0.754 to 0.539 (**−0.215**).
Expansion added far more noise than signal. Negative results like this rarely get
published; this one is in the benchmark in full, with the sub-analysis of why.

## And the end-to-end RAG honesty check

Plugging retrieval into a full RAG pipeline (Qwen2.5-7B, n=300): the stemmer raises
retrieval hit@3 from 0.737 to 0.803, but the **end-to-end answer accuracy gain is not
statistically significant** (McNemar p=0.63). The bottleneck downstream is the
Kazakh-language generator, not the retriever. The stemmer's value is proven *at the
retrieval level* — and I say so rather than overclaiming the whole pipeline.

## Takeaways

- For low-resource, morphologically rich languages, **don't skip the cheap lexical
  baseline** — and definitely don't skip morphological normalization. It can beat
  multilingual embeddings outright.
- **Measure per query type.** Aggregate scores hide that dense models can ace easy queries
  and collapse on hard ones.
- **Publish negative results.** They save everyone else the experiment.

## Links

- **Dataset (this page):** https://huggingface.co/datasets/Tim2190/kaz-rag-search-benchmark
- **Code & full results (GitHub):** https://github.com/Tim2190/Kaz-RAG-search-benchmark
- **Preprint with DOI (Zenodo):** https://doi.org/10.5281/zenodo.20605663
- **Kazakh Stemmer:** https://qaz-api.vercel.app/

*Disclosure: I'm the developer of the Kazakh stemmer evaluated here (System 2). All systems
were run on identical queries, corpus, and metrics; everything is public and independently
reproducible; and results unfavorable to my system — its statistical insignificance on
vocabulary-gap vs BM25, and E5's higher overall score — are reported in full.*
