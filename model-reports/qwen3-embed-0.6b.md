# Qwen3-Embedding-0.6B on Kazakh IR Benchmark

**Model:** `Qwen/Qwen3-Embedding-0.6B`  
**Architecture:** LLM-based dense embedder (Alibaba); 0.6B parameters  
**Task encoding:** Instruction prefix for queries:
`"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "`  
**Vocabulary:** 151 643 tokens (Qwen3 BPE), fertility = 6.20

Numbers in this file are a summary; canonical tables are in
[`../../results/RESULTS.md`](../../results/RESULTS.md) (Wikipedia) and
[`../../results/akorda/AKORDA_RESULTS.md`](../../results/akorda/AKORDA_RESULTS.md) (Akorda).

---

## §1 Wikipedia Results (n=300)

| Category | nDCG@10 |
|----------|--------:|
| inflected | 0.792 |
| natural | 0.927 |
| vocabulary-gap | 0.352 |
| **ALL** | **0.690** |

Qwen3-0.6B matches unstemmed BM25 (0.690 ≈ 0.690) but is significantly below every
competitive system in the benchmark.

**Statistical significance (paired bootstrap, 10 000 resamples, nDCG@10, n=300):**

| Comparison | Δ | p-value |
|------------|--:|--------:|
| BGE-M3 vs Qwen3-0.6B | −0.175 | p<0.001 ✅ |
| Jina v3 vs Qwen3-0.6B | −0.131 | p<0.001 ✅ |
| E5 vs Qwen3-0.6B | −0.094 | p<0.001 ✅ |
| BM25+Stemmer vs Qwen3-0.6B | −0.064 | p=0.010 ✅ |

Negative Δ = Qwen3 is worse. Qwen3 is significantly below BM25+Stemmer (the lexical
baseline) — a dense LLM-based embedder failing to beat an unstemmed BM25 would be
surprising in most contexts, but the tokenizer analysis below offers a clear explanation.

---

## §2 Akorda Results (n=244)

| Category | nDCG@10 |
|----------|--------:|
| factoid | 0.516 |
| paraphrase | 0.256 |
| low_overlap | 0.220 |
| **ALL** | **0.330** |

**Statistical significance vs. key Akorda systems (n=10 000 resamples):**

| Comparison | Δ | p-value |
|------------|--:|--------:|
| BGE-M3 vs Qwen3-0.6B | −0.349 | p<0.001 ✅ |
| Jina v3 vs Qwen3-0.6B | −0.283 | p<0.001 ✅ |
| E5 vs Qwen3-0.6B | −0.179 | p<0.001 ✅ |
| BM25+Stemmer vs Qwen3-0.6B | −0.186 | p<0.001 ✅ |

**Cross-domain drop:** Wiki 0.690 → Akorda 0.330 (−0.360) — the **largest absolute
cross-domain drop** among all tested models, including models that perform worse overall.
The collapse is steepest on semantic categories (paraphrase 0.927 → 0.256; low_overlap not
directly comparable across domains but Akorda 0.220 is very low).

**Akorda hybrid (RRF k=60):** both pre-registered criteria fail.
- hybrid ALL = 0.464 < max channel (BM25 0.517) → criterion 1 FAILS
- hybrid low_overlap = 0.277 < BM25 low_overlap (0.332) → criterion 2 FAILS

The hybrid is significantly better than Qwen3 alone (+0.133, p<0.001) but significantly
worse than BM25+Stemmer alone (−0.053, p=0.003). Pattern is analogous to Granite R2-97M:
the dense channel is too weak on factoid queries to offset BM25's strength there.

---

## §3 Tokenizer Analysis

Qwen3-0.6B uses a BPE vocabulary of **151 643 tokens** but exhibits the highest sub-word
fertility of all models tested (6.20 sub-words per word on 100 frequent Kazakh words ≥9
chars). The fragmentation is ~3.4× higher than the XLM-R cluster (BGE-M3, E5, Jina at
1.81) and higher than even the fragmented Granite R2 models (4.00–4.20).

The example outputs are particularly revealing: `«халықаралық»` (international) is split
into 8 pieces, `«мүмкіндіктерін»` (of possibilities, acc.) into 11 pieces — each Kazakh
morpheme is further fragmented at the byte level (the displayed tokens are UTF-8 byte
sequences, not readable Cyrillic). This is not just oversegmentation; it is tokenization
at the byte level for Cyrillic characters the vocabulary did not internalize as units.

| Tokenizer | Vocab size | Fertility (Wiki) | Fertility (Akorda) | Δ |
|-----------|----------:|----------------:|------------------:|--:|
| BGE-M3 | 250 002 | 1.81 | 1.82 | +0.01 |
| Jina v3 | 250 002 | 1.81 | 1.82 | +0.01 |
| E5-base | 250 002 | 1.81 | 1.82 | +0.01 |
| Granite R1 | 250 002 | 1.81 | 1.82 | +0.01 |
| Granite R2-97M | 179 934 | 4.00 | 4.29 | +0.29 |
| Granite R2-311M | 262 144 | 4.20 | 4.43 | +0.23 |
| Nomic v1.5 | 30 522 | 3.49† | 4.01† | — |
| **Qwen3-0.6B** | 151 643 | **6.20** | **6.27** | +0.07 |

> † Nomic fertility is misleading ([UNK] issue); see `nomic-v1.5.md`.

The Akorda domain shift for Qwen3 (Δ=+0.07) is small in absolute fertility terms, but the
baseline fragmentation (6.20) is so high that even small additional fragmentation compounds
meaningfully. This is the plausible candidate mechanism for the largest cross-domain drop
(−0.360) in the benchmark, though this is correlation, not proven causation.

**Conclusion:** Qwen3-0.6B's Qwen3 BPE vocabulary did not develop stable sub-word units
for agglutinative Kazakh morphology. The model compensates with byte-level fallback
tokenization which inflates sequence lengths and degrades the semantic coherence of
embeddings — especially on the formal, morphologically rich vocabulary of Akorda texts.
