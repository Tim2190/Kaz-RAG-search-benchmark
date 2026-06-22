# BGE-M3 on Kazakh IR Benchmark

**Model:** `BAAI/bge-m3`  
**Architecture:** XLM-R backbone, multi-lingual, trained on 100+ languages  
**Task encoding:** Standard SentenceTransformer (no instruction prefix required)  
**Vocabulary:** 250 002 tokens (XLM-R SentencePiece), fertility = 1.81

Numbers in this file are a summary; canonical tables are in
[`../results/RESULTS.md`](../results/RESULTS.md) (Wikipedia) and
[`../results/akorda/AKORDA_RESULTS.md`](../results/akorda/AKORDA_RESULTS.md) (Akorda).

---

## §1 Wikipedia Results (n=300)

| Category | nDCG@10 |
|----------|--------:|
| inflected | 0.948 |
| natural | 0.977 |
| vocabulary-gap | 0.672 |
| **ALL** | **0.866** |

BGE-M3 is the strongest single model on Wikipedia (0.866), surpassing every prior system
including the previous best single model (Jina v3, 0.821) and the best fusion
(Hybrid ⊕ kazakh-e5, 0.808). A dense-only model now outperforms all hybrid systems
evaluated to date.

**Statistical significance (paired bootstrap, 10 000 resamples, nDCG@10, n=300):**

| Comparison | Δ | p-value |
|------------|--:|--------:|
| BGE-M3 vs Jina v3 | +0.045 | p=0.0001 ✅ |
| BGE-M3 vs E5 | +0.081 | p<0.0001 ✅ |
| BGE-M3 vs BM25+Stemmer | +0.111 | p<0.0001 ✅ |

---

## §2 Akorda Results (n=244)

| Category | nDCG@10 |
|----------|--------:|
| factoid | 0.792 |
| paraphrase | 0.636 |
| low_overlap | 0.610 |
| **ALL** | **0.679** |

BGE-M3 is also the best single system on Akorda (0.679), surpassing the previous best
(Jina v3, 0.613) as a single dense model.

**Statistical significance (paired bootstrap, 10 000 resamples, nDCG@10, n=244):**

| Comparison | Δ | p-value |
|------------|--:|--------:|
| BGE-M3 vs Jina v3 | +0.066 | p<0.001 ✅ |
| BGE-M3 vs E5 | +0.170 | p<0.001 ✅ |
| BGE-M3 vs BM25+Stemmer | +0.162 | p<0.001 ✅ |
| BGE-M3 hybrid vs BGE-M3 alone | −0.046 | p=0.014 ✅ |

Unusually, the BGE-M3 hybrid (0.633) is **significantly weaker** than BGE-M3 alone
(Δ=−0.046, p=0.014) — the first model in this benchmark where fusion with BM25 measurably
hurts. Mechanism: BGE-M3 dominates BM25 on semantic categories (paraphrase 0.636 vs 0.314,
p<0.001; low_overlap 0.610 vs 0.332, p<0.001). BM25's factoid gain in fusion (+0.100,
p<0.001) does not compensate for the semantic dilution (paraphrase −0.103, p=0.003;
low_overlap −0.136, p<0.001). The verdict is robust across the RRF k-sweep (hybrid stays
below 0.679 at every k ∈ {10, 30, 60, 100}).

**Cross-domain drop:** Wiki 0.866 → Akorda 0.679 (−0.187). This is the smallest absolute
drop among dense models tested, suggesting BGE-M3 generalises better to formal political
text than smaller or domain-specific models.

---

## §3 Tokenizer Analysis

BGE-M3 uses the **same XLM-R SentencePiece vocabulary** as E5 and Jina v3:
- Vocabulary size: 250 002 tokens
- Mean sub-words per Kazakh word (len≥9): **1.81** (same as E5 and Jina v3)
- Domain shift (Akorda vs Wiki fertility): Δ=+0.01 (negligible)

This rules out tokenizer adaptation as the source of BGE-M3's advantage over E5 and Jina v3.
The gain is **architectural / training-data**, not vocabulary coverage.

| Tokenizer | Vocab size | Fertility (Wiki) | Fertility (Akorda) | Δ |
|-----------|----------:|----------------:|------------------:|--:|
| BGE-M3 | 250 002 | 1.81 | 1.82 | +0.01 |
| Jina v3 | 250 002 | 1.81 | 1.82 | +0.01 |
| E5-base | 250 002 | 1.81 | 1.82 | +0.01 |
| Granite R1 | 250 002 | 1.81 | 1.82 | +0.01 |
| Granite R2-311M | 262 144 | 4.20 | 4.43 | +0.23 |
| Granite R2-97M | 179 934 | 4.00 | 4.29 | +0.29 |
| Nomic v1.5 | 30 522 | 3.49† | 4.01† | +0.52† |
| Qwen3-0.6B | 151 643 | 6.20 | 6.27 | +0.07 |

> † Nomic fertility figure is misleading — Kazakh-specific chars become `[UNK]`, single
> token regardless of word length. Not low fragmentation; zero coverage.
