# KazEmbed-V5 on Kazakh IR Benchmark

**Model:** `Nurlykhan/kazembed-v5`
**Base:** `intfloat/multilingual-e5-base` (fine-tuned)
**Prefixes:** `query: ` / `passage: ` (same as e5; required per model card)
**Tokenizer:** XLM-R SentencePiece (250 002 tokens, inherited — fine-tuning does not change tokenizer)
**Fertility:** 1.81 (Wiki) / 1.82 (Akorda) — identical to e5, BGE-M3, Jina v3

Numbers in this file are a summary; canonical tables are in
[`../results/RESULTS.md`](../results/RESULTS.md) (Wikipedia) and
[`../results/akorda/AKORDA_RESULTS.md`](../results/akorda/AKORDA_RESULTS.md) (Akorda).

---

## §1 Wikipedia Results (n=300)

| Category | nDCG@10 |
|----------|--------:|
| inflected | 0.778 |
| natural | 0.865 |
| vocabulary-gap | 0.284 |
| **ALL** | **0.642** |

**Statistical significance (paired bootstrap, 10 000 resamples, nDCG@10, n=300):**

| Comparison | Δ | p-value |
|------------|--:|--------:|
| BGE-M3 vs KazEmbed-V5 | +0.223 | p<0.001 ✅ |
| Jina v3 vs KazEmbed-V5 | +0.179 | p<0.001 ✅ |
| Cohere embed-v4.0 vs KazEmbed-V5 | +0.158 | p<0.001 ✅ |
| E5 vs KazEmbed-V5 | +0.142 | p<0.001 ✅ |
| BM25+Stemmer vs KazEmbed-V5 | +0.112 | p<0.001 ✅ |
| kazakh-e5 vs KazEmbed-V5 | +0.105 | p<0.001 ✅ |
| Qwen3-0.6B vs KazEmbed-V5 | +0.048 | p=0.003 ✅ |
| Granite R1 vs KazEmbed-V5 | +0.030 | p=0.032 ✅ |
| Granite R2-311M vs KazEmbed-V5 | +0.017 | p=0.153 — n.s. |
| KazEmbed-V5 vs Granite R2-97M | +0.054 | p=0.002 ✅ |
| KazEmbed-V5 vs Nomic v1.5 | +0.471 | p<0.001 ✅ |

Positive Δ = the other system is better.

**Headline Wikipedia result:** kazembed-v5 is **significantly below the base model e5** on
every comparison above (Δ=+0.142, p<0.001), and also significantly below kazakh-e5
(Δ=+0.105, p<0.001) — a different fine-tune of the same base model. It is statistically
tied with Granite R2-311M (p=0.153, n.s.) and significantly above only Granite R2-97M and
Nomic v1.5. vocabulary-gap score (0.284) is the third-weakest of all models after Nomic
(0.071) and Granite R2-97M (0.175).

---

## §2 Akorda Results (n=244)

| Category | nDCG@10 |
|----------|--------:|
| factoid | 0.505 |
| paraphrase | 0.369 |
| low_overlap | 0.295 |
| **ALL** | **0.389** |

**Statistical significance vs. key Akorda systems (10 000 resamples):**

| Comparison | Δ | p-value |
|------------|--:|--------:|
| BGE-M3 vs KazEmbed-V5 | +0.290 | p<0.001 ✅ |
| Jina v3 vs KazEmbed-V5 | +0.224 | p<0.001 ✅ |
| E5 vs KazEmbed-V5 | +0.120 | p<0.001 ✅ |
| BM25+Stemmer vs KazEmbed-V5 | +0.128 | p<0.001 ✅ |
| kazakh-e5 vs KazEmbed-V5 | +0.037 | p<0.001 ✅ |
| Granite R1 vs KazEmbed-V5 | +0.041 | p=0.045 ✅ |
| Granite R2-311M vs KazEmbed-V5 | +0.010 | p=0.346 — n.s. |
| KazEmbed-V5 vs Cohere embed-v4.0 | +0.022 | p=0.215 — n.s. |
| KazEmbed-V5 vs Qwen3-0.6B | +0.059 | p=0.021 ✅ |

On Akorda, kazembed-v5 is again significantly below e5 (Δ=+0.120, p<0.001) and kazakh-e5
(Δ=+0.037, p<0.001). It is statistically tied with Cohere embed-v4.0 (p=0.215) and
Granite R2-311M (p=0.346), and significantly better than Qwen3-0.6B (p=0.021).

**Cross-domain drop:** Wiki 0.642 → Akorda 0.389 (**−0.253**). This is a moderate drop,
similar in magnitude to Granite R2-311M (−0.260) and smaller than shyngys-e5 (−0.321),
Qwen3 (−0.360), and Cohere (−0.433). Unlike those models, kazembed-v5's drop is not
explained by tokenizer fragmentation (fertility 1.81 = same as e5, which only drops −0.276).

**Akorda hybrid (RRF k=60):** criterion 1 fails, criterion 2 passes.
- hybrid ALL = 0.493 < max channel (BM25 0.517) → criterion 1 FAILS
- hybrid low_overlap = 0.352 > BM25 low_overlap (0.332) → criterion 2 PASSES

| Slice | hybrid | Δ vs bm25 | p | Δ vs dense | p |
|-------|-------:|----------:|--:|-----------:|--:|
| factoid (n=81) | 0.709 | −0.197 | **<0.001** | +0.204 | **<0.001** |
| paraphrase (n=81) | 0.419 | +0.105 | **<0.001** | +0.050 | 0.126 n.s. |
| low_overlap (n=82) | 0.352 | +0.020 | 0.259 n.s. | +0.057 | 0.059 n.s. |
| **ALL (n=244)** | **0.493** | −0.024 | 0.123 n.s. | **+0.104** | **<0.001** |

The hybrid significantly improves on kazembed-v5 dense (+0.104, p<0.001) and on paraphrase
beats BM25+stemmer significantly (+0.105, p<0.001). But factoid dominates the overall result
(BM25 0.906 vs dense 0.505), keeping the hybrid below BM25 overall. The overall loss vs BM25
is not significant (p=0.123) — the same pattern as Cohere hybrid (p=0.176).

---

## §3 Tokenizer Note

KazEmbed-V5 inherits the `intfloat/multilingual-e5-base` tokenizer: XLM-R SentencePiece with
250 002 tokens. **Fine-tuning does not change the tokenizer**, so fertility is identical to
the entire XLM-R cluster (e5, BGE-M3, Jina v3, kazakh-e5, Granite R1): **1.81 (Wiki) /
1.82 (Akorda)**. Tokenizer fragmentation cannot explain why kazembed-v5 underperforms e5
or kazakh-e5 — the cause is purely in the fine-tuning.

---

## §4 Summary

KazEmbed-V5 is fine-tuned from `intfloat/multilingual-e5-base` on Kazakh retrieval data
(KazQAD + Powerful-Kazakh-Dialogue, 61 255 pairs) and claims +2.1% MRR over e5-base on the
KazQAD test set. **That in-domain gain does not transfer to this benchmark on either domain.**
On Wikipedia (n=300, OOD from training), kazembed-v5 (0.642) is significantly below the
base model (0.785, Δ=+0.142, p<0.001) and below kazakh-e5 (0.747), a competing Kazakh
fine-tune. On Akorda (formal political Kazakh, strongly OOD), the same pattern holds:
kazembed-v5 (0.389) < e5 (0.509, p<0.001) < kazakh-e5 (0.426, p<0.001).

The likely mechanism is **training data mismatch**: KazQAD consists of short factoid
question-answer pairs from Kazakh Wikipedia, while this benchmark queries Kazakh Wikipedia
(OOD from KazQAD's specific splits) and formal Akorda political prose. Fine-tuning on
in-distribution retrieval data with aggressive MultipleNegativesRankingLoss can suppress
the broad multilingual representations of the base model, harming OOD performance.

**Practical takeaway:** for Kazakh retrieval in production, prefer the base
`intfloat/multilingual-e5-base` or `shyngys879/kazakh-e5-rag-embedding` over kazembed-v5
unless deploying on data closely resembling KazQAD. Benchmark on your actual domain before
adopting a domain-specific fine-tune.
