# Cohere embed-v4.0 on Kazakh IR Benchmark

**Model:** `cohere/embed-v4.0` (API-only)
**Provider:** Cohere (2025); marketed for cross-lingual retrieval across 100+ languages
**Task encoding:** `input_type=search_query` / `search_document` (no string prefixes)
**Tokenizer:** byte-level BPE; sub-word fertility = 6.20 (Wiki) / 6.27 (Akorda) — **identical to Qwen3-0.6B**

Numbers in this file are a summary; canonical tables are in
[`../results/RESULTS.md`](../results/RESULTS.md) (Wikipedia) and
[`../results/akorda/AKORDA_RESULTS.md`](../results/akorda/AKORDA_RESULTS.md) (Akorda).

---

## §1 Wikipedia Results (n=300)

| Category | nDCG@10 |
|----------|--------:|
| inflected | 0.864 |
| natural | 0.965 |
| vocabulary-gap | 0.570 |
| **ALL** | **0.800** |

On Wikipedia, Cohere embed-v4.0 is the **3rd-strongest single system** (0.800), behind
BGE-M3 (0.866) and statistically level with Jina v3 and E5.

**Statistical significance (paired bootstrap, 10 000 resamples, nDCG@10, n=300):**

| Comparison | Δ | p-value |
|------------|--:|--------:|
| BGE-M3 vs Cohere | +0.066 | p<0.001 ✅ |
| Jina v3 vs Cohere | +0.021 | p=0.104 — n.s. |
| E5 vs Cohere | −0.015 | p=0.179 — n.s. |
| BM25+Stemmer vs Cohere | −0.046 | p=0.028 ✅ |
| Qwen3-0.6B vs Cohere | −0.110 | p<0.001 ✅ |

Positive Δ = the other system is better. Cohere is significantly below BGE-M3, tied with
Jina v3 and E5, and significantly above both BM25+Stemmer and Qwen3. A solid — if not
leading — Wikipedia result.

---

## §2 Akorda Results (n=244)

| Category | nDCG@10 |
|----------|--------:|
| factoid | 0.627 |
| paraphrase | 0.263 |
| low_overlap | 0.213 |
| **ALL** | **0.367** |

**Statistical significance vs. key Akorda systems (10 000 resamples):**

| Comparison | Δ | p-value |
|------------|--:|--------:|
| BGE-M3 vs Cohere | +0.312 | p<0.001 ✅ |
| Jina v3 vs Cohere | +0.246 | p<0.001 ✅ |
| E5 vs Cohere | +0.142 | p<0.001 ✅ |
| BM25+Stemmer vs Cohere | +0.150 | p<0.001 ✅ |
| Qwen3-0.6B vs Cohere | −0.037 | p=0.058 — n.s. |

On Akorda, Cohere ranks 8th of the dense+lexical systems — below BM25+Stemmer, E5, and all
three Granite variants, and only statistically tied with Qwen3 (the next-weakest). It beats
only Qwen3 and Nomic outright.

**Cross-domain drop:** Wiki 0.800 → Akorda 0.367 (**−0.433**) — the **largest absolute
cross-domain drop in the entire benchmark**, exceeding Qwen3 (−0.360) and far beyond the
XLM-R cluster (BGE-M3 −0.187, Jina −0.208, E5 −0.276). This is the headline result: a model
explicitly marketed for cross-lingual retrieval is the least domain-robust dense model
tested on Kazakh.

**Akorda hybrid (RRF k=60):** both pre-registered criteria fail.
- hybrid ALL = 0.501 < max channel (BM25 0.517) → criterion 1 FAILS
- hybrid low_overlap = 0.311 < BM25 low_overlap (0.332) → criterion 2 FAILS

Unlike Qwen3, the overall hybrid loss vs BM25 is **not** significant (Δ=−0.016, p=0.176),
and on paraphrase the hybrid significantly beats both channels (0.386, p=0.005 vs BM25).
The hybrid significantly improves on Cohere dense alone (+0.134, p<0.001) — fusion rescues
the weak dense channel but cannot lift it past the strong lexical baseline (BM25 reaches
0.906 on factoid, where Cohere manages only 0.627).

---

## §3 Tokenizer Analysis

Cohere embed-v4.0 is API-only, so its tokenizer is not available as a HuggingFace
`AutoTokenizer` and was instead probed through the Cohere tokenize API
(`src/eval/cohere_tokenization.py`) on the **same 100 frequent long Kazakh words** used for
every other model (`fertility_compare.py`), guaranteeing comparability.

| Tokenizer | Vocab | Fertility (Wiki) | Fertility (Akorda) |
|-----------|------:|-----------------:|-------------------:|
| XLM-R cluster (BGE-M3, Jina, E5) | 250 002 | 1.81 | 1.82 |
| Granite R2-311M | 262 144 | 4.20 | 4.43 |
| Nomic v1.5 | 30 522 | 3.49 † | 4.01 † |
| Qwen3-0.6B | 151 643 | 6.20 | 6.27 |
| **Cohere embed-v4.0** | byte-level BPE | **6.20** | **6.27** |

> † Nomic fertility is misleading ([UNK] artifact); see `nomic-v1.5.md`.

The result is **identical to Qwen3-0.6B to three significant figures** (6.20 / 6.27); token
counts are identical on all four probe words (the Cohere tokenize API returns IDs only, not
string pieces, so splits are inferred from counts):

| Word | tokens |
|------|-------:|
| сөзжасам | 6 |
| мүмкіндіктерін | 11 |
| ұйымдастырушылық | 9 |
| халықаралық | 8 |

This is strong evidence that Cohere embed-v4.0 uses the **same byte-level BPE fallback** for
Kazakh Cyrillic as Qwen3: each character is encoded as 2–3 byte-piece tokens. Importantly,
this is **not** the Nomic `[UNK]` failure (where absent characters collapse to one token,
*compressing* the count). Here every character *is* tokenized — just at byte granularity —
so the fertility 6.20 reflects genuine sequence-length inflation, not a measurement artifact.
(The Cohere tokenize API returns token IDs but not display strings, so per-word counts are
shown rather than the byte-mojibake pieces; the counts match Qwen3 exactly.)

**Conclusion.** Cohere embed-v4.0's strong Wikipedia score (0.800, #3) does not transfer to
formal political Kazakh: it suffers the benchmark's largest cross-domain collapse (−0.433)
and its byte-level tokenizer fragments Kazakh morphology identically to Qwen3. Tokenizer
fertility is a *candidate mechanism* — the two byte-level-BPE models (Cohere, Qwen3) show
the two largest drops while the XLM-R cluster (fertility 1.81) stays domain-robust — but
this is a consistent correlation, not a controlled ablation. The practical takeaway: vendor
claims of broad multilingual / cross-lingual coverage should be validated on the specific
target language and register, not assumed from a high aggregate or encyclopedic-domain score.
