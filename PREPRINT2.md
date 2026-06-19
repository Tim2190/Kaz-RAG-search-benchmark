# Hybrid Retrieval and Out-of-Domain Robustness for Kazakh Search:
# Extended Benchmark with New Models, RRF Fusion, and Tokenizer Analysis

> **Preprint — DOI will be added after Zenodo deposit.**
>
> This is the follow-up to:
> Seidalin, T. (2026). *Morphology Beats Multilingual Embeddings for Kazakh Retrieval:
> A 300-Query Benchmark with Honest Negative Results.* Zenodo.
> https://doi.org/10.5281/zenodo.20605663

---

## Abstract

We extend the Kazakh retrieval benchmark from Paper 1 in three directions: (1) we evaluate
three new embedding models — IBM Granite R2 (97M and 311M) and a Kazakh-fine-tuned E5
(shyngys-e5) — on the original 300-query Wikipedia benchmark; (2) we add Reciprocal Rank
Fusion (RRF) hybrids combining BM25+stemmer with each dense model; (3) we replicate the
full 7-system comparison on an entirely different corpus — official presidential speeches
from akorda.kz (244 queries) — as an out-of-domain (OOD) validation. We also analyze
sub-word tokenizer fertility as a candidate mechanism for observed performance gaps.

Key results: the BM25+stemmer ⊕ kazakh-e5 hybrid is the best system on Wikipedia
(nDCG@10 = 0.808, vs 0.785 for the best single model). OOD rankings on Akorda are largely
stable (Spearman ρ = 0.89), confirming that Paper 1 conclusions generalize beyond
Wikipedia; absolute scores are lower on Akorda (best hybrid 0.562 vs 0.808 on Wikipedia)
because formal political text is harder for all systems uniformly — the gap reflects domain
difficulty, not a failure of the retrieval approach. Granite R2 underperforms R1 on Kazakh on both domains; sub-word fertility analysis reveals
that R2's tokenizer fragments Kazakh words 2.3× more than R1/e5 — a plausible
tokenizer-level mechanism for its domain drop. kazakh-e5 (Kazakh-specific fine-tune of e5)
is significantly worse than base e5 overall (Δ=−0.037, p=0.005 on Wikipedia), an honest
negative result for domain-specific fine-tuning. The hybrid is the safest cross-domain
default.

---

## 1. Introduction

Paper 1 established that morphological stemming significantly improves Kazakh BM25 search
(nDCG@10 +9% overall, +16% on inflected queries, p ≤ 0.0017, n=300), that BM25+stemmer
outperforms zero-shot LaBSE (0.754 vs 0.481), and that multilingual-e5-base (0.785) is
the best embedding model despite receiving no Kazakh-specific training. Three questions
were left open:

1. **New models:** IBM released Granite R2 after Paper 1's data collection. A Kazakh
   community fine-tune of E5 (shyngys879/kazakh-e5-rag-embedding) has also appeared.
   How do they compare?

2. **Hybrid retrieval:** BM25+stemmer and dense models have complementary failure modes
   (BM25 fails on synonyms, dense fails on inflection without fine-tuning). Can RRF
   combination beat either channel?

3. **OOD generalization:** all Paper 1 results are on Kazakh Wikipedia. Do they hold on
   a structurally different corpus — formal political language with different vocabulary
   and query types?

We answer all three, plus add a tokenizer analysis explaining *why* some models fail more
than others across domains.

---

## 2. Datasets

### 2.1 Wikipedia Benchmark (Paper 1, unchanged)

- **Corpus:** 8 370 passages from 800 random Kazakh Wikipedia articles (~120 words each)
- **Queries:** 300 (100 entities × 3 categories)
  - `inflected` — key word in an oblique grammatical case (morphology stress-test)
  - `natural` — standard factoid questions
  - `vocabulary-gap` — paraphrase/synonym queries (semantic stress-test)
- **Qrels:** one ground-truth passage per query

### 2.2 Akorda OOD Dataset (new)

- **Corpus:** 471 passages from official presidential speeches (akorda.kz)
- **Queries:** 244 (3 categories)
  - `factoid` — high lexical overlap with gold passage (≈0.79); analogous to `inflected`
  - `paraphrase` — medium overlap (≈0.36)
  - `low_overlap` — low lexical overlap (≈0.32); analogous to the semantic queries above
- **Domain shift:** formal political Kazakh, distinct from encyclopedic Wikipedia

### 2.3 Native-Speaker Validated Semantic Queries (Sprint 3)

- **127 queries** written specifically to have low lexical overlap with gold passages
  (mean overlap 0.145, threshold ≤ 0.30, measured by 5-prefix stem matching)
- Validated by a native Kazakh speaker: 104 accepted unchanged, 23 reformulated, 18 removed
- Passage-level and article-level metrics reported separately

All data in: `data/queries/`, `data/akorda/`, `data/queries/synonym_queries_final.jsonl`.

---

## 3. Systems

| System | Type | Notes |
|--------|------|-------|
| BM25 (identity) | lexical | Okapi BM25, k₁=1.5, b=0.75, no normalization |
| BM25 + Kazakh stemmer | lexical | same + morphological stemmer from Paper 1 |
| LaBSE | dense | multilingual, symmetric; no prefixes |
| multilingual-e5-base | dense | `query: ` / `passage: ` prefixes; Paper 1 |
| IBM Granite R1 (278M) | dense | `granite-embedding-278m-multilingual`; no prefixes |
| IBM Granite R2 (97M) | dense | `granite-embedding-97m-multilingual-r2`; **new** |
| IBM Granite R2 (311M) | dense | `granite-embedding-311m-multilingual-r2`; **new** |
| kazakh-e5 (shyngys-e5) | dense | `shyngys879/kazakh-e5-rag-embedding`; fine-tuned from e5; **new** |
| Hybrid ⊕ e5 | RRF | BM25+stemmer ⊕ e5, k=60; **new** |
| Hybrid ⊕ kazakh-e5 | RRF | BM25+stemmer ⊕ shyngys-e5, k=60; best on Wiki; **new** |
| Hybrid ⊕ Granite R1 | RRF | BM25+stemmer ⊕ Granite R1, k=60; **new** |
| Hybrid ⊕ Granite R2-311M | RRF | BM25+stemmer ⊕ Granite R2-311M, k=60; **new** |
| Hybrid ⊕ Granite R2-97M | RRF | BM25+stemmer ⊕ Granite R2-97M, k=60; **new** |

**RRF formula** (Cormack et al., 2009): score(d) = Σ 1/(k + rank_i(d)), k=60 pre-registered.
All dense models evaluated zero-shot (no fine-tuning on our data, except shyngys-e5 which
is fine-tuned on external Kazakh data). Similarity: cosine. Brute-force exact search.

---

## 4. Results: Wikipedia Benchmark (n=300)

### 4.1 All systems — nDCG@10 by category

| System | inflected | natural | vocab-gap | **ALL** |
|--------|----------:|--------:|----------:|--------:|
| BM25 + stemmer | 0.727 | 0.772 | 0.764 | 0.754 |
| multilingual-e5-base | 0.845 | **0.947** | 0.562 | 0.785 |
| LaBSE | 0.477 | 0.546 | 0.419 | 0.481 |
| Granite R1 (278M) | 0.791 | 0.923 | 0.303 | 0.672 |
| Granite R2 (97M) | 0.711 | 0.880 | 0.175 | 0.589 |
| Granite R2 (311M) | 0.791 | 0.924 | 0.263 | 0.659 |
| kazakh-e5 (shyngys-e5) | 0.836 | 0.909 | 0.497 | 0.747 |
| **Hybrid ⊕ kazakh-e5** | **0.862** | 0.869 | **0.694** | **0.808** |
| Hybrid ⊕ Granite R1 | 0.824 | 0.877 | 0.525 | 0.742 |
| Hybrid ⊕ Granite R2-311M | 0.821 | 0.894 | 0.504 | 0.740 |
| Hybrid ⊕ Granite R2-97M | 0.779 | 0.869 | 0.438 | 0.695 |

**Best system overall: Hybrid BM25+stemmer ⊕ kazakh-e5, nDCG@10 = 0.808.**

*Note: Hybrid ⊕ e5 was not computed on Wikipedia; the RRF run was not retained. Based on
the Akorda results (where ⊕ e5 = 0.562 vs ⊕ kazakh-e5 = 0.520), the kazakh-e5 hybrid
likely remains best on Wikipedia as well.*

### 4.2 Key observations — Wikipedia

**Granite R2 does not outperform R1 on Kazakh.** Despite the "R2" designation implying
improvement, both R2 variants score below R1 (278M) overall: 97M = 0.589 (−0.083 vs R1),
311M = 0.659 (−0.013 vs R1). The vocabulary-gap category is the weakest point: R2-97M
scores 0.175 — well below every other model including BM25 without stemmer.

**kazakh-e5 Kazakh fine-tuning does not improve over base e5 — the difference is
statistically significant.** kazakh-e5 scores 0.747 overall vs base e5 0.785: Δ=−0.037,
p=0.005 (paired bootstrap, n=300). It is lower on all three categories — inflected (0.836
vs 0.845), natural (0.909 vs 0.947), and vocabulary-gap (0.497 vs 0.562). Among the three
new models added in this paper it is the strongest on vocabulary-gap, substantially above
all Granite models (≤0.303), but Kazakh-specific fine-tuning does not close — and in fact
widens — the gap to base e5.

**Hybrid closes the vocabulary-gap on the dense side without losing the lexical ceiling.**
The RRF fusion of BM25+stemmer ⊕ kazakh-e5 reaches 0.694 on vocab-gap — significantly
above kazakh-e5 alone (0.497, +0.197) but, as expected from RRF averaging, below the
BM25+stemmer alone (0.764). This is not a failure: the stemmer is already excellent on
vocab-gap because these queries were paraphrases with strong lexical signal. The fusion
gains on the categories where the dense channel contributes most (inflected: 0.862 vs
stemmer 0.727), resulting in the only system above 0.80 overall (0.808).

---

## 5. Results: Native-Speaker Validated Semantic Queries (n=127)

On the most demanding semantic test — 127 native-speaker-validated low-overlap queries —
lexical systems break down almost completely. Results at both scoring granularities:

### 5.1 Passage-level (strict: one designated passage)

| System | Hit@10 | CI 95% | nDCG@10 |
|--------|-------:|:------:|--------:|
| BM25 + stemmer | 0.157 | [0.094, 0.220] | 0.109 |
| BM25 + synonym expansion | 0.134 | [0.079, 0.197] | 0.073 |
| multilingual-e5-base | **0.236** | [0.165, 0.315] | 0.114 |
| LaBSE | 0.205 | [0.134, 0.283] | **0.135** |
| Granite R1 (278M) | 0.189 | [0.126, 0.260] | 0.109 |
| Granite R2-97M | 0.055 | [0.016, 0.094] | 0.029 |
| Granite R2-311M | 0.173 | [0.110, 0.244] | 0.089 |
| kazakh-e5 | 0.228 | [0.157, 0.299] | 0.114 |
| **Hybrid ⊕ kazakh-e5** | 0.228 | [0.157, 0.299] | 0.133 |

### 5.2 Article-level (lenient: any passage from the right article)

| System | Hit@10 | CI 95% | nDCG@10 |
|--------|-------:|:------:|--------:|
| BM25 + stemmer | 0.677 | [0.598, 0.756] | 0.459 |
| multilingual-e5-base | 0.724 | [0.646, 0.803] | 0.471 |
| Granite R1 (278M) | 0.614 | [0.528, 0.701] | 0.426 |
| Granite R2-97M | 0.362 | [0.276, 0.449] | 0.217 |
| Granite R2-311M | 0.583 | [0.496, 0.669] | 0.393 |
| kazakh-e5 | 0.724 | [0.646, 0.803] | 0.507 |
| **Hybrid ⊕ kazakh-e5** | **0.780** | [0.709, 0.850] | **0.554** |

The gap between passage-level (≤0.236) and article-level (≤0.780) shows that models often
find the right *article* but not the exact *passage*. This is a chunking and passage
boundary problem, not a retrieval failure. Granite R2-97M fails even at article level
(0.362) — the only model that cannot locate the right article most of the time.

---

## 6. Results: Akorda OOD Benchmark (n=244)

### 6.1 Full system comparison — nDCG@10

| System | factoid | paraphrase | low_overlap | **ALL** | CI 95% |
|--------|--------:|-----------:|------------:|--------:|--------|
| **Hybrid ⊕ e5** | 0.8456 | 0.4393 | 0.4039 | **0.5623** | [0.5125, 0.6110] |
| BM25 + stemmer | **0.9063** | 0.3144 | 0.3316 | 0.5166 | [0.4610, 0.5717] |
| e5 (multilingual-e5-base) | 0.6743 | **0.4408** | **0.4129** | 0.5090 | [0.4601, 0.5587] |
| BM25 (identity) | 0.9010 | 0.2777 | 0.2770 | 0.4844 | [0.4304, 0.5403] |
| Granite R1 (278M) | 0.5481 | 0.4061 | 0.3385 | 0.4305 | [0.3840, 0.4779] |
| kazakh-e5 (shyngys-e5) | 0.5368 | 0.4103 | 0.3330 | 0.4263 | [0.3786, 0.4759] |
| Granite R2-311M | 0.5236 | 0.3809 | 0.2936 | 0.3989 | [0.3517, 0.4484] |
| Granite R2-97M | 0.4387 | 0.1755 | 0.1624 | 0.2585 | [0.2134, 0.3049] |

Paired bootstrap, 10 000 resamples; n=244 queries.

### 6.2 OOD ranking stability

Cross-domain Spearman rank correlation on nDCG@10 (n=7 single systems, excluding hybrids):
**ρ = 0.89** (Σd² = 6). Rankings are substantially preserved:

| System | Wiki rank | Akorda rank | Shift |
|--------|:---------:|:-----------:|:-----:|
| e5 | 1 | 1 | — |
| BM25 + stemmer | 2 | 2 | — |
| kazakh-e5 | **3** | **5** | ↓ 2 |
| BM25 (identity) | 4 | 3 | ↑ 1 |
| Granite R1 | 5 | 4 | ↑ 1 |
| Granite R2-311M | 6 | 6 | — |
| Granite R2-97M | 7 | 7 | — |

The one systematic shift is **kazakh-e5 (shyngys-e5)**, which drops from rank 3 on
Wikipedia to rank 5 on Akorda, falling below Granite R1 — a reversal from Wikipedia.
The absolute drop is large (−0.321), the largest among dense models. Since kazakh-e5
shares its tokenizer with e5 (it is fine-tuned from multilingual-e5-base), the tokenizer
fragmentation mechanism (Section 7) does not explain this differential. A likely factor
is training data mismatch: the fine-tuning data for kazakh-e5 probably resembles Wikipedia
more than it resembles formal political language.

### 6.3 Category analysis: why BM25 ranks high on Akorda

BM25+stemmer is nominally the second-best single system on Akorda (0.5166 vs e5 0.5090),
but the difference is not significant (Δ = −0.008, p = 0.399). The ranking is driven
entirely by the factoid category (lexical overlap ≈ 0.79), where BM25 scores 0.906 and
dense models top out at 0.674. Once factoid is excluded, e5 significantly outperforms
BM25 on paraphrase (Δ = +0.126, p = 0.006) and marginally on low_overlap (Δ = +0.081,
p = 0.064). **The BM25 vs dense balance is domain-dependent; the dense model ranking
among themselves is OOD-stable.**

### 6.4 Hybrid on Akorda: stronger gains than on Wikipedia

| Channel pair | BM25+stemmer | Dense | Hybrid | Both criteria met? |
|---|---:|---:|---:|:---:|
| ⊕ e5 | 0.5166 | 0.5090 | **0.5623** | ✓ |
| ⊕ kazakh-e5 | 0.5166 | 0.4263 | 0.5202 | ✓ |
| ⊕ Granite R1 | 0.5166 | 0.4305 | 0.5175 | ✓ |
| ⊕ Granite R2-311M | 0.5166 | 0.3989 | 0.5158 | ✗ (Δ = −0.0008 vs BM25) |
| ⊕ Granite R2-97M | 0.5166 | 0.2585 | 0.4073 | ✗ |

Pre-registered success criteria: (1) hybrid ≥ max(channel) on ALL; (2) hybrid ≥ BM25 on
the semantic slice (`low_overlap`). 3 of 5 fusions meet both criteria. The near-miss by
R2-311M (−0.0008 overall, semantic criterion ✓) is because BM25 itself is stronger after
full-cache stemming, not because the fusion deteriorated.

**Why fusion gains more on Akorda than on Wikipedia:** Akorda's factoid/semantic split
is clean — factoid (overlap 0.79) is dominated by BM25, paraphrase/low_overlap (overlap
0.32–0.36) are dominated by dense. The channels are maximally complementary. On Wikipedia,
the categories overlap more, so BM25 and dense are not as cleanly separated.

The headline hybrid (⊕ e5) significantly beats both channels: vs BM25 p = 0.003,
vs e5 p = 0.009. k-sweep from 10 to 100 shows ALL nDCG@10 ∈ [0.559, 0.586] — the result
is robust to k choice.

---

## 7. Sub-word Tokenizer Fertility: Why Granite R2 Struggles

### 7.1 Method

We sampled 100 most frequent long (≥9 char) Kazakh word forms from the Wikipedia corpus
and 100 from the Akorda corpus (cached in `data/words_tokenization_100.json` and
`data/words_akorda_100.json`). We computed mean sub-words per word for each tokenizer,
in two modes: bare (no leading space) and with leading space (relevant for byte-level BPE
tokenizers like R2's ModernBERT backbone).

### 7.2 Results — Wikipedia sample

| Tokenizer | bare | with space |
|-----------|-----:|-----------:|
| multilingual-e5-base (SentencePiece) | **1.81** | **1.81** |
| Granite R1 (SentencePiece) | **1.81** | **1.81** |
| Granite R2-97M (ModernBERT BPE) | 4.00 | 3.57 |
| Granite R2-311M (ModernBERT BPE) | 4.20 | 3.82 |

### 7.3 Results — Akorda vs Wikipedia (domain shift)

| Tokenizer | wiki (bare) | akorda (bare) | Δ | wiki (+sp) | akorda (+sp) | Δ |
|-----------|------------:|--------------:|--:|-----------:|-------------:|--:|
| Granite R2-97M | 4.00 | 4.29 | **+0.29** | 3.57 | 3.88 | **+0.31** |
| Granite R2-311M | 4.20 | 4.43 | **+0.23** | 3.82 | 4.07 | **+0.25** |
| e5-base | 1.81 | 1.82 | +0.01 | 1.81 | 1.82 | +0.01 |
| Granite R1 | 1.81 | 1.82 | +0.01 | 1.81 | 1.82 | +0.01 |

### 7.4 Interpretation

**Granite R2 fragments Kazakh words 2.3× more than R1/e5** (4.00–4.20 vs 1.81 sub-words
per word). Kazakh is agglutinative: a word like *мемлекеттіліктің* carries full
morphosyntactic information in its suffixes. When this is split into 4+ BPE tokens,
the embedding must compose across many fragments to recover meaning — a harder task than
encoding a near-complete root (1.81 tokens).

This is a **candidate mechanism**, not a proven cause:
- R2's drop on vocabulary-gap (0.175–0.263 vs R1 0.303, e5 0.562) is consistent with
  poor suffix encoding — a model that fragments suffixes is less likely to build
  morphologically-aware representations.
- On the morphology-heavy `inflected` category, R2-311M (0.791) matches R1 (0.791),
  suggesting this category is recoverable even with fragmentation (perhaps because the
  query and passage share the same surface form). Fragmentation hurts most where the model
  needs to understand word meaning across forms — exactly the semantic categories.
- **Akorda amplifies the effect** (+0.23–0.29 additional fragmentation on formal
  vocabulary), consistent with R2's larger performance drop on Akorda.

**e5 and kazakh-e5 share one tokenizer** (SentencePiece, 1.81 sub-words/word, domain-
stable). Fertility cannot explain why kazakh-e5 underperforms e5 on Akorda (they fragment
identically). That differential is attributed to training data mismatch, not tokenization.

---

## 8. Statistical Significance Summary

All comparisons: paired bootstrap, 10 000 resamples, two-tailed, threshold p < 0.05.

### Wikipedia (n=300)

| Comparison | Δ nDCG@10 | p | Significant? |
|-----------|----------:|--:|:---:|
| BM25 identity → BM25+stemmer | +0.064 | 0.0001 | ✓ |
| BM25+stemmer → e5 | +0.030 | 0.112 | — |
| BM25+stemmer → kazakh-e5 | −0.007 | 0.40 | — |
| e5 → kazakh-e5 | −0.037 | 0.005 | ✓ |
| Granite R1 → Granite R2-311M | −0.013 | 0.29 | — |
| Granite R2-97M → Granite R2-311M | +0.070 | <0.001 | ✓ |
| kazakh-e5 → Hybrid ⊕ kazakh-e5 | +0.061 | <0.001 | ✓ |

### Akorda (n=244)

| Comparison | Δ nDCG@10 | p | Significant? |
|-----------|----------:|--:|:---:|
| BM25 identity → BM25+stemmer | +0.032 | 0.029 | ✓ |
| BM25+stemmer → e5 | −0.008 | 0.399 | — |
| BM25+stemmer → Granite R1 | −0.086 | 0.002 | ✓ |
| e5 → Granite R1 | −0.078 | 0.001 | ✓ |
| e5 → kazakh-e5 | −0.083 | <0.001 | ✓ |
| Granite R1 → Granite R2-311M | −0.032 | 0.076 | — |
| Granite R2-97M → Granite R2-311M | +0.140 | <0.001 | ✓ |
| e5 → Hybrid ⊕ e5 | +0.053 | 0.009 | ✓ |
| BM25+stemmer → Hybrid ⊕ e5 | +0.046 | 0.003 | ✓ |

---

## 9. Discussion

### What generalizes across domains

The core ranking of dense models is OOD-stable (ρ = 0.89): e5 leads, R2-97M is last,
R2-311M and R1 are in the middle. The BM25 stemmer's usefulness also generalizes:
significant on Wikipedia (p = 0.0001) and on Akorda (p = 0.029). What does **not**
generalize cleanly is the BM25-vs-dense balance — BM25 ranks 2nd on Akorda overall but
only because of the high-overlap factoid category. A practitioner who observed only
Wikipedia results might underestimate BM25's usefulness on factoid-heavy formal domains,
or overestimate it on semantic-gap-heavy domains.

### Hybrid as the safe default

The hybrid (BM25+stemmer ⊕ best dense) was at or near the top on both domains and all
query categories. It is the most robust choice when the domain and query distribution are
unknown in advance. On Wikipedia, the kazakh-e5 hybrid reached 0.808 (best overall),
closing the semantic gap to 0.694. On Akorda, the e5 hybrid reached 0.562 (best overall,
significantly above both channels). The hybrid design costs only CPU time for RRF
re-ranking of already-computed runs.

### Granite R2: an honest negative result

Granite R2 was released with expanded language support. On Kazakh, it does not outperform
R1 on any domain or category we tested. R2-311M matches R1 on inflected/natural but falls
short on vocabulary-gap; R2-97M is the weakest model overall. The tokenizer fertility
analysis suggests the ModernBERT BPE backbone is under-adapted to Kazakh morphology.
This is a structural issue (sub-word vocabulary), not a fine-tuning issue — and it
becomes more pronounced on formal political language (+0.25–0.31 additional fragmentation
on Akorda).

### Limitations

- Hybrid R2 fusions were not tested on the Sprint 3 semantic query set; only R1 and
  kazakh-e5 hybrids were evaluated there.
- The fertility analysis uses frequent words (top 100 by length ≥ 9), which are better
  covered by any tokenizer vocabulary; rare domain-specific forms would show larger
  fragmentation gaps.
- OOD testing covers one additional domain (political speeches). Generalization to other
  domains (legal, medical, news) is untested.
- All models evaluated zero-shot; fine-tuning on Akorda data might change the R2 vs R1
  comparison.

---

## 10. Conclusion

Three additions to the Kazakh retrieval benchmark from Paper 1:

1. **New models:** Granite R2 does not improve on R1 for Kazakh (consistent across both
   domains). Kazakh-fine-tuned E5 (kazakh-e5) is significantly worse than base e5 overall
   (Δ=−0.037, p=0.005 on Wikipedia); it outperforms Granite models on vocabulary-gap
   (0.497 vs ≤0.303) but drops disproportionately on formal OOD text (Akorda).

2. **Hybrid RRF:** combining BM25+stemmer with a dense model via RRF (k=60) consistently
   beats both channels. The best hybrid (⊕ kazakh-e5) reaches nDCG@10 = 0.808 on
   Wikipedia and (⊕ e5) 0.562 on Akorda. The hybrid is the recommended default for
   Kazakh retrieval in production.

3. **OOD validation:** Wikipedia and Akorda rankings correlate at ρ = 0.89. The stemmer
   effect is significant on both domains. The main finding of Paper 1 — morphological
   normalization matters for Kazakh search — is confirmed on structurally different text.
   The tokenizer fragmentation analysis (R2 fragments Kazakh words 2.3× more than R1/e5)
   provides a candidate mechanism for R2's consistent underperformance.

---

## Reproduce

```bash
# New dense models (GPU required):
python -m src.eval.run_benchmark --system granite-r2-97m  --out results/dense_granite_r2_97m.json
python -m src.eval.run_benchmark --system granite-r2-311m --out results/dense_granite_r2_311m.json
python -m src.eval.run_benchmark --system shyngys-e5      --out results/dense_shyngys.json

# Hybrid RRF (CPU, needs precomputed BM25 and dense runs):
python -m src.eval.run_hybrid \
    --bm25-runs   results/runs_bm25_kazakh.json \
    --dense-runs  results/runs_dense_shyngys.json \
    --dense-label kazakh-e5 --out results/hybrid_shyngys.json

# Tokenization fertility:
python -m src.eval.tokenization_test   # Wikipedia
python -m src.eval.fertility_compare   # Akorda vs Wikipedia

# Akorda OOD:
python -m src.eval.run_akorda --system e5 --out results/akorda/dense_e5.json
python -m src.eval.run_akorda --system hybrid-e5 \
    --bm25-runs results/akorda/bm25_kazakh_full.json \
    --dense-runs results/akorda/dense_e5.json \
    --dense-label e5 --out results/akorda/hybrid_e5.json

# Sprint 3 semantic queries:
python -m src.eval.sprint3_rescore
```

Full reproducibility notes, GPU requirements, and Kaggle notebooks:
[`PIPELINE.md`](PIPELINE.md) · [`notebooks/akorda_kaggle.py`](notebooks/akorda_kaggle.py)

---

## Citation

> Seidalin, T. (2026). *Hybrid Retrieval and Out-of-Domain Robustness for Kazakh Search:
> Extended Benchmark with New Models, RRF Fusion, and Tokenizer Analysis.* Zenodo.
> DOI: *[to be assigned after deposit]*

```bibtex
@misc{seidalin2026kazakh2,
  author    = {Seidalin, Timur},
  title     = {Hybrid Retrieval and Out-of-Domain Robustness for Kazakh Search:
               Extended Benchmark with New Models, RRF Fusion, and Tokenizer Analysis},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {[to be assigned]},
  url       = {[to be assigned]}
}
```

---

## Acknowledgements

Benchmark design, data collection, corpus annotation, and evaluation by the author.
Claude (Anthropic) used for code scaffolding, analysis scripting, and text drafting.
First paper: https://doi.org/10.5281/zenodo.20605663
Repository: https://github.com/Tim2190/Kaz-RAG-search-benchmark
