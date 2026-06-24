# Kazakh IR Benchmark — Leaderboard

**Corpus:** 8 370 passages (Kazakh Wikipedia, n=300 queries) +
471 passages (Akorda official speeches, n=244 queries)  
**Primary metric:** nDCG@10 · **Significance:** paired bootstrap 10 000 resamples

> This is a summary view only. For full metrics, per-category breakdown, and significance
> tables see the canonical files — every score below links to its source:
> - Wikipedia: [`RESULTS.md`](RESULTS.md)
> - Akorda (OOD): [`akorda/AKORDA_RESULTS.md`](akorda/AKORDA_RESULTS.md)
> - Per-model deep dives: [`../model-reports/INDEX.md`](../model-reports/INDEX.md)

---

## 1. Single Models (dense + lexical baselines)

Sorted by Wikipedia nDCG@10. Each score links to its full-metrics section.

| # | Model | HF id | Wiki nDCG@10 | Akorda nDCG@10 | Vocab | Fertility | Report | Headline finding |
|--:|-------|-------|:------------:|:--------------:|------:|----------:|:------:|-----------------|
| 1 | **BGE-M3** | `BAAI/bge-m3` | [**0.866**](RESULTS.md#dense-bge-m3--n300) | [**0.679**](akorda/AKORDA_RESULTS.md#main-results--ndcg10) | 250 002 | 1.81 | [📄](../model-reports/bge-m3.md) | Best on both domains; beats Jina (Wiki Δ=+0.045 p=0.0001; Akorda Δ=+0.066 p<0.001); first dense model to beat its own hybrid |
| 2 | **Jina v3** | `jinaai/jina-embeddings-v3` | [0.821](RESULTS.md#main-result-ndcg10-n300) | [0.613](akorda/AKORDA_RESULTS.md#main-results--ndcg10) | 250 002 | 1.81 | [📄](../model-reports/jina-v3.md) | 2nd best; gain over e5 is purely semantic — identical XLM-R tokenizer |
| 3 | **Cohere embed-v4.0** | `cohere/embed-v4.0` (API) | [0.800](RESULTS.md#dense-cohere-embed-v40--n300) | [0.367](akorda/AKORDA_RESULTS.md#main-results--ndcg10) | ~byte-BPE | 6.20 § | [📄](../model-reports/cohere-embed-v4.md) | Ties Jina/E5 on Wiki, but **largest cross-domain drop in benchmark (−0.433)**; collapses on Akorda despite marketing for cross-lingual retrieval (100+ langs) |
| 4 | **E5** | `intfloat/multilingual-e5-base` | [0.785](RESULTS.md#dense-e5--n300) | [0.509](akorda/AKORDA_RESULTS.md#main-results--ndcg10) | 250 002 | 1.81 | — | Strong baseline; same XLM-R vocab as BGE-M3 and Jina |
| 5 | **BM25 + Stemmer** | *(lexical)* | [0.754](RESULTS.md#bm25--stemmer--n300) | [0.517](akorda/AKORDA_RESULTS.md#main-results--ndcg10) | — | — | — | Most balanced; never below 0.727 on any Wiki slice; best on Wiki vocab-gap † |
| 6 | **kazakh-e5** | `shyngys879/kazakh-e5-rag-embedding` | [0.747](RESULTS.md#main-result-ndcg10-n300) | [0.426](akorda/AKORDA_RESULTS.md#main-results--ndcg10) | 250 002 | 1.81 | — | Fine-tuned from e5; **below base e5 on both domains** (Wiki 0.747 vs 0.785; Akorda 0.426 vs 0.509) — Akorda drop is disproportionately larger than Wiki gap |
| 7 | **BM25** | *(lexical)* | [0.690](RESULTS.md#bm25--n300) | [0.484](akorda/AKORDA_RESULTS.md#main-results--ndcg10) | — | — | — | Unstemmed lexical baseline |
| 8 | **Qwen3-0.6B** | `Qwen/Qwen3-Embedding-0.6B` | [0.690](RESULTS.md#dense-qwen3-06b--n300) | [0.330](akorda/AKORDA_RESULTS.md#main-results--ndcg10) | 151 643 | 6.20 § | [📄](../model-reports/qwen3-embed-0.6b.md) | 2nd-largest cross-domain drop (−0.360); byte-level BPE fragmentation (6.20 §); below BM25+stemmer on both domains despite claimed 100+ language support |
| 9 | **Granite R1 (278M)** | `ibm-granite/granite-embedding-278m-multilingual` | [0.672](RESULTS.md#dense-granite--n300) | [0.431](akorda/AKORDA_RESULTS.md#main-results--ndcg10) | 250 002 | 1.81 | [📄](../model-reports/INDEX.md) | Strong on morphology/natural; collapses on Wiki vocab-gap (0.303) |
| 10 | **Granite R2-311M** | `ibm-granite/granite-embedding-311m-multilingual-r2` | [0.659](RESULTS.md#main-result-ndcg10-n300) | [0.399](akorda/AKORDA_RESULTS.md#main-results--ndcg10) | 262 144 | 4.20 | [📄](../model-reports/INDEX.md) | ModernBERT backbone; fragments Kazakh ~2.3× more than R1; no gain over R1 |
| 11 | **KazEmbed-V5** | `Nurlykhan/kazembed-v5` | [0.642](RESULTS.md#dense-kazembed-v5--n300) | [0.389](akorda/AKORDA_RESULTS.md#main-results--ndcg10) | 250 002 | 1.81 | [📄](../model-reports/kazembed-v5.md) | Fine-tuned from e5-base for Kazakh; **below the base model on both domains** (Wiki Δ=−0.143, p<0.001; Akorda Δ=−0.120, p<0.001) — in-domain KazQAD gain does not transfer |
| 12 | **Granite R2-97M** | `ibm-granite/granite-embedding-97m-multilingual-r2` | [0.589](RESULTS.md#main-result-ndcg10-n300) | [0.259](akorda/AKORDA_RESULTS.md#main-results--ndcg10) | 179 934 | 4.00 | [📄](../model-reports/INDEX.md) | Smaller R2; among the weakest dense models — only LaBSE and Nomic score lower on Wiki |
| 13 | **LaBSE** | `sentence-transformers/LaBSE` | [0.481](RESULTS.md#dense-labse--n300) | — | 501 153 | — | — | Naive multilingual; beaten by BM25+stemmer on every Wiki category (Akorda not run) |
| 14 | **Nomic v1.5** | `nomic-ai/nomic-embed-text-v1.5` | [0.171](RESULTS.md#dense-nomic-v15--n300) | [0.066](akorda/AKORDA_RESULTS.md#main-results--ndcg10) | 30 522 | 3.49 ‡ | [📄](../model-reports/nomic-v1.5.md) | Weakest; English BERT WordPiece — Kazakh-specific Cyrillic entirely `[UNK]` |

---

## 2. Hybrid Systems (RRF, BM25+Stemmer ⊕ dense, k=60)

Reciprocal Rank Fusion of the lexical channel (BM25+Stemmer) with each dense channel.
Sorted by Akorda nDCG@10 (more complete coverage). Scores link to the hybrid sections.

| Dense channel | Wiki hybrid | Akorda hybrid | Notes |
|---------------|:-----------:|:-------------:|-------|
| **bge-m3** | — | [0.633](akorda/AKORDA_RESULTS.md#hybrid-rrf-bm25stemmer--dense) | Hybrid **worse** than BGE-M3 alone (0.679, Δ=−0.046 p=0.014) — fusion hurts when dense dominates |
| **jina-v3** | — | [0.615](akorda/AKORDA_RESULTS.md#hybrid-rrf-bm25stemmer--dense) | Statistical tie with Jina alone (Δ=+0.0015, p=0.475); meets both criteria |
| **e5** | — | [0.562](akorda/AKORDA_RESULTS.md#hybrid-rrf-bm25stemmer--dense) | Significantly beats **both** channels (vs BM25 p=0.003, vs e5 p=0.009) |
| **kazakh-e5** | [0.808](RESULTS.md#main-result-ndcg10-n300) | [0.520](akorda/AKORDA_RESULTS.md#hybrid-rrf-bm25stemmer--dense) | Best Wiki hybrid; only Wiki fusion to beat its best single channel |
| **granite-r1** | [0.742](RESULTS.md#hybrid-retrieval-rrf-can-bm25stemmer-and-granite-be-combined-n300) | [0.518](akorda/AKORDA_RESULTS.md#hybrid-rrf-bm25stemmer--dense) | Wiki: falsified (below BM25 on ALL, vocab-gap leaks Granite's 0.303 collapse) |
| **granite-r2-311m** | [0.740](RESULTS.md#hybrid-retrieval-rrf-can-bm25stemmer-and-granite-be-combined-n300) | [0.516](akorda/AKORDA_RESULTS.md#hybrid-rrf-bm25stemmer--dense) | Narrowly misses ALL ≥ max(channel) on Akorda |
| **cohere-v4** | — | [0.501](akorda/AKORDA_RESULTS.md#cohere-embed-v40-hybrid-both-criteria-fail) | Both criteria fail — dense (0.367) too weak; but overall loss vs BM25 not significant (p=0.176), and fusion beats dense alone (+0.134) |
| **kazembed-v5** | — | [0.493](akorda/AKORDA_RESULTS.md#kazembed-v5-hybrid-criterion-1-fails-criterion-2-passes) | Criterion 1 fails, criterion 2 passes — low_overlap hybrid (0.352) > BM25 (0.332); overall loss vs BM25 not significant (p=0.123); hybrid beats dense alone (+0.104, p<0.001) |
| **qwen3-0.6b** | — | [0.464](akorda/AKORDA_RESULTS.md#hybrid-rrf-bm25stemmer--dense) | Both criteria fail — dense (0.330) too weak; hybrid worse than BM25+stemmer alone (p=0.003) |
| **granite-r2-97m** | [0.695](RESULTS.md#hybrid-retrieval-rrf-can-bm25stemmer-and-granite-be-combined-n300) | [0.407](akorda/AKORDA_RESULTS.md#hybrid-rrf-bm25stemmer--dense) | Both criteria fail — dense channel too weak |
| **nomic-v1.5** | — | [0.228](akorda/AKORDA_RESULTS.md#hybrid-rrf-bm25stemmer--dense) | Both criteria fail — near-zero dense channel drags BM25 down |

> **BGE-M3, Jina v3, E5, Cohere embed-v4, Qwen3, Nomic, and KazEmbed-V5 hybrids were not evaluated on Wikipedia** (only Akorda); the Wiki
> hybrid study (`RESULTS.md`) pre-dates them and used kazakh-e5 + the three Granite variants.

---

## 3. Notes & Benchmark Caveats

**Metric & method.** Primary metric is nDCG@10. All significance tests are paired bootstrap
with 10 000 resamples, p<0.05 two-tailed. Hybrid = BM25+Kazakh-Stemmer ⊕ dense via
Reciprocal Rank Fusion at the pre-registered k=60.

**Fertility** = mean sub-words per Kazakh word over 100 frequent long words (len≥9) from the
Wikipedia corpus, tokenized without a leading space. Lower = less fragmentation. Models
sharing the XLM-R SentencePiece vocabulary (BGE-M3, Jina, E5, kazakh-e5, KazEmbed-V5, Granite R1) all
sit at 1.81 — so for those models fertility **cannot** explain ranking differences; the
gaps are architectural/training, not tokenizer.

> **‡ Nomic fertility is misleading.** Kazakh-specific characters (ә, і, ң, ғ, ү, ұ, қ, ө, һ)
> map to a single `[UNK]` token regardless of word length, artificially *compressing* the
> count to 3.49. This is zero coverage, not low fragmentation — the root cause of Nomic's
> collapse. See [`../model-reports/nomic-v1.5.md`](../model-reports/nomic-v1.5.md).

> **§ Byte-level BPE over-fragmentation (Qwen3 **and** Cohere embed-v4.0).** Qwen3-0.6B's
> 151 643-token BPE vocabulary falls back to byte-level tokenization for Kazakh Cyrillic: each
> character encodes as 2–3 byte-piece tokens displayed as UTF-8 byte sequences (e.g.,
> `«халықаралық»` → `['Ñħ', 'Ð°Ð»', 'Ñĭ', 'ÒĽ', 'Ð°ÑĢ', 'Ð°Ð»', 'Ñĭ', 'ÒĽ']`, 8 tokens).
> Unlike Nomic's `[UNK]` collapse (absent characters → 1 token each, *compressing* count),
> these models genuinely tokenize every character — but at byte granularity, producing true
> over-fragmentation; the fertility count 6.20 correctly reflects sequence-length inflation,
> not an artifact. **Cohere embed-v4.0 is empirically identical to Qwen3** here: fertility
> 6.20 / 6.27 (measured via the Cohere tokenize API on the same 100 words); token counts are
> identical on all four probe words (the API returns IDs only, not string pieces, so splits are
> inferred from counts) — consistent with the same byte-level BPE mechanism for Kazakh.
> See [`../model-reports/qwen3-embed-0.6b.md`](../model-reports/qwen3-embed-0.6b.md) and
> [`../model-reports/cohere-embed-v4.md`](../model-reports/cohere-embed-v4.md).

> **† The Wikipedia `vocab-gap` category does not measure what its name suggests.** Despite
> being designed as the semantic-gap test, it was found to have the *highest* query↔gold
> lexical overlap of the three Wiki categories (≈0.56, vs 0.51 natural / 0.47 inflected):
> the LLM-generated "encyclopedic riddle" queries unintentionally reused key terms from the
> gold passage. So BM25+Stemmer topping this column (0.764) reflects **strong lexical signal,
> not closing a semantic gap** — and dense models "collapsing" here (e.g. Granite R1 at 0.303)
> are penalised on a lexically-favorable category, not a genuinely semantic one.

> **The genuine low-overlap test is the Akorda `low_overlap` category** (≈0.32 overlap, by
> construction). There the ordering matches theory: dense models (BGE-M3 0.610, Jina 0.546,
> e5 0.413) beat BM25+Stemmer (0.332). When reading the two domains together, treat Akorda
> `low_overlap` — not Wiki `vocab-gap` — as the semantic-retrieval benchmark.

**Coverage gaps.** LaBSE was not run on Akorda. Wikipedia hybrids exist only for kazakh-e5
and the three Granite variants (the hybrid study pre-dates BGE-M3/Jina/Cohere/KazEmbed-V5).
All other systems in §1 have complete results on both domains.

**Cross-domain.** Dense-model rank ordering is largely OOD-stable (Wiki↔Akorda Spearman
ρ=0.89 over the original 7 systems); the BM25-vs-dense balance is what shifts with the
domain's lexical-overlap distribution. BGE-M3 shows the smallest absolute drop
(Wiki→Akorda −0.187) among dense models, while the two byte-level-BPE models show the two
largest: **Cohere embed-v4.0 (−0.433)** and Qwen3 (−0.360). The sharpest single finding of
the Cohere run is that a model explicitly marketed for cross-lingual retrieval (100+ languages)
ranks #3 on Wikipedia (0.800) yet collapses to 10th on Akorda (0.367, below BM25+stemmer,
E5, and six other dense models) — strong Wikipedia performance does not guarantee robustness
on formal, morphologically dense Kazakh.

---

**Side study — fixing the input without fine-tuning (attempt #1).** Can stemming or
Cyrillic→Latin transliteration repair byte-fallback tokenizers (Qwen3) without training?
A self-contained study (4 lines × e5/qwen3 × Wiki/Akorda, paired bootstrap) finds **no**:
every transform hurts (11 of 12 line-vs-baseline comparisons significantly), and a direct fertility measurement shows the
byte-fallback failure is representational, not script-level (transliteration even gives
Qwen3 *fewer* tokens, yet retrieval still drops). Self-contained in
[`../input-preprocessing/`](../input-preprocessing/PREPROCESSING.en.md); not part of the
rankings above.
