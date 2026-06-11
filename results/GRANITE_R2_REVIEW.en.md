# IBM Granite Embedding R2 on Kazakh Retrieval — A Review

## Abstract

We evaluate the IBM Granite Embedding R2 models (97M and 311M, ModernBERT
backbone) against the previous R1 generation (278M, XLM-RoBERTa backbone) on
Kazakh passage retrieval, using two independent test sets: a 300-query
categorized benchmark and a separate 127-query native-speaker-validated
semantic set. Across both, R2 does not improve over R1: the 311M-R2 matches R1
on morphological and natural queries but is lower on semantically-distant
queries, and the 97M-R2 regresses on every category. A tokenization analysis
shows the ModernBERT tokenizer fragments Kazakh words 2.1–2.3× more than the
R1/e5 tokenizer (robust to the leading-space convention), which is consistent
with the observed retrieval gap. We report methodology, confidence intervals,
and limitations in full.

---

## 1. Models

| Model | Generation | Params | Backbone | Context | Kazakh |
|-------|-----------|-------:|----------|--------:|--------|
| `granite-embedding-278m-multilingual` | R1 | 278M | XLM-RoBERTa | 512 | multilingual (implicit) |
| `granite-embedding-97m-multilingual-r2` | R2 | 97M | ModernBERT | 32K | listed (kk among 52) |
| `granite-embedding-278m-multilingual-r2` (311M) | R2 | 311M | ModernBERT | 32K | listed (kk among 52) |

## 2. Main benchmark (n=300), nDCG@10 by category

300 queries over 100 Kazakh-Wikipedia entities, three query categories
(*inflected* = morphological variation; *natural* = fluent paraphrase;
*vocabulary-gap* = see Limitations §6). Corpus: 8,370 passages.

| System | inflected | natural | vocabulary-gap | **ALL** |
|--------|----------:|--------:|---------------:|--------:|
| BM25 + stemmer | 0.727 | 0.772 | 0.764 | 0.754 |
| multilingual-e5-base | 0.845 | 0.947 | 0.562 | **0.785** |
| LaBSE | 0.477 | 0.546 | 0.419 | 0.481 |
| **Granite R1-278M** | 0.791 | 0.923 | 0.303 | 0.672 |
| **Granite R2-97M** | 0.711 | 0.880 | 0.175 | 0.589 |
| **Granite R2-311M** | 0.791 | 0.924 | 0.263 | 0.659 |
| Granite R2-311M ⊕ BM25+stemmer (RRF) | 0.821 | 0.894 | 0.504 | 0.740 |

recall@10 (ALL): R1-278M = 0.773, R2-97M = 0.697, R2-311M = 0.770.

**Observations.** R2-311M equals R1-278M on *inflected* (0.791) and *natural*
(0.923/0.924) but is lower on *vocabulary-gap* (0.263 vs 0.303) and on the
aggregate (0.659 vs 0.672). R2-97M is below R1 in every category. Neither R2
model reaches base multilingual-e5 (0.785), and an RRF hybrid of R2-311M with a
lexical channel (0.740) still does not.

## 3. Tokenization analysis

For 100 frequent, morphologically rich Kazakh surface forms (length ≥ 9
characters) drawn from the corpus, we measured the mean number of sub-word
tokens per word (lower = closer to the morpheme; less fragmentation). Frequent
words are, if anything, *better* known to a tokenizer, so this sampling is
conservative — it tends to understate rather than inflate any gap. We report two
conditions per word — the word in isolation and the word with a leading space —
because R2 (ModernBERT) uses byte-level BPE, in which a leading space is part of
the token, whereas R1/e5 (SentencePiece) are nearly insensitive to it.

| Tokenizer | sub-words / word (isolated) | sub-words / word (leading space) |
|-----------|---------------------------:|---------------------------------:|
| multilingual-e5-base | 1.81 | 1.81 |
| Granite R1-278M | 1.81 | 1.81 |
| Granite R2-97M | 4.00 | 3.57 |
| Granite R2-311M | 4.20 | 3.82 |

A leading space reduces the R2 counts slightly (byte-level BPE absorbs it) and
leaves the SentencePiece R1/e5 counts unchanged, as expected. Under either
condition R2 fragments Kazakh substantially more: ≈2.3× in isolation and ≈2.1×
with a leading space (R2-311M 3.82 vs 1.81). The gap is robust to the space
convention. Example splits (isolated; ▁ marks a SentencePiece word boundary):

| word | e5 / R1-278M | R2-311M | R2-97M |
|------|-------------|---------|-------:|
| халықаралық | `▁халықаралық` (1) | `ха·лы·қа·ра·лық` (5) | 4 |
| мүмкіндіктерін | `▁мүмкіндіктері·н` (2) | `м·үм·кі·нді·кте·рін` (6) | 6 |
| ұйымдастырушылық | `▁ұйымдастыру·шылық` (2) | `ұ·йы·м·да·сты·ру·шы·лық` (8) | 8 |

(R2-97M uses byte-level tokens that do not render as readable Cyrillic, so only
the count is shown; reproduce with `python -m src.eval.tokenization_test`.)

We treat this fragmentation as **one factor** contributing to R2's overall
regression on Kazakh, not as the sole cause and not as an explanation of any
single query category — see Limitations §6, where we note that the larger
R2-311M does *not* regress on the morphological category.

## 4. Validated semantic set (n=127), independent replication

The *vocabulary-gap* category above turned out not to isolate semantic
generalization (Limitations §6). We therefore built a separate, native-speaker-
validated set of 127 queries with low lexical overlap to their gold passage
(mean stemmed-token overlap 0.145, threshold ≤ 0.30) and re-ran the same models.
This is an independent replication on a different query distribution.

We report two relevance definitions with 95% bootstrap confidence intervals on
Hit@10: **passage-level** (only the assigned gold passage counts) and
**article-level** (any passage of the gold Wikipedia article counts — the
standard document-retrieval criterion).

| System | passage Hit@10 (CI95) | passage nDCG@10 | article Hit@10 (CI95) | article nDCG@10 |
|--------|:---------------------:|----------------:|:---------------------:|----------------:|
| **Granite R1-278M** | 0.189 [0.126, 0.260] | 0.109 | 0.614 [0.528, 0.701] | 0.426 |
| **Granite R2-311M** | 0.173 [0.110, 0.244] | 0.089 | 0.583 [0.496, 0.669] | 0.393 |
| **Granite R2-97M** | 0.055 [0.016, 0.094] | 0.029 | 0.362 [0.276, 0.449] | 0.217 |
| multilingual-e5-base (reference) | 0.236 [0.165, 0.315] | 0.114 | 0.724 [0.646, 0.803] | 0.471 |

The R1 ≥ R2-311M ≫ R2-97M ordering replicates on this independent set,
strengthening the conclusion that it is not an artifact of the n=300 query
construction. Confidence intervals for R1-278M and R2-311M overlap; the two are
statistically comparable rather than R2 being an improvement.

## 5. Methodology

> - **Main benchmark:** 300 queries (100 entities × 3 categories), 8,370 Kazakh-
>   Wikipedia passages. Metric: nDCG@10 / MRR@10 / recall@{1,5,10}.
> - **Validated set:** 127 queries. Construction: candidate queries were filtered
>   to lexical overlap ≤ 0.30 with the gold passage (prefix-5 stemming), then
>   reviewed by a native Kazakh speaker over 145 candidates — 104 accepted as-is,
>   23 reworded, 18 removed. Mean overlap of the final set: 0.145.
> - **Dense encoding:** all sentence-transformer models capped at
>   `max_seq_length = 512`. The R2 (ModernBERT) default context window (up to 32K)
>   triggers out-of-memory allocation on a 16 GB T4; passages are short, so the
>   cap removes no information. GPU memory was released between models.
> - **Statistics:** 95% confidence intervals via bootstrap (2,000 resamples) on
>   Hit@10. Two relevance definitions (passage-level, article-level) are reported
>   as lower and upper bounds.
> - **Reproducibility:** raw runs in `results/sprint3_runs.json`; scoring via
>   `python -m src.eval.sprint3_rescore`. n=300 results and tokenization test in
>   `results/SPRINT2_NEW_MODELS.md`.

## 6. Limitations

- **The n=300 `vocabulary-gap` category does not measure synonymy.** On
  validation it had the *highest* query–gold lexical overlap of the three
  categories (0.70), and the answer was present in the gold passage in 93% of
  cases. These are long descriptive questions dense with rare keywords, not
  synonym substitutions. The n=300 `vocabulary-gap` column should therefore be
  read as a keyword-overlap stress test, not a semantic-gap test. Section 4 (the
  validated set) is the corrected semantic measurement.
- **Sample size.** The validated set has 127 queries; Hit@10 confidence
  intervals are wide and the intervals of the top systems overlap. Differences
  between adjacent systems should be treated as suggestive, not significant. The
  R1-vs-R2 ordering is supported by agreement across the two independent sets
  rather than by significance on either alone.
- **Single-gold vs. document-level relevance.** We did not perform manual
  multi-passage pooling. We bracket the result instead: passage-level
  (lower bound) and article-level (upper bound, standard document retrieval).
- **Domain.** The corpus is Kazakh Wikipedia only; results may not transfer to
  other domains or registers.
- **Tokenization metric is one factor, not a single cause.** The 2.3× figure is a
  descriptive sub-word count on a 100-word sample, offered as a contributing
  factor, not a controlled ablation. Notably, R2-311M does not regress on the
  *inflected* (morphology) category (0.791, equal to R1), which shows that
  fragmentation alone does not determine category-level outcomes — model capacity
  evidently compensates in the 311M case. The clearest fragmentation-linked
  regression is the small R2-97M, which is weaker across the board. We also report
  isolated-word and leading-space tokenizations separately, since byte-level BPE
  (R2) and SentencePiece (R1/e5) handle a leading space differently.

## 7. Summary

On Kazakh retrieval, IBM Granite Embedding R2 does not improve over R1 on either
test set. R2-311M is comparable to R1-278M on morphological and natural queries
and lower on semantically-distant ones; R2-97M regresses across the board. The
ModernBERT tokenizer's heavier fragmentation of Kazakh surface forms is a
consistent candidate explanation. For Kazakh retrieval among the systems tested,
base multilingual-e5 and lexical–dense RRF hybrids remain the stronger choices.

---

*Companion documents: `results/SPRINT2_NEW_MODELS.md` (n=300 + tokenization),
`results/SPRINT3_SYNONYM.md` (validated set, all systems), `results/sprint3_final.json`
(scored runs with CI). Russian version: `results/GRANITE_R2_REVIEW.md`.
For a broader tokenization comparison including a dedicated Kazakh morpheme tokenizer, see
`results/TOKENIZATION.md`.*
