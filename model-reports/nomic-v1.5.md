# Finding: `nomic-embed-text-v1.5` on Kazakh Retrieval

**Model:** [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
(prefixes: `search_query:` / `search_document:`)
**Evaluated:** zero-shot, two Kazakh domains, identical protocol to every other system in
the benchmark (nDCG@10, paired bootstrap 10 000 resamples, 512-token context).
**Status:** measured, committed. Vendor not yet contacted.

---

## TL;DR

Nomic v1.5 scores **nDCG@10 = 0.171** on Wikipedia Kazakh (300 queries) and **0.066** on
Akorda OOD (244 queries) — the weakest result in the benchmark, significantly below every
other system including BM25 without a stemmer. The root cause is the tokenizer:
`nomic-embed-text-v1.5` uses `nomic-bert-2048` with a **BERT English WordPiece vocabulary
(30 522 tokens)**. Kazakh-specific Cyrillic characters (ә, і, ң, ғ, ү, ұ, қ, ө, һ) are
entirely out-of-vocabulary and tokenize as `[UNK]`. With most query and document tokens
collapsed to `[UNK]`, the embedding space is effectively random on Kazakh text.

---

## 1. Results vs baselines

| Domain | Nomic v1.5 | e5 (best prev.) | BM25+stemmer | Δ vs e5 | p |
|--------|----------:|----------------:|-------------:|--------:|---|
| Wikipedia (n=300) | 0.171 | 0.785 | 0.754 | −0.614 | **<0.001** ✓ |
| Akorda OOD (n=244) | 0.066 | 0.509 | 0.517 | −0.443 | **<0.001** ✓ |

Comparison to every prior system:
[`../results/RESULTS.md`](../results/RESULTS.md) (Wiki),
[`../results/akorda/AKORDA_RESULTS.md`](../results/akorda/AKORDA_RESULTS.md) (Akorda).

## 2. Root cause: tokenizer has no Kazakh coverage

Sub-word fertility test (`src/eval/tokenization_test.py`, 100 long Kazakh wordforms):

| Tokenizer | vocab | fertility (bare) | Kazakh chars |
|-----------|------:|-----------------:|:-------------|
| e5-base / jina-v3 | 250 002 | 1.81 | covered (XLM-R vocab) |
| nomic-v1.5 | **30 522** | 3.49 | **[UNK] for Kazakh-specific chars** |
| granite-r2-311m | 262 144 | 4.20 | covered |

Nomic's vocabulary is BERT English WordPiece. Every token containing ә, і, ң, ғ, ү, ұ,
қ, ө, or һ becomes `[UNK]`. Example:
```
«мүмкіндіктерін» → [UNK]   (e5: ['▁мүмкіндіктері', 'н'])
«халықаралық»    → [UNK]   (e5: ['▁халықаралық'])
```
The fertility of 3.49 is measured on the subset of sampled words that happen to contain
only standard Cyrillic (shared with Russian); words with Kazakh-specific characters collapse
to a single `[UNK]` token regardless of length.

## 3. Hybrid does not rescue performance

RRF fusion with BM25+stemmer (k=60) reaches **0.228** on Akorda — significantly better
than Nomic alone (Δ=+0.162, p<0.001) but significantly **worse** than BM25 alone
(Δ=−0.289, p<0.001). The Nomic channel is so noisy that fusion degrades BM25. Both
pre-registered success criteria fail.

## 4. Reproduce

```bash
# Wikipedia (n=300)
python -m src.eval.run_dense --model nomic-v1.5 \
    --out results/dense_nomic_300.json \
    --runs-out results/runs_dense_nomic.json \
    --max-seq-len 512

# Akorda OOD (n=244)
python -m src.eval.run_akorda --system nomic-v1.5 --out results/akorda/dense_nomic.json

# Tokenizer fertility
python -m src.eval.tokenization_test
```
Kaggle notebook: [`../notebooks/nomic_kaggle.py`](../notebooks/nomic_kaggle.py).
Per-query runs committed: `results/runs_dense_nomic.json`, `results/akorda/dense_nomic.json`.

*Reference: preprint DOI [10.5281/zenodo.20781386](https://doi.org/10.5281/zenodo.20781386).*
