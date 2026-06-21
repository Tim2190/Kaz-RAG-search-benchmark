# Finding: `jina-embeddings-v3` on Kazakh Retrieval

**Model:** [`jinaai/jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3)
(task-LoRA encoding: `retrieval.query` / `retrieval.passage`)
**Evaluated:** zero-shot, two Kazakh domains, identical protocol to every other system in
the benchmark (nDCG@10, paired bootstrap 10 000 resamples, 512-token context).
**Status:** measured, committed. Vendor not yet contacted.

---

## TL;DR

Jina v3 is the **strongest single dense model in the benchmark on both domains**, and the
margin over `multilingual-e5-base` is **statistically significant** in each case. It is the
first dense model to beat the lexical baseline (BM25 + Kazakh stemmer) on the out-of-domain
formal-speech corpus. Its remaining weakness is high-surface-overlap queries where the
morphological BM25 baseline still wins — the actionable hook for the Jina team on low-resource
Turkic languages.

---

## 1. Significant win over e5 on both domains

| Domain | Jina v3 nDCG@10 | e5 | Δ (Jina − e5) | p (paired bootstrap) |
|--------|----------------:|----:|--------------:|:---------------------:|
| Wikipedia (n=300) | **0.821** | 0.785 | +0.036 | **0.009** ✓ |
| Akorda OOD (n=244) | **0.613** | 0.509 | +0.104 | **<0.001** ✓ |

It also significantly beats the strong lexical baseline (BM25 + Kazakh stemmer):
Wikipedia Δ=+0.067 (p=0.004), Akorda Δ=+0.096 (p<0.001). On Akorda this makes Jina the
**first dense model to overtake the lexical baseline overall** — earlier dense models
(e5, Granite R1/R2, kazakh-e5) all lost to BM25+stemmer on that domain.

Full tables: [`../results/RESULTS.md`](../results/RESULTS.md) (Wiki),
[`../results/akorda/AKORDA_RESULTS.md`](../results/akorda/AKORDA_RESULTS.md) (Akorda).

## 2. Where it wins, by category

| Domain · category | Jina v3 | best competitor | note |
|-------------------|--------:|----------------:|------|
| Wiki · inflected | **0.910** | e5 0.845 | morphology handled well (+0.065 vs e5, p=0.008) |
| Wiki · natural | **0.957** | e5 0.947 | ceiling; tie (Δ=+0.010, p=0.26 n.s.) |
| Akorda · factoid | 0.704 | BM25+stem 0.906 | lexical still wins high-overlap factoid |
| Akorda · paraphrase | **0.590** | e5 0.441 | strong semantic bridging |
| Akorda · low_overlap | **0.546** | e5 0.413 | best on the genuine semantic-gap slice |

## 3. The actionable weakness (ping hook)

Jina's remaining gap is **high-surface-overlap queries** where the Kazakh morphological BM25
baseline still wins: on Wiki BM25+stemmer 0.764 vs Jina 0.596. The mechanism candidate is
sub-word under-adaptation to Kazakh morphology — on a low-resource Turkic language a simple
morphological normalizer still out-retrieves a SOTA multilingual embedder when surface lexical
signal is present, pointing at the tokenizer rather than the semantic model itself.

Sub-word fertility (mean sub-words/word, 100 long Kazakh wordforms) for Jina v3 vs
`multilingual-e5-base` is measured in `src/eval/tokenization_test.py`.

## 4. Reproduce

```bash
# Wikipedia (n=300)
python -m src.eval.run_dense --model jina-v3 \
    --out results/dense_jina_300.json --runs-out results/runs_dense_jina.json --max-seq-len 512

# Akorda OOD (n=244)
python -m src.eval.run_akorda --system jina-v3 --out results/akorda/dense_jina.json

# Sub-word fertility
python -m src.eval.tokenization_test
python -m src.eval.fertility_compare
```
Kaggle notebook: [`../notebooks/jina_kaggle.py`](../notebooks/jina_kaggle.py).
Per-query runs committed: `results/runs_dense_jina.json`, `results/akorda/dense_jina.json`.

## 5. Draft ping

> Hi — I maintain an independent Kazakh retrieval benchmark (Wikipedia + out-of-domain
> presidential speeches, BEIR-format, bootstrap significance). I ran `jina-embeddings-v3`
> zero-shot: it's the strongest dense model I've tested, significantly beating
> `multilingual-e5` on both domains and the first to beat a Kazakh morphological BM25
> baseline out-of-domain. Where it still trails lexical search is high-surface-overlap
> queries — consistent with sub-word under-adaptation to Kazakh morphology. Full numbers +
> per-query runs public: [https://github.com/Tim2190/Kaz-RAG-search-benchmark]. Happy to
> share the breakdown.

*Reference: preprint DOI [10.5281/zenodo.20781386](https://doi.org/10.5281/zenodo.20781386).*
