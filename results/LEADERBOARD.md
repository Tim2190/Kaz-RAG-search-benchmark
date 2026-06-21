# Kazakh IR Benchmark — Leaderboard

**Corpus:** 8 370 passages (Kazakh Wikipedia, n=300 queries) +
471 passages (Akorda official speeches, n=244 queries)  
**Primary metric:** nDCG@10 · **Significance:** paired bootstrap 10 000 resamples

> This is a summary view only. For full metrics, per-category breakdown, and
> significance tables see the canonical files:
> - Wikipedia: [`RESULTS.md`](RESULTS.md)
> - Akorda (OOD): [`akorda/AKORDA_RESULTS.md`](akorda/AKORDA_RESULTS.md)

---

## All Models

| Model | HF id | Wiki nDCG@10 | Akorda nDCG@10 | Vocab size | Fertility (Wiki) | Headline finding |
|-------|-------|-------------:|---------------:|-----------:|-----------------:|-----------------|
| **Jina v3** | `jinaai/jina-embeddings-v3` | **0.821** | **0.613** | 250 002 | 1.81 | Best single model on both domains; gain over e5 is purely semantic — same XLM-R tokenizer |
| **Hybrid ⊕ Jina v3** | *(RRF k=60)* | — | **0.615** | — | — | Only Akorda evaluated; meets both pre-registered success criteria |
| **Dense E5** | `intfloat/multilingual-e5-base` | 0.785 | 0.509 | 250 002 | 1.81 | Strong baseline; same XLM-R vocabulary as Jina |
| **Hybrid ⊕ kazakh-e5** | *(RRF k=60)* | 0.808 | 0.562 | — | — | Best hybrid on Wikipedia; BM25 saves vocab-gap where dense is weak |
| **BM25 + Stemmer** | *(lexical)* | 0.754 | 0.517 | — | — | Most balanced system; never drops below 0.727 on any Wiki slice |
| **Dense kazakh-e5** | `shyngys879/kazakh-e5-rag-embedding` | 0.747 | 0.426 | 250 002 | 1.81 | Fine-tuned from e5; underperforms base e5 on Akorda (domain shift) |
| **Dense Granite R1** | `ibm-granite/granite-embedding-278m-multilingual` | 0.672 | 0.431 | 128 000 | 2.32 | Strong on morphology/natural; collapses on vocab-gap (0.303) |
| **BM25** | *(lexical)* | 0.690 | 0.484 | — | — | Unstemmed baseline |
| **Dense Granite R2-311M** | `ibm-granite/granite-embedding-311m-multilingual-r2` | 0.659 | 0.399 | 128 000 | 5.30 | ModernBERT backbone; fragments Kazakh 2.3× more than R1 tokenizer |
| **Dense Granite R2-97M** | `ibm-granite/granite-embedding-97m-multilingual-r2` | 0.589 | 0.259 | 128 000 | 5.30 | Smaller R2 variant; largest gap vs R1 on Akorda |
| **Dense LaBSE** | `sentence-transformers/LaBSE` | 0.481 | — | 501 153 | — | Naive multilingual; beaten by BM25+stemmer on every category |
| **Dense BGE-M3** | `BAAI/bge-m3` | TBD | TBD | TBD | TBD | — |
| **Dense Qwen3-0.6B** | `Qwen/Qwen3-Embedding-0.6B` | TBD | TBD | TBD | TBD | — |
| **Dense Nomic v1.5** | `nomic-ai/nomic-embed-text-v1.5` | 0.171 | 0.066 | 30 522 | 3.49† | Weakest model; English BERT WordPiece — Kazakh-specific Cyrillic entirely [UNK] |

> † Nomic fertility figure is misleading — Kazakh-specific chars (ә, і, ң…) become
> `[UNK]` (single token), artificially compressing the count. The model has no
> Kazakh coverage, not "low fragmentation".

---

## Notes

- **Fertility** = mean sub-words per word on 100 frequent Kazakh words (len≥9) from the
  Wikipedia corpus, tokenized without a leading space. Lower = less fragmentation.
- **Hybrid** = BM25+Kazakh-Stemmer ⊕ dense, RRF k=60 (pre-registered).
- LaBSE Akorda and several hybrid variants not yet evaluated (marked —).
- BGE-M3 and Qwen3-0.6B results pending Kaggle runs (see
  [`../notebooks/bge_m3_kaggle.py`](../notebooks/bge_m3_kaggle.py) and
  [`../notebooks/qwen3_embed_kaggle.py`](../notebooks/qwen3_embed_kaggle.py)).
- For per-model deep dives see [`../model-reports/INDEX.md`](../model-reports/INDEX.md).
