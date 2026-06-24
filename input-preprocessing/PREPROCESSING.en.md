# input-preprocessing: fixing the input without fine-tuning (stemmer + transliteration)

*Русская версия: [PREPROCESSING.md](PREPROCESSING.md)*

Can we improve Kazakh retrieval by changing **only the input text** fed to the
model (text-to-text), with no fine-tuning at all? This layer answers that.

## Hypothesis

"Broken" embedders with byte-fallback tokenization (Qwen3, Cohere) shred Kazakh
**Cyrillic** into byte soup → collapse. Two input fixes:

1. **Stemmer** — morphological normalization (strip agglutinative suffixes so
   query and document meet at the stem).
2. **Cyrillic→Latin transliteration** (official 2021 standard) — tokenizers saw
   Latin more often in pretraining; byte-fallback may not trigger.

## Design: 4 lines, all vs baseline

| Line | Text transform before the model |
|------|---------------------------------|
| **0. baseline**   | raw text (reference point) |
| **1. stem**       | Kazakh stemmer |
| **2. latin**      | Cyrillic→Latin transliteration |
| **3. stem_latin** | stemmer, then transliteration |

**Symmetry (critical):** in every line the query AND the corpus go through the
same function — otherwise a Latin query against a Cyrillic corpus is an
artificial false failure. `src/preprocess/input_preprocess.py` returns one
`transform` that is applied to both sides.

## Models and domains

- **e5** (`intfloat/multilingual-e5-base`) — "good" model (XLM-R tokenizer):
  control for whether preprocessing hurts a strong model.
- **qwen3-0.6b** (`Qwen/Qwen3-Embedding-0.6B`) — "broken" (byte-fallback):
  the main test — does Latin rescue it?
- Domains: **Wikipedia** (n=300) and **Akorda** (n=244).

Model prefixes (`query: `/`passage: `, `Instruct: …`) are NOT preprocessed —
they are English instructions, appended after the Kazakh text is transformed.

## Protocol

- Metric: **nDCG@10**. Significance: **paired bootstrap, 10,000** (`*` = p<0.05).
- Raw per-query rankings: `runs/`. Per-line embedding cache: `emb/` (gitignored).
- Reproducibility: transliteration is a hard-coded 2021-standard table
  (`src/preprocess/translit.py`), no network / no external library.

## Results

Full sweep: 4 lines × 2 models × 2 domains (GPU T4, Kaggle). Numbers reconcile:
`Δ = line − baseline`.

### Wikipedia (n=300) — nDCG@10

| Model | baseline | stem (Δ) | latin (Δ) | stem_latin (Δ) |
|-------|---------:|---------:|----------:|---------------:|
| e5         | 0.7845 | 0.6671 (−0.117 *) | 0.4743 (−0.310 *) | 0.3994 (−0.385 *) |
| qwen3-0.6b | 0.6903 | 0.6701 (−0.020) | 0.6142 (−0.076 *) | 0.5866 (−0.104 *) |

### Akorda (n=244) — nDCG@10

| Model | baseline | stem (Δ) | latin (Δ) | stem_latin (Δ) |
|-------|---------:|---------:|----------:|---------------:|
| e5         | 0.5090 | 0.4277 (−0.081 *) | 0.2058 (−0.303 *) | 0.1837 (−0.325 *) |
| qwen3-0.6b | 0.3304 | 0.3011 (−0.029 *) | 0.2162 (−0.114 *) | 0.1555 (−0.175 *) |

Of 16 comparisons (line vs baseline, ALL), **15 are significantly negative**;
the only non-significant one is wiki/qwen3 stem (Δ −0.020, p=0.081). At best the
preprocessing is neutral, typically it hurts.

### Headline: you cannot fix the input by preprocessing without fine-tuning

1. **Transliteration does not rescue byte-fallback** (measured). The main
   hypothesis fails: Latin significantly *worsens* qwen3 on both domains
   (wiki −0.076, akorda −0.114) rather than lifting it. This is one of the
   pre-registered publishable outcomes.
   *Mechanism* (likely, not proven by retrieval alone): the tokenizer never saw
   Kazakh Latin in pretraining, so transliteration does not yield "wholer"
   tokens. This is inferred from the result, not measured directly — checked
   separately with a fertility measurement (below), not asserted.

2. **Stemmer preprocessing hurts dense models.** Stripping agglutinative
   suffixes destroys the sub-word signal embeddings rely on. The "good" e5
   suffers most (wiki −0.117, akorda −0.081). For qwen3 stemming is near-neutral
   on wiki (p=0.081) but significantly negative on akorda.

3. **Stacking both transforms is worst** in all 4 configs (stem_latin is always
   the lowest), as expected.

### A curious nuance: transliteration "levels" the models

| domain | model | baseline | latin |
|--------|-------|---------:|------:|
| wiki   | e5         | **0.7845** | 0.4743 |
| wiki   | qwen3-0.6b | 0.6903 | **0.6142** |
| akorda | e5         | **0.5090** | 0.2058 |
| akorda | qwen3-0.6b | 0.3304 | **0.2162** |

In Cyrillic, e5 is clearly ahead of qwen3. Under Latin they **swap**: qwen3 ≥ e5
on both domains. e5's edge is its strong Cyrillic tokenizer (XLM-R);
transliteration throws that edge away, so e5 collapses harder than byte-fallback
qwen3 (which treated everything as bytes anyway). No practical gain, but a vivid
demonstration of *where* a good tokenizer's strength on Kazakh comes from.

### Hardest slice (vocabulary-gap / low_overlap)

The hardest slice drops the most under Latin:

| config | baseline | latin (Δ) |
|--------|---------:|----------:|
| wiki / e5     · vocabulary-gap | 0.5622 | 0.2208 (−0.341 *) |
| wiki / qwen3  · vocabulary-gap | 0.3515 | 0.2343 (−0.117 *) |
| akorda / e5    · low_overlap   | 0.4129 | 0.0992 (−0.314 *) |
| akorda / qwen3 · low_overlap   | 0.2203 | 0.1720 (−0.048)   |

### Direct measurement: does Latin shred like Cyrillic?

The claim "the tokenizer doesn't know Kazakh Latin" is indirect (read off the
retrieval result). To avoid passing it off as fact, we measure **fertility**
(sub-words per word) of Cyrillic vs its transliteration on the same 100 words:

```bash
python -m src.eval.translit_fertility   # Kaggle/Colab, Internet ON (needs HF)
```

`src/eval/translit_fertility.py` compares Qwen3 and e5 on Cyrillic vs Latin.
Expectation: if fertility(Latin) ≈ or > fertility(Cyrillic), transliteration
does not give the tokenizer wholer tokens — confirming the result with data.
(Note: in raw UTF-8 bytes Latin is actually *shorter*, so this is not about byte
count but about learned BPE merges — which is exactly what the measurement tests.)

| tokenizer | domain | cyr (bare) | lat (bare) | Δ | verdict |
|---|---|---:|---:|---:|---|
| qwen3-0.6b | wiki | _TBD_ | _TBD_ | _TBD_ | _fill from Kaggle_ |
| qwen3-0.6b | akorda | _TBD_ | _TBD_ | _TBD_ | |
| e5-base | wiki | _TBD_ | _TBD_ | _TBD_ | |
| e5-base | akorda | _TBD_ | _TBD_ | _TBD_ | |

### Practical takeaway

Improving Kazakh retrieval for free with text-to-text preprocessing (stemmer /
transliteration) **does not work** — you need either a proper Cyrillic tokenizer
(XLM-R family) or fine-tuning for Kazakh. Kazakhstan's switch to Latin will
likely not fix the model on its own: per these results (and pending the
fertility measurement above), transliteration does not give the tokenizer wholer
Kazakh tokens.

> ⚠️ This layer is a standalone negative result; it is NOT merged into the main
> leaderboard. After review, a link can be added from `results/LEADERBOARD.md`.
