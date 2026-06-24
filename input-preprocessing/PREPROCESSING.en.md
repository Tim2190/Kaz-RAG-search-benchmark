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
`transform` applied to both sides.

## Models and domains

- **e5** (`intfloat/multilingual-e5-base`) — "good" model (XLM-R tokenizer):
  control for whether preprocessing hurts a strong model.
- **qwen3-0.6b** (`Qwen/Qwen3-Embedding-0.6B`) — "broken" (byte-fallback):
  the main test — does Latin rescue it?
- Domains: **Wikipedia** (n=300) and **Akorda** (n=244).

Model prefixes (`query: `/`passage: `, `Instruct: …`) are NOT preprocessed —
they are English instructions, appended after the Kazakh text is transformed.

## Protocol

- Metric **nDCG@10**, significance via paired bootstrap, 10,000 (`*` = p<0.05).
- Main comparison: Δ of each line (1,2,3) vs baseline (0), per slice.
- Transliteration is a hard-coded 2021-standard table
  (`src/preprocess/translit.py`), deterministic, no network / no external library.
- Run: `notebooks/preprocessing_kaggle.py` (GPU T4). Raw per-query rankings and
  `*_compare.json` are in `runs/`.

## Results

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

## Headline: you cannot fix the input by preprocessing without fine-tuning

1. **Transliteration does not rescue byte-fallback.** The main hypothesis fails:
   Latin significantly *worsens* qwen3 on both domains (wiki −0.076,
   akorda −0.114) rather than lifting it. The mechanism is not fragmentation: the
   fertility measurement (below) shows that for qwen3 Latin yields *fewer* tokens
   (6.2→4.5), yet retrieval still drops. So wholer Latin tokens carry no Kazakh
   meaning — the byte-fallback problem is **representational**, not about the
   writing system.

2. **Stemmer preprocessing hurts dense models.** Stripping agglutinative suffixes
   destroys the sub-word signal embeddings rely on. The "good" e5 suffers most
   (wiki −0.117, akorda −0.081). For qwen3 stemming is near-neutral on wiki
   (p=0.081) but significantly negative on akorda.

3. **Stacking both transforms is worst** in all 4 configs (stem_latin is always
   the lowest).

## Nuance: transliteration "levels" the models

| domain | model | baseline | latin |
|--------|-------|---------:|------:|
| wiki   | e5         | **0.7845** | 0.4743 |
| wiki   | qwen3-0.6b | 0.6903 | **0.6142** |
| akorda | e5         | **0.5090** | 0.2058 |
| akorda | qwen3-0.6b | 0.3304 | **0.2162** |

In Cyrillic, e5 is clearly ahead of qwen3. Under Latin they **swap**: qwen3 ≥ e5
on both domains. e5's edge is its strong Cyrillic tokenizer (XLM-R);
transliteration throws that edge away, so e5 collapses harder than byte-fallback
qwen3. No practical gain, but a vivid demonstration of where a good tokenizer's
strength on Kazakh comes from.

## Hardest slice (vocabulary-gap / low_overlap)

The hardest slice drops the most under Latin:

| config | baseline | latin (Δ) |
|--------|---------:|----------:|
| wiki / e5     · vocabulary-gap | 0.5622 | 0.2208 (−0.341 *) |
| wiki / qwen3  · vocabulary-gap | 0.3515 | 0.2343 (−0.117 *) |
| akorda / e5    · low_overlap   | 0.4129 | 0.0992 (−0.314 *) |
| akorda / qwen3 · low_overlap   | 0.2203 | 0.1720 (−0.048)   |

## Direct measurement: does Latin shred like Cyrillic?

To separate a tokenization cause from a semantic one, we measure **fertility**
(sub-words per word) of Cyrillic vs its transliteration on the same 100 frequent
forms per domain (`python -m src.eval.translit_fertility`). In raw UTF-8 bytes
Latin is shorter than Cyrillic, so the measurement isolates learned BPE merges,
not byte count.

| tokenizer | domain | cyr (bare) | lat (bare) | Δ |
|-----------|--------|-----------:|-----------:|---:|
| qwen3-0.6b | wiki   | 6.20 | 4.46 | **−1.74** |
| qwen3-0.6b | akorda | 6.27 | 4.75 | **−1.52** |
| e5-base    | wiki   | 1.81 | 3.61 | **+1.80** |
| e5-base    | akorda | 1.82 | 3.96 | **+2.14** |

The measurement separates two causes of the Latin collapse:

- **e5** splits Cyrillic almost one-token-per-word (1.81 — excellent
  XLM-R/SentencePiece coverage). Transliteration **doubles** fragmentation
  (3.6–4.0) → e5's collapse is **loss of its strong Cyrillic tokenizer**
  (tokenization cause).
- **qwen3** fragments Cyrillic catastrophically (6.20 — 3.4× worse than e5; hence
  its low baseline). But Latin gives it **fewer** tokens (4.46), wholer — **and
  retrieval still dropped significantly**. So qwen3's failure is **not explained
  by fragmentation**: wholer Latin tokens carry no Kazakh meaning
  (representational cause).

## Practical takeaway

Improving Kazakh retrieval for free with text-to-text preprocessing (stemmer /
transliteration) does not work — you need either a proper Cyrillic tokenizer
(XLM-R family) or fine-tuning for Kazakh. Kazakhstan's switch to Latin will not
fix the model on its own: even where transliteration gives qwen3 wholer tokens,
they carry no meaning and retrieval does not improve. It is not about the writing
system or token count, but about whether the model has seen Kazakh.
