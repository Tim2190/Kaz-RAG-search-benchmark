# Tokenization Fertility: Kazakh Sub-Word Fragmentation — 5-Tokenizer Comparison

## Setup

**Word list:** 100 most-frequent Kazakh surface forms of length ≥ 9 characters from the
benchmark corpus (Kazakh Wikipedia, 8,370 passages). Frequent long forms are, if anything,
*better known* to a tokenizer than rare ones, so this sampling is conservative — it tends to
*understate* rather than inflate fragmentation gaps.  
**Word list cache:** `data/words_tokenization_100.json` (generated on first run, fixed for
reproducibility — all 5 tokenizers run on the identical set).

**Two conditions per word:**
- **Isolated** — the bare word string `"сөзжасам"`
- **Leading space** — the word with a prepended space `" сөзжасам"`

The distinction matters because byte-level BPE tokenizers (R2, TilQazyna) encode a leading
space as part of the token, while SentencePiece tokenizers (R1/e5) are nearly insensitive
to it. Reporting both conditions makes the comparison robust to the tokenization convention.

**Metric:** mean sub-words per word (lower = less fragmentation = morphology more intact).

**Reproduce:**

```bash
python -m src.eval.tokenization_test
# Kaggle/Colab: see notebooks/tokenization_kaggle.py
```

---

## Fertility Table (mean sub-words / word, n=100)

| Tokenizer | HuggingFace ID | Architecture | Isolated | +Leading Space |
|-----------|----------------|-------------|--------:|--------------:|
| multilingual-e5-base | `intfloat/multilingual-e5-base` | SentencePiece | 1.81 | 1.81 |
| Granite R1-278M | `ibm-granite/granite-embedding-278m-multilingual` | SentencePiece | 1.81 | 1.81 |
| Granite R2-97M | `ibm-granite/granite-embedding-97m-multilingual-r2` | byte-level BPE | 4.00 | 3.57 |
| Granite R2-311M | `ibm-granite/granite-embedding-311m-multilingual-r2` | byte-level BPE | 4.20 | 3.82 |
| **TilQazyna morphBPE-256k** | `stukenov/sozkz-morphbpe-256k-kk-v1` | byte-level BPE (256K vocab) | **1.64** | **1.28** |

---

## Split Examples (isolated word; ▁ marks SentencePiece word boundary)

| Word | e5 / R1-278M | R2-311M | R2-97M | TilQazyna |
|------|:-------------|:--------|-------:|:---------:|
| халықаралық | `▁халықаралық` (1) | `ха·лы·қа·ра·лық` (5) | *(byte-level, 4)* | *(byte-level, **1**)* |
| мүмкіндіктерін | `▁мүмкіндіктері·н` (2) | `м·үм·кі·нді·кте·рін` (6) | *(byte-level, 6)* | *(byte-level, **1**)* |
| ұйымдастырушылық | `▁ұйымдастыру·шылық` (2) | `ұ·йы·м·да·сты·ру·шы·лық` (8) | *(byte-level, 8)* | *(byte-level, **1**)* |
| сөзжасам | `▁сөзжасам` (1) | `с·өз·жа·сам` (4) | *(byte-level, 5)* | *(byte-level, **1**)* |

> R2-97M and TilQazyna both use byte-level BPE and output tokens that do not render as
> readable Cyrillic — only counts are shown. The critical difference is in those counts:
> R2-97M produces 4–8 tokens per word; TilQazyna produces **1 token per word** for all
> four examples, its 256K Kazakh-specific vocabulary absorbing whole words / long surface
> forms as single tokens. (We measure fragmentation — token *counts*. Whether those single
> tokens correspond to morphemes is TilQazyna's design claim, not something visible in the
> byte-level output here.) Reproduce with `python -m src.eval.tokenization_test`.

---

## Contrast Conclusion

### Two architectures, opposite outcomes

Both R2 (ModernBERT-based model) and TilQazyna use **byte-level BPE** — the same underlying algorithm.
The outputs look superficially similar: both produce unreadable byte-level token strings for
Cyrillic. The vocabulary sizes are also in the same ballpark (179K–262K). Yet their
fragmentation rates are opposite extremes — which makes the contrast sharper, not weaker:

| Tokenizer | Architecture | Vocab size | Isolated | +Space |
|-----------|-------------|----------:|--------:|------:|
| Granite R2-311M | byte-level BPE | 262,144 | 4.20 | 3.82 |
| Granite R2-97M | byte-level BPE | 179,934 | 4.00 | 3.57 |
| e5-base / R1-278M | SentencePiece | 250,002 | 1.81 | 1.81 |
| **TilQazyna morphBPE-256k** | byte-level BPE | **256,000** | **1.64** | **1.28** |

> Vocab sizes printed from `tokenizer.vocab_size` on the loaded tokenizers (reproduce with
> `python -m src.eval.tokenization_test`).

The cause is vocabulary allocation across languages, not vocabulary size. R2's vocabulary
is actually large — granite-311m-r2 has 262,144 entries, more than e5/R1 (250,002) and
TilQazyna (256,000) — yet it fragments Kazakh far more. The difference is coverage: R2's
vocabulary is multilingual and spreads its budget across many languages and scripts, leaving
Kazakh under-represented — which the high fragmentation directly demonstrates. e5/R1's
similarly-sized multilingual SentencePiece vocabulary (XLM-R lineage) happens to cover
Cyrillic and many Kazakh surface forms well, and TilQazyna's 256K vocabulary is dedicated
entirely to Kazakh. So for a low-resource language, what matters is how much of the
vocabulary actually covers that language, not its raw size. IBM describes the r2 tokenizer
change as motivated by better low-resource coverage; for Kazakh specifically the measured
fragmentation is higher than r1, not lower.

### The fragmentation gap

- **R2 vs e5/R1 (isolated):** 4.20 / 1.81 = **2.32×** more fragmentation
- **R2 vs TilQazyna (isolated):** 4.20 / 1.64 = **2.56×** more fragmentation
- **R2 vs TilQazyna (+space):** 3.82 / 1.28 = **2.98×** more fragmentation

TilQazyna outperforms even SentencePiece-based e5/R1 on isolated words (1.64 < 1.81),
and substantially better with a leading space (1.28 vs 1.81). The +space advantage in
TilQazyna is pronounced because its large vocabulary encodes the space+word combination
as a single entry, eliminating the boundary split entirely.

### The core contrast

> **A multilingual general-purpose byte-level BPE whose vocabulary is multilingual but
> Kazakh-sparse (R2) fragments Kazakh words ≥2.5× more than a dedicated Kazakh tokenizer
> (TilQazyna) — despite R2-311M having the larger vocabulary (262K vs 256K).** The fragmentation problem is
> not inherent to byte-level BPE, and it is not a matter of raw vocabulary size: it is a
> consequence of how little of the vocabulary covers Kazakh. SentencePiece tokenizers (e5/R1)
> occupy a middle ground: their broad multilingual vocabularies incidentally preserve many
> Kazakh forms, giving fertility close to TilQazyna (1.81 vs 1.64 isolated).

### Scope of this analysis

This is a **descriptive fertility measurement**, not a controlled ablation:

- Fragmentation is **one factor** contributing to R2's retrieval regression on Kazakh — not
  the sole cause. R2-311M does not regress on the morphological query category (nDCG@10 =
  0.791, equal to R1), which shows that model capacity partially compensates for fragmentation;
  the clearest regression is in R2-97M, which is weaker across the board.
- TilQazyna (`sozkz-morphbpe-256k-kk-v1`) is a **tokenizer-only analysis** here. The
  associated Til-Core-0.5B model is a causal language model, not an embedding model, and
  is not evaluated in the IR retrieval benchmark.
- Frequent words are better known to any tokenizer (conservative sampling direction — tends
  to understate gaps).

---

## Related documents

- `results/GRANITE_R2_REVIEW.en.md` — Granite-specific analysis (R1 vs R2, n=300 + n=127)
- `results/SPRINT2_NEW_MODELS.md` — full n=300 comparison across 10 systems (RU)
- `results/SPRINT3_SYNONYM.md` — validated semantic query set, 13 systems
- `src/eval/tokenization_test.py` — source script (5 tokenizers, both conditions)
- `notebooks/tokenization_kaggle.py` — standalone Kaggle/Colab version
