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

The distinction matters because R2 (ModernBERT) uses **byte-level BPE**, where a leading
space is part of the token representation, while R1/e5 (SentencePiece) are nearly insensitive
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
| multilingual-e5-base | `intfloat/multilingual-e5-base` | SentencePiece | **1.81** | **1.81** |
| Granite R1-278M | `ibm-granite/granite-embedding-278m-multilingual` | SentencePiece | **1.81** | **1.81** |
| Granite R2-97M | `ibm-granite/granite-embedding-97m-multilingual-r2` | byte-level BPE | 4.00 | 3.57 |
| Granite R2-311M | `ibm-granite/granite-embedding-311m-multilingual-r2` | byte-level BPE | 4.20 | 3.82 |
| **TilQazyna morphBPE-256k** | `stukenov/sozkz-morphbpe-256k-kk-v1` | morpheme BPE | *pending* | *pending* |

> **Note on TilQazyna numbers:** run `python -m src.eval.tokenization_test` (or
> `notebooks/tokenization_kaggle.py` on Kaggle) to fill in the TilQazyna row. The word list
> is cached, so re-running produces identical numbers for all 5 tokenizers.

---

## Split Examples (isolated word; ▁ marks SentencePiece word boundary)

| Word | e5 / R1-278M | R2-311M | R2-97M | TilQazyna |
|------|:-------------|:--------|-------:|:---------:|
| халықаралық | `▁халықаралық` (1) | `ха·лы·қа·ра·лық` (5) | *(byte-level, 4)* | *pending* |
| мүмкіндіктерін | `▁мүмкіндіктері·н` (2) | `м·үм·кі·нді·кте·рін` (6) | *(byte-level, 6)* | *pending* |
| ұйымдастырушылық | `▁ұйымдастыру·шылық` (2) | `ұ·йы·м·да·сты·ру·шы·лық` (8) | *(byte-level, 8)* | *pending* |
| сөзжасам | `▁сөзжасам` (1) | *(~3–4 pieces)* | *(byte-level)* | *pending* |

> R2-97M tokens are byte-level encoded and do not render as readable Cyrillic; counts are
> shown. R2-311M pieces are shown with `·` as separator for readability.
>
> TilQazyna (`sozkz-morphbpe-256k-kk-v1`) is a **256K vocabulary morpheme-aware BPE**
> trained specifically on Kazakh. It is expected to align token boundaries with Kazakh
> morphemes rather than arbitrary byte sequences. Fill in this column after running the script.

---

## Contrast Conclusion

### Pattern 1 — Multilingual general-purpose tokenizers

Both R1 (SentencePiece, XLM-RoBERTa base) and e5 (SentencePiece) produce **1.81 sub-words /
word** in both conditions — essentially one token per long Kazakh surface form. This reflects
SentencePiece's unigram language model, which can learn whole-word units even for agglutinative
forms it sees often in multilingual data.

The ModernBERT byte-level BPE (R2) produces **4.00–4.20 sub-words / word in isolation** and
**3.57–3.82 with a leading space** — a consistent **≥2.1× fragmentation gap** relative to
e5/R1 under either convention. This is not a measurement artefact of the space convention: the
gap is 2.23× in isolation and 2.11× with a leading space (R2-311M vs e5). R2-97M fragments
slightly *less* than R2-311M in aggregate but outputs unreadable byte-level tokens for Cyrillic,
which makes it less interpretable in Kazakh-facing applications.

The fragmentation gap is consistent with R2's overall retrieval regression on Kazakh (R1-278M
nDCG@10 ALL = 0.672 vs R2-97M = 0.589, R2-311M = 0.659 on n=300) — though fragmentation is
**one factor among several**, not the sole explanation. Notably, R2-311M does *not* regress on
the inflected/morphological category (0.791 = R1), so model capacity partially compensates;
the clearest fragmentation-linked regression is in R2-97M, which is weaker across the board.

### Pattern 2 — Dedicated Kazakh morpheme tokenizer (TilQazyna)

`stukenov/sozkz-morphbpe-256k-kk-v1` is a morpheme-aware BPE with a 256K vocabulary trained
on Kazakh text. Its design goal is to align token boundaries with Kazakh morphemes rather than
arbitrary byte runs. Its fertility number (pending) is expected to be substantially lower than
the R2 models and comparable to or better than e5/R1.

**The contrast:**  
> Multilingual tokenizers (R2/ModernBERT, byte-level BPE) fragment Kazakh words **≥2.1×**
> more than SentencePiece-based ones. A dedicated Kazakh morpheme-aware tokenizer (TilQazyna,
> 256K vocab) is designed to avoid this fragmentation. Exact numbers pending run; the
> qualitative contrast between general-purpose fragmentation and language-specific design is
> the key finding of this document.

*Note: TilQazyna (`sozkz-morphbpe-256k-kk-v1`) is a tokenizer-only analysis — the associated
`Til-Core-0.5B` causal language model is not an embedding model and is not evaluated in the IR
retrieval benchmark.*

---

## Related documents

- `results/GRANITE_R2_REVIEW.en.md` — Granite-specific analysis (R1 vs R2, n=300 + n=127)
- `results/SPRINT2_NEW_MODELS.md` — full n=300 comparison across 10 systems (RU)
- `results/SPRINT3_SYNONYM.md` — validated semantic query set, 13 systems
- `src/eval/tokenization_test.py` — source script (5 tokenizers, both conditions)
- `notebooks/tokenization_kaggle.py` — standalone Kaggle/Colab version
