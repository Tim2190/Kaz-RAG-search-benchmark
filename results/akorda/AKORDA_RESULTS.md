# Akorda OOD Benchmark — Results

**Dataset:** Akorda (official presidential speeches, akorda.kz)  
**Protocol:** identical to the main Wiki benchmark — same systems, same metrics, same prefixes  
**n = 244 queries** · **471 passages** · **paired bootstrap 10 000 resamples**

---

## Dataset Categories

| Category | n | Lexical overlap (avg) | Description |
|----------|--:|-----------------------:|-------------|
| factoid | 81 | ≈ 0.79 | Direct factoid questions, high BM25-friendly overlap |
| paraphrase | 81 | ≈ 0.36 | Paraphrased questions |
| low_overlap | 82 | ≈ 0.32 | Low lexical overlap (corrected semantic category; distinct from the Wiki vocab-gap category, which had unintentionally high overlap by construction) |

---

## Main Results — nDCG@10

| System | nDCG@10 | CI 95% | factoid | paraphrase | low_overlap |
|--------|--------:|--------|--------:|-----------:|------------:|
| **hybrid: bm25+stemmer ⊕ e5** (RRF) | **0.5623** | [0.5125, 0.6110] | 0.8456 | 0.4393 | 0.4039 |
| bm25+stemmer | 0.5166 | [0.4610, 0.5717] | **0.9063** | 0.3144 | 0.3316 |
| e5 (multilingual-e5-base) | 0.5090 | [0.4601, 0.5587] | 0.6743 | **0.4408** | **0.4129** |
| bm25+identity | 0.4844 | [0.4304, 0.5403] | 0.9010 | 0.2777 | 0.2770 |
| granite-r1 (278M) | 0.4305 | [0.3840, 0.4779] | 0.5481 | 0.4061 | 0.3385 |
| shyngys-e5 | 0.4263 | [0.3786, 0.4759] | 0.5368 | 0.4103 | 0.3330 |
| granite-r2-311m | 0.3989 | [0.3517, 0.4484] | 0.5236 | 0.3809 | 0.2936 |
| granite-r2-97m | 0.2585 | [0.2134, 0.3049] | 0.4387 | 0.1755 | 0.1624 |

BM25+stemmer numbers reflect the **100%-cache** run (full stemmer coverage). The single best
system on Akorda is the **hybrid** (BM25+stemmer ⊕ e5). BM25+stemmer and e5 are not
significantly different overall (Δ=−0.008, p=0.399) — see Findings 2–3.

---

## Full Metrics Table

| System | nDCG@10 | MRR@10 | Recall@5 | Recall@10 |
|--------|--------:|-------:|---------:|----------:|
| hybrid: bm25+stemmer ⊕ e5 | 0.5623 | 0.4981 | 0.6475 | 0.7664 |
| bm25+stemmer | 0.5166 | 0.4754 | 0.5492 | 0.6516 |
| e5 | 0.5090 | 0.4388 | 0.6025 | 0.7336 |
| bm25+identity | 0.4844 | 0.4421 | 0.5246 | 0.6230 |
| granite-r1 (278M) | 0.4305 | 0.3536 | 0.5369 | 0.6762 |
| shyngys-e5 | 0.4263 | 0.3627 | 0.5287 | 0.6270 |
| granite-r2-311m | 0.3989 | 0.3291 | 0.4713 | 0.6270 |
| granite-r2-97m | 0.2585 | 0.2142 | 0.3115 | 0.4016 |

---

## Statistical Significance (paired bootstrap, nDCG@10, n=10 000 resamples)

| Comparison (A → B) | Δ | p-value | Significant? |
|--------------------|--:|--------:|:------------:|
| bm25-identity → bm25+stemmer | +0.0323 | 0.029 | ✓ |
| bm25+stemmer → e5 | −0.0077 | 0.399 | — |
| bm25+stemmer → granite-r1 | −0.0861 | 0.002 | ✓ |
| bm25+stemmer → shyngys-e5 | −0.0903 | 0.002 | ✓ |
| e5 → granite-r1 | −0.0784 | 0.001 | ✓ |
| e5 → granite-r2-311m | −0.1100 | <0.001 | ✓ |
| e5 → granite-r2-97m | −0.2505 | <0.001 | ✓ |
| e5 → shyngys-e5 | −0.0826 | <0.001 | ✓ |
| granite-r1 → granite-r2-311m | −0.0316 | 0.076 | — |
| granite-r2-97m → granite-r2-311m | +0.1404 | <0.001 | ✓ |

Positive Δ = B is better than A. Threshold: p < 0.05 (two-tailed paired bootstrap).

### By-category significance (nDCG@10)

| Comparison | factoid (n=81) | paraphrase (n=81) | low_overlap (n=82) | Overall (n=244) |
|------------|:--------------:|:-----------------:|:-----------------:|:---------------:|
| bm25+stemmer → e5 | Δ=−0.232 **p<0.001** | Δ=+0.126 **p=0.006** | Δ=+0.081 p=0.064 n.s. | Δ=−0.008 p=0.399 n.s. |
| e5 → granite-r1 | Δ=−0.126 **p<0.001** | Δ=−0.035 p=0.195 n.s. | Δ=−0.074 **p=0.048** | Δ=−0.078 **p=0.001** |
| bm25-identity → e5 | Δ=−0.227 **p<0.001** | Δ=+0.163 **p<0.001** | Δ=+0.136 **p=0.004** | Δ=+0.025 p=0.195 n.s. |

---

## Hybrid RRF (BM25+stemmer ⊕ dense)

Same protocol as the Wiki hybrid (`src/eval/run_hybrid.py`): Reciprocal Rank Fusion at
the pre-registered **k=60** (Cormack et al., 2009), fusing the lexical channel
(BM25+stemmer, 100%-cache) with each dense channel. RRF merges *ranks*, not raw scores,
so no score calibration is involved. Pure CPU re-ranking of the already-computed runs.

### All five fusions — nDCG@10 overall

| dense channel | bm25+stemmer | dense | **hybrid** | hybrid ≥ max(channel)? | low_overlap hybrid ≥ bm25? |
|---------------|------------:|------:|-----------:|:----------------------:|:--------------------------:|
| **e5** | 0.5166 | 0.5090 | **0.5623** | ✓ | ✓ |
| shyngys-e5 | 0.5166 | 0.4263 | 0.5202 | ✓ | ✓ |
| granite-r1 | 0.5166 | 0.4305 | 0.5175 | ✓ | ✓ |
| granite-r2-311m | 0.5166 | 0.3989 | 0.5158 | ✗ | ✓ |
| granite-r2-97m | 0.5166 | 0.2585 | 0.4073 | ✗ | ✗ |

Both pre-registered success criteria are met for **3 of 5** dense channels. The r2-311m
hybrid (0.5158) narrowly misses the "ALL ≥ max(channel)" bar by 0.0008 — because BM25
itself is now stronger (0.517) after full-cache stemming — while still meeting the semantic
criterion. The r2-97m failure is structural: its dense channel (0.259) is so weak it
drags the fusion below BM25 alone.

### Headline hybrid: BM25+stemmer ⊕ e5

This is the best system on Akorda (nDCG@10 = 0.5623). It significantly beats **both** of
its own channels — not just the weaker one:

| Slice | hybrid | Δ vs bm25 | p | Δ vs e5 | p |
|-------|-------:|----------:|--:|--------:|--:|
| factoid (n=81) | 0.8456 | −0.0607 | **0.006** | +0.1713 | **<0.001** |
| paraphrase (n=81) | 0.4393 | +0.1249 | **<0.001** | −0.0016 | 0.491 n.s. |
| low_overlap (n=82) | 0.4039 | +0.0723 | **0.005** | −0.0090 | 0.423 n.s. |
| **ALL (n=244)** | **0.5623** | **+0.0456** | **0.003** | **+0.0533** | **0.009** |

The mechanism is visible in the slices: the hybrid recovers most of BM25's factoid
strength (0.846 vs BM25 0.906, only −0.061) while retaining e5's semantic categories
(paraphrase/low_overlap differences vs e5 are not significant). Overall the hybrid is
significantly above both channels: vs BM25 p=0.003, vs e5 p=0.009.

### Sensitivity to k (not cherry-picked)

nDCG@10 of the headline hybrid is stable across the RRF constant; k=60 is the
pre-registered canonical value, not tuned for the best score:

| k | factoid | paraphrase | low_overlap | ALL |
|---|--------:|-----------:|------------:|----:|
| 10 | 0.8539 | 0.4642 | 0.4410 | 0.5858 |
| 30 | 0.8482 | 0.4393 | 0.4144 | 0.5667 |
| 60 | 0.8456 | 0.4393 | 0.4039 | 0.5623 |
| 100 | 0.8401 | 0.4369 | 0.4033 | 0.5595 |

ALL stays in [0.559, 0.586] across the full sweep — the hybrid's advantage does not
depend on the choice of k.

### Contrast with Wiki

On **Wiki**, fusion helped far less: only the BM25⊕shyngys hybrid beat its best single
channel on ALL (0.808 vs 0.754); the BM25⊕granite-r1, ⊕r2-311m and ⊕r2-97m hybrids all
fell *below* the BM25 channel on ALL. On **Akorda**, 3 of 5 (and a near-miss 4th)
hybrids beat their best channel. The reason ties directly to Finding 5: Akorda's categories
cleanly separate lexical-favoring (factoid, overlap ≈0.79) from dense-favoring
(paraphrase/low_overlap, overlap ≈0.32–0.36), so the two channels are more complementary
and fusion gains more. The domain-dependence of the BM25-vs-dense balance also makes
the **hybrid** the safest default across domains: it was at or near the top on both.

---

## Key Findings

### 1. BM25 factoid anomaly — not an anomaly

BM25 nDCG@10 = **0.906** on factoid queries (lexical overlap ≈ 0.79). This is expected by design: factoid queries were constructed with high word-form overlap with the passage, making exact-match BM25 near-optimal. Dense models top out at 0.674 (e5) on this category. The BM25 "overall" lead over e5 evaporates once factoid is disaggregated: on paraphrase (p=0.006) e5 significantly outperforms BM25+stemmer; on low_overlap the gap is p=0.064 (marginal).

### 2. BM25+stemmer and e5 are a statistical tie at the top

With 100%-cache stemming, BM25+stemmer nDCG@10 = 0.5166 and e5 = 0.5090. The overall difference is Δ=−0.008 (BM25 nominally ahead), p=0.399 — not significant. The result is domain-split: BM25+stemmer dominates on factoid (Δ=−0.232, p<0.001) while e5 significantly leads on paraphrase (Δ=+0.126, p=0.006) and marginally on low_overlap (Δ=+0.081, p=0.064). Among dense systems, e5 is significantly better than every other model (p<0.001). The hybrid BM25+stemmer ⊕ e5 surpasses both (0.562, p=0.003 vs BM25, p=0.009 vs e5) — see Finding 7.

### 3. Stemmer effect is significant with full cache

bm25-identity → bm25+stemmer (100%-cache): Δ=+0.032, **p=0.029**. The effect is concentrated in semantic categories: paraphrase Δ=+0.037 (p=0.142 per slice, pooled significance from overall bootstrap) and low_overlap Δ=+0.055 (p=0.067 per slice). Factoid is unaffected (Δ=+0.005, p=0.362) — as expected, since stemmer helps semantic recall, not exact-match lexical precision. For reference, the stemmer effect on Wiki was Δ=+0.064, p<0.01 (larger; Wiki corpus is morphologically more diverse). An earlier 78%-cache run gave Δ=+0.014, p=0.196 (n.s.); the non-significance was a cache artifact, not a property of the domain.

### 4. BM25+stemmer outperforms granite-r1 and shyngys-e5

Both dense systems fail to match BM25+stemmer on Akorda overall (p=0.002 each). The factoid category accounts for most of this: BM25 nDCG@10 = 0.906 vs granite-r1 = 0.548 on factoid.

### 5. Dense model ranking is OOD-stable; BM25 vs dense balance is domain-dependent

Cross-dataset Spearman rank correlation on nDCG@10 (n=7 systems): **ρ=0.89** (Σd²=6). The ranking is substantially preserved — e5 is #1 on both domains, granite-r2-97m is last on both — but not perfectly identical. The one systematic shift is shyngys-e5, which moves from rank 3 on Wiki to rank 5 on Akorda:

| System | Wiki nDCG@10 | Wiki rank | Akorda nDCG@10 | Akorda rank | Shift |
|--------|-------------:|:---------:|---------------:|:-----------:|:-----:|
| e5 | 0.785 | 1 | 0.509 | 1 | — |
| bm25+stemmer | 0.754 | 2 | 0.517 | 2 | — |
| shyngys-e5 | 0.747 | **3** | 0.426 | **5** | ↓ 2 |
| bm25+identity | 0.690 | 4 | 0.484 | 3 | ↑ 1 |
| granite-r1 | 0.672 | 5 | 0.431 | 4 | ↑ 1 |
| granite-r2-311m | 0.659 | 6 | 0.399 | 6 | — |
| granite-r2-97m | 0.589 | 7 | 0.259 | 7 | — |

**shyngys-e5** drops disproportionately on formal political text (−0.321 absolute, the second largest drop after r2-97m), falling below granite-r1 on Akorda — a relationship reversed on Wiki. On Akorda the shyngys-e5 vs granite-r1 difference is not significant (p>0.05), so the practical gap is small, but the direction reversal is real.

**BM25 vs dense** is the other domain-sensitive relationship. On Wiki, BM25+stemmer was clearly behind e5 and shyngys-e5 in absolute terms. On Akorda, BM25+stemmer ranks #2 overall, nominally tied with e5 (p=0.399), — but this is driven entirely by the factoid category (lexical overlap ≈0.79): when factoid is excluded, e5 leads on paraphrase and low_overlap. The core finding is that **the ranking of dense models among themselves is OOD-stable, while the BM25-vs-dense balance depends on the domain's lexical overlap distribution**.

### 6. R2 does not outperform R1 on any domain

Granite R2-97M nDCG@10 = 0.259 — more than 2× below e5, and significantly worse than all other systems. R2-311M (0.399) is significantly better than R2-97M (p<0.001) but still significantly below e5 (p<0.001). On neither domain does R2 outperform R1: on Wiki R2-311M scores lower (0.659 vs R1 0.672); on Akorda the gap is similar in direction (0.399 vs 0.431) but does not reach significance (p=0.076, n=244). The sub-word fertility analysis (see below) suggests a tokenizer-level candidate mechanism: Granite R2 fragments Akorda formal vocabulary +0.23–0.31 sub-words/word more than Wiki, while e5/R1 are domain-stable (Δ≈+0.01). The pattern is consistent — R2 is not an improvement over R1 for Kazakh — but the Akorda result should be read as "within noise" rather than a confirmed gap.

### 7. The hybrid is the best system — and the safest cross-domain default

On Akorda the single best system is the **hybrid BM25+stemmer ⊕ e5** (nDCG@10 = 0.562), which significantly beats *both* of its channels overall (vs BM25 p=0.003, vs e5 p=0.009). 3 of 5 dense channels yield a hybrid that beats their best single channel on ALL; a 4th (r2-311m) misses by 0.0008 but still meets the semantic criterion. Only r2-97m fails both criteria. This is a stronger fusion result than on Wiki, where only the shyngys hybrid beat its best channel. The reason follows directly from Finding 5: Akorda separates lexical-favoring (factoid) from dense-favoring (paraphrase/low_overlap) cleanly, so the channels are complementary. Because the BM25-vs-dense balance is domain-dependent, the hybrid is the safest default — it was at or near the top on both domains, neither pure channel was. Full breakdown, significance, and k-sensitivity in the **Hybrid RRF** section above.

---

## Sub-word Fertility: Akorda vs Wiki

*Candidate mechanism for R2's domain drop. Run: `python -m src.eval.fertility_compare`*

Sample: 100 most frequent long words (≥9 chars) from each corpus.  
Method: average sub-words per word (bare = no leading space; +sp = with leading space).  
TilQazyna tokenizer skipped (gated, 401). Note: shyngys-e5 is fine-tuned from multilingual-e5 and shares e5's tokenizer; their fertility values are identical by construction.

| Tokenizer | wiki (bare) | akorda (bare) | Δbare | wiki (+sp) | akorda (+sp) | Δ+sp |
|-----------|------------:|--------------:|------:|-----------:|-------------:|-----:|
| granite-97m-r2 | 4.00 | 4.29 | **+0.29** | 3.57 | 3.88 | **+0.31** |
| granite-311m-r2 | 4.20 | 4.43 | **+0.23** | 3.82 | 4.07 | **+0.25** |
| e5-base | 1.81 | 1.82 | +0.01 | 1.81 | 1.82 | +0.01 |
| granite-278m-r1 | 1.81 | 1.82 | +0.01 | 1.81 | 1.82 | +0.01 |

**Interpretation (cautious):**

- **Granite R2** fragments Akorda formal vocabulary +0.23–0.31 sub-words/word more than Wiki. This is a *candidate mechanism* for R2's universal domain drop: increased fragmentation degrades embedding coherence, consistent with the observed performance gap.
- **e5 / Granite R1** are tokenizer-domain-stable (Δ≈+0.01). Their performance drop on Akorda is not explained by tokenizer fragmentation — other factors (training data distribution, model capacity) dominate.
- **shyngys-e5 and e5 share one tokenizer**: fertility cannot explain why shyngys underperforms e5 specifically on Akorda. That differential (Finding 5, rank 3→5) requires a different explanation (e.g., fine-tuning data mismatch with formal political language).
- This is correlation + theoretical plausibility, not proven causation. Frequent words are over-represented in the tokenizer vocab, so Δ is likely *underestimated* on rarer domain-specific forms.

---

## Comparison with Wiki Benchmark

| System | Wiki nDCG@10 | Akorda nDCG@10 | Drop |
|--------|-------------:|---------------:|-----:|
| e5 | 0.785 | 0.509 | −0.276 |
| bm25+stemmer | 0.754 | 0.517 | −0.237 |
| bm25+identity | 0.690 | 0.484 | −0.206 |
| granite-r1 | 0.672 | 0.431 | −0.241 |
| shyngys-e5 | 0.747 | 0.426 | −0.321 |
| granite-r2-311m | 0.659 | 0.399 | −0.260 |
| granite-r2-97m | 0.589 | 0.259 | −0.330 |

All systems drop substantially on Akorda. The largest drops are granite-r2-97m (−0.330) and shyngys-e5 (−0.321), smallest is bm25+identity (−0.206). The relative ordering is largely preserved (Spearman ρ=0.89, n=7), with one systematic shift: shyngys-e5 moves from rank 3 on Wiki to rank 5 on Akorda (see Finding 5).

---

## Reproduce

```bash
# BM25 (CPU):
python -m src.eval.run_akorda --system bm25         --out results/akorda/bm25_identity.json
python -m src.eval.run_akorda --system bm25-stemmer --out results/akorda/bm25_kazakh_full.json

# Dense (GPU):
python -m src.eval.run_akorda --system e5              --out results/akorda/dense_e5.json
python -m src.eval.run_akorda --system granite-r1      --out results/akorda/dense_granite_r1.json
python -m src.eval.run_akorda --system granite-r2-97m  --out results/akorda/dense_granite_r2_97m.json
python -m src.eval.run_akorda --system granite-r2-311m --out results/akorda/dense_granite_r2_311m.json
python -m src.eval.run_akorda --system shyngys-e5      --out results/akorda/dense_shyngys.json

# Hybrid RRF (CPU, fuses already-computed runs; --dense-label sets channel name):
python -m src.eval.run_akorda --system hybrid-e5 \
    --bm25-runs results/akorda/bm25_kazakh_full.json \
    --dense-runs results/akorda/dense_e5.json \
    --dense-label e5 --out results/akorda/hybrid_e5.json
# (likewise hybrid-granite-r1, hybrid-granite-r2-97m, hybrid-granite-r2-311m, hybrid-shyngys-e5)

# Fertility comparison (Wiki vs Akorda, requires HF tokenizer downloads):
python -m src.eval.fertility_compare

# See also: notebooks/akorda_kaggle.py
```
