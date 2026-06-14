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
| **e5** (multilingual-e5-base) | **0.5090** | [0.4601, 0.5587] | 0.6743 | **0.4408** | **0.4129** |
| bm25+stemmer | 0.4981 | [0.4427, 0.5554] | **0.9051** | 0.2796 | 0.3119 |
| bm25+identity | 0.4844 | [0.4304, 0.5403] | 0.9010 | 0.2777 | 0.2770 |
| granite-r1 (278M) | 0.4305 | [0.3840, 0.4779] | 0.5481 | 0.4061 | 0.3385 |
| shyngys-e5 | 0.4263 | [0.3786, 0.4759] | 0.5368 | 0.4103 | 0.3330 |
| granite-r2-311m | 0.3989 | [0.3517, 0.4484] | 0.5236 | 0.3809 | 0.2936 |
| granite-r2-97m | 0.2585 | [0.2134, 0.3049] | 0.4387 | 0.1755 | 0.1624 |

---

## Full Metrics Table

| System | nDCG@10 | MRR@10 | Recall@5 | Recall@10 |
|--------|--------:|-------:|---------:|----------:|
| e5 | 0.5090 | 0.4388 | 0.6025 | 0.7336 |
| bm25+stemmer | 0.4981 | 0.4642 | 0.5369 | 0.6066 |
| bm25+identity | 0.4844 | 0.4421 | 0.5246 | 0.6230 |
| granite-r1 (278M) | 0.4305 | 0.3536 | 0.5369 | 0.6762 |
| shyngys-e5 | 0.4263 | 0.3627 | 0.5287 | 0.6270 |
| granite-r2-311m | 0.3989 | 0.3291 | 0.4713 | 0.6270 |
| granite-r2-97m | 0.2585 | 0.2142 | 0.3115 | 0.4016 |

---

## Statistical Significance (paired bootstrap, nDCG@10, n=10 000 resamples)

| Comparison (A → B) | Δ | p-value | Significant? |
|--------------------|--:|--------:|:------------:|
| bm25-identity → bm25+stemmer | +0.0137 | 0.196 | — |
| bm25+stemmer → e5 | +0.0109 | 0.353 | — |
| bm25+stemmer → granite-r1 | −0.0676 | 0.011 | ✓ |
| bm25+stemmer → shyngys-e5 | −0.0718 | 0.013 | ✓ |
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
| bm25+stemmer → e5 | Δ=−0.231 **p<0.001** | Δ=+0.161 **p<0.001** | Δ=+0.101 **p=0.028** | Δ=+0.011 p=0.353 n.s. |
| e5 → granite-r1 | Δ=−0.126 **p<0.001** | Δ=−0.035 p=0.195 n.s. | Δ=−0.074 **p=0.048** | Δ=−0.078 **p=0.001** |
| bm25-identity → e5 | Δ=−0.227 **p<0.001** | Δ=+0.163 **p<0.001** | Δ=+0.136 **p=0.004** | Δ=+0.025 p=0.195 n.s. |

---

## Key Findings

### 1. BM25 factoid anomaly — not an anomaly

BM25 nDCG@10 = **0.905** on factoid queries (lexical overlap ≈ 0.79). This is expected by design: factoid queries were constructed with high word-form overlap with the passage, making exact-match BM25 near-optimal. Dense models top out at 0.674 (e5) on this category. The BM25 "overall" lead over e5 evaporates once factoid is disaggregated: on paraphrase (p<0.001) and low_overlap (p=0.028) e5 significantly outperforms BM25+stemmer.

### 2. e5 is the best semantic system

e5 is the only system that is simultaneously: best at paraphrase (0.441), best at low_overlap (0.413), and best overall (0.509). Its overall edge over BM25+stemmer (Δ=+0.011) is not significant (p=0.353) because factoid pulls BM25 up. Among all dense systems, e5 is significantly better than every other model (p<0.001).

### 3. Stemmer effect not significant on Akorda (preliminary)

bm25-identity → bm25+stemmer: Δ=+0.014, p=0.196. **This result is preliminary**: BM25+stemmer was run with a 78%-cache stemmer (identity fallback for uncached tokens), meaning 22% of tokens were not stemmed. The non-significance could therefore be an artifact of incomplete stemming rather than a property of the language or domain. A clean run with full cache coverage is needed to confirm. For reference, the effect was significant on Wiki (Δ=+0.064, p<0.01), where full cache was available.

### 4. BM25+stemmer outperforms granite-r1 and shyngys-e5

Both dense systems fail to match BM25+stemmer on Akorda overall (p=0.011 and p=0.013 respectively). The factoid category accounts for most of this: BM25 nDCG@10 = 0.905 vs granite-r1 = 0.548 on factoid.

### 5. Dense model ranking is OOD-stable; BM25 vs dense balance is domain-dependent

Cross-dataset Spearman rank correlation on nDCG@10 (n=7 systems): **ρ=0.89** (Σd²=6). The ranking is substantially preserved — e5 is #1 on both domains, granite-r2-97m is last on both — but not perfectly identical. The one systematic shift is shyngys-e5, which moves from rank 3 on Wiki to rank 5 on Akorda:

| System | Wiki nDCG@10 | Wiki rank | Akorda nDCG@10 | Akorda rank | Shift |
|--------|-------------:|:---------:|---------------:|:-----------:|:-----:|
| e5 | 0.785 | 1 | 0.509 | 1 | — |
| bm25+stemmer | 0.754 | 2 | 0.498 | 2 | — |
| shyngys-e5 | 0.747 | **3** | 0.426 | **5** | ↓ 2 |
| bm25+identity | 0.690 | 4 | 0.484 | 3 | ↑ 1 |
| granite-r1 | 0.672 | 5 | 0.431 | 4 | ↑ 1 |
| granite-r2-311m | 0.659 | 6 | 0.399 | 6 | — |
| granite-r2-97m | 0.589 | 7 | 0.259 | 7 | — |

**shyngys-e5** drops disproportionately on formal political text (−0.321 absolute, the second largest drop after r2-97m), falling below granite-r1 on Akorda — a relationship reversed on Wiki. On Akorda the shyngys-e5 vs granite-r1 difference is not significant (p>0.05), so the practical gap is small, but the direction reversal is real.

**BM25 vs dense** is the other domain-sensitive relationship. On Wiki, BM25+stemmer was clearly behind e5 and shyngys-e5 in absolute terms. On Akorda, BM25+stemmer ranks #2 overall, above all dense models except e5 — but this is driven entirely by the factoid category (lexical overlap ≈0.79): when factoid is excluded, e5 leads on paraphrase and low_overlap (both p<0.05). The core finding is that **the ranking of dense models among themselves is OOD-stable, while the BM25-vs-dense balance depends on the domain's lexical overlap distribution**.

### 6. R2 does not outperform R1 on any domain

Granite R2-97M nDCG@10 = 0.259 — more than 2× below e5, and significantly worse than all other systems. R2-311M (0.399) is significantly better than R2-97M (p<0.001) but still significantly below e5 (p<0.001). On neither domain does R2 outperform R1: on Wiki R2-311M scores lower (0.659 vs R1 0.672); on Akorda the gap is similar in direction (0.399 vs 0.431) but does not reach significance (p=0.076, n=244). The pattern is consistent — R2 is not an improvement over R1 for Kazakh — but the Akorda result should be read as "within noise" rather than a confirmed gap, pending a larger sample or a full-cache BM25+stemmer run for cleaner context.

---

## Comparison with Wiki Benchmark

| System | Wiki nDCG@10 | Akorda nDCG@10 | Drop |
|--------|-------------:|---------------:|-----:|
| e5 | 0.785 | 0.509 | −0.276 |
| bm25+stemmer | 0.754 | 0.498 | −0.256 |
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
python -m src.eval.run_akorda --system bm25-stemmer --out results/akorda/bm25_kazakh.json

# Dense (GPU):
python -m src.eval.run_akorda --system e5              --out results/akorda/dense_e5.json
python -m src.eval.run_akorda --system granite-r1      --out results/akorda/dense_granite_r1.json
python -m src.eval.run_akorda --system granite-r2-97m  --out results/akorda/dense_granite_r2_97m.json
python -m src.eval.run_akorda --system granite-r2-311m --out results/akorda/dense_granite_r2_311m.json
python -m src.eval.run_akorda --system shyngys-e5      --out results/akorda/dense_shyngys.json

# See also: notebooks/akorda_kaggle.py
```
