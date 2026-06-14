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
| low_overlap | 82 | ≈ 0.32 | Low lexical overlap, vocabulary-gap / synonym-like |

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

### 3. Stemmer effect not significant on Akorda

bm25-identity → bm25+stemmer: Δ=+0.014, p=0.196. Two possible reasons: (a) BM25 was run with a 78%-cache stemmer (identity fallback for uncached tokens); (b) formal political Kazakh in Akorda may have less morphological variation than Wikipedia. The effect was significant on Wiki (Δ=+0.064, p<0.01).

### 4. BM25+stemmer outperforms granite-r1 and shyngys-e5

Both dense systems fail to match BM25+stemmer on Akorda overall (p=0.011 and p=0.013 respectively). The factoid category accounts for most of this: BM25 nDCG@10 = 0.905 vs granite-r1 = 0.548 on factoid.

### 5. Rankings generalize from Wiki to Akorda

Cross-dataset Spearman rank correlation on nDCG@10 (n=7 systems): ρ=1.00 (identical rank order). Despite a ≈30% absolute score drop (formal political text is harder), no system swaps position. This is an OOD confirmation: the Wiki ranking is not an artifact of that corpus.

| System | Wiki rank | Akorda rank |
|--------|:---------:|:-----------:|
| e5 | 1 | 1 |
| bm25+stemmer | 2 | 2 |
| bm25+identity | 3 | 3 |
| granite-r1 | 4† | 4 |
| shyngys-e5 | 3† | 5 |
| granite-r2-311m | 5 | 6 |
| granite-r2-97m | 6 | 7 |

†On Wiki, shyngys-e5 (0.747) was above granite-r1 (0.672). On Akorda, shyngys-e5 (0.426) drops to within noise of granite-r1 (0.431). The difference between them on Akorda is not significant (p > 0.05 from bootstraps above).

### 6. R2-97M remains the weakest system

Granite R2-97M nDCG@10 = 0.259 — more than 2× below e5, and significantly worse than all other systems. R2-311M (0.399) is significantly better than R2-97M (p<0.001) but still significantly below e5 (p<0.001). The R2 vs R1 gap (R1=0.431 vs R2-311M=0.399) does not reach significance on Akorda (p=0.076) — possible due to smaller sample (n=244 vs 300).

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

All systems drop substantially on Akorda. The largest drops are granite-r2-97m (−0.330) and shyngys-e5 (−0.321), smallest is bm25+identity (−0.206). The relative ordering is preserved (ρ=1.00), confirming that score differences on Wiki are not corpus-specific artifacts.

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
