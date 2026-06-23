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

| System | nDCG@10 | factoid | paraphrase | low_overlap |
|--------|--------:|--------:|-----------:|------------:|
| **bge-m3** | **0.6790** | 0.7920 | **0.6359** | **0.6100** |
| hybrid: bm25+stemmer ⊕ bge-m3 (RRF) | 0.6326 | 0.8925 | 0.5331 | 0.4742 |
| hybrid: bm25+stemmer ⊕ jina-v3 (RRF) | 0.6145 | 0.8655 | 0.5158 | 0.4641 |
| jina-v3 | 0.6130 | 0.7036 | 0.5898 | 0.5465 |
| hybrid: bm25+stemmer ⊕ e5 (RRF) | 0.5623 | 0.8456 | 0.4393 | 0.4039 |
| bm25+stemmer | 0.5166 | **0.9063** | 0.3144 | 0.3316 |
| e5 (multilingual-e5-base) | 0.5090 | 0.6743 | 0.4408 | 0.4129 |
| hybrid: bm25+stemmer ⊕ cohere-v4 (RRF) | 0.5007 | 0.8073 | 0.3860 | 0.3111 |
| hybrid: bm25+stemmer ⊕ kazembed-v5 (RRF) | 0.4927 | 0.7085 | 0.4194 | 0.3518 |
| bm25+identity | 0.4844 | 0.9010 | 0.2777 | 0.2770 |
| granite-r1 (278M) | 0.4305 | 0.5481 | 0.4061 | 0.3385 |
| shyngys-e5 | 0.4263 | 0.5368 | 0.4103 | 0.3330 |
| granite-r2-311m | 0.3989 | 0.5236 | 0.3809 | 0.2936 |
| kazembed-v5 | 0.3892 | 0.5045 | 0.3694 | 0.2948 |
| cohere embed-v4.0 | 0.3670 | 0.6266 | 0.2633 | 0.2131 |
| qwen3-0.6b | 0.3304 | 0.5163 | 0.2560 | 0.2203 |
| granite-r2-97m | 0.2585 | 0.4387 | 0.1755 | 0.1624 |
| nomic-v1.5 | 0.0658 | 0.1268 | 0.0292 | 0.0416 |

BM25+stemmer numbers reflect the **100%-cache** run (full stemmer coverage). **BGE-M3 is
the new best system on Akorda (0.679)**, surpassing the previous best Jina hybrid (0.615)
as a single dense model. Unusually, the BGE-M3 hybrid (0.633) is *worse* than BGE-M3 alone
(0.679) — when the dense model is this strong, BM25 fusion dilutes performance rather than
complementing it. Full significance pending JSON upload; narrative in Key Findings below.

---

## Full Metrics Table

| System | nDCG@10 | MRR@10 | Recall@5 | Recall@10 |
|--------|--------:|-------:|---------:|----------:|
| bge-m3 | 0.6790 | 0.6124 | 0.8279 | 0.8811 |
| hybrid: bm25+stemmer ⊕ bge-m3 | 0.6326 | 0.5757 | 0.7254 | 0.8115 |
| hybrid: bm25+stemmer ⊕ jina-v3 | 0.6145 | 0.5491 | 0.7336 | 0.8197 |
| jina-v3 | 0.6130 | 0.5309 | 0.7787 | 0.8689 |
| hybrid: bm25+stemmer ⊕ e5 | 0.5623 | 0.4981 | 0.6475 | 0.7664 |
| bm25+stemmer | 0.5166 | 0.4754 | 0.5492 | 0.6516 |
| e5 | 0.5090 | 0.4388 | 0.6025 | 0.7336 |
| hybrid: bm25+stemmer ⊕ cohere-v4 | 0.5007 | — | — | — |
| bm25+identity | 0.4844 | 0.4421 | 0.5246 | 0.6230 |
| granite-r1 (278M) | 0.4305 | 0.3536 | 0.5369 | 0.6762 |
| shyngys-e5 | 0.4263 | 0.3627 | 0.5287 | 0.6270 |
| granite-r2-311m | 0.3989 | 0.3291 | 0.4713 | 0.6270 |
| kazembed-v5 | 0.3892 | 0.3308 | 0.4590 | 0.5779 |
| cohere embed-v4.0 | 0.3670 | 0.3120 | 0.4631 | 0.5410 |
| qwen3-0.6b | 0.3304 | 0.2793 | 0.3852 | 0.4959 |
| granite-r2-97m | 0.2585 | 0.2142 | 0.3115 | 0.4016 |
| nomic-v1.5 | 0.0658 | 0.0501 | 0.0656 | 0.1189 |

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
| e5 → jina-v3 | +0.1041 | <0.001 | ✓ |
| bm25+stemmer → jina-v3 | +0.0964 | <0.001 | ✓ |
| jina-v3 → hybrid ⊕ jina-v3 | +0.0015 | 0.475 | — |
| e5 → nomic-v1.5 | −0.4432 | <0.001 | ✓ |
| bm25+stemmer → nomic-v1.5 | −0.4509 | <0.001 | ✓ |
| jina-v3 → bge-m3 | +0.0659 | <0.001 | ✓ |
| e5 → bge-m3 | +0.1700 | <0.001 | ✓ |
| bm25+stemmer → bge-m3 | +0.1623 | <0.001 | ✓ |
| bge-m3 → hybrid ⊕ bge-m3 | −0.0464 | 0.014 | ✓ |
| bm25+stemmer → qwen3-0.6b | −0.1862 | <0.001 | ✓ |
| e5 → qwen3-0.6b | −0.1785 | <0.001 | ✓ |
| jina-v3 → qwen3-0.6b | −0.2826 | <0.001 | ✓ |
| bge-m3 → qwen3-0.6b | −0.3486 | <0.001 | ✓ |
| bm25+stemmer → cohere-v4 | −0.1496 | <0.001 | ✓ |
| e5 → cohere-v4 | −0.1419 | <0.001 | ✓ |
| jina-v3 → cohere-v4 | −0.2460 | <0.001 | ✓ |
| bge-m3 → cohere-v4 | −0.3120 | <0.001 | ✓ |
| qwen3-0.6b → cohere-v4 | +0.0366 | 0.058 | — |
| bm25+stemmer → kazembed-v5 | −0.1275 | <0.001 | ✓ |
| e5 → kazembed-v5 | −0.1198 | <0.001 | ✓ |
| shyngys-e5 → kazembed-v5 | −0.0372 | <0.001 | ✓ |
| granite-r1 → kazembed-v5 | −0.0414 | 0.045 | ✓ |
| granite-r2-311m → kazembed-v5 | −0.0097 | 0.346 | — |
| kazembed-v5 → cohere-v4 | +0.0221 | 0.215 | — |
| kazembed-v5 → qwen3-0.6b | +0.0587 | 0.021 | ✓ |

Positive Δ = B is better than A. Threshold: p < 0.05 (two-tailed paired bootstrap).
**BGE-M3 is significantly the best single model on Akorda** — it beats Jina v3
(Δ=+0.066, p<0.001), e5 (Δ=+0.170, p<0.001), and BM25+stemmer (Δ=+0.162, p<0.001).
Uniquely, the BGE-M3 hybrid is **significantly worse** than BGE-M3 alone
(Δ=−0.046, p=0.014) — the first model where fusion with BM25 measurably hurts.
By contrast the Jina hybrid is statistically tied with Jina alone (Δ=+0.0015, p=0.475),
and the e5 hybrid significantly beat both its channels.

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

### All fusions — nDCG@10 overall

| dense channel | bm25+stemmer | dense | **hybrid** | hybrid ≥ max(channel)? | low_overlap hybrid ≥ bm25? |
|---------------|------------:|------:|-----------:|:----------------------:|:--------------------------:|
| **bge-m3** | 0.5166 | **0.6790** | 0.6326 | ✗ | ✓ |
| **jina-v3** | 0.5166 | 0.6130 | **0.6145** | ✓ | ✓ |
| **e5** | 0.5166 | 0.5090 | 0.5623 | ✓ | ✓ |
| shyngys-e5 | 0.5166 | 0.4263 | 0.5202 | ✓ | ✓ |
| granite-r1 | 0.5166 | 0.4305 | 0.5175 | ✓ | ✓ |
| granite-r2-311m | 0.5166 | 0.3989 | 0.5158 | ✗ | ✓ |
| granite-r2-97m | 0.5166 | 0.2585 | 0.4073 | ✗ | ✗ |
| qwen3-0.6b | 0.5166 | 0.3304 | 0.4637 | ✗ | ✗ |
| cohere-v4 | 0.5166 | 0.3670 | 0.5007 | ✗ | ✗ |
| kazembed-v5 | 0.5166 | 0.3892 | 0.4927 | ✗ | ✓ |
| nomic-v1.5 | 0.5166 | 0.0658 | 0.2278 | ✗ | ✗ |

Both pre-registered success criteria are met for **4 of 11** dense channels
(jina-v3, e5, shyngys-e5, granite-r1). bge-m3, granite-r2-311m, and kazembed-v5 each meet
criterion 2 only (low_overlap hybrid ≥ BM25) but fail criterion 1 (hybrid ALL < max channel).
**BGE-M3 alone (0.679) is now the best system overall**, but its hybrid (0.633) *fails*
criterion 1 — the dense channel is stronger than the fusion. This is the first model in
this benchmark where the dense model alone beats BM25 cleanly enough that RRF with BM25
drags performance down. The r2-311m hybrid (0.5158) narrowly misses because BM25 itself
is strong after full-cache stemming. The r2-97m and nomic failures are structural (channels
too weak). See the BGE-M3 hybrid section below.

### Best overall: BGE-M3

BGE-M3 is the new best system on Akorda as a single dense model (0.679), surpassing the
previous best (Jina v3 hybrid, 0.615) by a clear margin.

| Slice | hybrid | Δ vs bm25 | p | Δ vs bge-m3 | p |
|-------|-------:|----------:|--:|------------:|--:|
| factoid (n=81) | 0.8925 | −0.0138 | 0.227 n.s. | +0.1005 | **<0.001** |
| paraphrase (n=81) | 0.5331 | +0.2187 | **<0.001** | −0.1028 | **0.003** |
| low_overlap (n=82) | 0.4742 | +0.1426 | **<0.001** | −0.1358 | **<0.001** |
| **ALL (n=244)** | **0.6326** | **+0.1159** | **<0.001** | **−0.0464** | **0.014** |

The pattern is the inverse of all previous models: **BGE-M3 alone (0.679) is significantly
stronger than its hybrid (0.633)** (Δ=−0.046, p=0.014). Criterion 1 fails because BM25
(0.517) is much weaker than the dense model (0.679), so fusion with BM25 dilutes the strong
semantic signal — significantly on paraphrase (0.636 → 0.533, p=0.003) and low_overlap
(0.610 → 0.474, p<0.001). The factoid gain from BM25 (0.792 → 0.893, +0.100, p<0.001) does
not compensate for the semantic losses.

**Practical implication:** BGE-M3 alone is sufficient on Akorda. RRF fusion only helps when
the two channels are complementary (each strong where the other is weak). Once the dense
model dominates BM25 across the board, the fusion penalty on semantic categories exceeds
the lexical gain on factoid queries.

**Sensitivity to k** (pre-registered k=60; sweep for honesty): the hybrid stays below
BGE-M3 alone at every k — ALL = 0.657 (k=10), 0.639 (k=30), 0.633 (k=60), 0.630 (k=100),
none reaching BGE-M3's 0.679. The dense-beats-hybrid verdict is robust to the RRF constant.

### Jina v3 (and its hybrid)

Jina v3 is the previous best system on Akorda — both as a single dense model (0.6130) and
in fusion (hybrid 0.6145). Per-category breakdown of the Jina hybrid vs its channels:

| Slice | hybrid | Δ vs bm25 | p | Δ vs jina | p |
|-------|-------:|----------:|--:|----------:|--:|
| factoid (n=81) | 0.8655 | −0.0408 | **0.037** | +0.1619 | **<0.001** |
| paraphrase (n=81) | 0.5158 | +0.2014 | **<0.001** | −0.0740 | **0.024** |
| low_overlap (n=82) | 0.4641 | +0.1325 | **<0.001** | −0.0825 | **0.029** |
| **ALL (n=244)** | **0.6145** | **+0.0979** | **<0.001** | +0.0015 | 0.475 n.s. |

Unlike the e5 hybrid, the Jina hybrid does **not** significantly improve over its dense
channel overall (Δ=+0.0015, p=0.475). The fusion trades a large factoid gain (+0.162)
for paraphrase/low_overlap losses (−0.074, −0.083), netting to a statistical tie with
Jina alone. **Practical implication:** when the dense model is this strong, Jina v3 alone
is sufficient on Akorda — fusion adds robustness on factoid queries but no aggregate lift.

### e5 hybrid (previous headline): BM25+stemmer ⊕ e5

The e5 hybrid (nDCG@10 = 0.5623) significantly beats **both** of its own channels — not
just the weaker one:

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

### Nomic v1.5 hybrid: dense channel too weak for RRF to help

The Nomic hybrid (nDCG@10 = 0.2278) is significantly **better** than Nomic alone
(Δ=+0.162, p<0.001) but significantly **worse** than BM25+stemmer alone (Δ=−0.289, p<0.001).
RRF fusion with a near-zero dense channel does not rescue performance — it degrades BM25.
Both pre-registered criteria fail: ALL hybrid (0.228) < max channel BM25 (0.517), and
low_overlap hybrid (0.134) < BM25 (0.332).

| Slice | hybrid | Δ vs bm25 | p | Δ vs nomic | p |
|-------|-------:|----------:|--:|-----------:|--:|
| factoid (n=81) | 0.3896 | −0.5167 | **<0.001** | +0.2628 | **<0.001** |
| paraphrase (n=81) | 0.1613 | −0.1530 | **<0.001** | +0.1321 | **<0.001** |
| low_overlap (n=82) | 0.1337 | −0.1979 | **<0.001** | +0.0921 | **<0.001** |
| **ALL (n=244)** | 0.2278 | **−0.2888** | **<0.001** | **+0.1621** | **<0.001** |

Root cause: Nomic's tokenizer has no Kazakh-specific Cyrillic coverage (all [UNK]).
See `model-reports/nomic-v1.5.md`.

### Qwen3-Embedding-0.6B hybrid: both criteria fail

| Slice | hybrid | Δ vs bm25 | p | Δ vs qwen3 | p |
|-------|-------:|----------:|--:|-----------:|--:|
| factoid (n=81) | 0.7312 | −0.1751 | **<0.001** | +0.2149 | **<0.001** |
| paraphrase (n=81) | 0.3855 | +0.0712 | **0.007** | +0.1295 | **<0.001** |
| low_overlap (n=82) | 0.2766 | −0.0550 | **0.047** | +0.0563 | **0.047** |
| **ALL (n=244)** | 0.4637 | **−0.0530** | **0.003** | **+0.1332** | **<0.001** |

The hybrid is significantly better than Qwen3 dense alone (+0.133, p<0.001) but
significantly **worse** than BM25+stemmer alone (−0.053, p=0.003). Both pre-registered
criteria fail: ALL hybrid (0.464) < BM25 (0.517), and low_overlap hybrid (0.277) <
BM25 (0.332). Pattern is similar to granite-r2-97m: the dense channel (0.330) is too
weak relative to BM25 to contribute positively on factoid queries, and its semantic
advantage on paraphrase/low_overlap is insufficient to compensate. k-sweep: hybrid stays
below BM25 at all k ∈ {10, 30, 60, 100} (best at k=10: 0.491, still below 0.517).

Root cause candidate: Qwen3-0.6B has the highest sub-word fertility of all tested models
(6.20 sub-words per Kazakh word), fragmenting morphologically rich Akorda vocabulary into
many small pieces. This is plausibly worse on formal political text (Akorda) than on
encyclopedic text (Wiki), consistent with the severe cross-domain drop (Wiki 0.690 →
Akorda 0.330, −0.360 — the second-largest absolute drop, behind only Cohere embed-v4.0).
See `model-reports/qwen3-embed-0.6b.md`.

### Cohere embed-v4.0 hybrid: both criteria fail

| Slice | hybrid | Δ vs bm25 | p | Δ vs cohere | p |
|-------|-------:|----------:|--:|------------:|--:|
| factoid (n=81) | 0.8073 | −0.0990 | **<0.001** | +0.1807 | **<0.001** |
| paraphrase (n=81) | 0.3860 | +0.0716 | **0.005** | +0.1227 | **<0.001** |
| low_overlap (n=82) | 0.3111 | −0.0205 | 0.260 | +0.0980 | **<0.001** |
| **ALL (n=244)** | 0.5007 | −0.0160 | 0.176 | **+0.1337** | **<0.001** |

The Cohere hybrid (0.501) is the same pattern as Qwen3 and granite-r2-97m — both
pre-registered criteria fail: ALL hybrid (0.501) < BM25 (0.517), and low_overlap hybrid
(0.311) < BM25 (0.332). Notably, unlike Qwen3, the overall hybrid loss vs BM25 is *not*
significant (Δ=−0.016, p=0.176), and on paraphrase the hybrid significantly beats both
channels (0.386, p=0.005 vs BM25). But the dense channel (0.367) is too weak on factoid
(where BM25 reaches 0.906) for fusion to clear the bar. The hybrid is significantly above
Cohere dense alone (+0.134, p<0.001) — fusion *rescues* the weak dense channel but cannot
lift it past the strong lexical baseline.

### KazEmbed-V5 hybrid: criterion 1 fails, criterion 2 passes

| Slice | hybrid | Δ vs bm25 | p | Δ vs dense | p |
|-------|-------:|----------:|--:|-----------:|--:|
| factoid (n=81) | 0.709 | −0.197 | **<0.001** | +0.204 | **<0.001** |
| paraphrase (n=81) | 0.419 | +0.105 | **<0.001** | +0.050 | 0.126 n.s. |
| low_overlap (n=82) | 0.352 | +0.020 | 0.259 n.s. | +0.057 | 0.059 n.s. |
| **ALL (n=244)** | **0.493** | −0.024 | 0.123 n.s. | **+0.104** | **<0.001** |

The hybrid significantly improves on KazEmbed-V5 dense (+0.104, p<0.001), but falls short
of BM25+stemmer overall (numerically −0.024, not significant at p=0.123). Criterion 1 fails
(hybrid 0.493 < BM25 0.517), but **criterion 2 passes**: low_overlap hybrid (0.352) exceeds
BM25+stemmer (0.332), unlike Cohere and Qwen3 where both criteria fail.

On paraphrase the hybrid significantly beats BM25+stemmer (+0.105, p<0.001). On factoid
BM25+stemmer dominates strongly (BM25 0.906 vs dense 0.505), dragging the fusion below BM25
overall. The KazEmbed-V5 dense (0.389) is comparable to Cohere dense (0.367, not significantly
different, p=0.215), and the hybrid pattern is similar: rescues the weak dense channel on
paraphrase but cannot overcome BM25's factoid advantage.

The **key finding for KazEmbed-V5** is the Wikipedia comparison: despite being fine-tuned
specifically for Kazakh retrieval, kazembed-v5 (0.642) is significantly below both the base
model e5 (0.785, Δ=+0.143, p<0.001) and the other Kazakh e5 fine-tune shyngys-e5 (0.747,
Δ=+0.105, p<0.001). On Akorda, shyngys-e5 (0.426) also significantly outperforms
kazembed-v5 (0.389, Δ=+0.037, p<0.001). The in-domain KazQAD gain (+2.1% MRR) does not
transfer to this benchmark on either domain.

See `model-reports/kazembed-v5.md`.

The striking result is the dense model itself: **Cohere embed-v4.0, marketed specifically
for cross-lingual retrieval (100+ languages), has the largest absolute cross-domain drop in
the entire benchmark — Wiki 0.800 → Akorda 0.367, −0.433** — exceeding even Qwen3 (−0.360).
Its sub-word fertility on Kazakh is 6.20 (Wiki) / 6.27 (Akorda) — *identical* to Qwen3 to
three significant figures, with identical token counts on every probe word (string pieces not
available via API) — consistent with the same byte-level BPE fallback for Kazakh Cyrillic. The strong Wikipedia score does not
transfer to formal, morphologically dense political prose.
See `model-reports/cohere-embed-v4.md`.

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

**shyngys-e5** drops disproportionately on formal political text (−0.321 absolute, the second largest drop after r2-97m *among these original 7 systems*; Cohere and Qwen3, added later, both drop more), falling below granite-r1 on Akorda — a relationship reversed on Wiki. On Akorda the shyngys-e5 vs granite-r1 difference is not significant (p>0.05), so the practical gap is small, but the direction reversal is real.

**BM25 vs dense** is the other domain-sensitive relationship. On Wiki, BM25+stemmer was clearly behind e5 and shyngys-e5 in absolute terms. On Akorda, BM25+stemmer ranks #2 overall, nominally tied with e5 (p=0.399), — but this is driven entirely by the factoid category (lexical overlap ≈0.79): when factoid is excluded, e5 leads on paraphrase and low_overlap. The core finding is that **the ranking of dense models among themselves is OOD-stable, while the BM25-vs-dense balance depends on the domain's lexical overlap distribution**.

### 6. R2 does not outperform R1 on any domain

Granite R2-97M nDCG@10 = 0.259 — more than 2× below e5, and significantly worse than all other systems. R2-311M (0.399) is significantly better than R2-97M (p<0.001) but still significantly below e5 (p<0.001). On neither domain does R2 outperform R1: on Wiki R2-311M scores lower (0.659 vs R1 0.672); on Akorda the gap is similar in direction (0.399 vs 0.431) but does not reach significance (p=0.076, n=244). The sub-word fertility analysis (see below) suggests a tokenizer-level candidate mechanism: Granite R2 fragments Akorda formal vocabulary +0.23–0.31 sub-words/word more than Wiki, while e5/R1 are domain-stable (Δ≈+0.01). The pattern is consistent — R2 is not an improvement over R1 for Kazakh — but the Akorda result should be read as "within noise" rather than a confirmed gap.

### 7. BGE-M3 is the new best system — and the hybrid pattern reverses at this level

**BGE-M3 (nDCG@10 = 0.679) is now the best system on Akorda overall**, surpassing the
previous best (Jina v3 hybrid, 0.615) as a single dense model. Its hybrid (0.633) *fails*
criterion 1 for the first time in this benchmark: BGE-M3 alone beats the fusion because
BM25's factoid gain (+0.100 on factoid) is outweighed by semantic dilution on paraphrase
(−0.103) and low_overlap (−0.137). **When the dense model dominates BM25 overall, RRF
fusion is net-negative.**

For models where dense and BM25 are complementary (e5, Jina), the hybrid is still
the safest choice: the e5 hybrid (0.562) significantly beats both channels, and the Jina
hybrid (0.615) matches Jina alone while gaining robustness on factoid. 4 of 11 dense
channels (jina-v3, e5, shyngys-e5, granite-r1) meet both pre-registered criteria and yield
a hybrid that numerically beats their best single channel. **BGE-M3 alone is the single
best choice for Akorda**; whether that generalises to other formal domains requires further
evaluation.

---

## Sub-word Fertility: Akorda vs Wiki

*Candidate mechanism for R2's domain drop. Run: `python -m src.eval.fertility_compare`*

Sample: 100 most frequent long words (≥9 chars) from each corpus.  
Method: average sub-words per word (bare = no leading space; +sp = with leading space).  
Note: shyngys-e5 and kazembed-v5 are both fine-tuned from multilingual-e5 and share e5's tokenizer; their fertility values are identical to e5 by construction (1.81 / 1.82).

| Tokenizer | wiki (bare) | akorda (bare) | Δbare | wiki (+sp) | akorda (+sp) | Δ+sp |
|-----------|------------:|--------------:|------:|-----------:|-------------:|-----:|
| granite-97m-r2 | 4.00 | 4.29 | **+0.29** | 3.57 | 3.88 | **+0.31** |
| granite-311m-r2 | 4.20 | 4.43 | **+0.23** | 3.82 | 4.07 | **+0.25** |
| e5-base | 1.81 | 1.82 | +0.01 | 1.81 | 1.82 | +0.01 |
| granite-278m-r1 | 1.81 | 1.82 | +0.01 | 1.81 | 1.82 | +0.01 |
| jina-v3 | 1.81 | 1.82 | +0.01 | 1.81 | 1.82 | +0.01 |
| bge-m3 | 1.81 | 1.82 | +0.01 | 1.81 | 1.82 | +0.01 |
| qwen3-0.6b | 6.20 | 6.27 | +0.07 | 6.50 | 6.63 | +0.13 |
| cohere embed-v4.0 ‡ | 6.20 | 6.27 | +0.07 | — | — | — |
| tilcore_morph256k § | 1.64 | 1.64 | +0.00 | 1.28 | 1.40 | +0.12 |
| nomic-v1.5 | 3.49 | — | — | 3.49 | — | — |

‡ Cohere `embed-v4.0` is API-only; fertility was measured via the Cohere tokenize API
(`src/eval/cohere_tokenization.py`) on the same 100 words, bare only (the API does not
expose a leading-space variant). The result is **identical to Qwen3-0.6B** (6.20 / 6.27),
with identical token counts on all four probe words (сөзжасам→6, мүмкіндіктерін→11,
ұйымдастырушылық→9, халықаралық→8; API returns IDs only, not string pieces) — consistent
with the same byte-level BPE fallback for Kazakh Cyrillic.

§ `tilcore_morph256k` (`stukenov/sozkz-morphbpe-256k-kk-v1`, TilQazyna morphBPE) is a
**tokenizer-only reference**, not a retrieval model in this benchmark — there is no Akorda
nDCG@10 for it. It is the **least-fragmenting tokenizer measured** (1.64 bare / 1.28 +sp),
below even the XLM-R cluster (1.81), because its 256K vocabulary is dedicated entirely to
Kazakh. Like Qwen3/Cohere it is byte-level BPE (tokens render as mojibake), which shows that
byte-level BPE per se does **not** cause over-fragmentation — Kazakh vocabulary coverage does.
Full analysis in [`../TOKENIZATION.md`](../TOKENIZATION.md).

Note: Nomic v1.5 uses a BERT English WordPiece tokenizer (30 522 tokens); Kazakh-specific
Cyrillic characters tokenize entirely as `[UNK]`, so the fertility of 3.49 reflects only
words with standard (non-Kazakh-specific) Cyrillic characters in the 100-word sample. The
OOV issue is the primary explanation for Nomic's nDCG@10=0.066 on Akorda (not fragmentation).
Akorda fertility not measured for Nomic (numbers not meaningful when most tokens are [UNK]).

**Interpretation (cautious):**

- **Granite R2** fragments Akorda formal vocabulary +0.23–0.31 sub-words/word more than Wiki. This is a *candidate mechanism* for R2's universal domain drop: increased fragmentation degrades embedding coherence, consistent with the observed performance gap.
- **e5 / Granite R1** are tokenizer-domain-stable (Δ≈+0.01). Their performance drop on Akorda is not explained by tokenizer fragmentation — other factors (training data distribution, model capacity) dominate.
- **shyngys-e5 and e5 share one tokenizer**: fertility cannot explain why shyngys underperforms e5 specifically on Akorda. That differential (Finding 5, rank 3→5) requires a different explanation (e.g., fine-tuning data mismatch with formal political language).
- This is correlation + theoretical plausibility, not proven causation. Frequent words are over-represented in the tokenizer vocab, so Δ is likely *underestimated* on rarer domain-specific forms.

---

## Comparison with Wiki Benchmark

| System | Wiki nDCG@10 | Akorda nDCG@10 | Drop |
|--------|-------------:|---------------:|-----:|
| bge-m3 | 0.866 | 0.679 | −0.187 |
| jina-v3 | 0.821 | 0.613 | −0.208 |
| e5 | 0.785 | 0.509 | −0.276 |
| bm25+stemmer | 0.754 | 0.517 | −0.237 |
| shyngys-e5 | 0.747 | 0.426 | −0.321 |
| bm25+identity | 0.690 | 0.484 | −0.206 |
| granite-r1 | 0.672 | 0.431 | −0.241 |
| granite-r2-311m | 0.659 | 0.399 | −0.260 |
| granite-r2-97m | 0.589 | 0.259 | −0.330 |
| cohere embed-v4.0 | 0.800 | 0.367 | **−0.433** |
| qwen3-0.6b | 0.690 | 0.330 | −0.360 |
| kazembed-v5 | 0.642 | 0.389 | −0.253 |
| nomic-v1.5 | 0.171 | 0.066 | −0.105 |

All systems drop substantially on Akorda. BGE-M3 shows the **smallest absolute drop among
dense models (−0.187)**, possibly reflecting stronger generalisation across formal political
language. **Cohere embed-v4.0 has the largest absolute drop in the benchmark (−0.433)**,
ahead of Qwen3-0.6B (−0.360). This is the most pointed result of the Cohere run: a model
marketed specifically for cross-lingual retrieval (100+ languages) is the 3rd-strongest
system on Wikipedia (0.800) yet collapses to 0.367 on formal political Kazakh — below
BM25+stemmer, e5, and the Granite models. The cross-domain fragility of both Cohere and
Qwen3 is consistent with their shared byte-level BPE tokenizer (fertility 6.20, identical
to three significant figures), which fragments morphologically complex formal Kazakh into
byte-level pieces. Tokenizer fertility is a *candidate mechanism*, not proven causation:
BGE-M3/Jina/e5 share the XLM-R tokenizer (1.81) and stay domain-robust, while the two
byte-level-BPE models show the two largest drops — a consistent but correlational pattern.

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
