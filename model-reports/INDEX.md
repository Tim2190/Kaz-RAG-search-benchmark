# Model Reports

> For the project overview see [`../README.md`](../README.md).
> For the canonical benchmark numbers see
> [`../results/RESULTS.md`](../results/RESULTS.md) (Wikipedia) and
> [`../results/akorda/AKORDA_RESULTS.md`](../results/akorda/AKORDA_RESULTS.md) (Akorda).

Per-model deep-dive reports — each isolates a concrete, reproducible finding about a
specific third-party embedding model, with significance testing and a mechanism check.
Numbers are not duplicated here; each report links to the canonical result tables above.

| Model | File | Headline finding |
|-------|------|-----------------|
| IBM Granite (R1/R2) | *(in preprints)* | R2 underperforms R1 on Kazakh; tokenizer fragments 2.3× more than R1 |
| Jina v3 | [`jina-v3.md`](jina-v3.md) | Strongest dense model on both domains; gain over e5 is purely semantic — identical tokenizer fertility (1.81) |
| Nomic v1.5 | [`nomic-v1.5.md`](nomic-v1.5.md) | Weakest model tested (Wiki 0.171, Akorda 0.066); English BERT tokenizer — Kazakh-specific Cyrillic entirely [UNK] |
| BGE-M3 | [`bge-m3.md`](bge-m3.md) | Best model on both domains (Wiki 0.866, Akorda 0.679); beats Jina v3 (Δ=+0.045, p=0.0001 on Wiki); first model where dense alone beats its hybrid |
| Qwen3-Embedding-0.6B | [`qwen3-embed-0.6b.md`](qwen3-embed-0.6b.md) | Largest cross-domain drop (Wiki 0.690 → Akorda 0.330, −0.360); highest tokenizer fragmentation (6.20 sub-words/word) — byte-level BPE fallback for Kazakh |
