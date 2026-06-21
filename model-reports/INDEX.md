# Model Reports — Index

> **This is not the project README.** For the project overview see
> [`../README.md`](../README.md). For the canonical benchmark numbers of **every**
> model see [`../results/RESULTS.md`](../results/RESULTS.md) (Wikipedia) and
> [`../results/akorda/AKORDA_RESULTS.md`](../results/akorda/AKORDA_RESULTS.md) (Akorda).

This folder holds **per-model deep-dive reports** for specific third-party embedding
models — each one a self-contained artifact prepared for contacting that model's team.
One file per model (`<model>.md`).

These reports do **not** restate the full result tables — they link to the canonical
sources above so numbers never desync. Each isolates a concrete, reproducible finding
about *that* model: a significant result **and** a specific, mechanism-checked weakness.

## Index

| Model | File | Headline finding | Outreach status |
|-------|------|------------------|-----------------|
| IBM Granite (R1/R2) | *(in preprints)* | R2 underperforms R1 on Kazakh; tokenizer fragments 2.3× | contacted (prior) |
| Jina v3 | [`jina-v3.md`](jina-v3.md) | Beats e5 on both domains; identical tokenizer to e5, so the gain is purely semantic | **not yet** |

## Queue (one at a time — open the next only after the current plays out)

1. **Jina** — `jinaai/jina-embeddings-v3` ← current
2. Nomic — `nomic-ai/nomic-embed-text-v1.5` (pending)
3. Cohere — `embed-multilingual-v3.0` (pending, API)

## How a new model gets added

When a new model is run, results land in **two independent places**:

1. **Canonical numbers** (same as every other model) →
   - `../results/RESULTS.md` + raw JSON in `../results/` (Wikipedia)
   - `../results/akorda/AKORDA_RESULTS.md` + raw JSON in `../results/akorda/` (Akorda)
2. **This folder** → a new `<model>.md` report (copy the structure of `jina-v3.md`),
   plus one new row in the index table above and a shift in the queue.

The report here is a *derived* artifact — it cites the canonical numbers, never
duplicates them.
