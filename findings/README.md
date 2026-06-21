# External Model Findings — Vendor Pings

Focused, self-contained write-ups of what this benchmark measured about **specific
third-party embedding models**, prepared as artifacts for contacting each vendor's team.

Protocol (one channel at a time): run the model on both Kazakh domains → isolate a
concrete, reproducible finding about *their* model (a significant result **and** a
specific weakness) → ping the team with evidence, not a request. The weakness is the
conversation hook: "here is exactly where your model underperforms on a low-resource
Turkic language, and why."

These files do **not** restate the full result tables — they link to the canonical
sources so numbers never desync:
- Wikipedia (n=300): [`../results/RESULTS.md`](../results/RESULTS.md)
- Akorda OOD (n=244): [`../results/akorda/AKORDA_RESULTS.md`](../results/akorda/AKORDA_RESULTS.md)

## Index

| Model | File | Headline finding | Ping status |
|-------|------|------------------|-------------|
| IBM Granite (R1/R2) | *(in preprints)* | R2 underperforms R1 on Kazakh; tokenizer fragments 2.3× | contacted (prior) |
| Jina v3 | [`jina-v3.md`](jina-v3.md) | Significantly beats e5 on both domains; weak on genuine low-overlap | **not yet** |

## Queue (one at a time — open the next only after the current plays out)

1. **Jina** — `jinaai/jina-embeddings-v3` ← current
2. Nomic — `nomic-ai/nomic-embed-text-v1.5` (pending)
3. Cohere — `embed-multilingual-v3.0` (pending, API)
