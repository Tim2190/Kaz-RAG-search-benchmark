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
