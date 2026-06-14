"""
Прогон dense-систем на датасете Akorda — Kaggle/Colab.

# ── Cell 1: install & clone ──────────────────────────────────────────────────
# !pip install -q transformers sentence-transformers torch
# !git clone -b claude/akorda-benchmark https://github.com/tim2190/kaz-rag-search-benchmark.git
# %cd kaz-rag-search-benchmark

# ── Cell 2: прогоны (по одному на cell, чтобы GPU-память освобождалась) ─────
#
# !python -m src.eval.run_akorda --system e5              --out results/akorda/dense_e5.json
# !python -m src.eval.run_akorda --system granite-r1      --out results/akorda/dense_granite_r1.json
# !python -m src.eval.run_akorda --system granite-r2-97m  --out results/akorda/dense_granite_r2_97m.json
# !python -m src.eval.run_akorda --system granite-r2-311m --out results/akorda/dense_granite_r2_311m.json
# !python -m src.eval.run_akorda --system shyngys-e5      --out results/akorda/dense_shyngys.json
#
# ── Cell 3: hybrid (после того как готовы оба runs) ─────────────────────────
#
# Пример: BM25+stemmer ⊕ e5
# !python -m src.eval.run_akorda --system hybrid-e5 \
#     --bm25-runs  results/akorda/bm25_kazakh.json \
#     --dense-runs results/akorda/dense_e5.json \
#     --out results/akorda/hybrid_e5.json
#
# ── Cell 4: BM25+stemmer с полным кэшем (опционально) ────────────────────────
# На Kaggle API стеммера доступен и кэш пополнится до 100%.
# !python -m src.eval.run_akorda --system bm25-stemmer \
#     --out results/akorda/bm25_kazakh_full.json
#
# ── Cell 5: коммит результатов ───────────────────────────────────────────────
# !git add results/akorda/
# !git commit -m "Akorda: dense + hybrid results"
# !git push origin claude/akorda-benchmark
#
# ── Ожидаемый формат вывода ──────────────────────────────────────────────────
# nDCG@10 overall: X.XXXX
#   factoid    : X.XXXX
#   low_overlap: X.XXXX
#   paraphrase : X.XXXX

# ── Standalone-блок: просмотр уже сохранённых результатов ───────────────────

import json, os, glob

results_dir = "results/akorda"
files = sorted(glob.glob(f"{results_dir}/*.json"))
if not files:
    print("Нет результатов в results/akorda/ — запусти прогоны выше.")
else:
    print(f"{'система':30s} {'nDCG@10':>8s} {'factoid':>9s} {'paraphrase':>11s} {'low_overlap':>12s}")
    print("-" * 75)
    for path in files:
        r = json.load(open(path))
        sys_name = r.get("system", os.path.basename(path))
        ov = r["overall"].get("ndcg@10", "?")
        by_cat = r.get("by_category", {})
        fac = by_cat.get("factoid", {}).get("ndcg@10", "—")
        par = by_cat.get("paraphrase", {}).get("ndcg@10", "—")
        low = by_cat.get("low_overlap", {}).get("ndcg@10", "—")
        def fmt(v): return f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"{sys_name:30s} {fmt(ov):>8s} {fmt(fac):>9s} {fmt(par):>11s} {fmt(low):>12s}")
"""
