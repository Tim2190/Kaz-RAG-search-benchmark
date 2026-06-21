"""
Прогон BGE-M3 на обоих датасетах (Wikipedia n=300 + Akorda n=244).

ТРЕБОВАНИЯ: Kaggle GPU T4 + Internet ON.
Каждый "# %%" — отдельная ячейка.

BGE-M3 (BAAI/bge-m3) — флагманская мультиязычная модель BAAI.
Использует стандартный SentenceTransformer API без инструкционных префиксов
(модель документирована как «instruction-free» для поискового режима).
Restart НЕ нужен.

# %% [1] Установка зависимостей и клонирование
# !pip install -q sentence-transformers torch
# !git clone -b claude/laughing-ramanujan-LsMa8 https://github.com/Tim2190/Kaz-RAG-search-benchmark.git
# %cd Kaz-RAG-search-benchmark

# ─── WIKIPEDIA (n=300) ───────────────────────────────────────────────────────

# %% [2] BGE-M3 на Wikipedia (~15 мин на T4)
# !python -m src.eval.run_dense \
#     --model bge-m3 \
#     --out results/dense_bge_m3_300.json \
#     --runs-out results/runs_dense_bge_m3.json \
#     --max-seq-len 512

# %% [3] Проверка Wikipedia
# import json
# s = json.load(open('results/dense_bge_m3_300.json'))
# print(f"nDCG@10: {s['overall']['ndcg@10']:.4f}")
# r = json.load(open('results/runs_dense_bge_m3.json'))
# print(f"Queries in run: {len(r)}")   # должно быть 300

# ─── AKORDA (n=244) ──────────────────────────────────────────────────────────

# %% [4] BGE-M3 на Akorda (~7 мин на T4)
# !python -m src.eval.run_akorda \
#     --system bge-m3 \
#     --out results/akorda/dense_bge_m3.json

# %% [5] Проверка Akorda
# import json
# s = json.load(open('results/akorda/dense_bge_m3.json'))
# print(f"nDCG@10 ALL: {s['overall']['ndcg@10']:.4f}")
# for cat, m in s.get('by_category', {}).items():
#     print(f"  {cat}: {m['ndcg@10']:.4f}")

# ─── HYBRID AKORDA ───────────────────────────────────────────────────────────

# %% [6] Гибрид BM25+стеммер ⊕ BGE-M3 на Akorda
# bm25_kazakh_full.json уже в репо (578 KB).
# Нужны per-query rankings для dense — получаем запуском run_akorda с --top-k 100.
#
# Шаг 6a: dense runs с top-k=100
# !python -m src.eval.run_akorda \
#     --system bge-m3 \
#     --out results/akorda/dense_bge_m3.json
#
# Шаг 6b: hybrid
# !python -m src.eval.run_akorda \
#     --system hybrid-bge-m3 \
#     --bm25-runs  results/akorda/bm25_kazakh_full.json \
#     --dense-runs results/akorda/dense_bge_m3.json \
#     --dense-label bge-m3 \
#     --out results/akorda/hybrid_bge_m3.json

# ─── FERTILITY ───────────────────────────────────────────────────────────────

# %% [7] Sub-word fertility: BGE-M3 токенизатор vs остальные
# !python -m src.eval.tokenization_test
# !python -m src.eval.fertility_compare

# ─── КОММИТ ─────────────────────────────────────────────────────────────────

# %% [8] Коммит результатов
# !git config user.email "9189920ts@gmail.com"
# !git config user.name "Tim2190"
# !git add results/dense_bge_m3_300.json \
#           results/runs_dense_bge_m3.json \
#           results/akorda/dense_bge_m3.json \
#           results/akorda/hybrid_bge_m3.json
# !git commit -m "BGE-M3: benchmark results on Wikipedia (n=300) and Akorda (n=244)"
# !git push origin claude/laughing-ramanujan-LsMa8
"""

import json, os

def _show(path, label):
    if not os.path.exists(path):
        print(f"  {label}: нет файла ({path})")
        return
    d = json.load(open(path))
    ov = d.get("overall", {})
    print(f"  {label}: nDCG@10={ov.get('ndcg@10','?'):.4f}  "
          f"MRR@10={ov.get('mrr@10','?'):.4f}")
    for cat, m in d.get("by_category", {}).items():
        print(f"    {cat}: {m.get('ndcg@10','?'):.4f}")

print("=== BGE-M3 результаты ===")
_show("results/dense_bge_m3_300.json",    "Wiki  (dense)")
_show("results/akorda/dense_bge_m3.json", "Akorda (dense)")
_show("results/akorda/hybrid_bge_m3.json","Akorda (hybrid)")
