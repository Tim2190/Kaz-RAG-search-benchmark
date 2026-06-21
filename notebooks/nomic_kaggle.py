"""
Прогон Nomic v1.5 на обоих датасетах (Wikipedia n=300 + Akorda n=244).

ТРЕБОВАНИЯ: Kaggle GPU T4 + Internet ON.
Каждый "# %%" — отдельная ячейка.

Nomic v1.5 — стандартная SentenceTransformer-модель с префиксами
  search_query:    для запросов
  search_document: для пассажей
и флагом trust_remote_code=True (кастомный attention-pooling код).
Работает с любой современной версией transformers — restart НЕ нужен.

# %% [1] Установка зависимостей и клонирование
# !pip install -q sentence-transformers torch einops
# !git clone https://github.com/Tim2190/Kaz-RAG-search-benchmark.git
# %cd Kaz-RAG-search-benchmark

# ─── WIKIPEDIA (n=300) ───────────────────────────────────────────────────────

# %% [2] Nomic v1.5 на Wikipedia (~10 мин на T4)
# !python -m src.eval.run_dense \
#     --model nomic-v1.5 \
#     --out results/dense_nomic_300.json \
#     --runs-out results/runs_dense_nomic.json \
#     --max-seq-len 512

# %% [3] Проверка Wikipedia
# import json
# s = json.load(open('results/dense_nomic_300.json'))
# print(f"nDCG@10: {s['overall']['ndcg@10']:.4f}")
# r = json.load(open('results/runs_dense_nomic.json'))
# print(f"Queries in run: {len(r)}")   # должно быть 300

# ─── AKORDA (n=244) ──────────────────────────────────────────────────────────

# %% [4] Nomic v1.5 на Akorda (~5 мин на T4)
# !python -m src.eval.run_akorda \
#     --system nomic-v1.5 \
#     --out results/akorda/dense_nomic.json

# %% [5] Проверка Akorda
# import json
# s = json.load(open('results/akorda/dense_nomic.json'))
# print(f"nDCG@10 ALL: {s['overall']['ndcg@10']:.4f}")
# for cat, m in s.get('by_category', {}).items():
#     print(f"  {cat}: {m['ndcg@10']:.4f}")

# ─── HYBRID AKORDA ───────────────────────────────────────────────────────────

# %% [6] Гибрид BM25+стеммер ⊕ Nomic v1.5 на Akorda
# bm25_kazakh_full.json уже в репо (578 KB).
#
# !python -m src.eval.run_akorda \
#     --system hybrid-nomic-v1.5 \
#     --bm25-runs  results/akorda/bm25_kazakh_full.json \
#     --dense-runs results/akorda/dense_nomic.json \
#     --dense-label nomic-v1.5 \
#     --out results/akorda/hybrid_nomic.json

# ─── FERTILITY ───────────────────────────────────────────────────────────────

# %% [7] Sub-word fertility: Nomic токенизатор vs остальные
# !python -m src.eval.tokenization_test
# !python -m src.eval.fertility_compare

# ─── КОММИТ ─────────────────────────────────────────────────────────────────

# %% [8] Коммит результатов
# !git config user.email "9189920ts@gmail.com"
# !git config user.name "Tim2190"
# !git add results/dense_nomic_300.json \
#           results/runs_dense_nomic.json \
#           results/akorda/dense_nomic.json \
#           results/akorda/hybrid_nomic.json
# !git commit -m "Nomic v1.5: benchmark results on Wikipedia (n=300) and Akorda (n=244)"
# !git push origin main
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

print("=== Nomic v1.5 результаты ===")
_show("results/dense_nomic_300.json",    "Wiki  (dense)")
_show("results/akorda/dense_nomic.json", "Akorda (dense)")
_show("results/akorda/hybrid_nomic.json","Akorda (hybrid)")
