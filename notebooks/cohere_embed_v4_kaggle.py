"""
Прогон Cohere embed-v4.0 на обоих датасетах (Wikipedia n=300 + Akorda n=244).

ТРЕБОВАНИЯ: Kaggle (CPU или GPU) + Internet ON + секрет COHERE_API_KEY.
Каждый "# %%" — отдельная ячейка.

Cohere embed-v4.0 — флагманская мультиязычная модель Cohere (2025).
Специально обучена для кросс-язычного поиска (arXiv), 100+ языков,
128K-токенный контекст, 1024-dim. API-only (не HuggingFace).
Restart НЕ нужен.

# %% [1] Установка зависимостей и клонирование
# !pip install -q cohere>=5
# !git clone -b claude/laughing-ramanujan-LsMa8 https://github.com/Tim2190/Kaz-RAG-search-benchmark.git
# %cd Kaz-RAG-search-benchmark

# %% [2] Установка API-ключа (из Kaggle Secrets → Add-on → Secrets)
# from kaggle_secrets import UserSecretsClient
# import os
# os.environ["COHERE_API_KEY"] = UserSecretsClient().get_secret("COHERE_API_KEY")

# ─── WIKIPEDIA (n=300) ───────────────────────────────────────────────────────

# %% [3] Cohere embed-v4.0 на Wikipedia (~3 мин, ~92 API-вызова)
# !python -m src.eval.run_cohere \
#     --model embed-v4.0 \
#     --out results/dense_cohere_v4_300.json \
#     --runs-out results/runs_dense_cohere_v4.json \
#     --top-k 100

# %% [4] Проверка Wikipedia
# import json
# s = json.load(open('results/dense_cohere_v4_300.json'))
# print(f"nDCG@10: {s['overall']['ndcg@10']:.4f}")
# r = json.load(open('results/runs_dense_cohere_v4.json'))
# print(f"Queries in run: {len(r)}")   # должно быть 300

# ─── AKORDA (n=244) ──────────────────────────────────────────────────────────

# %% [5] Cohere embed-v4.0 на Akorda (~1 мин, ~9 API-вызовов)
# !python -m src.eval.run_akorda \
#     --system cohere-v4 \
#     --out results/akorda/dense_cohere_v4.json

# %% [6] Проверка Akorda
# import json
# s = json.load(open('results/akorda/dense_cohere_v4.json'))
# print(f"nDCG@10 ALL: {s['overall']['ndcg@10']:.4f}")
# for cat, m in s.get('by_category', {}).items():
#     print(f"  {cat}: {m['ndcg@10']:.4f}")

# ─── HYBRID AKORDA ───────────────────────────────────────────────────────────

# %% [7] Гибрид BM25+стеммер ⊕ Cohere embed-v4.0 на Akorda
# bm25_kazakh_full.json уже в репо.
# dense runs с top-k=100 получены на шаге [5] (run_akorda сохраняет run внутри JSON).
#
# !python -m src.eval.run_akorda \
#     --system hybrid-cohere-v4 \
#     --bm25-runs  results/akorda/bm25_kazakh_full.json \
#     --dense-runs results/akorda/dense_cohere_v4.json \
#     --dense-label cohere-v4 \
#     --out results/akorda/hybrid_cohere_v4.json

# ─── КОММИТ ─────────────────────────────────────────────────────────────────

# %% [8] Коммит результатов
# !git config user.email "9189920ts@gmail.com"
# !git config user.name "Tim2190"
# !git add results/dense_cohere_v4_300.json \
#           results/runs_dense_cohere_v4.json \
#           results/akorda/dense_cohere_v4.json \
#           results/akorda/hybrid_cohere_v4.json
# !git commit -m "Cohere embed-v4.0: benchmark results on Wikipedia (n=300) and Akorda (n=244)"
# !git push origin claude/laughing-ramanujan-LsMa8
"""

import json, os


def _show(path, label):
    if not os.path.exists(path):
        print(f"  {label}: нет файла ({path})")
        return
    d = json.load(open(path))
    ov = d.get("overall", {})
    print(f"  {label}: nDCG@10={ov.get('ndcg@10', '?'):.4f}  "
          f"MRR@10={ov.get('mrr@10', '?'):.4f}")
    for cat, m in d.get("by_category", {}).items():
        print(f"    {cat}: {m.get('ndcg@10', '?'):.4f}")


print("=== Cohere embed-v4.0 результаты ===")
_show("results/dense_cohere_v4_300.json",    "Wiki   (dense)")
_show("results/akorda/dense_cohere_v4.json", "Akorda (dense)")
_show("results/akorda/hybrid_cohere_v4.json","Akorda (hybrid)")
