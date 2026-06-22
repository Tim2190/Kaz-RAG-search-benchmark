"""
Прогон Qwen3-Embedding-0.6B на обоих датасетах (Wikipedia n=300 + Akorda n=244).

ТРЕБОВАНИЯ: Kaggle GPU T4 + Internet ON.
Каждый "# %%" — отдельная ячейка.

Qwen3-Embedding-0.6B (Qwen/Qwen3-Embedding-0.6B) — LLM-based embeds от Alibaba.
Использует инструкционный префикс для запросов:
  "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
Документы индексируются без префикса.
Работает через стандартный SentenceTransformer API. Restart НЕ нужен.

# %% [1] Установка зависимостей и клонирование
# !pip install -q sentence-transformers torch
# !git clone -b claude/laughing-ramanujan-LsMa8 https://github.com/Tim2190/Kaz-RAG-search-benchmark.git
# %cd Kaz-RAG-search-benchmark

# ─── WIKIPEDIA (n=300) ───────────────────────────────────────────────────────

# %% [2] Qwen3-0.6B на Wikipedia (~20 мин на T4)
# !python -m src.eval.run_dense \
#     --model qwen3-0.6b \
#     --out results/dense_qwen3_0.6b_300.json \
#     --runs-out results/runs_dense_qwen3_0.6b.json \
#     --max-seq-len 512

# %% [3] Проверка Wikipedia
# import json
# s = json.load(open('results/dense_qwen3_0.6b_300.json'))
# print(f"nDCG@10: {s['overall']['ndcg@10']:.4f}")
# r = json.load(open('results/runs_dense_qwen3_0.6b.json'))
# print(f"Queries in run: {len(r)}")   # должно быть 300

# ─── AKORDA (n=244) ──────────────────────────────────────────────────────────

# %% [4] Qwen3-0.6B на Akorda (~10 мин на T4)
# !python -m src.eval.run_akorda \
#     --system qwen3-0.6b \
#     --out results/akorda/dense_qwen3_0.6b.json

# %% [5] Проверка Akorda
# import json
# s = json.load(open('results/akorda/dense_qwen3_0.6b.json'))
# print(f"nDCG@10 ALL: {s['overall']['ndcg@10']:.4f}")
# for cat, m in s.get('by_category', {}).items():
#     print(f"  {cat}: {m['ndcg@10']:.4f}")

# ─── HYBRID AKORDA ───────────────────────────────────────────────────────────

# %% [6] Гибрид BM25+стеммер ⊕ Qwen3-0.6B на Akorda
# bm25_kazakh_full.json уже в репо (578 KB).
#
# Шаг 6a: dense runs уже сделаны в ячейке [4]
#
# Шаг 6b: hybrid
# !python -m src.eval.run_akorda \
#     --system hybrid-qwen3-0.6b \
#     --bm25-runs  results/akorda/bm25_kazakh_full.json \
#     --dense-runs results/akorda/dense_qwen3_0.6b.json \
#     --dense-label qwen3-0.6b \
#     --out results/akorda/hybrid_qwen3_0.6b.json

# ─── FERTILITY ───────────────────────────────────────────────────────────────

# %% [7] Sub-word fertility: Qwen3 токенизатор vs остальные
# (если уже запускали для другой модели — пропустить; кэш слов уже есть)
# !python -m src.eval.tokenization_test
# !python -m src.eval.fertility_compare

# ─── КОММИТ ─────────────────────────────────────────────────────────────────

# %% [8] Коммит результатов
# !git config user.email "9189920ts@gmail.com"
# !git config user.name "Tim2190"
# !git add results/dense_qwen3_0.6b_300.json \
#           results/runs_dense_qwen3_0.6b.json \
#           results/akorda/dense_qwen3_0.6b.json \
#           results/akorda/hybrid_qwen3_0.6b.json
# !git commit -m "Qwen3-Embedding-0.6B: benchmark results on Wikipedia (n=300) and Akorda (n=244)"
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

print("=== Qwen3-Embedding-0.6B результаты ===")
_show("results/dense_qwen3_0.6b_300.json",    "Wiki  (dense)")
_show("results/akorda/dense_qwen3_0.6b.json", "Akorda (dense)")
_show("results/akorda/hybrid_qwen3_0.6b.json","Akorda (hybrid)")
