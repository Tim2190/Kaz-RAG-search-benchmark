"""
Прогон Nurlykhan/kazembed-v5 на обоих датасетах (Wikipedia n=300 + Akorda n=244).

ТРЕБОВАНИЯ: Kaggle GPU T4 + Internet ON.
Каждый "# %%" — отдельная ячейка.

KazEmbed-V5 (Nurlykhan/kazembed-v5) — дообученный intfloat/multilingual-e5-base
для казахского retrieval/RAG (KazQAD + Powerful-Kazakh-Dialogue, 61 255 пар).
Использует e5-префиксы:
  запрос    → "query: "
  документ  → "passage: "
Работает через стандартный SentenceTransformer API. Restart НЕ нужен.
Базовая модель и схема префиксов идентичны нашему e5 / shyngys-e5 — поэтому
сравнение с ними прямое и честное (один токенизатор, один base).

# %% [1] Установка зависимостей и клонирование
# !pip install -q sentence-transformers torch
# !git clone -b claude/laughing-ramanujan-LsMa8 https://github.com/Tim2190/Kaz-RAG-search-benchmark.git
# %cd Kaz-RAG-search-benchmark

# ─── WIKIPEDIA (n=300) ───────────────────────────────────────────────────────

# %% [2] KazEmbed-V5 на Wikipedia (~5 мин на T4; base-size 278M)
# !python -m src.eval.run_dense \
#     --model kazembed-v5 \
#     --out results/dense_kazembed_v5_300.json \
#     --runs-out results/runs_dense_kazembed_v5.json \
#     --max-seq-len 512

# %% [3] Проверка Wikipedia
# import json
# s = json.load(open('results/dense_kazembed_v5_300.json'))
# print(f"nDCG@10: {s['overall']['ndcg@10']:.4f}")
# r = json.load(open('results/runs_dense_kazembed_v5.json'))
# print(f"Queries in run: {len(r)}")   # должно быть 300

# ─── AKORDA (n=244) ──────────────────────────────────────────────────────────

# %% [4] KazEmbed-V5 на Akorda (~3 мин на T4)
# !python -m src.eval.run_akorda \
#     --system kazembed-v5 \
#     --out results/akorda/dense_kazembed_v5.json

# %% [5] Проверка Akorda
# import json
# s = json.load(open('results/akorda/dense_kazembed_v5.json'))
# print(f"nDCG@10 ALL: {s['overall']['ndcg@10']:.4f}")
# for cat, m in s.get('by_category', {}).items():
#     print(f"  {cat}: {m['ndcg@10']:.4f}")

# ─── HYBRID AKORDA ───────────────────────────────────────────────────────────

# %% [6] Гибрид BM25+стеммер ⊕ KazEmbed-V5 на Akorda (RRF k=60)
# bm25_kazakh_full.json уже в репо.
# dense runs уже сделаны в ячейке [4] (top_k=100 по умолчанию).
# !python -m src.eval.run_akorda \
#     --system hybrid-kazembed-v5 \
#     --bm25-runs  results/akorda/bm25_kazakh_full.json \
#     --dense-runs results/akorda/dense_kazembed_v5.json \
#     --dense-label kazembed-v5 \
#     --out results/akorda/hybrid_kazembed_v5.json

# ─── FERTILITY ───────────────────────────────────────────────────────────────

# %% [7] Sub-word fertility — ПРОПУСТИТЬ.
# KazEmbed-V5 наследует токенизатор multilingual-e5-base (XLM-R, 250 002 токена),
# который уже измерен (fertility 1.81 / 1.82). Дообучение НЕ меняет токенизатор,
# поэтому отдельный прогон не нужен — значение берётся из XLM-R-кластера.

# ─── КОММИТ ─────────────────────────────────────────────────────────────────

# %% [8] Коммит результатов
# !git config user.email "9189920ts@gmail.com"
# !git config user.name "Tim2190"
# !git add results/dense_kazembed_v5_300.json \
#           results/runs_dense_kazembed_v5.json \
#           results/akorda/dense_kazembed_v5.json \
#           results/akorda/hybrid_kazembed_v5.json
# !git commit -m "KazEmbed-V5: benchmark results on Wikipedia (n=300) and Akorda (n=244)"
# !git push origin claude/laughing-ramanujan-LsMa8
#
# Либо просто скачайте 4 JSON и пришлите в чат — я обновлю все таблицы,
# посчитаю paired bootstrap significance и пересоберу лидерборд.
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

print("=== KazEmbed-V5 результаты ===")
_show("results/dense_kazembed_v5_300.json",    "Wiki  (dense)")
_show("results/akorda/dense_kazembed_v5.json", "Akorda (dense)")
_show("results/akorda/hybrid_kazembed_v5.json","Akorda (hybrid)")
