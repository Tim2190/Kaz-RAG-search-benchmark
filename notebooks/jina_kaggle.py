"""
Прогон Jina v3 на обоих датасетах (Wikipedia n=300 + Akorda n=244).

ТРЕБОВАНИЯ: Kaggle GPU T4 + Internet ON.
Каждый "# %%" — отдельная ячейка.

# %% [1] Установка зависимостей и клонирование
# ВАЖНО: Jina v3 несовместима с новейшим transformers (>=4.56) — там
# переписан загрузчик весов (core_model_loading.dot_natural_key), и кастомный
# код Jina падает с TypeError "'<' not supported between str and int".
# Пиним рабочую версию. После установки ОБЯЗАТЕЛЬНО Restart kernel (Run → Restart),
# иначе останется уже импортированный новый transformers.
# !pip install -q "transformers==4.49.0" "sentence-transformers==3.3.1" torch einops
#
# --- ПОСЛЕ restart выполнить клонирование заново: ---
# !git clone https://github.com/Tim2190/Kaz-RAG-search-benchmark.git
# %cd Kaz-RAG-search-benchmark

# ─── WIKIPEDIA (n=300) ───────────────────────────────────────────────────────

# %% [2] Jina v3 на Wikipedia (~15 мин на T4)
# Модель: jinaai/jina-embeddings-v3
# Encoding: task='retrieval.query' для запросов, 'retrieval.passage' для пассажей
# Требует trust_remote_code=True (кастомный код в репо модели).
# Лимит контекста 512 токенов — для сравнимости с e5/Granite.
#
# !python -m src.eval.run_dense \
#     --model jina-v3 \
#     --out results/dense_jina_300.json \
#     --runs-out results/runs_dense_jina.json \
#     --max-seq-len 512

# %% [3] Проверка Wikipedia
# import json
# s = json.load(open('results/dense_jina_300.json'))
# print(f"nDCG@10: {s['overall']['ndcg@10']:.4f}")
# r = json.load(open('results/runs_dense_jina.json'))
# print(f"Queries in run: {len(r)}")   # должно быть 300

# ─── AKORDA (n=244) ──────────────────────────────────────────────────────────

# %% [4] Jina v3 на Akorda (~10 мин на T4)
# Стеммер нужен только для BM25-канала; для Jina — только dense
# (эмбеддинги корпуса пересчитываются с нуля, кэш не совместим с Wiki)
#
# !python -m src.eval.run_akorda \
#     --system jina-v3 \
#     --out results/akorda/dense_jina.json

# %% [5] Проверка Akorda
# import json
# s = json.load(open('results/akorda/dense_jina.json'))
# print(f"nDCG@10 ALL: {s['overall']['ndcg@10']:.4f}")
# by_cat = s.get('by_category', {})
# for cat, m in by_cat.items():
#     print(f"  {cat}: {m['ndcg@10']:.4f}")

# ─── HYBRID AKORDA (опционально, после [4]) ──────────────────────────────────

# %% [6] Гибрид BM25+стеммер ⊕ Jina v3 на Akorda
# Для гибрида нужен сохранённый bm25_kazakh_full.json (full-cache BM25+stemmer run).
# Этот файл уже есть в репо (578 KB), скачивать не нужно.
#
# !python -m src.eval.run_akorda \
#     --system hybrid-jina-v3 \
#     --bm25-runs  results/akorda/bm25_kazakh_full.json \
#     --dense-runs results/akorda/dense_jina.json \
#     --dense-label jina-v3 \
#     --out results/akorda/hybrid_jina.json

# ─── FERTILITY (опционально) ─────────────────────────────────────────────────

# %% [7] Sub-word fertility: Jina токенизатор vs остальные
# Сравнивает среднее число sub-word на казахское слово по обоим корпусам.
# Требует trust_remote_code=True (загружается автоматически в TOKENIZERS).
#
# !python -m src.eval.tokenization_test   # Wiki
# !python -m src.eval.fertility_compare   # Wiki vs Akorda

# ─── КОММИТ ─────────────────────────────────────────────────────────────────

# %% [8] Коммит всех результатов
# !git config user.email "9189920ts@gmail.com"
# !git config user.name "Tim2190"
# !git add results/dense_jina_300.json \
#           results/runs_dense_jina.json \
#           results/akorda/dense_jina.json \
#           results/akorda/hybrid_jina.json
# !git commit -m "Jina v3: benchmark results on Wikipedia (n=300) and Akorda (n=244)"
# !git push origin main

# ─── Ожидаемый формат вывода [2] ─────────────────────────────────────────────
# === BM25 (стеммер=dense:jina-v3) | 8370 пассажей, 300 запросов ===
# категория         recall@1  recall@5   mrr@10  ndcg@10
# ВСЕ                  X.XXX     X.XXX    X.XXX    X.XXX
# inflected            X.XXX     X.XXX    X.XXX    X.XXX
# natural              X.XXX     X.XXX    X.XXX    X.XXX
# vocabulary-gap       X.XXX     X.XXX    X.XXX    X.XXX
"""

# ── Standalone-блок: просмотр уже сохранённых результатов ───────────────────

import json, os, glob

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

print("=== Jina v3 результаты ===")
_show("results/dense_jina_300.json",         "Wiki  (dense)")
_show("results/akorda/dense_jina.json",      "Akorda (dense)")
_show("results/akorda/hybrid_jina.json",     "Akorda (hybrid)")
