"""
Перепрогон e5 на Wikipedia-бенчмарке (300 запросов) с сохранением per-query ранжировок.

Зачем: файл dense_e5_300.json из Sprint 1 содержит только итоговые метрики, без
ранжировок на каждый запрос. Для парного бутстрэпа (BM25+stemmer vs e5,
e5 vs kazakh-e5) нужен файл runs_dense_e5.json аналогичный уже существующим
runs_dense_shyngys.json / runs_dense_granite.json.

ТРЕБОВАНИЯ: Kaggle GPU T4 + Internet ON.
Каждый "# %%" — отдельная ячейка.

# %% [1] Зависимости + клонирование
# !pip install -q sentence-transformers torch
# !git clone https://github.com/Tim2190/Kaz-RAG-search-benchmark.git
# %cd Kaz-RAG-search-benchmark

# %% [2] Прогон e5 с сохранением per-query ранжировок (~15 мин на Kaggle T4)
# !python -m src.eval.run_dense \
#     --model e5 \
#     --out results/dense_e5_300.json \
#     --runs-out results/runs_dense_e5.json

# %% [3] Проверка
# import json
# run = json.load(open('results/runs_dense_e5.json'))
# print(f"Queries: {len(run)}")       # должно быть 300
# print(f"Sample: {list(run.items())[0]}")
# summary = json.load(open('results/dense_e5_300.json'))
# print(f"nDCG@10: {summary['overall']['ndcg@10']:.4f}")  # должно быть ~0.7845

# %% [4] Коммит и пуш
# !git add results/runs_dense_e5.json results/dense_e5_300.json
# !git commit -m "Add e5 per-query runs for Wikipedia benchmark (n=300)"
# !git push origin main
"""

# ── Standalone-блок: просмотр результата после прогона ──────────────────────
import json, os

path = "results/runs_dense_e5.json"
if os.path.exists(path):
    run = json.load(open(path))
    print(f"runs_dense_e5.json: {len(run)} запросов")
    first_qid = list(run.keys())[0]
    print(f"Пример ({first_qid}): {run[first_qid][:5]}")
else:
    print(f"Файл {path} не найден — запусти ячейку [2] выше.")

summary_path = "results/dense_e5_300.json"
if os.path.exists(summary_path):
    s = json.load(open(summary_path))
    print(f"nDCG@10 overall: {s['overall']['ndcg@10']:.4f}")
