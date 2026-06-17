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
# ── Cell 4: BM25+stemmer с ПОЛНЫМ кэшем (ПРИОРИТЕТ 1) ────────────────────────
# Закрывает единственную дыру в AKORDA_RESULTS.md (Finding 3): сейчас результат
# на 78%-кэше. На Kaggle с Internet ON API стеммера доступен → warm() догонит
# недостающие ~2338 токенов (22%) до 100%, и стеммер сработает на всех токенах.
# НИЧЕГО НЕ МЕНЯЕМ в коде — тот же существующий стеммер, тот же run_akorda.
# !python -m src.eval.run_akorda --system bm25-stemmer \
#     --out results/akorda/bm25_kazakh_full.json
#
# После прогона сравнить с 78%-версией:
#   import json
#   a=json.load(open('results/akorda/bm25_kazakh.json'))['overall']['ndcg@10']
#   b=json.load(open('results/akorda/bm25_kazakh_full.json'))['overall']['ndcg@10']
#   print(f'78%-кэш: {a:.4f}  |  100%-кэш: {b:.4f}  |  Δ: {b-a:+.4f}')
# Если Δ заметна — обновить Finding 3 и таблицы; если ≈0 — снять пометку
# "preliminary" (стеммер честно не помогает на формальном тексте).
# (опц.) закоммитить пополненный кэш, чтобы 78%→100% было и в репо:
# !git add data/resources/stem_cache.json results/akorda/bm25_kazakh_full.json
#
# ── Cell 5: Fertility Akorda vs Wiki (ПРИОРИТЕТ 2) ───────────────────────────
# Тот же инструмент, что и токенизационный тест, но на ДВУХ корпусах.
# Словники уже закэшированы в репо (data/words_*_100.json) — сети для них не надо,
# только для загрузки 5 токенизаторов с HF (Internet ON; TilQazyna — gated, нужен
# HF-токен с принятым доступом, иначе строка [skip] и считаем по 4 токенизаторам).
# !python -m src.eval.fertility_compare
#
# Подавать как CANDIDATE MECHANISM для общей доменной просадки (НЕ для shyngys vs e5
# отдельно — у них общий токенизатор; см. докстринг fertility_compare.py).
#
# ── Cell 6: коммит результатов ───────────────────────────────────────────────
# !git add results/akorda/
# !git commit -m "Akorda: full-cache BM25 + fertility comparison"
# !git push origin claude/akorda-hybrid
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
