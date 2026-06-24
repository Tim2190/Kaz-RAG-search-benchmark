"""
Слой input-preprocessing: 4 линии (baseline/stem/latin/stem_latin) × 2 модели
(e5, qwen3-0.6b) × 2 домена (Wikipedia n=300, Akorda n=244).

Гипотеза: транслитерация кириллица→латиница и/или стеммер-предобработка чинят
вход «сломанных» (byte-fallback) эмбеддеров БЕЗ дообучения. e5 — «нормальная»
контрольная модель (проверяем, не вредит ли предобработка хорошей модели);
qwen3-0.6b — «сломанная» (главная гипотеза).

ТРЕБОВАНИЯ: Kaggle GPU T4 + Internet ON (нужен и HF, и API казахского стеммера
для линий stem/stem_latin). Каждый "# %%" — отдельная ячейка.

КРИТИЧНО: в каждой линии запрос и корпус обрабатываются ОДНОЙ функцией
(симметрия). Эмбеддинг-кэш раздельный на линию. Главный результат — compare:
Δ nDCG@10 линий 1–3 против baseline (0) с paired bootstrap.

# %% [1] Установка зависимостей и клонирование
# !pip install -q sentence-transformers torch
# !git clone -b claude/tender-goldberg-snsxap https://github.com/Tim2190/Kaz-RAG-search-benchmark.git
# %cd Kaz-RAG-search-benchmark

# ─── BASELINE (0): wiki переиспользуем готовые ранжировки (без лишнего GPU) ───

# %% [2] Wiki baseline из готовых dense-ранжировок
# !python -m src.eval.run_preprocessing run --domain wiki --model e5 --line baseline \
#     --reuse-run results/runs_dense_e5.json \
#     --out input-preprocessing/runs/wiki_e5_baseline.json
# !python -m src.eval.run_preprocessing run --domain wiki --model qwen3-0.6b --line baseline \
#     --reuse-run results/runs_dense_qwen3_0.6b.json \
#     --out input-preprocessing/runs/wiki_qwen3-0.6b_baseline.json

# %% [3] Akorda baseline (dense-ранжировок в репо нет — считаем; ~10 мин/модель)
# !python -m src.eval.run_preprocessing run --domain akorda --model e5 --line baseline \
#     --out input-preprocessing/runs/akorda_e5_baseline.json
# !python -m src.eval.run_preprocessing run --domain akorda --model qwen3-0.6b --line baseline \
#     --out input-preprocessing/runs/akorda_qwen3-0.6b_baseline.json

# ─── ЛИНИЯ 1: STEM ───────────────────────────────────────────────────────────
# (первая stem-линия прогреет стем-кэш по корпусу; ETA печатается)

# %% [4] stem — e5 + qwen3, оба домена
# for DOM in wiki akorda:
#   for M in e5 qwen3-0.6b:
# !python -m src.eval.run_preprocessing run --domain wiki   --model e5         --line stem --out input-preprocessing/runs/wiki_e5_stem.json
# !python -m src.eval.run_preprocessing run --domain wiki   --model qwen3-0.6b --line stem --out input-preprocessing/runs/wiki_qwen3-0.6b_stem.json
# !python -m src.eval.run_preprocessing run --domain akorda --model e5         --line stem --out input-preprocessing/runs/akorda_e5_stem.json
# !python -m src.eval.run_preprocessing run --domain akorda --model qwen3-0.6b --line stem --out input-preprocessing/runs/akorda_qwen3-0.6b_stem.json

# ─── ЛИНИЯ 2: LATIN (главная гипотеза) ───────────────────────────────────────

# %% [5] latin — e5 + qwen3, оба домена
# !python -m src.eval.run_preprocessing run --domain wiki   --model e5         --line latin --out input-preprocessing/runs/wiki_e5_latin.json
# !python -m src.eval.run_preprocessing run --domain wiki   --model qwen3-0.6b --line latin --out input-preprocessing/runs/wiki_qwen3-0.6b_latin.json
# !python -m src.eval.run_preprocessing run --domain akorda --model e5         --line latin --out input-preprocessing/runs/akorda_e5_latin.json
# !python -m src.eval.run_preprocessing run --domain akorda --model qwen3-0.6b --line latin --out input-preprocessing/runs/akorda_qwen3-0.6b_latin.json

# ─── ЛИНИЯ 3: STEM + LATIN ───────────────────────────────────────────────────

# %% [6] stem_latin — e5 + qwen3, оба домена
# !python -m src.eval.run_preprocessing run --domain wiki   --model e5         --line stem_latin --out input-preprocessing/runs/wiki_e5_stem_latin.json
# !python -m src.eval.run_preprocessing run --domain wiki   --model qwen3-0.6b --line stem_latin --out input-preprocessing/runs/wiki_qwen3-0.6b_stem_latin.json
# !python -m src.eval.run_preprocessing run --domain akorda --model e5         --line stem_latin --out input-preprocessing/runs/akorda_e5_stem_latin.json
# !python -m src.eval.run_preprocessing run --domain akorda --model qwen3-0.6b --line stem_latin --out input-preprocessing/runs/akorda_qwen3-0.6b_stem_latin.json

# ─── СРАВНЕНИЕ: Δ линий против baseline + paired bootstrap ────────────────────

# %% [7] compare (главный результат)
# !python -m src.eval.run_preprocessing compare --domain wiki   --model e5         --out input-preprocessing/runs/wiki_e5_compare.json
# !python -m src.eval.run_preprocessing compare --domain wiki   --model qwen3-0.6b --out input-preprocessing/runs/wiki_qwen3-0.6b_compare.json
# !python -m src.eval.run_preprocessing compare --domain akorda --model e5         --out input-preprocessing/runs/akorda_e5_compare.json
# !python -m src.eval.run_preprocessing compare --domain akorda --model qwen3-0.6b --out input-preprocessing/runs/akorda_qwen3-0.6b_compare.json

# %% [8] Сводка (печать таблиц для PREPROCESSING.md) — см. _summary() ниже
# !python notebooks/preprocessing_kaggle.py

# ─── КОММИТ ──────────────────────────────────────────────────────────────────

# %% [9] Коммит результатов
# !git config user.email "9189920ts@gmail.com"
# !git config user.name "Tim2190"
# !git add input-preprocessing/runs/*_compare.json data/resources/stem_cache.json
# !git commit -m "input-preprocessing: 4 lines x e5/qwen3 x wiki/akorda (compare + bootstrap)"
# !git push origin claude/tender-goldberg-snsxap
"""

import json
import os

RUNS = "input-preprocessing/runs"
MODELS = ("e5", "qwen3-0.6b")
DOMAINS = ("wiki", "akorda")
LINES = ("stem", "latin", "stem_latin")


def _summary():
    """Печатает Δ nDCG@10 (ALL) каждой линии против baseline по всем
    compare-файлам — удобно копировать в input-preprocessing/PREPROCESSING.md."""
    for dom in DOMAINS:
        for m in MODELS:
            p = os.path.join(RUNS, f"{dom}_{m}_compare.json")
            if not os.path.exists(p):
                print(f"  {dom}/{m}: нет {p}")
                continue
            c = json.load(open(p))
            base = c["baseline"]["ALL"]
            print(f"\n[{dom} / {m}]  baseline ALL nDCG@10 = {base:.4f}")
            for line in LINES:
                v = c["lines"].get(line, {}).get("ALL")
                if not v:
                    continue
                mark = " *" if v["sig"] else ""
                print(f"   {line:<11} {v['line']:.4f}  Δ={v['delta']:+.4f}  "
                      f"p={v['p']:.3f}{mark}")


if __name__ == "__main__":
    print("=== input-preprocessing: Δ против baseline (ALL, nDCG@10) ===")
    _summary()
