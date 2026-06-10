# Sprint 3 — ДОПРОГОН одной модели: shyngys879/kazakh-e5-rag-embedding
# ============================================================================
# Зачем: эта казахская дообученная E5 была зарегистрирована в Sprint 2, но так
#   и не прогонялась. Здесь гоним ТОЛЬКО её на тех же 127 валидированных
#   запросах, что и основной прогон. Гибрид (стеммер ⊕ shyngys-e5) считается
#   локально из уже готового sprint3_runs.json — на Kaggle его делать не надо.
#
# Результат: /kaggle/working/shyngys_run.json  →  {"shyngys_e5": {qid: [doc_id...]}}
#   Скачайте и пришлите в чат. Я вмержу в sprint3_runs.json и пересчитаю.
#
# ТРЕБОВАНИЯ: Kaggle GPU T4 + Internet ON. Каждый "# %%" — отдельная ячейка.

# %% [1] Зависимости + окружение
import subprocess, sys, os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "sentence-transformers>=2.6"])

# %% [2] Репозиторий (нужны корпус + валидированные запросы)
import json
from typing import Dict, List, Set, Tuple

REPO = "https://github.com/Tim2190/Kaz-RAG-search-benchmark.git"
BRANCH = "claude/laughing-ramanujan-LsMa8"
REPO_DIR = "/kaggle/working/repo"
if not os.path.exists(REPO_DIR):
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "--branch", BRANCH, REPO, REPO_DIR])
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
from src.eval import metrics

TOP_K = 20

# %% [3] Данные: корпус + те же 127 валидированных запросов
corpus_pairs: List[Tuple[str, str]] = []
with open("data/corpus/corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        corpus_pairs.append((d["doc_id"], d["text"]))
print(f"Корпус: {len(corpus_pairs)} пассажей")

queries: Dict[str, str] = {}
qrels: Dict[str, Set[str]] = {}
with open("data/queries/synonym_queries_final.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        queries[item["qid"]] = item["query"]
        qrels[item["qid"]] = {item["gold_doc_id"]}
print(f"Запросов: {len(queries)} (носительски валидированы)")

# %% [4] Прогон shyngys879/kazakh-e5-rag-embedding (E5-префиксы query:/passage:)
import numpy as np, gc, torch
from sentence_transformers import SentenceTransformer

MODEL = "shyngys879/kazakh-e5-rag-embedding"
QPREF, DPREF = "query: ", "passage: "   # дообучена от multilingual-e5
MAX_SEQ, BATCH = 512, 32

print(f"  модель: {MODEL}")
model = SentenceTransformer(MODEL)
try:
    model.max_seq_length = min(int(model.max_seq_length or MAX_SEQ), MAX_SEQ)
except Exception:
    model.max_seq_length = MAX_SEQ

doc_emb = model.encode([DPREF + t for _, t in corpus_pairs],
                       batch_size=BATCH, show_progress_bar=True,
                       normalize_embeddings=True)
doc_ids = [d for d, _ in corpus_pairs]
doc_emb = np.asarray(doc_emb, dtype=np.float32)
nrm = np.linalg.norm(doc_emb, axis=1, keepdims=True); nrm[nrm == 0] = 1; doc_emb /= nrm

qids = sorted(queries)
q_emb = np.asarray(model.encode([QPREF + queries[q] for q in qids],
                                batch_size=BATCH, normalize_embeddings=True),
                   dtype=np.float32)
run = {}
for i, qid in enumerate(qids):
    sc = doc_emb @ q_emb[i]
    top = np.argpartition(-sc, TOP_K)[:TOP_K]
    top = top[np.argsort(-sc[top])]
    run[qid] = [doc_ids[j] for j in top]
del model; gc.collect(); torch.cuda.empty_cache()

# контрольные метрики (passage-level, single-gold — для сверки на глаз)
m = metrics.evaluate_run({q: r[:10] for q, r in run.items()},
                         qrels, metrics=("hit", "mrr", "ndcg"), ks=(1, 5, 10))
print(f"  shyngys-e5  Hit@1={m['hit@1']:.3f} Hit@10={m['hit@10']:.3f} "
      f"nDCG@10={m['ndcg@10']:.3f}")

# %% [5] Сохранение
OUT = "/kaggle/working/shyngys_run.json"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"shyngys_e5": run}, f, ensure_ascii=False)
print(f"\nСохранено → {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)")
print("Скачайте и пришлите в чат — вмержу в sprint3_runs.json и пересчитаю "
      "(passage + article + CI95), гибрид посчитаю локально.")
