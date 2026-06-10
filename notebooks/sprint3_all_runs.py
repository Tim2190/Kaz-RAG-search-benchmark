# Sprint 3 (v2) — ЕДИНЫЙ прогон: все системы, все запросы (v1+v2), полные выдачи
# ============================================================================
# Отличия от прежних ноутбуков:
#   * запросы = synonym_queries.jsonl (70) + synonym_queries_v2.jsonl (75) = 145
#   * КАЖДАЯ система сохраняет полную выдачу top-20 -> один sprint3_runs.json
#     (нужен для pooling-разметки и article-level пересчёта, см.
#      src/eval/sprint3_pool.py и src/eval/sprint3_rescore.py)
#   * системы: BM25 identity / prefix-5 / каз.стеммер / synonym-expansion,
#              LaBSE, E5-base, Granite-R1-278M, Granite-R2-97M, Granite-R2-311M,
#              Hybrid RRF (стеммер ⊕ R1) и (стеммер ⊕ R2-311M)
#
# ТРЕБОВАНИЯ: Kaggle GPU T4 + Internet ON.
# Каждый "# %%" — отдельная ячейка.

# %% [1] Зависимости + окружение
import subprocess, sys, os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def pip(*a):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *a])

pip("sentence-transformers>=2.6")

# %% [2] Клонирование репо
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

from src.preprocess.tokenize import tokenize
from src.preprocess.stemmer import get_stemmer, stem_tokens
from src.preprocess.synonyms import SynonymExpander
from src.retrieval.bm25 import BM25Index, default_analyzer
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.eval import metrics

TOP_K = 20          # глубина сохраняемых выдач (pooling использует top-10)
RUNS: Dict[str, Dict[str, List[str]]] = {}   # system -> {qid: [doc_id,...]}

# %% [3] Данные: корпус + объединённые запросы v1+v2
corpus_pairs: List[Tuple[str, str]] = []
with open("data/corpus/corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        corpus_pairs.append((d["doc_id"], d["text"]))
print(f"Корпус: {len(corpus_pairs)} пассажей")

queries: Dict[str, str] = {}
qrels: Dict[str, Set[str]] = {}
qid_n = 0
for fname in ("data/queries/synonym_queries.jsonl",
              "data/queries/synonym_queries_v2.jsonl"):
    with open(fname, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = f"syn_{qid_n:03d}"
            queries[qid] = item["query"]
            qrels[qid] = {item["gold_doc_id"]}
            qid_n += 1
print(f"Запросов: {len(queries)} (v1+v2)")

def show(name: str, run: Dict[str, List[str]]):
    run10 = {q: r[:10] for q, r in run.items()}
    m = metrics.evaluate_run(run10, qrels, metrics=("hit", "mrr", "ndcg"), ks=(1, 5, 10))
    print(f"  {name:<22} Hit@1={m['hit@1']:.3f} Hit@10={m['hit@10']:.3f} "
          f"nDCG@10={m['ndcg@10']:.3f}")
    return m

# %% [4] Казахский стеммер (Cloud Run + кэш)
print("Стеммер: прогрев...")
stemmer = get_stemmer("kazakh")
toks = set()
for _, t in corpus_pairs:
    toks.update(tokenize(t))
for t in queries.values():
    toks.update(tokenize(t))
stemmer.warm(toks)
print("  готов.")

# %% [5] BM25 x4
def prefix5_analyzer(text: str) -> List[str]:
    return [w[:5] for w in tokenize(text)]

print("\n[1/9] BM25 identity...")
idx_id = BM25Index().index(corpus_pairs)
RUNS["bm25_identity"] = idx_id.run(queries, top_k=TOP_K)
show("BM25 identity", RUNS["bm25_identity"])

print("[2/9] BM25 prefix-5...")
idx_p5 = BM25Index(analyzer=prefix5_analyzer).index(corpus_pairs)
RUNS["bm25_prefix5"] = idx_p5.run(queries, top_k=TOP_K)
show("BM25 prefix-5", RUNS["bm25_prefix5"])

print("[3/9] BM25 + каз. стеммер...")
idx_stem = BM25Index(analyzer=default_analyzer(stemmer)).index(corpus_pairs)
RUNS["bm25_stemmer"] = idx_stem.run(queries, top_k=TOP_K)
show("BM25+стеммер", RUNS["bm25_stemmer"])

print("[4/9] BM25 + synonym expansion...")
query_stems = {qid: stem_tokens(tokenize(t), stemmer) for qid, t in queries.items()}
all_stems = set().union(*query_stems.values())
expander = SynonymExpander(cache_path="data/resources/synonym_cache.json")
expander.warm(all_stems)
expanded = {qid: expander.expand(s) for qid, s in query_stems.items()}
RUNS["bm25_synonym"] = idx_stem.run_terms(expanded, top_k=TOP_K)
show("BM25+synonym", RUNS["bm25_synonym"])

# %% [6] Dense x5
import numpy as np, gc, torch
from sentence_transformers import SentenceTransformer

def dense_run(model_name, query_prefix="", doc_prefix="", cache_tag="",
              max_seq_length=512, batch_size=32):
    print(f"  модель: {model_name}")
    model = SentenceTransformer(model_name)
    try:
        model.max_seq_length = min(int(model.max_seq_length or max_seq_length),
                                   max_seq_length)
    except Exception:
        model.max_seq_length = max_seq_length
    npy = f"/kaggle/working/emb_{cache_tag}.npy"
    ids = f"/kaggle/working/emb_{cache_tag}_ids.json"
    if os.path.exists(npy) and os.path.exists(ids):
        doc_ids = json.load(open(ids)); doc_emb = np.load(npy)
    else:
        doc_emb = model.encode([doc_prefix + t for _, t in corpus_pairs],
                               batch_size=batch_size, show_progress_bar=True,
                               normalize_embeddings=True)
        doc_ids = [d for d, _ in corpus_pairs]
        np.save(npy, doc_emb); json.dump(doc_ids, open(ids, "w"))
    doc_emb = np.asarray(doc_emb, dtype=np.float32)
    n = np.linalg.norm(doc_emb, axis=1, keepdims=True); n[n == 0] = 1; doc_emb /= n
    qids = sorted(queries)
    q_emb = np.asarray(model.encode([query_prefix + queries[q] for q in qids],
                                    batch_size=batch_size,
                                    normalize_embeddings=True), dtype=np.float32)
    out = {}
    for i, qid in enumerate(qids):
        sc = doc_emb @ q_emb[i]
        top = np.argpartition(-sc, TOP_K)[:TOP_K]
        top = top[np.argsort(-sc[top])]
        out[qid] = [doc_ids[j] for j in top]
    del model; gc.collect(); torch.cuda.empty_cache()
    return out

print("\n[5/9] LaBSE...")
RUNS["labse"] = dense_run("sentence-transformers/LaBSE", cache_tag="labse")
show("LaBSE", RUNS["labse"])

print("[6/9] multilingual-e5-base...")
RUNS["e5_base"] = dense_run("intfloat/multilingual-e5-base",
                            query_prefix="query: ", doc_prefix="passage: ",
                            cache_tag="e5")
show("E5-base", RUNS["e5_base"])

print("[7/9] Granite-278M (R1)...")
RUNS["granite_r1_278m"] = dense_run(
    "ibm-granite/granite-embedding-278m-multilingual", cache_tag="granite_r1")
show("Granite-R1-278M", RUNS["granite_r1_278m"])

print("[8/9] Granite-97M-R2...")
RUNS["granite_r2_97m"] = dense_run(
    "ibm-granite/granite-embedding-97m-multilingual-r2", cache_tag="granite_r2_97")
show("Granite-R2-97M", RUNS["granite_r2_97m"])

print("[9/9] Granite-311M-R2...")
RUNS["granite_r2_311m"] = dense_run(
    "ibm-granite/granite-embedding-311m-multilingual-r2", cache_tag="granite_r2_311")
show("Granite-R2-311M", RUNS["granite_r2_311m"])

# %% [7] Гибриды (RRF k=60)
print("\nHybrid RRF...")
RUNS["hybrid_rrf_r1"] = reciprocal_rank_fusion(
    {"bm25": RUNS["bm25_stemmer"], "dense": RUNS["granite_r1_278m"]},
    k=60, top_k=TOP_K)
show("Hybrid ⊕ R1", RUNS["hybrid_rrf_r1"])

RUNS["hybrid_rrf_r2_311m"] = reciprocal_rank_fusion(
    {"bm25": RUNS["bm25_stemmer"], "dense": RUNS["granite_r2_311m"]},
    k=60, top_k=TOP_K)
show("Hybrid ⊕ R2-311M", RUNS["hybrid_rrf_r2_311m"])

# %% [8] Сохранение: ОДИН файл с полными выдачами
OUT = "/kaggle/working/sprint3_runs.json"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(RUNS, f, ensure_ascii=False)
size_mb = os.path.getsize(OUT) / 1e6
print(f"\nСохранено → {OUT} ({size_mb:.1f} MB, {len(RUNS)} систем x {len(queries)} запросов)")
print("Скачайте этот файл и передайте в чат — дальше всё считается локально:")
print("  python -m src.eval.sprint3_pool    --runs sprint3_runs.json --out annotation/pool_relevance.tsv")
print("  python -m src.eval.sprint3_rescore --runs sprint3_runs.json --pool annotation/pool_relevance.tsv")
