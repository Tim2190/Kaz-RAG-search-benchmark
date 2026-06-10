# Sprint 3 — Synonym / Vocabulary-Gap Benchmark
# Запускать в Kaggle (GPU T4/P100) или Colab с GPU.
# Прогоняет 5 систем на synonym_queries.jsonl и выводит таблицу Hit@k.
#
# СТРУКТУРА NOTEBOOK (каждый комментарий "# %%" — новая ячейка):
#   1. Установка зависимостей
#   2. Клонирование репо и импорты
#   3. Вспомогательные функции (BM25, метрики)
#   4. Загрузка корпуса и запросов
#   5. BM25 identity (без стеммера)
#   6. BM25 prefix-5 (5-char prefix stem — тот же, что фильтр)
#   7. Dense: LaBSE
#   8. Dense: multilingual-e5-base
#   9. Dense: IBM Granite-278M
#  10. Итоговая таблица

# %% [1] Установка зависимостей
# ============================================================
import subprocess, sys

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

pip("sentence-transformers>=2.6")
pip("torch")          # обычно уже есть в Kaggle

# %% [2] Клонирование репо и базовые импорты
# ============================================================
import os, json, math, re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Sequence

REPO = "https://github.com/Tim2190/Kaz-RAG-search-benchmark.git"
REPO_DIR = "/kaggle/working/repo"

if not os.path.exists(REPO_DIR):
    subprocess.check_call(["git", "clone", "--depth", "1", REPO, REPO_DIR])

CORPUS_PATH  = f"{REPO_DIR}/data/corpus/corpus.jsonl"
QUERIES_PATH = f"{REPO_DIR}/data/queries/synonym_queries.jsonl"

# %% [3] Вспомогательные функции
# ============================================================

# --- токенизация (Казахский Кириллица) ---
_WORD = re.compile(r"[А-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]+")

def tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())

def prefix5(text: str) -> List[str]:
    """5-char prefix stemmer — тот же, что synonym_filter.py."""
    return [w[:5] for w in tokenize(text)]

def identity(text: str) -> List[str]:
    return tokenize(text)

# --- BM25 Okapi ---
class BM25:
    def __init__(self, analyzer=identity, k1=1.5, b=0.75):
        self.analyzer = analyzer
        self.k1 = k1; self.b = b
        self.ids: List[str] = []
        self.freqs: List[Counter] = []
        self.df: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        self.lens: List[int] = []
        self.avgdl = 0.0

    def index(self, docs: List[Tuple[str, str]]) -> "BM25":
        for did, text in docs:
            toks = self.analyzer(text)
            self.ids.append(did)
            self.freqs.append(Counter(toks))
            self.lens.append(len(toks))
            for t in set(toks):
                self.df[t] += 1
        n = len(self.ids)
        self.avgdl = sum(self.lens) / n if n else 0.0
        for t, df in self.df.items():
            self.idf[t] = math.log(1 + (n - df + 0.5) / (df + 0.5))
        return self

    def search(self, query: str, k: int = 10) -> List[str]:
        q = self.analyzer(query)
        scores = []
        for i, did in enumerate(self.ids):
            fr = self.freqs[i]
            dl = self.lens[i]
            norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
            s = sum(
                self.idf.get(t, 0) * fr[t] * (self.k1 + 1) / (fr[t] + norm)
                for t in q if fr[t]
            )
            if s > 0:
                scores.append((did, s))
        scores.sort(key=lambda x: -x[1])
        return [d for d, _ in scores[:k]]

    def run(self, queries: Dict[str, str], k: int = 10) -> Dict[str, List[str]]:
        return {qid: self.search(text, k) for qid, text in queries.items()}

# --- метрики ---
def hit_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    return 1.0 if any(d in gold for d in ranked[:k]) else 0.0

def mrr(ranked: List[str], gold: Set[str], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k], 1):
        if d in gold:
            return 1.0 / i
    return 0.0

def ndcg(ranked: List[str], gold: Set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0

def evaluate(run: Dict[str, List[str]], qrels: Dict[str, Set[str]]) -> Dict:
    ks = [1, 5, 10]
    results = {f"Hit@{k}": [] for k in ks}
    results["MRR@10"] = []
    results["nDCG@10"] = []
    for qid, ranked in run.items():
        gold = qrels.get(qid, set())
        for k in ks:
            results[f"Hit@{k}"].append(hit_at_k(ranked, gold, k))
        results["MRR@10"].append(mrr(ranked, gold))
        results["nDCG@10"].append(ndcg(ranked, gold))
    return {k: sum(v) / len(v) if v else 0.0 for k, v in results.items()}

# %% [4] Загрузка корпуса и запросов
# ============================================================

print("Загрузка корпуса...")
corpus: List[Tuple[str, str]] = []
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        corpus.append((d["doc_id"], d["text"]))
print(f"  {len(corpus)} пассажей")

print("Загрузка synonym запросов...")
queries: Dict[str, str] = {}   # qid -> query text
qrels: Dict[str, Set[str]] = {}  # qid -> {gold_doc_id}
overlaps: Dict[str, float] = {}

with open(QUERIES_PATH, encoding="utf-8") as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        qid = f"syn_{i:03d}"
        queries[qid] = item["query"]
        qrels[qid] = {item["gold_doc_id"]}
        overlaps[qid] = item.get("overlap", 0.0)

print(f"  {len(queries)} запросов")
print(f"  средний overlap: {sum(overlaps.values())/len(overlaps):.3f}")

# %% [5] BM25 — без стеммера (baseline)
# ============================================================
print("\n[1/5] BM25 identity (без стеммера)...")
idx_id = BM25(analyzer=identity).index(corpus)
run_id = idx_id.run(queries, k=10)
m_id = evaluate(run_id, qrels)
print("  Hit@1={Hit@1:.3f}  Hit@5={Hit@5:.3f}  Hit@10={Hit@10:.3f}  nDCG@10={nDCG@10:.3f}".format(**m_id))

# %% [6] BM25 — 5-char prefix stemmer
# ============================================================
print("\n[2/5] BM25 prefix-5 (5-char prefix stem)...")
idx_p5 = BM25(analyzer=prefix5).index(corpus)
run_p5 = idx_p5.run(queries, k=10)
m_p5 = evaluate(run_p5, qrels)
print("  Hit@1={Hit@1:.3f}  Hit@5={Hit@5:.3f}  Hit@10={Hit@10:.3f}  nDCG@10={nDCG@10:.3f}".format(**m_p5))

# %% [7] Dense — LaBSE
# ============================================================
import numpy as np
from sentence_transformers import SentenceTransformer

def dense_run(model_name: str, query_prefix: str = "", doc_prefix: str = "",
              cache_tag: str = "") -> Dict[str, List[str]]:
    """Кодирует корпус и запросы, возвращает топ-10 по косинусу."""
    print(f"  Загрузка модели {model_name}...")
    model = SentenceTransformer(model_name)

    cache_path = f"/kaggle/working/emb_{cache_tag}.npy"
    ids_path   = f"/kaggle/working/emb_{cache_tag}_ids.json"

    if os.path.exists(cache_path) and os.path.exists(ids_path):
        print("  Загрузка эмбеддингов корпуса из кэша...")
        doc_ids = json.load(open(ids_path))
        doc_emb = np.load(cache_path)
    else:
        print(f"  Кодирование {len(corpus)} пассажей...")
        texts = [doc_prefix + t for _, t in corpus]
        doc_emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True)
        doc_ids = [did for did, _ in corpus]
        np.save(cache_path, doc_emb)
        json.dump(doc_ids, open(ids_path, "w"))

    doc_emb = np.asarray(doc_emb, dtype=np.float32)
    norms = np.linalg.norm(doc_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    doc_emb = doc_emb / norms

    print(f"  Кодирование {len(queries)} запросов...")
    q_texts = [query_prefix + queries[qid] for qid in sorted(queries)]
    q_emb = model.encode(q_texts, batch_size=64, show_progress_bar=False,
                         normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)

    run = {}
    k = 10
    for i, qid in enumerate(sorted(queries)):
        scores = doc_emb @ q_emb[i]
        top = np.argpartition(-scores, range(k))[:k]
        top = top[np.argsort(-scores[top])]
        run[qid] = [doc_ids[j] for j in top]
    return run


print("\n[3/5] Dense LaBSE...")
run_labse = dense_run("sentence-transformers/LaBSE", cache_tag="labse")
m_labse = evaluate(run_labse, qrels)
print("  Hit@1={Hit@1:.3f}  Hit@5={Hit@5:.3f}  Hit@10={Hit@10:.3f}  nDCG@10={nDCG@10:.3f}".format(**m_labse))

# %% [8] Dense — multilingual-e5-base
# ============================================================
print("\n[4/5] Dense multilingual-e5-base...")
run_e5 = dense_run("intfloat/multilingual-e5-base",
                   query_prefix="query: ", doc_prefix="passage: ",
                   cache_tag="e5")
m_e5 = evaluate(run_e5, qrels)
print("  Hit@1={Hit@1:.3f}  Hit@5={Hit@5:.3f}  Hit@10={Hit@10:.3f}  nDCG@10={nDCG@10:.3f}".format(**m_e5))

# %% [9] Dense — IBM Granite-278M
# ============================================================
print("\n[5/5] Dense IBM Granite-278M...")
run_granite = dense_run("ibm-granite/granite-embedding-278m-multilingual",
                        cache_tag="granite")
m_granite = evaluate(run_granite, qrels)
print("  Hit@1={Hit@1:.3f}  Hit@5={Hit@5:.3f}  Hit@10={Hit@10:.3f}  nDCG@10={nDCG@10:.3f}".format(**m_granite))

# %% [10] Итоговая таблица + сохранение
# ============================================================
systems = {
    "BM25 identity":   m_id,
    "BM25 prefix-5":   m_p5,
    "LaBSE":           m_labse,
    "E5-base":         m_e5,
    "Granite-278M":    m_granite,
}

print("\n" + "="*70)
print(f"SPRINT 3 — Synonym / Vocabulary-Gap Benchmark  (n={len(queries)} запросов)")
print(f"avg lexical overlap с gold passage: {sum(overlaps.values())/len(overlaps):.3f}")
print("="*70)
header = f"{'Система':<20}  {'Hit@1':>7}  {'Hit@5':>7}  {'Hit@10':>7}  {'MRR@10':>7}  {'nDCG@10':>8}"
print(header)
print("-" * len(header))
for name, m in systems.items():
    print(f"{name:<20}  {m['Hit@1']:>7.3f}  {m['Hit@5']:>7.3f}  {m['Hit@10']:>7.3f}  {m['MRR@10']:>7.3f}  {m['nDCG@10']:>8.3f}")
print("="*70)

# --- сохраняем JSON ---
out = {
    "sprint": 3,
    "benchmark": "synonym_vocabulary_gap",
    "n_queries": len(queries),
    "avg_overlap": round(sum(overlaps.values()) / len(overlaps), 4),
    "systems": {name: m for name, m in systems.items()},
    "per_query": {}
}

for qid in sorted(queries):
    gold = next(iter(qrels[qid]))
    out["per_query"][qid] = {
        "query": queries[qid],
        "gold_doc_id": gold,
        "overlap": overlaps[qid],
        "hits": {
            name: {
                "rank": next((i+1 for i, d in enumerate(run[qid]) if d == gold), None)
            }
            for name, run in [
                ("BM25_identity", run_id),
                ("BM25_prefix5",  run_p5),
                ("LaBSE",         run_labse),
                ("E5_base",       run_e5),
                ("Granite_278M",  run_granite),
            ]
        }
    }

OUT_PATH = "/kaggle/working/sprint3_synonym_results.json"
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\nРезультаты сохранены → {OUT_PATH}")
print("Скопируйте этот файл в data/results/ репозитория.")
