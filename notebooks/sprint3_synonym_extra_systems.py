# Sprint 3 (доп.) — недостающие системы на synonym / vocabulary-gap бенчмарке
# ==========================================================================
# Первый ноутбук (sprint3_synonym_benchmark.py) прогнал 5 «чистых» ретриверов
# (BM25 identity / prefix-5 / LaBSE / E5 / Granite). Здесь добавляем ТРИ системы
# из основного исследования, которых там не было, используя НАСТОЯЩИЕ компоненты
# из src/ (а не self-contained заглушки):
#
#   6. BM25 + казахский стеммер   (Cloud Run стеммер, stem_cache.json)
#   7. BM25 + synonym expansion   (тот же стеммер + SynonymExpander)
#   8. Hybrid (RRF, k=60)         (BM25+стеммер ⊕ Granite-278M)
#
# ТРЕБОВАНИЯ:
#   * Интернет ВКЛючён (Kaggle: Settings → Internet → On) — нужен для стеммер/
#     synonym API. Кэши уже в репо, дозапрашивается всего ~63 слова и ~404 стема.
#   * GPU — только для канала Granite в гибриде (LaBSE/E5 здесь не нужны).
#
# Каждый "# %%" — отдельная ячейка.

# %% [1] Установка зависимостей
import subprocess, sys, os

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

pip("sentence-transformers>=2.6")

# %% [2] Клонирование репо + sys.path + chdir
import json, math
from typing import Dict, List, Set, Tuple

REPO = "https://github.com/Tim2190/Kaz-RAG-search-benchmark.git"
BRANCH = "claude/laughing-ramanujan-LsMa8"
REPO_DIR = "/kaggle/working/repo"

if not os.path.exists(REPO_DIR):
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "--branch", BRANCH, REPO, REPO_DIR])

# repo на sys.path -> работают импорты `import src...`
sys.path.insert(0, REPO_DIR)
# chdir в repo -> относительные пути кэшей ("data/resources/...") резолвятся
os.chdir(REPO_DIR)

CORPUS_PATH  = "data/corpus/corpus.jsonl"
QUERIES_PATH = "data/queries/synonym_queries.jsonl"

# --- настоящие компоненты бенчмарка ---
from src.preprocess.tokenize import tokenize
from src.preprocess.stemmer import get_stemmer, stem_tokens
from src.preprocess.synonyms import SynonymExpander
from src.retrieval.bm25 import BM25Index, default_analyzer
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.eval import metrics

# %% [3] Загрузка корпуса и synonym-запросов
corpus_pairs: List[Tuple[str, str]] = []
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        corpus_pairs.append((d["doc_id"], d["text"]))
print(f"Корпус: {len(corpus_pairs)} пассажей")

queries: Dict[str, str] = {}
qrels: Dict[str, Set[str]] = {}
overlaps: Dict[str, float] = {}
with open(QUERIES_PATH, encoding="utf-8") as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        qid = f"syn_{i:03d}"
        queries[qid] = item["query"]
        qrels[qid] = {item["gold_doc_id"]}
        overlaps[qid] = item.get("overlap", 0.0)
print(f"Запросов: {len(queries)}, avg overlap={sum(overlaps.values())/len(overlaps):.3f}")

def show(name, run):
    m = metrics.evaluate_run(run, qrels, metrics=("hit", "mrr", "ndcg"), ks=(1, 5, 10))
    print(f"  {name:<22} Hit@1={m['hit@1']:.3f}  Hit@5={m['hit@5']:.3f}  "
          f"Hit@10={m['hit@10']:.3f}  MRR@10={m['mrr@10']:.3f}  nDCG@10={m['ndcg@10']:.3f}")
    return m

# %% [4] Прогрев казахского стеммера (Cloud Run + stem_cache.json)
# Весь корпус уже в кэше из основного исследования; новыми будут ~63 слова из
# synonym-запросов. warm() пропускает закэшированные и дозапрашивает только их.
print("Инициализация казахского стеммера...")
stemmer = get_stemmer("kazakh")   # cache_path по умолчанию data/resources/stem_cache.json

warm = getattr(stemmer, "warm", None)
if callable(warm):
    toks = set()
    for _, text in corpus_pairs:
        toks.update(tokenize(text))
    for text in queries.values():
        toks.update(tokenize(text))
    print(f"  прогрев: {len(toks)} уникальных токенов (кэшированные пропускаются)...")
    warm(toks)
print("  стеммер готов.")

# %% [5] Система 6 — BM25 + казахский стеммер
print("\n[6] BM25 + казахский стеммер...")
print(f"  индексация {len(corpus_pairs)} пассажей со стеммером...")
idx_stem = BM25Index(analyzer=default_analyzer(stemmer)).index(corpus_pairs)
run_stem_10  = idx_stem.run(queries, top_k=10)
run_stem_100 = idx_stem.run(queries, top_k=100)   # для гибрида
m_stem = show("BM25+стеммер", run_stem_10)

# %% [6] Система 7 — BM25 + synonym expansion
# Логика зеркалит src/eval/run_synonyms.py: стеммим запрос, расширяем синонимами
# из словаря (уже стеммлены -> в то же пространство, что корпус), ищем по терминам.
print("\n[7] BM25 + synonym expansion...")
query_stems: Dict[str, List[str]] = {
    qid: stem_tokens(tokenize(text), stemmer) for qid, text in queries.items()
}
all_stems: Set[str] = set()
for s in query_stems.values():
    all_stems.update(s)

expander = SynonymExpander(cache_path="data/resources/synonym_cache.json")
print(f"  прогрев synonym-кэша: {len(all_stems)} стемов запросов (новые дозапросятся)...")
expander.warm(all_stems)

expanded: Dict[str, List[str]] = {}
tot_o = tot_e = 0
for qid, stems in query_stems.items():
    exp = expander.expand(stems)
    expanded[qid] = exp
    tot_o += len(stems); tot_e += len(exp)
print(f"  расширение: {tot_o/len(query_stems):.1f} → {tot_e/len(query_stems):.1f} токенов/запрос")

run_syn = idx_stem.run_terms(expanded, top_k=10)
m_syn = show("BM25+synonym", run_syn)

# %% [7] Dense-каналы Granite: R1 (278M) + R2 (97M, 311M — казахский kk явно обучен)
import numpy as np
from sentence_transformers import SentenceTransformer

def dense_run(model_name, top_k=100, query_prefix="", doc_prefix="", cache_tag=""):
    print(f"  модель: {model_name}")
    model = SentenceTransformer(model_name)
    npy = f"/kaggle/working/emb_{cache_tag}.npy"
    ids = f"/kaggle/working/emb_{cache_tag}_ids.json"
    if os.path.exists(npy) and os.path.exists(ids):
        doc_ids = json.load(open(ids)); doc_emb = np.load(npy)
    else:
        texts = [doc_prefix + t for _, t in corpus_pairs]
        doc_emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True)
        doc_ids = [d for d, _ in corpus_pairs]
        np.save(npy, doc_emb); json.dump(doc_ids, open(ids, "w"))
    doc_emb = np.asarray(doc_emb, dtype=np.float32)
    n = np.linalg.norm(doc_emb, axis=1, keepdims=True); n[n == 0] = 1; doc_emb /= n
    qids = sorted(queries)
    q_emb = model.encode([query_prefix + queries[q] for q in qids],
                         batch_size=64, normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    out = {}
    for i, qid in enumerate(qids):
        sc = doc_emb @ q_emb[i]
        top = np.argpartition(-sc, top_k)[:top_k]
        top = top[np.argsort(-sc[top])]
        out[qid] = [doc_ids[j] for j in top]
    return out

# --- R1 (старое поколение, БЕЗ явного обучения на казахском) ---
print("\n[канал] Granite-278M (R1)...")
run_granite_100 = dense_run("ibm-granite/granite-embedding-278m-multilingual",
                            top_k=100, cache_tag="granite")
m_granite = show("Granite-278M (R1)", {q: r[:10] for q, r in run_granite_100.items()})

# --- R2 (новое поколение, казахский kk В ЧИСЛЕ 52 ЯЗЫКОВ с явной поддержкой) ---
print("\n[канал] Granite-97M-R2 (multilingual)...")
run_r2_97_100 = dense_run("ibm-granite/granite-embedding-97m-multilingual-r2",
                          top_k=100, cache_tag="granite_r2_97")
m_r2_97 = show("Granite-97M-R2", {q: r[:10] for q, r in run_r2_97_100.items()})

print("\n[канал] Granite-311M-R2 (multilingual)...")
run_r2_311_100 = dense_run("ibm-granite/granite-embedding-311m-multilingual-r2",
                           top_k=100, cache_tag="granite_r2_311")
m_r2_311 = show("Granite-311M-R2", {q: r[:10] for q, r in run_r2_311_100.items()})

# %% [8] Гибриды (RRF, k=60) = BM25+стеммер ⊕ Granite (R1 и лучший R2)
print("\n[8] Hybrid (RRF, k=60)...")
run_hybrid = reciprocal_rank_fusion(
    {"bm25_stemmer": run_stem_100, "granite": run_granite_100}, k=60, top_k=10)
m_hybrid = show("Hybrid ⊕ Granite-R1", run_hybrid)

run_hybrid_r2 = reciprocal_rank_fusion(
    {"bm25_stemmer": run_stem_100, "granite_r2": run_r2_311_100}, k=60, top_k=10)
m_hybrid_r2 = show("Hybrid ⊕ Granite-311M-R2", run_hybrid_r2)

# %% [9] Итоговая таблица + сохранение
systems = {
    "BM25 + Kazakh stemmer":       m_stem,
    "BM25 + synonym exp.":         m_syn,
    "Granite-278M (R1)":           m_granite,
    "Granite-97M-R2":              m_r2_97,
    "Granite-311M-R2":             m_r2_311,
    "Hybrid ⊕ Granite-R1":         m_hybrid,
    "Hybrid ⊕ Granite-311M-R2":    m_hybrid_r2,
}
print("\n" + "=" * 78)
print(f"SPRINT 3 (доп.) — недостающие системы на synonym-бенчмарке (n={len(queries)})")
print(f"avg lexical overlap с gold: {sum(overlaps.values())/len(overlaps):.3f}")
print("=" * 78)
hdr = f"{'Система':<24} {'Hit@1':>7} {'Hit@5':>7} {'Hit@10':>7} {'MRR@10':>7} {'nDCG@10':>8}"
print(hdr); print("-" * len(hdr))
for name, m in systems.items():
    print(f"{name:<24} {m['hit@1']:>7.3f} {m['hit@5']:>7.3f} {m['hit@10']:>7.3f} "
          f"{m['mrr@10']:>7.3f} {m['ndcg@10']:>8.3f}")
print("=" * 78)

out = {
    "sprint": 3,
    "benchmark": "synonym_vocabulary_gap",
    "note": "extra systems: real Kazakh stemmer, synonym expansion, Granite R1+R2, hybrids",
    "n_queries": len(queries),
    "avg_overlap": round(sum(overlaps.values()) / len(overlaps), 4),
    "systems": {
        "BM25_Kazakh_stemmer": m_stem,
        "BM25_synonym_expansion": m_syn,
        "Granite_278M_R1": m_granite,
        "Granite_97M_R2": m_r2_97,
        "Granite_311M_R2": m_r2_311,
        "Hybrid_RRF_k60_R1": m_hybrid,
        "Hybrid_RRF_k60_R2_311M": m_hybrid_r2,
    },
}
OUT = "/kaggle/working/sprint3_synonym_extra_results.json"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\nСохранено → {OUT}")
print("Скопируйте этот JSON, я добавлю результаты в data/results/ и RESULTS.md.")
