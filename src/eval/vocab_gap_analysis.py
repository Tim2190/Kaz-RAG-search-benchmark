"""
Анализ vocabulary-gap: покрытые vs непокрытые запросы.

Разбивает категорию vocabulary-gap на подгруппы по тому, нашёл ли
синонимайзер что-то полезное для данного запроса:

  uncovered    — для всех стемов запроса словарь вернул пустой список
  covered      — хотя бы один стем получил синонимы; далее делится:
    └ bridge   — хотя бы один из синонимов встречается в gold-пассаже
                 (значит, мост через семантический разрыв реально построен)
    └ noise    — синонимы найдены, но ни один не попал в gold-пассаж
                 (лишние токены, мост не построен)

Это позволяет разделить две гипотезы:
  H1 "Словарь не покрывает нужные слова" → падение только в uncovered
  H2 "Механизм размывает сигнал"         → падение даже в covered/bridge

ЗАПУСК (нужны кэш синонимов + corpus.jsonl + оба rankings файла):
    python -m src.eval.vocab_gap_analysis \\
        --synonym-cache data/resources/synonym_cache.json \\
        --kazakh-runs   results/runs_bm25_kazakh.json \\
        --synonym-runs  results/runs_bm25_synonym.json
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Set, Tuple

from ..preprocess.tokenize import tokenize
from ..preprocess.stemmer import get_stemmer, stem_tokens
from ..queries import dataset
from ..eval import metrics

KS = (1, 3, 5, 10)
METRIC_KS = (1, 5, 10)


def _stem_text(text: str, stemmer) -> Set[str]:
    return set(stem_tokens(tokenize(text), stemmer))


def _load_run(path: str) -> Dict[str, List[str]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _classify_query(
    query_stems: Set[str],
    gold_text: str,
    gold_stems: Set[str],
    synonym_cache: Dict[str, List[str]],
) -> str:
    """
    Возвращает: 'uncovered', 'covered_noise', 'covered_bridge'.

    bridge = расширение добавило хотя бы один стем, который есть в gold-пассаже.
    noise  = расширение добавило стемы, но ни один не попал в gold-пассаж.
    """
    added: Set[str] = set()
    for stem in query_stems:
        syns = synonym_cache.get(stem, [])
        for s in syns:
            if s not in query_stems:
                added.add(s)

    if not added:
        return "uncovered"

    if added & gold_stems:
        return "covered_bridge"

    return "covered_noise"


def _hit_at_k(run: Dict[str, List[str]], qrels: Dict[str, Set[str]],
              qids: List[str], k: int) -> float:
    valid = [q for q in qids if q in run and qrels.get(q)]
    if not valid:
        return 0.0
    return sum(1 for q in valid if qrels[q] & set(run[q][:k])) / len(valid)


def _ndcg10(run, qrels, qids):
    sub_qrels = {q: qrels[q] for q in qids if q in qrels and qrels[q]}
    if not sub_qrels:
        return 0.0
    sub_run = {q: run.get(q, []) for q in sub_qrels}
    m = metrics.evaluate_run(sub_run, sub_qrels,
                             metrics=("ndcg",), ks=(10,))
    return m.get("ndcg@10", 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="vocabulary-gap: covered vs uncovered vs bridge/noise"
    )
    ap.add_argument("--queries",        default="data/queries/queries.jsonl")
    ap.add_argument("--corpus",         default="data/corpus/corpus.jsonl")
    ap.add_argument("--synonym-cache",  default="data/resources/synonym_cache.json")
    ap.add_argument("--kazakh-runs",    default="results/runs_bm25_kazakh.json")
    ap.add_argument("--synonym-runs",   default="results/runs_bm25_synonym.json")
    args = ap.parse_args()

    # --- загрузка ---
    queries   = dataset.load_queries(args.queries)
    corpus    = dict(dataset.load_corpus(args.corpus))
    qrels     = dataset.qrels_from_queries(queries)
    cats      = dataset.categories_from_queries(queries)

    with open(args.synonym_cache, encoding="utf-8") as f:
        synonym_cache: Dict[str, List[str]] = json.load(f)

    run_kaz = _load_run(args.kazakh_runs)
    run_syn = _load_run(args.synonym_runs)

    stemmer = get_stemmer("kazakh")
    print("Стемминг запросов и gold-пассажей…", flush=True)

    # --- классификация vocabulary-gap запросов ---
    vg_queries = [q for q in queries if cats.get(q["query_id"]) == "vocabulary-gap"]

    groups: Dict[str, List[str]] = {
        "uncovered":      [],
        "covered_noise":  [],
        "covered_bridge": [],
    }

    for q in vg_queries:
        qid = q["query_id"]
        q_stems = _stem_text(q["text"], stemmer)

        gold_ids = list(qrels.get(qid, []))
        gold_text = corpus.get(gold_ids[0], "") if gold_ids else ""
        gold_stems = _stem_text(gold_text, stemmer) if gold_text else set()

        label = _classify_query(q_stems, gold_text, gold_stems, synonym_cache)
        groups[label].append(qid)

    total_vg = len(vg_queries)
    print(f"\nvocabulary-gap: {total_vg} запросов")
    for g, ids in groups.items():
        pct = 100 * len(ids) / total_vg if total_vg else 0
        print(f"  {g:<18}: {len(ids):3d} запросов ({pct:.0f}%)")

    # --- метрики по подгруппам ---
    print("\n" + "="*72)
    print(f"{'подгруппа':<18} {'n':>3} | "
          f"{'hit@3':>6} {'hit@5':>6} {'ndcg@10':>8} | "
          f"{'hit@3':>6} {'hit@5':>6} {'ndcg@10':>8} | "
          f"{'Δ ndcg':>8}")
    print(f"{'':18} {'':3}   "
          f"{'——— kazakh ———':>22}   "
          f"{'— synonym ———':>22}   "
          f"{'':8}")
    print("-"*72)

    for g, qids in [
        ("ALL vocab-gap",    [q["query_id"] for q in vg_queries]),
        ("uncovered",        groups["uncovered"]),
        ("covered_noise",    groups["covered_noise"]),
        ("covered_bridge",   groups["covered_bridge"]),
    ]:
        n = len(qids)
        if n == 0:
            print(f"  {g:<18} {n:3d}   (пусто)")
            continue

        h3k = _hit_at_k(run_kaz, qrels, qids, 3)
        h5k = _hit_at_k(run_kaz, qrels, qids, 5)
        nd_k = _ndcg10(run_kaz, qrels, qids)

        h3s = _hit_at_k(run_syn, qrels, qids, 3)
        h5s = _hit_at_k(run_syn, qrels, qids, 5)
        nd_s = _ndcg10(run_syn, qrels, qids)

        delta = nd_s - nd_k
        sign  = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "≈")
        print(f"  {g:<18} {n:3d}   "
              f"{h3k:6.3f} {h5k:6.3f} {nd_k:8.3f}   "
              f"{h3s:6.3f} {h5s:6.3f} {nd_s:8.3f}   "
              f"{sign}{abs(delta):7.3f}")

    print("\nЛегенда:")
    print("  uncovered     — словарь не нашёл синонимов ни для одного стема запроса")
    print("  covered_noise — синонимы найдены, но ни один не попал в gold-пассаж")
    print("  covered_bridge— синонимы найдены И хотя бы один есть в gold-пассаже")
    print()
    print("Ключевой вопрос:")
    print("  Если covered_bridge ▼ → механизм размывает сигнал даже при правильном мосте")
    print("  Если covered_bridge ▲ + uncovered ▼ → проблема в coverage словаря")


if __name__ == "__main__":
    main()
