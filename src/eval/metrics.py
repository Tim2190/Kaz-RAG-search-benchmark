"""
Метрики ранжирования для бенчмарка поиска — без внешних зависимостей.

Договорённости о формате:
- qrels: Dict[query_id, Set[doc_id]] — множество РЕЛЕВАНТНЫХ документов
  (бинарная релевантность).
- run:   Dict[query_id, List[doc_id]] — РАНЖИРОВАННЫЙ список выдачи
  (от самого релевантного к наименее), уже обрезанный или полный.

Все функции считают метрику для ОДНОГО запроса; агрегатор evaluate_run
усредняет по всем запросам и умеет считать статзначимость разницы
между двумя системами (paired bootstrap).
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Set, Sequence, Tuple


# ---------- метрики на один запрос ----------

def hit_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """1.0, если хотя бы один релевантный документ попал в топ-k, иначе 0.0."""
    return 1.0 if any(doc in relevant for doc in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """Доля релевантных документов, найденных в топ-k."""
    if not relevant:
        return 0.0
    found = sum(1 for doc in ranked[:k] if doc in relevant)
    return found / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """1/позиция первого релевантного документа в топ-k (MRR-слагаемое)."""
    for idx, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """nDCG@k для бинарной релевантности."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for idx, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            dcg += 1.0 / math.log2(idx + 1)
    # идеальный DCG: все релевантные на первых позициях
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------- агрегатор по набору запросов ----------

#: имя метрики -> функция (ranked, relevant, k) -> float
_METRIC_FUNCS = {
    "hit": hit_at_k,
    "recall": recall_at_k,
    "mrr": reciprocal_rank,
    "ndcg": ndcg_at_k,
}


def per_query_scores(
    run: Dict[str, List[str]],
    qrels: Dict[str, Set[str]],
    metric: str,
    k: int,
) -> Dict[str, float]:
    """Считает метрику для каждого запроса. Запросы без qrels пропускаются."""
    if metric not in _METRIC_FUNCS:
        raise ValueError(f"Неизвестная метрика: {metric}. Доступны: {list(_METRIC_FUNCS)}")
    func = _METRIC_FUNCS[metric]
    scores: Dict[str, float] = {}
    for qid, relevant in qrels.items():
        if not relevant:
            continue
        ranked = run.get(qid, [])
        scores[qid] = func(ranked, relevant, k)
    return scores


def evaluate_run(
    run: Dict[str, List[str]],
    qrels: Dict[str, Set[str]],
    metrics: Sequence[str] = ("recall", "mrr", "ndcg"),
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Возвращает усреднённые метрики по всем запросам в виде
    {"recall@1": ..., "recall@5": ..., "mrr@10": ..., ...}.
    """
    results: Dict[str, float] = {}
    for metric in metrics:
        for k in ks:
            scores = per_query_scores(run, qrels, metric, k)
            mean = sum(scores.values()) / len(scores) if scores else 0.0
            results[f"{metric}@{k}"] = mean
    return results


# ---------- статистическая значимость (До/После) ----------

def paired_bootstrap(
    run_a: Dict[str, List[str]],
    run_b: Dict[str, List[str]],
    qrels: Dict[str, Set[str]],
    metric: str = "ndcg",
    k: int = 10,
    n_resamples: int = 10000,
    seed: int = 13,
) -> Tuple[float, float, float]:
    """
    Парный bootstrap для разницы (B - A) по одной метрике.

    Возвращает (delta_mean, p_value, ci95_low..high упакованы как delta и p).
    p_value — двусторонняя доля бутстрэп-выборок, где знак дельты меняется
    относительно наблюдаемого (грубая оценка значимости прироста B над A).
    """
    scores_a = per_query_scores(run_a, qrels, metric, k)
    scores_b = per_query_scores(run_b, qrels, metric, k)
    qids = [q for q in scores_a if q in scores_b]
    if not qids:
        return 0.0, 1.0, 0.0

    diffs = [scores_b[q] - scores_a[q] for q in qids]
    observed = sum(diffs) / len(diffs)

    rng = random.Random(seed)
    n = len(diffs)
    count_opposite = 0
    boot_means: List[float] = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        m = sum(sample) / n
        boot_means.append(m)
        # если знак бутстрэп-среднего противоположен наблюдаемому
        if (m <= 0) if observed > 0 else (m >= 0):
            count_opposite += 1

    p_value = count_opposite / n_resamples
    return observed, p_value, observed
