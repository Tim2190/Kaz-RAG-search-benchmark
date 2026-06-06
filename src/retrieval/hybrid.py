"""
Reciprocal Rank Fusion (RRF) — слияние ранжировок нескольких ретриверов.

Идея гибрида: лексический канал (BM25 + казахский стеммер) и плотный канал
(Granite-278M) ошибаются в разных местах. BM25+стеммер чинит морфологию, но
не видит синонимов; Granite силён на семантике, но проваливается на
vocabulary-gap. RRF объединяет их РАНГИ (а не сырые несопоставимые скоры),
поэтому документ, уверенно найденный любым каналом, всплывает наверх.

Формула (Cormack et al., 2009):
    score(d) = Σ_channels 1 / (k + rank_i(d))
где rank нумеруется с 1; документ, которого канал не вернул, даёт по этому
каналу 0. Дефолт k=60 — каноническое значение из оригинальной статьи,
зафиксировано заранее (pre-registered), а не подобрано под результат.

Слияние детерминированно: при равенстве скоров порядок задаётся doc_id,
поэтому одинаковый вход всегда даёт одинаковый выход.

Зависимостей нет — чистый Python, тестируется без сети и GPU.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

#: {channel_name: {query_id: [doc_id, ...ранжировано...]}}
Channels = Dict[str, Dict[str, List[str]]]
#: {query_id: [doc_id, ...]}
Run = Dict[str, List[str]]

DEFAULT_K = 60


def fuse_scores(channels: Channels, k: int = DEFAULT_K) -> Dict[str, Dict[str, float]]:
    """
    Считает RRF-скоры по каждому запросу.

    Возвращает {query_id: {doc_id: score}}. Документ, отсутствующий в канале,
    не добавляет вклад (эквивалентно нулю по этому каналу).
    """
    if k <= 0:
        raise ValueError(f"RRF k должен быть > 0, получено {k}")

    # объединение всех query_id по всем каналам
    qids = set()
    for ranking in channels.values():
        qids.update(ranking.keys())

    scores: Dict[str, Dict[str, float]] = {}
    for qid in qids:
        acc: Dict[str, float] = {}
        for ranking in channels.values():
            ranked = ranking.get(qid, [])
            for rank, doc_id in enumerate(ranked, start=1):
                acc[doc_id] = acc.get(doc_id, 0.0) + 1.0 / (k + rank)
        scores[qid] = acc
    return scores


def reciprocal_rank_fusion(
    channels: Channels,
    k: int = DEFAULT_K,
    top_k: Optional[int] = None,
) -> Run:
    """
    Сливает каналы в единую ранжировку на запрос.

    channels: {имя_канала: {query_id: [doc_id, ...]}}.
    k:        константа RRF (дефолт 60, pre-registered).
    top_k:    если задан — обрезать выдачу до top_k документов.

    Возвращает {query_id: [doc_id, ...]} по убыванию RRF-скора. Тай-брейк по
    doc_id (по возрастанию) — слияние полностью детерминированно.
    """
    scores = fuse_scores(channels, k=k)
    fused: Run = {}
    for qid, doc_scores in scores.items():
        # сортировка: сначала по убыванию скора, при равенстве — по doc_id
        ranked = sorted(doc_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        docs = [doc_id for doc_id, _ in ranked]
        fused[qid] = docs[:top_k] if top_k is not None else docs
    return fused


def load_run(path: str) -> Run:
    """Загружает дамп ранжировок {query_id: [doc_id, ...]} из JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Ожидался объект {{query_id: [...]}} в {path}")
    return {str(qid): list(docs) for qid, docs in data.items()}


def save_run(path: str, run: Run) -> None:
    """Сохраняет ранжировки {query_id: [doc_id, ...]} в JSON."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False)
