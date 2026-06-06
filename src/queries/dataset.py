"""
Загрузка датасета бенчмарка: корпус, запросы, qrels.

Форматы (см. data/README.md):
- corpus.jsonl: {doc_id, text, title, ...}
- queries.jsonl: {query_id, text, category, gold_doc_ids, validated}
- qrels берутся прямо из gold_doc_ids запросов (бинарная релевантность).
"""

from __future__ import annotations

import json
from typing import Dict, List, Set, Tuple


def load_corpus(path: str) -> List[Tuple[str, str]]:
    """Возвращает [(doc_id, text), ...] для индексации."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append((d["doc_id"], d["text"]))
    return out


def load_queries(path: str) -> List[Dict]:
    """Список запросов как словари."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def queries_as_map(queries: List[Dict]) -> Dict[str, str]:
    """{query_id: text} для прогона."""
    return {q["query_id"]: q["text"] for q in queries}


def qrels_from_queries(queries: List[Dict]) -> Dict[str, Set[str]]:
    """{query_id: {gold_doc_id, ...}} — релевантность из gold_doc_ids."""
    return {q["query_id"]: set(q.get("gold_doc_ids", [])) for q in queries}


def categories_from_queries(queries: List[Dict]) -> Dict[str, str]:
    """{query_id: category} для разбивки метрик по категориям."""
    return {q["query_id"]: q.get("category", "all") for q in queries}


def write_queries(path: str, queries: List[Dict]) -> None:
    """Сохраняет запросы в JSONL."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
