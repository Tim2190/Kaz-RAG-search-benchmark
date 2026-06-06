"""
BM25 (Okapi) — лексический baseline. Чистый Python, без зависимостей,
чтобы запускаться и тестироваться где угодно.

Ключевая идея бенчмарка: индекс и запрос проходят ОДИНАКОВУЮ нормализацию
(токенизация + опциональный стеммер). Подменяя стеммер с identity на
kazakh, мы меряем чистый вклад морфологической нормализации «до/после».
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Sequence, Tuple

from ..preprocess.tokenize import tokenize
from ..preprocess.stemmer import IdentityStemmer, Stemmer, stem_tokens


def default_analyzer(stemmer: Stemmer) -> Callable[[str], List[str]]:
    """Возвращает анализатор text -> токены с применённым стеммером."""
    def analyze(text: str) -> List[str]:
        return stem_tokens(tokenize(text), stemmer)
    return analyze


class BM25Index:
    """BM25 Okapi с настраиваемым анализатором (общим для корпуса и запроса)."""

    def __init__(
        self,
        analyzer: Callable[[str], List[str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.analyzer = analyzer or default_analyzer(IdentityStemmer())
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: List[Counter] = []
        self.df: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0

    def index(self, documents: Sequence[Tuple[str, str]]) -> "BM25Index":
        """documents: последовательность (doc_id, text)."""
        for doc_id, text in documents:
            toks = self.analyzer(text)
            self.doc_ids.append(doc_id)
            self.doc_tokens.append(toks)
            self.doc_freqs.append(Counter(toks))
            self.doc_len.append(len(toks))
            for term in set(toks):
                self.df[term] += 1
        n = len(self.doc_ids)
        self.avgdl = sum(self.doc_len) / n if n else 0.0
        # idf по Okapi (с +1 внутри log, чтобы не уходить в отрицательные/0)
        for term, df in self.df.items():
            self.idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))
        return self

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Возвращает топ-k [(doc_id, score)] по убыванию релевантности."""
        q_terms = self.analyzer(query)
        scores: List[Tuple[str, float]] = []
        for i, doc_id in enumerate(self.doc_ids):
            freqs = self.doc_freqs[i]
            dl = self.doc_len[i]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
            s = 0.0
            for term in q_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                s += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / (tf + denom_norm)
            if s > 0:
                scores.append((doc_id, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def run(self, queries: Dict[str, str], top_k: int = 10) -> Dict[str, List[str]]:
        """Прогон набора запросов -> {query_id: [doc_id, ...]} для метрик."""
        return {
            qid: [doc_id for doc_id, _ in self.search(text, top_k)]
            for qid, text in queries.items()
        }
