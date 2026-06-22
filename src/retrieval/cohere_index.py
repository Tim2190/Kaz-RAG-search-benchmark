"""
Cohere Embed API — dense index с тем же интерфейсом, что DenseIndex.

Поддерживаемые модели:
  embed-v4.0              — флагманская мультиязычная (2025, 100+ языков, 128K ctx)
  embed-multilingual-v3.0 — предыдущая (1024-dim, 512 токенов)

Требования:
  pip install cohere>=5
  export COHERE_API_KEY=<ваш ключ>

Cohere использует input_type ("search_query" / "search_document") вместо
строковых префиксов — семантически то же, что query:/passage: в e5.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_COHERE_BATCH = 96   # max texts per embed call (Cohere limit)
_INTER_BATCH_SLEEP = 0.3  # секунды между батчами (rate-limit buffer)
_RETRY_SLEEP_BASE  = 60   # базовое ожидание при rate-limit (секунды)
_MAX_RETRIES       = 5    # максимум попыток на один батч


class CohereIndex:
    def __init__(
        self,
        model_name: str = "embed-v4.0",
        api_key: Optional[str] = None,
        batch_size: int = 96,
    ) -> None:
        import cohere  # lazy import — не ломает импорт при отсутствии пакета
        self.model_name = model_name
        self.batch_size = min(batch_size, _COHERE_BATCH)
        key = api_key or os.environ.get("COHERE_API_KEY", "")
        if not key:
            raise ValueError(
                "COHERE_API_KEY не задан. "
                "Передайте api_key= или установите переменную окружения."
            )
        self.co = cohere.ClientV2(api_key=key)
        self.doc_ids: List[str] = []
        self.matrix: Optional[np.ndarray] = None

    # ── encode ───────────────────────────────────────────────────────────────

    def _embed_with_retry(
        self, batch: List[str], input_type: str
    ) -> List[List[float]]:
        """Один API-вызов с экспоненциальным backoff при HTTP 429.

        Базовое ожидание 60 с, удваивается каждую попытку:
        60 → 120 → 240 → 480 → 960 с (итого ≤5 попыток).
        Все другие ошибки пробрасываются немедленно.
        """
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self.co.embed(
                    texts=batch,
                    model=self.model_name,
                    input_type=input_type,
                    embedding_types=["float"],
                )
                # ClientV2: resp.embeddings.float_ ; fallback to resp.embeddings
                raw = getattr(resp.embeddings, "float_", None) or resp.embeddings
                return list(raw)
            except Exception as exc:
                is_rate_limit = (
                    getattr(exc, "status_code", None) == 429
                    or "429" in str(exc)
                    or "rate" in str(exc).lower()
                    or "too many" in str(exc).lower()
                )
                if not is_rate_limit or attempt == _MAX_RETRIES:
                    raise
                wait = _RETRY_SLEEP_BASE * (2 ** attempt)
                print(
                    f"  [rate-limit] батч {len(batch)} текстов заблокирован; "
                    f"ожидание {wait} с (попытка {attempt + 1}/{_MAX_RETRIES})…"
                )
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def _encode(self, texts: Sequence[str], input_type: str) -> np.ndarray:
        all_emb: List[List[float]] = []
        batches = [
            list(texts[i : i + self.batch_size])
            for i in range(0, len(texts), self.batch_size)
        ]
        for idx, batch in enumerate(batches):
            if idx > 0:
                time.sleep(_INTER_BATCH_SLEEP)
            all_emb.extend(self._embed_with_retry(batch, input_type))
        mat = np.asarray(all_emb, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    # ── публичный интерфейс (зеркалит DenseIndex) ────────────────────────────

    def index(
        self,
        documents: Sequence[Tuple[str, str]],
        show_progress: bool = True,
    ) -> "CohereIndex":
        self.doc_ids = [d for d, _ in documents]
        texts = [t for _, t in documents]
        if show_progress:
            print(
                f"Cohere/{self.model_name}: кодирование {len(texts)} документов "
                f"(input_type=search_document, {len(texts)//self.batch_size+1} батчей)…"
            )
        self.matrix = self._encode(texts, "search_document")
        return self

    def set_embeddings(
        self, doc_ids: List[str], matrix: np.ndarray
    ) -> "CohereIndex":
        self.doc_ids = list(doc_ids)
        self.matrix = np.asarray(matrix, dtype=np.float32)
        return self

    def run(
        self, queries: Dict[str, str], top_k: int = 10
    ) -> Dict[str, List[str]]:
        if self.matrix is None:
            raise RuntimeError("Индекс пуст: вызовите index() или set_embeddings().")
        q_ids = list(queries.keys())
        q_texts = [queries[q] for q in q_ids]
        q_mat = self._encode(q_texts, "search_query")
        scores = q_mat @ self.matrix.T
        result: Dict[str, List[str]] = {}
        for i, qid in enumerate(q_ids):
            top_idx = np.argsort(-scores[i])[:top_k]
            result[qid] = [self.doc_ids[j] for j in top_idx]
        return result
