"""
Dense-поиск (по смыслу) на эмбеддингах. Единый интерфейс под разные
мультиязычные модели: IBM Granite, Google LaBSE, multilingual-e5.

Поиск = косинусная близость (нормируем векторы, считаем скалярное
произведение). Энкодер инъектируется — поэтому логика поиска тестируется
без скачивания моделей (мок-энкодер).

ВАЖНО про префиксы: модели семейства e5 требуют префиксы
"query: " / "passage: " для запроса и документа — без них качество падает.
Для LaBSE/Granite префиксы пустые. Их задаёт run_dense по пресету модели.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class DenseIndex:
    def __init__(
        self,
        model_name: Optional[str] = None,
        encoder=None,
        batch_size: int = 64,
        device: Optional[str] = None,
        query_prefix: str = "",
        doc_prefix: str = "",
    ) -> None:
        self.model_name = model_name
        self._encoder = encoder
        self.batch_size = batch_size
        self.device = device
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.doc_ids: List[str] = []
        self.matrix: Optional[np.ndarray] = None

    # ---------- энкодер ----------

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name, device=self.device)
        return self._encoder

    def _encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        enc = self._get_encoder()
        emb = enc.encode(list(texts), batch_size=self.batch_size,
                         show_progress_bar=show_progress)
        emb = np.asarray(emb, dtype=np.float32)
        # L2-нормировка строк -> косинус = скалярное произведение
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    # ---------- индексация/поиск ----------

    def index(self, documents: Sequence[Tuple[str, str]], show_progress: bool = True) -> "DenseIndex":
        self.doc_ids = [doc_id for doc_id, _ in documents]
        texts = [self.doc_prefix + text for _, text in documents]
        self.matrix = self._encode(texts, show_progress=show_progress)
        return self

    def set_embeddings(self, doc_ids: List[str], matrix: np.ndarray) -> "DenseIndex":
        """Загрузить заранее посчитанные эмбеддинги (из кэша)."""
        self.doc_ids = list(doc_ids)
        self.matrix = np.asarray(matrix, dtype=np.float32)
        return self

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.matrix is None:
            raise RuntimeError("Индекс пуст: вызовите index() или set_embeddings().")
        q = self._encode([self.query_prefix + query])[0]
        scores = self.matrix @ q
        k = min(top_k, len(self.doc_ids))
        top = np.argpartition(-scores, range(k))[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.doc_ids[i], float(scores[i])) for i in top]

    def run(self, queries: Dict[str, str], top_k: int = 10) -> Dict[str, List[str]]:
        return {qid: [doc_id for doc_id, _ in self.search(text, top_k)]
                for qid, text in queries.items()}
