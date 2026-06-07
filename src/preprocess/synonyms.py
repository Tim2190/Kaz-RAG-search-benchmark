"""
Клиент к API расширения запросов синонимами (Query Expansion).

Endpoint: POST {base_url}/demo/expand/batch  {"words": [стеммленные_токены]}
Лимиты: <=50 слов на запрос, <=30 запросов/мин с одного IP.

Ответ: [{"word": "...", "stem": "...", "synonyms": ["syn1", ...], "found": true}, ...]
Синонимы уже стеммлены → кладутся в BM25 напрямую (то же пространство, что корпус).

Стратегия (зеркало stemmer_api.py):
  1. КЭШ на диск: {stem: [synonym, ...]} — found=false → пустой список.
  2. БАТЧИ по 50 уникальных некэшированных стемов.
  3. ТРОТТЛИНГ: не больше max_per_minute запросов в минуту.
  4. РЕТРАИ с экспоненциальной задержкой на 429/5xx.

Использование в пайплайне:
    expander = SynonymExpander()
    expander.warm(all_query_stems)        # заполнить кэш одним проходом
    expanded = expander.expand(stems)     # быстро, из кэша
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Optional

DEFAULT_BASE_URL = "https://kazakh-stemmer-590833642796.europe-west1.run.app"
MAX_WORDS_PER_REQUEST = 50
DEFAULT_MAX_PER_MINUTE = 28  # немного ниже лимита 30 для запаса


class SynonymExpander:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        cache_path: Optional[str] = "data/resources/synonym_cache.json",
        max_per_minute: int = DEFAULT_MAX_PER_MINUTE,
        timeout: int = 30,
        max_retries: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/demo/expand/batch"
        self.cache_path = cache_path
        self.timeout = timeout
        self.max_retries = max_retries
        self._min_interval = 60.0 / max_per_minute if max_per_minute > 0 else 0.0
        self._last_request_ts = 0.0
        self._cache: Dict[str, List[str]] = {}  # stem -> [synonym, ...]
        self._load_cache()

    def expand(self, stems: List[str]) -> List[str]:
        """
        Возвращает объединение оригинальных стемов и синонимов (без повторений).

        Порядок: сначала оригинальные стемы, затем синонимы в порядке первого
        встречанного источника. Дубликаты удаляются (чтобы не задваивать
        вклад в BM25).
        """
        seen: Dict[str, None] = {}
        for s in stems:
            seen.setdefault(s, None)
        for s in stems:
            for syn in self._cache.get(s, []):
                seen.setdefault(syn, None)
        return list(seen)

    def warm(self, stems: Iterable[str], save_every: int = 25,
             on_checkpoint=None) -> None:
        """
        Прогревает кэш для всех уникальных стемов батчами по 50.

        Устойчив к обрывам: уже закэшированные стемы пропускаются, можно
        прервать и продолжить. Печатает прогресс и ETA.
        """
        unique = [s for s in dict.fromkeys(stems)
                  if s and s not in self._cache]
        total = len(unique)
        if total == 0:
            self._save_cache()
            if on_checkpoint:
                on_checkpoint(0, 0)
            return
        start = time.time()
        done = 0
        for bi, i in enumerate(range(0, total, MAX_WORDS_PER_REQUEST)):
            batch = unique[i: i + MAX_WORDS_PER_REQUEST]
            results = self._expand_batch_remote(batch)
            for stem, syns in zip(batch, results):
                self._cache[stem] = syns
            done += len(batch)
            if bi % save_every == 0:
                self._save_cache()
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta_min = (total - done) / rate / 60 if rate > 0 else 0
                print(f"  синонимы: {done}/{total} стемов, ETA ~{eta_min:.0f} мин",
                      flush=True)
                if on_checkpoint:
                    on_checkpoint(done, total)
        self._save_cache()
        if on_checkpoint:
            on_checkpoint(total, total)
        print(f"  синонимы: {total} новых стемов закэшировано", flush=True)

    # ---------- сеть ----------

    def _expand_batch_remote(self, words: List[str]) -> List[List[str]]:
        """Один запрос к API (<=50 слов) с троттлингом и ретраями."""
        self._throttle()
        payload = json.dumps({"words": words}).encode("utf-8")
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                raw = self._post(payload)
                return self._parse_response(raw, words)
            except urllib.error.HTTPError as e:  # pragma: no cover - сеть
                last_err = e
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise
            except urllib.error.URLError as e:  # pragma: no cover - сеть
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(
            f"Synonym API недоступен после {self.max_retries} попыток: {last_err}"
        )

    def _post(self, payload: bytes) -> str:
        """Низкоуровневый POST. Вынесен отдельно — мокается в тестах."""
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        self._last_request_ts = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8")

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    @staticmethod
    def _parse_response(raw: str, words: List[str]) -> List[List[str]]:
        """
        Парсит ответ /demo/expand/batch.

        Ожидаемый формат:
          [{"word": "...", "stem": "...", "synonyms": ["syn1", ...], "found": bool}, ...]

        Возвращает List[List[str]] — синонимы для каждого входного слова.
        found=false или отсутствующий ключ synonyms → пустой список.
        """
        data = json.loads(raw)
        if not isinstance(data, list) or len(data) != len(words):
            raise ValueError(
                f"Неожиданный формат ответа synonym API: ожидался список из "
                f"{len(words)} элементов. Ответ: {raw[:500]}"
            )
        result = []
        for item in data:
            if not isinstance(item, dict):
                result.append([])
                continue
            syns = item.get("synonyms") or []
            result.append([s for s in syns if isinstance(s, str) and s])
        return result

    # ---------- кэш ----------

    def _load_cache(self) -> None:
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False)
        os.replace(tmp, self.cache_path)
