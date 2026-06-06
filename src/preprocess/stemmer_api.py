"""
Клиент к развёрнутому казахскому стеммеру (Cloud Run).

Endpoint: POST {base_url}/demo/stem/batch  с телом {"words": [...]}.
Лимиты сервиса: <=50 слов на запрос, <=30 запросов/мин с одного IP.

Стратегия экономии запросов (критично для корпуса в тысячи пассажей):
  1. КЭШ на диск: стемминг детерминированный, поэтому каждое уникальное
     слово запрашивается ровно один раз за всё время (и переживает рестарты).
  2. БАТЧИ по 50 уникальных некэшированных слов.
  3. ТРОТТЛИНГ: не больше max_per_minute запросов в минуту.
  4. РЕТРАИ с экспоненциальной задержкой на 429/5xx.

Использование в пайплайне:
    st = KazakhStemmerAPI()
    st.warm(all_unique_tokens)   # один проход — заполняет кэш батчами
    st.stem(token)               # дальше мгновенно из кэша

Зависимостей нет (urllib) — клиент тестируется через мок _post().
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
DEFAULT_MAX_PER_MINUTE = 30


class KazakhStemmerAPI:
    name = "kazakh"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        cache_path: Optional[str] = "data/resources/stem_cache.json",
        max_per_minute: int = DEFAULT_MAX_PER_MINUTE,
        timeout: int = 30,
        max_retries: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/demo/stem/batch"
        self.cache_path = cache_path
        self.timeout = timeout
        self.max_retries = max_retries
        # минимальный интервал между запросами, чтобы не превысить лимит
        self._min_interval = 60.0 / max_per_minute if max_per_minute > 0 else 0.0
        self._last_request_ts = 0.0
        self._cache: Dict[str, str] = {}
        self._load_cache()

    # ---------- публичный интерфейс ----------

    @staticmethod
    def _is_trivial(token: str) -> bool:
        """Числа и однобуквенные токены стеммить не нужно — экономим запросы."""
        return token.isdigit() or len(token) <= 1

    def stem(self, token: str) -> str:
        """Стем одного токена (из кэша; при промахе — одиночный запрос)."""
        if token in self._cache:
            return self._cache[token]
        if self._is_trivial(token):
            self._cache[token] = token
            return token
        self.warm([token])
        return self._cache.get(token, token)

    def warm(self, tokens: Iterable[str], save_every: int = 25,
             on_checkpoint=None) -> None:
        """
        Прогревает кэш для всех уникальных токенов батчами по 50.

        Устойчив к обрывам: сохраняет кэш на диск каждые save_every батчей,
        а уже закэшированные токены при повторном запуске пропускаются —
        т.е. прогрев можно прервать и продолжить с того же места.
        Печатает прогресс и ETA (прогрев большого корпуса идёт долго из-за
        лимита 30 запросов/мин).

        on_checkpoint(done, total): необязательный колбэк, вызывается после
        каждого сохранения кэша — например, для авто-коммита в GitHub.
        """
        unique = []
        for t in dict.fromkeys(tokens):
            if not t or t in self._cache:
                continue
            if self._is_trivial(t):
                self._cache[t] = t  # тривиальные — сразу, без сети
            else:
                unique.append(t)

        total = len(unique)
        if total == 0:
            self._save_cache()
            if on_checkpoint:
                on_checkpoint(0, 0)
            return
        start = time.time()
        done = 0
        for bi, i in enumerate(range(0, total, MAX_WORDS_PER_REQUEST)):
            batch = unique[i : i + MAX_WORDS_PER_REQUEST]
            stems = self._stem_batch_remote(batch)
            for word, stem in zip(batch, stems):
                self._cache[word] = stem
            done += len(batch)
            if bi % save_every == 0:
                self._save_cache()
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta_min = (total - done) / rate / 60 if rate > 0 else 0
                print(f"  стемминг: {done}/{total} слов, ETA ~{eta_min:.0f} мин", flush=True)
                if on_checkpoint:
                    on_checkpoint(done, total)
        self._save_cache()
        if on_checkpoint:
            on_checkpoint(total, total)
        print(f"  стемминг завершён: {total} новых слов закэшировано", flush=True)

    # ---------- сеть ----------

    def _stem_batch_remote(self, words: List[str]) -> List[str]:
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
        raise RuntimeError(f"Стеммер недоступен после {self.max_retries} попыток: {last_err}")

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
    def _parse_response(raw: str, words: List[str]) -> List[str]:
        """
        Парсер ответа стеммера.

        ПОДТВЕРЖДЁННЫЙ формат сервиса — список объектов:
          [{"word": "балаларымызда", "stem": "бала",
            "suffixes": ["да","ымыз","лар"],
            "found_in_dictionary": true, "confidence": "dictionary"}, ...]

        Дополнительно понимаются запасные формы (на случай смены API):
          ["стем", ...] / {"stems": [...]} / {"results": [...]}.
        Если формат иной — бросаем ошибку с сырым телом, чтобы зафиксировать.
        """
        data = json.loads(raw)

        def extract_list(obj):
            if isinstance(obj, list):
                # список строк
                if all(isinstance(x, str) for x in obj):
                    return obj
                # список объектов с ключом stem/lemma/result
                out = []
                for x in obj:
                    if isinstance(x, dict):
                        out.append(
                            x.get("stem") or x.get("lemma") or x.get("result") or x.get("word")
                        )
                    else:
                        out.append(x)
                return out
            return None

        if isinstance(data, list):
            result = extract_list(data)
        elif isinstance(data, dict):
            result = None
            for key in ("stems", "results", "stemmed", "words", "data"):
                if key in data:
                    result = extract_list(data[key])
                    if result is not None:
                        break
        else:
            result = None

        if result is None or len(result) != len(words):
            raise ValueError(
                "Не удалось разобрать ответ стеммера. Зафиксируй формат в "
                f"_parse_response. Сырой ответ: {raw[:500]}"
            )
        # подстраховка: пустые стемы заменяем исходным словом
        return [s if s else w for s, w in zip(result, words)]

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
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False)
        os.replace(tmp, self.cache_path)
