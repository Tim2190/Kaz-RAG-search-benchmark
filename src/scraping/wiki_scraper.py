"""
Сборщик корпуса из казахской Википедии (kk.wikipedia.org) через MediaWiki API.

Почему Википедия: официальный API отдаёт ЧИСТЫЙ текст статей (никаких
сайдбаров, меню и JS-проблем, на которых сломались новостные сайты).
Энциклопедические статьи удобны для Фазы 2 — по ним легко составлять
фактические вопросы с однозначным «правильным» документом для qrels.

Берём случайные статьи (list=random) и тянем их полный текст
(prop=extracts&explaintext). Фильтруем по длине и по казахскому языку.

ЗАПУСК (локально/Colab):
    python -m src.scraping.wiki_scraper --target 800 --out data/raw/wiki_kz.jsonl

Схема вывода совпадает с остальными сборщиками -> build_corpus без изменений:
    {"id","title","source_name","project","url","text","published_at"}
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Dict, List, Tuple

import requests

from .news_scraper import looks_kazakh

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API = "https://kk.wikipedia.org/w/api.php"
# Wikipedia требует осмысленный User-Agent
HEADERS = {"User-Agent": "KazRagBenchmark/1.0 (research; contact via github)"}
MIN_TEXT_LEN = 400          # символов; короче — заготовка-стаб, пропускаем
BATCH = 20                  # статей за один запрос extracts


def _get(session: requests.Session, params: Dict, max_retries: int = 6) -> Dict:
    """GET к API с экспоненциальным backoff и уважением к Retry-After на 429."""
    delay = 2.0
    for _ in range(max_retries):
        r = session.get(API, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            ra = r.headers.get("Retry-After", "")
            wait = float(ra) if ra.isdigit() else delay
            logger.warning(f"429 Too Many Requests — пауза {wait:.0f}с…")
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("429: исчерпаны попытки — Wikipedia лимитирует этот IP")


def parse_extracts(data: Dict) -> List[Tuple[int, str, str]]:
    """Разбирает ответ prop=extracts -> [(pageid, title, text), ...]."""
    pages = data.get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        pid = page.get("pageid")
        title = page.get("title", "")
        extract = (page.get("extract") or "").strip()
        if pid and extract:
            out.append((pid, title, extract))
    return out


def fetch_batch(session: requests.Session, cont: Dict = None):
    """
    Один запрос: пачка статей (generator=allpages) + их полный текст
    (prop=extracts) разом. Возвращает (записи, continue-токен или None).
    """
    params = {
        "action": "query", "format": "json",
        "generator": "allpages", "gapnamespace": 0, "gaplimit": BATCH,
        "gapfilterredir": "nonredirects",
        "prop": "extracts", "explaintext": 1, "exlimit": BATCH,
    }
    if cont:
        params.update(cont)
    data = _get(session, params)
    return parse_extracts(data), data.get("continue")


def collect(target: int, out_path: str, min_len: int = MIN_TEXT_LEN) -> int:
    """Главный цикл: листаем статьи через allpages -> JSONL, до target штук."""
    import os
    import random
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    session = requests.Session()
    total = 0
    seen = set()
    # стартуем со случайной буквы — для разнообразия тем между прогонами
    cont = {"gapfrom": random.choice("АӘБВГҒДЕЖЗИКҚЛМНОӨПРСТУҰҮФХЦЧШЫІ")}
    max_rounds = target * 3
    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(max_rounds):
            if total >= target:
                break
            try:
                records, cont = fetch_batch(session, cont)
            except Exception as e:
                logger.warning(f"запрос не удался: {e}")
                break
            for pid, title, text in records:
                if pid in seen:
                    continue
                seen.add(pid)
                if len(text) < min_len or not looks_kazakh(text):
                    continue
                rec = {
                    "id": f"wiki_{pid}",
                    "title": title,
                    "source_name": "kk.wikipedia.org",
                    "project": "wiki",
                    "url": f"https://kk.wikipedia.org/?curid={pid}",
                    "text": text,
                    "published_at": "",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                total += 1
                if total % 50 == 0:
                    logger.info(f"  собрано {total}/{target}…")
                if total >= target:
                    break
            if not cont:  # дошли до конца списка статей
                logger.info("allpages закончился, начинаем с другой буквы")
                cont = {"gapfrom": random.choice("АӘБВГҒДЕЖЗИКҚЛМНОӨПРСТУҰҮФХЦЧШЫІ")}
            time.sleep(0.5)  # вежливая пауза к API
    logger.info(f"🎯 Итого собрано: {total} статей → {out_path}")
    return total


def collect_from_hf(target: int, out_path: str, config: str = "20231101.kk",
                    min_len: int = MIN_TEXT_LEN) -> int:
    """
    РЕКОМЕНДУЕМЫЙ способ: берём ГОТОВЫЙ дамп казахской Википедии с
    Hugging Face (wikimedia/wikipedia) — чистый текст, без лимитов API.
    Стримим и останавливаемся на target (весь дамп качать не нужно).

    Требует: pip install datasets
    """
    import os
    from datasets import load_dataset

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    logger.info(f"Загружаю дамп wikimedia/wikipedia [{config}] (streaming)…")
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)

    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            text = (ex.get("text") or "").strip()
            if len(text) < min_len or not looks_kazakh(text):
                continue
            rec = {
                "id": f"wiki_{ex.get('id')}",
                "title": ex.get("title", ""),
                "source_name": "kk.wikipedia.org",
                "project": "wiki",
                "url": ex.get("url", ""),
                "text": text,
                "published_at": "",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
            if total % 100 == 0:
                logger.info(f"  собрано {total}/{target}…")
            if total >= target:
                break
    logger.info(f"🎯 Итого собрано: {total} статей → {out_path}")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Сбор корпуса из казахской Википедии")
    ap.add_argument("--target", type=int, default=800, help="число статей")
    ap.add_argument("--min-len", type=int, default=MIN_TEXT_LEN, help="мин. длина текста, симв.")
    ap.add_argument("--out", default="data/raw/wiki_kz.jsonl")
    ap.add_argument("--via", choices=["hf", "api"], default="hf",
                    help="hf = дамп с Hugging Face (рекомендуется, без лимитов); "
                         "api = живой MediaWiki API (лимитируется на общих IP)")
    ap.add_argument("--config", default="20231101.kk", help="снапшот дампа HF")
    args = ap.parse_args()
    if args.via == "hf":
        collect_from_hf(args.target, args.out, args.config, args.min_len)
    else:
        collect(args.target, args.out, args.min_len)


if __name__ == "__main__":
    main()
