"""
Сборщик казахских пресс-релизов с gov.kz — версия для бенчмарка.

Отличие от исходного scraper_gov (FastAPI): добавлена ПАГИНАЦИЯ, чтобы
набрать корпус в сотни документов, и CLI-запуск с записью в JSONL.

gov.kz — это SPA: список и тексты отдаёт закрытое API, которому нужны
заголовки hash+token. Их перехватывает Playwright (один раз на источник).

ЗАПУСК (локально, т.к. gov.kz закрыт в облачном окружении):
    pip install requests beautifulsoup4 playwright
    playwright install chromium
    python -m src.scraping.gov_scraper --target 800 --per-source 60 \
        --out data/raw/gov_news.jsonl

Результат — JSONL, по одной новости на строку:
    {"id","title","source_name","project","url","text","published_at"}
Дальше из него собирается корпус пассажей (src/corpus/build_corpus.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional

import requests
import urllib3
from bs4 import BeautifulSoup

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LANG = "kz"
PAGE_SIZE = 20  # размер страницы API gov.kz

# Источники: казахские версии сайтов госорганов (22 шт.)
SOURCES: List[Dict] = [
    {"name": "МинНацЭкономики", "project": "economy"},
    {"name": "МинФин", "project": "minfin"},
    {"name": "МИД РК", "project": "mfa"},
    {"name": "МВД РК", "project": "qriim"},
    {"name": "МинТруда", "project": "enbek"},
    {"name": "МинЗдрав", "project": "dsm"},
    {"name": "МинПросвещения", "project": "edu"},
    {"name": "МинНауки", "project": "sci"},
    {"name": "МинПромСтрой", "project": "mps"},
    {"name": "МинТранспорт", "project": "transport"},
    {"name": "МинЦифры", "project": "mdai"},
    {"name": "МинКультуры", "project": "mam"},
    {"name": "МинТуризм", "project": "tsm"},
    {"name": "МинЭкологии", "project": "ecogeo"},
    {"name": "МинСельХоз", "project": "moa"},
    {"name": "МинЭнерго", "project": "energo"},
    {"name": "МинЮст", "project": "adilet"},
    {"name": "МЧС РК", "project": "emer"},
    {"name": "МинТорговли", "project": "mti"},
    {"name": "Акимат Алматы", "project": "almaty"},
    {"name": "Акимат Астаны", "project": "astana"},
]
BASE_URL = "https://www.gov.kz"


async def fetch_tokens(verbose: bool = False) -> Optional[Dict]:
    """
    Запускает Playwright и ДОЖИДАЕТСЯ XHR-запроса браузера к API gov.kz,
    из которого читает hash+token. verbose=True печатает диагностику
    (финальный URL, заголовок страницы, какие запросы к API были видны).
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.error("Playwright не установлен: pip install playwright && playwright install chromium")
        return None
    tokens: Dict = {}
    seen_requests: List = []  # для диагностики: ключи заголовков замеченных запросов
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
            )
            context = await browser.new_context(
                user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"),
                locale="kk-KZ",
            )
            page = await context.new_page()

            def handle_request(request):
                if "api/v1/public/content-manager/news" in request.url:
                    h = request.headers
                    seen_requests.append(sorted(h.keys()))
                    if h.get("hash") and h.get("token"):
                        tokens.update(
                            hash=h["hash"], token=h["token"],
                            user_agent=h.get("user-agent", ""), obtained_at=time.time(),
                        )

            page.on("request", handle_request)

            try:
                await page.goto(f"{BASE_URL}/memleket/entities/economy/press/news?lang={LANG}",
                                wait_until="domcontentloaded", timeout=60000)
            except Exception as e:
                logger.error(f"goto не удался: {e}")

            # Прямо ждём запрос к API, в котором уже есть токен (а не просто ссылку)
            try:
                req = await page.wait_for_request(
                    lambda r: "api/v1/public/content-manager/news" in r.url
                    and bool(r.headers.get("token")),
                    timeout=45000,
                )
                h = req.headers
                tokens.update(
                    hash=h.get("hash"), token=h.get("token"),
                    user_agent=h.get("user-agent", ""), obtained_at=time.time(),
                )
            except Exception as e:
                logger.warning(f"Не дождались запроса с токеном: {e}")

            if verbose:
                try:
                    logger.info(f"🔎 Финальный URL: {page.url}")
                    logger.info(f"🔎 Заголовок страницы: {await page.title()}")
                except Exception:
                    pass
                logger.info(f"🔎 Запросов к news-API замечено: {len(seen_requests)}")
                if seen_requests:
                    logger.info(f"🔎 Ключи заголовков (первый запрос): {seen_requests[0]}")
                logger.info(f"🔎 hash найден: {bool(tokens.get('hash'))}, "
                            f"token найден: {bool(tokens.get('token'))}")

            await browser.close()
    except Exception as e:
        logger.error(f"Ошибка Playwright: {e}")
        return None
    return tokens if tokens.get("hash") and tokens.get("token") else None


def _headers(tokens: Dict, project: str) -> Dict:
    return {
        "accept": "application/json",
        "accept-language": LANG,
        "user-agent": tokens.get("user_agent", "Mozilla/5.0"),
        "referer": f"{BASE_URL}/memleket/entities/{project}/press/news?lang={LANG}",
        "hash": tokens["hash"],
        "token": tokens["token"],
        "origin": BASE_URL,
    }


def _extract_body_html(obj: Dict) -> str:
    if not isinstance(obj, dict):
        return ""
    return obj.get("body") or obj.get("text") or obj.get("content") or obj.get("description") or ""


def fetch_detail(news_id, project: str, tokens: Dict) -> str:
    """Полный текст одной новости через detail-эндпоинт."""
    url = f"{BASE_URL}/api/v1/public/content-manager/news/{news_id}?lang={LANG}"
    try:
        resp = requests.get(url, headers=_headers(tokens, project), timeout=15, verify=False)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        obj = data
        if isinstance(data, dict) and not _extract_body_html(data):
            obj = data.get("data") or data.get("content") or data
        raw = _extract_body_html(obj)
        return BeautifulSoup(raw, "html.parser").get_text(separator="\n").strip() if raw else ""
    except Exception as e:
        logger.warning(f"detail {news_id}: {e}")
        return ""


def fetch_page(project: str, page: int, tokens: Dict) -> List[Dict]:
    """Одна страница списочного API (PAGE_SIZE новостей)."""
    url = (f"{BASE_URL}/api/v1/public/content-manager/news"
           f"?sort-by=created_date:DESC&projects=eq:{project}&page={page}&size={PAGE_SIZE}&lang={LANG}")
    try:
        resp = requests.get(url, headers=_headers(tokens, project), timeout=15, verify=False)
        if resp.status_code != 200:
            logger.warning(f"{project} p{page}: код {resp.status_code}")
            return []
        data = resp.json()
        if isinstance(data, list):
            return data
        return data.get("content", []) or data.get("data", []) or data.get("items", [])
    except Exception as e:
        logger.warning(f"{project} p{page}: {e}")
        return []


def scrape_source(source: Dict, per_source: int, tokens: Dict) -> List[Dict]:
    """Тянет до per_source новостей из источника, ПЕРЕЛИСТЫВАЯ страницы."""
    name, project = source["name"], source["project"]
    out: List[Dict] = []
    seen = set()
    page = 1
    max_pages = (per_source // PAGE_SIZE) + 3  # запас на пустые/дубли
    while len(out) < per_source and page <= max_pages:
        items = fetch_page(project, page, tokens)
        if not items:
            break
        for item in items:
            if not isinstance(item, dict):
                continue
            news_id = item.get("id") or item.get("slug")
            title = (item.get("name") or item.get("title") or "").strip()
            if not news_id or not title or news_id in seen:
                continue
            seen.add(news_id)

            text = BeautifulSoup(_extract_body_html(item), "html.parser").get_text(separator="\n").strip()
            detail = fetch_detail(news_id, project, tokens)
            if len(detail) > len(text):
                text = detail
            time.sleep(0.3)
            if len(text) < 50:  # пропускаем пустышки
                continue

            out.append({
                "id": f"{project}_{news_id}",
                "title": title,
                "source_name": name,
                "project": project,
                "url": f"{BASE_URL}/memleket/entities/{project}/press/news/details/{news_id}?lang={LANG}",
                "text": text,
                "published_at": (item.get("created_date") or item.get("published_at") or ""),
            })
            if len(out) >= per_source:
                break
        page += 1
    logger.info(f"✅ {name}: {len(out)} текстов")
    return out


async def collect(target: int, per_source: int, out_path: str) -> int:
    """Главный цикл: по источникам, со свежими токенами, пишет JSONL инкрементально."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    total = 0
    seen_ids = set()
    with open(out_path, "w", encoding="utf-8") as f:
        for source in SOURCES:
            if total >= target:
                break
            tokens = await fetch_tokens()  # свежие токены на каждый источник
            if not tokens:
                logger.error(f"❌ {source['name']}: нет токенов, пропуск")
                continue
            remaining = min(per_source, target - total)
            for rec in scrape_source(source, remaining, tokens):
                if rec["id"] in seen_ids:
                    continue
                seen_ids.add(rec["id"])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                total += 1
            time.sleep(2)
    logger.info(f"🎯 Итого собрано: {total} → {out_path}")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Сбор корпуса пресс-релизов gov.kz (kz)")
    ap.add_argument("--target", type=int, default=800, help="целевое число документов")
    ap.add_argument("--per-source", type=int, default=60, help="максимум документов с одного источника")
    ap.add_argument("--out", default="data/raw/gov_news.jsonl", help="путь для JSONL")
    ap.add_argument("--diagnose", action="store_true",
                    help="разовая проверка получения токенов с диагностикой")
    args = ap.parse_args()

    if args.diagnose:
        tokens = asyncio.run(fetch_tokens(verbose=True))
        if tokens:
            logger.info("✅ Токены получены — можно запускать полный сбор.")
        else:
            logger.error("❌ Токены НЕ получены. Смотри диагностику выше "
                         "(финальный URL и заголовок страницы подскажут, "
                         "не отдаёт ли gov.kz блок/капчу для этого IP).")
        return

    asyncio.run(collect(args.target, args.per_source, args.out))


if __name__ == "__main__":
    main()
