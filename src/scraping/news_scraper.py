"""
Универсальный сборщик казахских новостей (baq.kz, inform.kz и любые другие).

Зачем не gov.kz: тот — SPA с закрытым API и токенами, хрупкий в сборе.
Для бенчмарка важен казахский ТЕКСТ, а не конкретный источник. Обычные
новостные сайты отдают готовый HTML, а trafilatura надёжно вытаскивает
из него основной текст + заголовок + дату на любом сайте без правки кода.

Поиск статей — через sitemap сайта (там тысячи ссылок), с фолбэком на
сбор ссылок прямо со стартовой страницы.

ЗАПУСК (локально/Colab):
    pip install trafilatura
    python -m src.scraping.news_scraper --target 800 \
        --sites https://baq.kz https://kaz.inform.kz \
        --out data/raw/news_kz.jsonl

Схема вывода совпадает с gov_scraper, чтобы build_corpus работал без изменений:
    {"id","title","source_name","project","url","text","published_at"}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Буквы, специфичные для казахского алфавита — для проверки языка текста.
_KAZAKH_LETTERS = set("әіңғүұқөһ")
MIN_TEXT_LEN = 200       # символов; короче — не статья
MIN_KAZAKH_RATIO = 0.01  # доля казах-специфичных букв в тексте


def looks_kazakh(text: str) -> bool:
    """Эвристика: в казахском тексте всегда заметна доля букв ә/қ/ң/ғ/… ."""
    if not text:
        return False
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return False
    kaz = sum(1 for c in letters if c in _KAZAKH_LETTERS)
    return (kaz / len(letters)) >= MIN_KAZAKH_RATIO


def site_slug(homepage: str) -> str:
    """baq.kz -> 'baq', kaz.inform.kz -> 'inform' (для id/source)."""
    host = urlparse(homepage).netloc or homepage
    host = host.replace("www.", "")
    parts = host.split(".")
    # берём самую содержательную часть домена
    return parts[-2] if len(parts) >= 2 else parts[0]


def discover_urls(homepage: str, limit: int) -> List[str]:
    """Находит ссылки на статьи через sitemap, фолбэк — ссылки со страницы."""
    import trafilatura
    from trafilatura.sitemaps import sitemap_search

    urls: List[str] = []
    try:
        urls = sitemap_search(homepage, target_lang="kk") or []
    except Exception as e:
        logger.warning(f"sitemap для {homepage} не сработал: {e}")

    if not urls:  # фолбэк: вытащить ссылки прямо со стартовой страницы
        try:
            downloaded = trafilatura.fetch_url(homepage)
            links = trafilatura.extract_links(downloaded) if downloaded else []
            host = urlparse(homepage).netloc
            urls = [u for u in (links or []) if urlparse(u).netloc == host]
        except Exception as e:
            logger.warning(f"Фолбэк-сбор ссылок для {homepage} не сработал: {e}")

    # уникализируем, сохраняя порядок
    seen, uniq = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    logger.info(f"{homepage}: найдено ссылок-кандидатов: {len(uniq)}")
    return uniq[: limit * 3]  # запас на отсев нестатей/неказахских


def scrape_article(url: str, source_name: str, project: str) -> Optional[Dict]:
    """Скачивает и извлекает текст+метаданные одной статьи."""
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    raw = trafilatura.extract(downloaded, output_format="json", with_metadata=True,
                              favor_precision=True)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    text = (data.get("text") or "").strip()
    if len(text) < MIN_TEXT_LEN or not looks_kazakh(text):
        return None

    # стабильный id из последней части пути
    slug = urlparse(url).path.rstrip("/").split("/")[-1] or str(abs(hash(url)))
    return {
        "id": f"{project}_{slug}",
        "title": (data.get("title") or "").strip(),
        "source_name": source_name,
        "project": project,
        "url": url,
        "text": text,
        "published_at": data.get("date") or "",
    }


def collect(homepages: List[str], target: int, per_site: int, out_path: str) -> int:
    """Главный цикл по сайтам, инкрементальная запись JSONL."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    total = 0
    seen_ids = set()
    with open(out_path, "w", encoding="utf-8") as f:
        for homepage in homepages:
            if total >= target:
                break
            project = site_slug(homepage)
            source_name = urlparse(homepage).netloc or homepage
            got = 0
            for url in discover_urls(homepage, per_site):
                if total >= target or got >= per_site:
                    break
                rec = scrape_article(url, source_name, project)
                time.sleep(0.4)  # вежливая пауза
                if not rec or rec["id"] in seen_ids:
                    continue
                seen_ids.add(rec["id"])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                total += 1
                got += 1
                if got % 25 == 0:
                    logger.info(f"  {source_name}: {got} статей…")
            logger.info(f"✅ {source_name}: собрано {got}")
    logger.info(f"🎯 Итого собрано: {total} → {out_path}")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Сбор казахских новостей (универсальный)")
    ap.add_argument("--sites", nargs="+",
                    default=["https://baq.kz", "https://kaz.inform.kz"],
                    help="стартовые URL сайтов")
    ap.add_argument("--target", type=int, default=800, help="всего документов")
    ap.add_argument("--per-site", type=int, default=500, help="максимум с одного сайта")
    ap.add_argument("--out", default="data/raw/news_kz.jsonl")
    args = ap.parse_args()
    collect(args.sites, args.target, args.per_site, args.out)


if __name__ == "__main__":
    main()
