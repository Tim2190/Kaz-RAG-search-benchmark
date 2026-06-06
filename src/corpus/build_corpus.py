"""
Сборка корпуса: сырые новости (data/raw/gov_news.jsonl) -> пассажи
для поиска (data/corpus/corpus.jsonl).

Зачем нарезка на пассажи: документы разной длины, а единицей поиска должен
быть короткий связный фрагмент — так честнее меряется retrieval и так
работает реальный RAG (находим пассаж, а не весь документ). Один документ
даёт несколько пассажей; релевантность в qrels тоже привязана к пассажу.

Модуль без сетевых зависимостей — чистка и чанкинг тестируются локально.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Iterator, List

from ..preprocess.tokenize import tokenize

# Строки-боилерплейт, которые мусорят в гос. релизах
_BOILERPLATE = re.compile(
    r"(подписывайтесь|telegram|whatsapp|instagram|facebook|поделиться|читайте также|"
    r"©|все права защищены)",
    re.IGNORECASE,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")

# Казах-специфичные буквы — для проверки языка пассажа
_KAZAKH_LETTERS = set("әіңғүұқөһ")
_CYRILLIC = set("абвгдежзийклмнопрстуфхцчшщъыьэюяёәіңғүұқөһ")


def is_kazakh_passage(text: str, min_kaz: float = 0.03, min_cyr: float = 0.5) -> bool:
    """
    Отсекает не-казахские пассажи (списки литературы на рус/англ, формулы).
    Требует минимальную долю казах-специфичных букв и кириллицы среди букв.
    Медианная доля казахских букв в настоящем казахском тексте ~0.15,
    поэтому порог 0.03 безопасно отбрасывает только инородные куски.
    """
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return False
    kaz = sum(1 for c in letters if c in _KAZAKH_LETTERS) / len(letters)
    cyr = sum(1 for c in letters if c in _CYRILLIC) / len(letters)
    return kaz >= min_kaz and cyr >= min_cyr


def clean_text(text: str) -> str:
    """Нормализует пробелы, выкидывает короткие/мусорные строки."""
    if not text:
        return ""
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < 15:           # обрывки меню/подписи
            continue
        if _BOILERPLATE.search(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def split_sentences(text: str) -> List[str]:
    """Грубое деление на предложения (по .!?… с пробелом)."""
    text = text.replace("\n", " ")
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def chunk_text(text: str, target_words: int = 120, overlap_words: int = 20) -> List[str]:
    """
    Режет текст на пассажи ~target_words слов по границам предложений,
    с перекрытием overlap_words (чтобы не терять контекст на стыках).
    """
    sentences = split_sentences(text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_n = 0
    for sent in sentences:
        n = len(tokenize(sent))
        if cur_n + n > target_words and cur:
            chunks.append(" ".join(cur))
            # перенос «хвоста» для перекрытия
            tail, tail_n = [], 0
            for s in reversed(cur):
                sn = len(tokenize(s))
                if tail_n + sn > overlap_words:
                    break
                tail.insert(0, s)
                tail_n += sn
            cur, cur_n = tail[:], tail_n
        cur.append(sent)
        cur_n += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def iter_passages(raw_path: str, target_words: int, overlap_words: int) -> Iterator[Dict]:
    """Читает сырые новости и выдаёт пассажи с метаданными."""
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            cleaned = clean_text(doc.get("text", ""))
            if not cleaned:
                continue
            source_doc_id = doc.get("id", "")
            for i, passage in enumerate(chunk_text(cleaned, target_words, overlap_words)):
                if len(tokenize(passage)) < 20:   # слишком короткий хвост
                    continue
                if not is_kazakh_passage(passage):  # отсев рус/англ библиографий
                    continue
                yield {
                    "doc_id": f"{source_doc_id}_p{i}",
                    "source_doc_id": source_doc_id,
                    "source_name": doc.get("source_name", ""),
                    "title": doc.get("title", ""),
                    "text": passage,
                    "lang": "kz",
                    "url": doc.get("url", ""),
                    "published_at": doc.get("published_at", ""),
                }


def build_corpus(raw_path: str, out_path: str, target_words: int = 120,
                 overlap_words: int = 20) -> int:
    """Главная функция: сырьё -> corpus.jsonl. Возвращает число пассажей.

    Дедуп: повторяющиеся тексты пассажей (боилерплейт-сайдбары и т.п.)
    отбрасываются — в корпус попадает только первое вхождение.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    seen_texts = set()
    with open(out_path, "w", encoding="utf-8") as out:
        for passage in iter_passages(raw_path, target_words, overlap_words):
            key = passage["text"].strip()
            if key in seen_texts:
                continue
            seen_texts.add(key)
            out.write(json.dumps(passage, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Сборка корпуса пассажей из сырых новостей")
    ap.add_argument("--raw", default="data/raw/gov_news.jsonl")
    ap.add_argument("--out", default="data/corpus/corpus.jsonl")
    ap.add_argument("--target-words", type=int, default=120)
    ap.add_argument("--overlap-words", type=int, default=20)
    args = ap.parse_args()
    n = build_corpus(args.raw, args.out, args.target_words, args.overlap_words)
    print(f"Готово: {n} пассажей -> {args.out}")


if __name__ == "__main__":
    main()
