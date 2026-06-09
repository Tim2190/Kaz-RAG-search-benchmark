"""
Спринт 3 — фильтр synonym-кандидатов по лексическому пересечению.

Берёт файл кандидатов (jsonl: query_id, text, gold_doc_ids, answer, category)
и для каждого считает overlap текста вопроса с его gold-пассажем тем же
методом, что и валидация Спринта 2 (префиксный стемминг, корень ≈ 5 букв).

ПРОПУСКАЕТ только вопросы с overlap < threshold — это и есть «настоящий
vocabulary gap»: слова вопроса не совпадают со словами пассажа.

ЗАПУСК:
    python -m src.eval.synonym_filter --in data/queries/synonym_candidates.jsonl \
        --out data/queries/synonym_queries.jsonl --threshold 0.30
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Set

from ..queries import dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CORPUS = os.path.join(ROOT, "data/corpus/corpus.jsonl")

_WORD = re.compile(r"[А-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]+")


def _stem(w: str) -> str:
    return w[:5]


def _toks(s: str) -> Set[str]:
    return set(_WORD.findall(s.lower()))


def overlap(qtext: str, gtext: str) -> float:
    q = {_stem(w) for w in _toks(qtext) if len(w) >= 3}
    g = {_stem(w) for w in _toks(gtext)}
    if not q:
        return 1.0
    return len(q & g) / len(q)


def main() -> None:
    ap = argparse.ArgumentParser(description="Фильтр synonym-кандидатов по overlap")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", default=None,
                    help="куда писать прошедших фильтр (jsonl)")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    corpus = dict(dataset.load_corpus(CORPUS))
    cands = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]

    kept, rejected = [], []
    for q in cands:
        gid = q["gold_doc_ids"][0]
        gtext = corpus.get(gid, "")
        if not gtext:
            print(f"[!] gold пассаж не найден: {gid} (вопрос {q.get('query_id')})")
            continue
        ov = overlap(q["text"], gtext)
        q["_overlap"] = round(ov, 3)
        (kept if ov < args.threshold else rejected).append(q)

    kept.sort(key=lambda x: x["_overlap"])
    print(f"Кандидатов: {len(cands)} | прошли (<{args.threshold}): {len(kept)} | "
          f"отсеяно: {len(rejected)}")
    if kept:
        avg = sum(q["_overlap"] for q in kept) / len(kept)
        print(f"Средний overlap прошедших: {avg:.3f} "
              f"(диапазон {kept[0]['_overlap']}–{kept[-1]['_overlap']})")

    if args.verbose:
        print("\n--- ПРОШЛИ ---")
        for q in kept:
            print(f"  [{q['_overlap']:.2f}] {q['text']}  →  {q.get('answer')}")
        print("\n--- ОТСЕЯНЫ (overlap высокий) ---")
        for q in sorted(rejected, key=lambda x: -x["_overlap"]):
            print(f"  [{q['_overlap']:.2f}] {q['text']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for q in kept:
                rec = {k: v for k, v in q.items() if k != "_overlap"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n→ Прошедшие фильтр сохранены: {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
