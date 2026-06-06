"""
Generate new queries for the benchmark using Claude API.

Selects candidate passages from the corpus that are not yet covered,
generates 3 queries per passage (natural / inflected / vocabulary-gap),
appends them to data/queries/queries.jsonl.

Usage:
    python -m src.queries.generate_queries --n 80 --out data/queries/queries.jsonl

Requires:
    ANTHROPIC_API_KEY environment variable
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Optional


PROMPT_TEMPLATE = """Саған қазақ Уикипедия мәтіні берілген. Осы мәтін негізінде 3 сұрақ жаз.

Мәтін:
\"\"\"
{text}
\"\"\"

Мынадай 3 сұрақ жасат (JSON массив ретінде жауап бер):
1. **natural** — мәтін туралы қарапайым сұрақ. Жауабы мәтіндегі сөйлемнен алынған 1-3 сөз болуы керек.
2. **inflected** — негізгі сөздің өзгерген (септелген/жіктелген) түрін қамтитын сұрақ. Жауабы 1-3 сөз.
3. **vocabulary-gap** — мәтіндегі сөздерді НЕ қолданбай, синоним немесе сипаттама арқылы жазылған сұрақ. Жауабы 1-3 сөз.

Ескерту:
- Жауап мәтінде ТІК кездесетін сөз(дер) болуы шарт (exact substring match үшін).
- Жауап 1–4 сөзден артық болмасын.
- Тек JSON массив қайтар, басқа ешнәрсе жазба.

Формат:
[
  {{"category": "natural",         "question": "...", "answer": "..."}},
  {{"category": "inflected",       "question": "...", "answer": "..."}},
  {{"category": "vocabulary-gap",  "question": "...", "answer": "..."}}
]"""


def load_covered(queries_path: str) -> set[str]:
    if not Path(queries_path).exists():
        return set()
    covered = set()
    for line in open(queries_path, encoding="utf-8"):
        q = json.loads(line)
        covered.update(q.get("gold_doc_ids", []))
    return covered


def load_passages(corpus_path: str) -> list[dict]:
    return [json.loads(l) for l in open(corpus_path, encoding="utf-8")]


def score_passage(p: dict) -> float:
    """Prefer passages with named entities, numbers, clear facts. Simple heuristic."""
    text = p["text"]
    score = 0.0
    # prefer moderate length (120-400 chars)
    ln = len(text)
    if 120 <= ln <= 500:
        score += 2.0
    elif ln < 120:
        score -= 2.0
    # reward capital letters (named entities in Kazakh)
    caps = sum(1 for c in text if c.isupper())
    score += min(caps * 0.05, 2.0)
    # reward digits (dates, numbers → precise answers)
    digits = sum(1 for c in text if c.isdigit())
    score += min(digits * 0.1, 2.0)
    return score


def pick_candidates(passages: list[dict], covered: set[str], n: int, seed: int = 42) -> list[dict]:
    p0s = [p for p in passages if p["doc_id"].endswith("_p0") and p["doc_id"] not in covered]
    scored = sorted(p0s, key=score_passage, reverse=True)
    top = scored[: n * 3]  # take top 3n, then sample n for diversity
    random.seed(seed)
    return random.sample(top, min(n, len(top)))


def generate_for_passage(client, passage: dict) -> Optional[list[dict]]:
    text = passage["text"][:600]  # cap at 600 chars to fit context
    prompt = PROMPT_TEMPLATE.format(text=text)
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # extract JSON array
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return None
        items = json.loads(m.group(0))
        return items if isinstance(items, list) and len(items) == 3 else None
    except Exception as exc:
        print(f"  API error: {exc}")
        return None


def next_query_id(queries_path: str) -> int:
    if not Path(queries_path).exists():
        return 1
    ids = []
    for line in open(queries_path, encoding="utf-8"):
        q = json.loads(line)
        try:
            ids.append(int(q["query_id"].lstrip("q")))
        except ValueError:
            pass
    return max(ids, default=0) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="Number of new entities to cover")
    ap.add_argument("--corpus", default="data/corpus/corpus.jsonl")
    ap.add_argument("--out", default="data/queries/queries.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    try:
        import anthropic
    except ImportError:
        raise SystemExit("pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)

    covered = load_covered(args.out)
    passages = load_passages(args.corpus)
    candidates = pick_candidates(passages, covered, args.n, args.seed)
    print(f"Generating queries for {len(candidates)} new passages → {args.out}")

    next_id = next_query_id(args.out)
    added = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for i, passage in enumerate(candidates, 1):
            print(f"[{i}/{len(candidates)}] {passage['doc_id']} | {passage['title'][:50]}")
            items = generate_for_passage(client, passage)
            if items is None:
                print("  skipped (bad response)")
                continue
            for item in items:
                q = {
                    "query_id": f"q{next_id:03d}",
                    "text": item["question"],
                    "category": item["category"],
                    "gold_doc_ids": [passage["doc_id"]],
                    "answer": item["answer"],
                    "generated_by": "claude",
                    "validated": False,
                }
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
                next_id += 1
                added += 1
            time.sleep(args.delay)

    print(f"\nDone. Added {added} queries. Total now: {added + len(covered) * 3}")


if __name__ == "__main__":
    main()
