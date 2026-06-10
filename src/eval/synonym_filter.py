"""
Sprint 3 — synonym / vocabulary-gap filter.
Reads synonym candidate queries from a JSONL file and keeps only those
where the query text has LOW lexical overlap with the gold passage.
"""
from __future__ import annotations
import argparse, json, os, re

_WORD = re.compile(r"[А-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]+")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CORPUS = os.path.join(ROOT, "data/corpus/corpus.jsonl")
_corpus_cache: dict[str, str] = {}


def _load_passage(doc_id: str) -> str:
    global _corpus_cache
    if not _corpus_cache:
        with open(CORPUS, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                _corpus_cache[d["doc_id"]] = d.get("text", "")
    return _corpus_cache.get(doc_id, "")


def _stem(w: str) -> str:
    return w[:5]


def _toks(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def overlap(qtext: str, gtext: str) -> float:
    q = {_stem(w) for w in _toks(qtext) if len(w) >= 3}
    g = {_stem(w) for w in _toks(gtext)}
    if not q:
        return 1.0
    return len(q & g) / len(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    passed, failed = [], []
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gtext = _load_passage(item["gold_doc_id"])
            ov = overlap(item["query"], gtext)
            item["overlap"] = round(ov, 4)
            if ov <= args.threshold:
                passed.append(item)
                if args.verbose:
                    print(f"PASS {ov:.3f}  {item['gold_doc_id']}  {item['query'][:60]}")
            else:
                failed.append(item)
                if args.verbose:
                    print(f"FAIL {ov:.3f}  {item['gold_doc_id']}  {item['query'][:60]}")

    with open(args.out, "w", encoding="utf-8") as f:
        for item in passed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nPASS: {len(passed)}  FAIL: {len(failed)}  total: {len(passed)+len(failed)}")


if __name__ == "__main__":
    main()
