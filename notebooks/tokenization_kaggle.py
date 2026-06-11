"""
Tokenization fertility test — Kaggle/Colab-ready version.
Copy each cell block into a notebook cell.

Models tested (tokenizer only, no weights downloaded):
  1. ibm-granite/granite-embedding-97m-multilingual-r2    (R2, byte-level BPE)
  2. ibm-granite/granite-embedding-311m-multilingual-r2   (R2, byte-level BPE)
  3. intfloat/multilingual-e5-base                        (SentencePiece)
  4. ibm-granite/granite-embedding-278m-multilingual      (R1, SentencePiece)
  5. stukenov/sozkz-morphbpe-256k-kk-v1                   (TilQazyna, morpheme BPE)

Expected run time: ~2-3 min (CPU, tokenizer-only).
"""

# ── Cell 1: install ────────────────────────────────────────────────────────────
# !pip install -q transformers huggingface_hub

# ── Cell 2: clone repo (Kaggle — skip if already present) ─────────────────────
# import os
# if not os.path.exists("Kaz-RAG-search-benchmark"):
#     !git clone https://github.com/tim2190/kaz-rag-search-benchmark.git Kaz-RAG-search-benchmark
# os.chdir("Kaz-RAG-search-benchmark")

# ── Cell 3: run tokenization test ─────────────────────────────────────────────
# Option A — run as module (preferred if repo is cloned):
# !python -m src.eval.tokenization_test

# Option B — standalone (paste the block below if running without the full repo):

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import List

_WORD = re.compile(r"[А-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]+")

TOKENIZERS = {
    "granite-97m-r2":    "ibm-granite/granite-embedding-97m-multilingual-r2",
    "granite-311m-r2":   "ibm-granite/granite-embedding-311m-multilingual-r2",
    "e5-base":           "intfloat/multilingual-e5-base",
    "granite-278m-r1":   "ibm-granite/granite-embedding-278m-multilingual",
    "tilcore_morph256k": "stukenov/sozkz-morphbpe-256k-kk-v1",
}

EXAMPLE_WORDS = ["сөзжасам", "мүмкіндіктерін", "ұйымдастырушылық", "халықаралық"]

# The fixed 100-word list (100 most-frequent Kazakh surface forms, len>=9,
# from the benchmark corpus). Embedded so the script runs standalone on
# Kaggle/Colab WITHOUT cloning the repo or any corpus file. All 5 tokenizers
# run on this identical set.
WORDS: List[str] = [
    "қазақстан", "мемлекеттік", "орналасқан", "солтүстік", "республикасы",
    "байланысты", "республикасының", "экономикалық", "қазақстанның", "халықаралық",
    "негізінен", "жергілікті", "арасындағы", "нәтижесінде", "облысының",
    "шаруашылығы", "әлеуметтік", "кездеседі", "әкімшілік", "президенті",
    "түркістан", "қаласында", "президент", "сондықтан", "дүниежүзілік",
    "республика", "жөніндегі", "этникалық", "аумағында", "қамтамасыз",
    "негізінде", "аралығында", "шығысында", "құрамында", "өнеркәсіп",
    "мемлекеті", "қаласының", "орынбасары", "иегерлері", "шаруашылық",
    "университеті", "оңтүстігінде", "қарағанды", "өнеркәсібі", "батысында",
    "қарағанда", "бөлігінде", "солтүстігінде", "ұлыбритания", "ғасырларда",
    "республикалық", "қазақтардың", "еуропалық", "қалыптасқан", "парламент",
    "қазақстанда", "мүмкіндік", "кеңесінің", "көпшілігі", "тәуелсіздік",
    "хандығының", "білдіреді", "экономика", "тайпалары", "көрсетеді",
    "әскерлері", "ортасында", "азаматтық", "демократиялық", "мемлекетінің",
    "музыкалық", "температурасы", "қызылорда", "жариялады", "мәдениеті",
    "температура", "империясының", "жарияланды", "географиялық", "бойындағы",
    "мәліметтер", "республиканың", "империясы", "қарамастан", "мекендеген",
    "комитетінің", "жартысында", "социалистік", "президентінің", "қабылданды",
    "аумағының", "өнімдерін", "техникалық", "мемлекеттің", "сақталған",
    "мақсатында", "парламенті", "жағалауында", "қалыптасты", "қазақстандағы",
]


def _harvest_words(corpus_path: str, n: int = 100, min_len: int = 9) -> List[str]:
    counter: Counter = Counter()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line).get("text", "")
            except Exception:
                continue
            for w in _WORD.findall(text.lower()):
                if len(w) >= min_len:
                    counter[w] += 1
    return [w for w, _ in counter.most_common(n)]


def _avg_subwords(tok, words: List[str], lead_space: bool) -> float:
    total = 0
    for w in words:
        s = (" " + w) if lead_space else w
        total += len(tok.encode(s, add_special_tokens=False))
    return total / max(len(words), 1)


def run(words: List[str] = WORDS) -> None:
    from transformers import AutoTokenizer

    print(f"Words: {len(words)}, examples: {', '.join(words[:5])} …\n")

    loaded = {}
    for alias, hf_id in TOKENIZERS.items():
        try:
            loaded[alias] = AutoTokenizer.from_pretrained(hf_id)
        except Exception:
            try:
                loaded[alias] = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            except Exception as e:
                print(f"[skip] {alias}: {e}")

    # ── Fertility table ────────────────────────────────────────────────────────
    print(f"\n{'tokenizer':22s} {'isolated':>10s} {'lead-space':>12s}")
    print("-" * 46)
    results = {}
    for alias, tok in loaded.items():
        a_bare = _avg_subwords(tok, words, lead_space=False)
        a_lead = _avg_subwords(tok, words, lead_space=True)
        results[alias] = (a_bare, a_lead)
        print(f"{alias:22s} {a_bare:10.2f} {a_lead:12.2f}")

    # ── Split examples ─────────────────────────────────────────────────────────
    print("\n--- Split examples (isolated; ▁/Ġ = word-boundary marker) ---")
    for w in EXAMPLE_WORDS:
        print(f"\n«{w}»")
        for alias, tok in loaded.items():
            bare = tok.tokenize(w)
            lead = tok.tokenize(" " + w)
            print(f"  {alias:22s} {len(bare)}: {bare}")
            print(f"  {'':22s} +space {len(lead)}: {lead}")

    # ── Summary ────────────────────────────────────────────────────────────────
    if results:
        worst = max(results, key=lambda k: results[k][0])
        best  = min(results, key=lambda k: results[k][0])
        print(f"\n{'─'*46}")
        print(f"Most fragmenting:  {worst}  ({results[worst][0]:.2f} / {results[worst][1]:.2f})")
        print(f"Least fragmenting: {best}   ({results[best][0]:.2f} / {results[best][1]:.2f})")
        ratio = results[worst][0] / max(results[best][0], 0.01)
        print(f"Ratio (isolated):  {ratio:.1f}×")


if __name__ == "__main__":
    run()
