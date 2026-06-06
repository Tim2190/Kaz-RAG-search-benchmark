"""
Прогрев стем-кэша для всего корпуса с АВТО-КОММИТОМ в GitHub.

Решает проблему обрывов Colab: кэш не просто пишется на диск, а каждые
~push_every слов коммитится и пушится в репозиторий. Если среда умрёт —
после `git pull` повторный запуск продолжит с последнего запушенного места.

ЗАПУСК (Colab/локально, репозиторий склонирован с токеном в URL):
    python -m src.preprocess.warm_corpus \
        --corpus data/corpus/corpus.jsonl \
        --queries data/queries/queries.jsonl

После завершения метрики со стеммером считаются мгновенно:
    python -m src.eval.run_benchmark --stemmer kazakh --out results/bm25_kazakh.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import List

from .tokenize import tokenize
from .stemmer_api import KazakhStemmerAPI, DEFAULT_BASE_URL


def _collect_tokens(corpus_path: str, queries_path: str) -> set:
    toks = set()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                toks.update(tokenize(json.loads(line)["text"]))
    try:
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    toks.update(tokenize(json.loads(line)["text"]))
    except FileNotFoundError:
        pass
    return toks


def _current_branch() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "HEAD"
    except Exception:
        return "HEAD"


def _git_push_cache(cache_path: str, branch: str) -> None:
    """Коммитит и пушит кэш. Ошибки не валят прогрев — просто логируются.
    Перед пушем подтягивает чужие коммиты (merge -X ours: наш кэш — главный,
    чужой код добавляется), чтобы авто-пуш не падал из-за параллельных правок."""
    try:
        subprocess.run(["git", "add", cache_path], capture_output=True, timeout=30)
        commit = subprocess.run(
            ["git", "commit", "-m", "Чекпоинт стем-кэша (авто)"],
            capture_output=True, text=True, timeout=30)
        if "nothing to commit" in (commit.stdout + commit.stderr):
            return
        # подтянуть удалённые изменения, при конфликте кэша оставить наш
        subprocess.run(["git", "pull", "--no-rebase", "-X", "ours", "origin", branch],
                       capture_output=True, text=True, timeout=120)
        push = subprocess.run(["git", "push", "origin", branch],
                             capture_output=True, text=True, timeout=120)
        if push.returncode == 0:
            print("  ✅ кэш запушен в GitHub", flush=True)
        else:
            print(f"  ⚠️ push не удался: {push.stderr.strip()[:200]}", flush=True)
    except Exception as e:
        print(f"  ⚠️ авто-пуш пропущен: {e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Прогрев стем-кэша с авто-коммитом в GitHub")
    ap.add_argument("--corpus", default="data/corpus/corpus.jsonl")
    ap.add_argument("--queries", default="data/queries/queries.jsonl")
    ap.add_argument("--cache", default="data/resources/stem_cache.json")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--push-every", type=int, default=5000,
                    help="пушить кэш в GitHub каждые N новых слов")
    ap.add_argument("--no-push", action="store_true", help="не пушить (только локальный кэш)")
    args = ap.parse_args()

    tokens = _collect_tokens(args.corpus, args.queries)
    print(f"Уникальных токенов для прогрева: {len(tokens)}", flush=True)

    stemmer = KazakhStemmerAPI(base_url=args.base_url, cache_path=args.cache)
    branch = _current_branch()
    state = {"last_push": 0}

    def on_checkpoint(done: int, total: int) -> None:
        if args.no_push:
            return
        if done - state["last_push"] >= args.push_every or done == total:
            _git_push_cache(args.cache, branch)
            state["last_push"] = done

    stemmer.warm(tokens, on_checkpoint=on_checkpoint)
    print(f"🎯 Прогрев завершён. Размер кэша: {len(stemmer._cache)} слов.", flush=True)


if __name__ == "__main__":
    main()
