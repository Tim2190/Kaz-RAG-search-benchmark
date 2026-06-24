"""
Прогон слоя input-preprocessing: 4 линии текст-в-текст × модель × домен.

Что делает
----------
Для одной (domain, model, line) тройки:
  1. грузит корпус+запросы домена (wiki | akorda);
  2. строит transform линии (baseline/stem/latin/stem_latin) и применяет её
     СИММЕТРИЧНО к корпусу и запросам (иначе ложный провал);
  3. кодирует dense-моделью (e5 | qwen3-0.6b) поверх ПРЕДОБРАБОТАННОГО текста;
  4. считает метрики и сохраняет per-query ранжировки.

Эмбеддинг-кэш РАЗДЕЛЬНЫЙ на линию (текст разный → вектора разные); путь
input-preprocessing/emb/{model}_{domain}_{line}. Без этого линия latin/stem
ошибочно подтянула бы baseline-эмбеддинги.

Сравнение
---------
Подкоманда `compare` грузит сохранённые ранжировки и для каждой линии (1,2,3)
считает Δ nDCG@10 ОТНОСИТЕЛЬНО baseline (0) + paired bootstrap (10 000), по
категориям и в целом. Это и есть главный результат слоя.

ЗАПУСК (GPU; Kaggle/Colab, см. input-preprocessing/code/preprocessing_kaggle.py):
    # один прогон линии
    python input-preprocessing/code/run_preprocessing.py run \\
        --domain wiki --model e5 --line latin \\
        --out input-preprocessing/runs/wiki_e5_latin.json

    # baseline можно переиспользовать из готовых ранжировок (без GPU):
    python input-preprocessing/code/run_preprocessing.py run \\
        --domain wiki --model e5 --line baseline \\
        --reuse-run results/runs_dense_e5.json \\
        --out input-preprocessing/runs/wiki_e5_baseline.json

    # сравнение линий против baseline
    python input-preprocessing/code/run_preprocessing.py compare \\
        --domain wiki --model e5 \\
        --runs-dir input-preprocessing/runs \\
        --out input-preprocessing/runs/wiki_e5_compare.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Set, Tuple

import sys

import numpy as np

# Соседние модули (input_preprocess) + пакет репозитория (src.*).
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from src.retrieval.dense import DenseIndex
from src.eval import metrics
from src.eval.run_dense import MODELS, MODEL_EXTRA
from input_preprocess import (
    LINES, build_transform, collect_tokens, needs_stemmer,
)


# ── загрузка доменов ──────────────────────────────────────────────────────────
# Возвращаем единый кортеж (corpus_pairs, qmap, qrels, cats) для обоих доменов.

def _load_wiki() -> Tuple[List[Tuple[str, str]], Dict[str, str],
                          Dict[str, Set[str]], Dict[str, str]]:
    from src.queries import dataset
    corpus = dataset.load_corpus(os.path.join(ROOT, "data/corpus/corpus.jsonl"))
    queries = dataset.load_queries(os.path.join(ROOT, "data/queries/queries.jsonl"))
    qmap = dataset.queries_as_map(queries)
    qrels = dataset.qrels_from_queries(queries)
    cats = dataset.categories_from_queries(queries)
    return corpus, qmap, qrels, cats


def _load_akorda() -> Tuple[List[Tuple[str, str]], Dict[str, str],
                            Dict[str, Set[str]], Dict[str, str]]:
    d = os.path.join(ROOT, "data", "akorda")
    corpus: List[Tuple[str, str]] = []
    with open(os.path.join(d, "passages.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            corpus.append((r["id"], r["text"]))
    qmap, cats = {}, {}
    with open(os.path.join(d, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qmap[r["query_id"]] = r["query"]
            cats[r["query_id"]] = r["type"]
    qrels: Dict[str, Set[str]] = {}
    with open(os.path.join(d, "qrels.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qrels.setdefault(r["query_id"], set()).add(r["passage_id"])
    return corpus, qmap, qrels, cats


def load_domain(domain: str):
    if domain == "wiki":
        return _load_wiki()
    if domain == "akorda":
        return _load_akorda()
    raise ValueError(f"неизвестный домен: {domain}. Доступны: wiki | akorda")


# ── стеммер с фолбэком на cache-only (если API недоступен) ────────────────────

def _make_stemmer(corpus_texts, query_texts):
    """Готовит казахский стеммер: прогрев кэша по всем токенам.
    Если API стеммера недоступен — переключаемся на cache-only (как run_akorda).
    """
    from src.preprocess.stemmer import get_stemmer
    stemmer = get_stemmer("kazakh")
    toks = collect_tokens(list(corpus_texts) + list(query_texts))
    print(f"Прогрев стеммера: {len(toks)} уникальных токенов…", flush=True)
    try:
        stemmer.warm(toks)
    except Exception as e:  # pragma: no cover - сеть
        print(f"  API стеммера недоступен ({type(e).__name__}): cache-only mode "
              f"(некэшированные токены → identity).", flush=True)
        cache = stemmer._cache

        class _CacheOnly:
            name = "kazakh-cached"

            def stem(self, token: str) -> str:
                return cache.get(token, token)

        stemmer = _CacheOnly()
    return stemmer


# ── метрики ───────────────────────────────────────────────────────────────────

def _subset(qrels: Dict[str, Set[str]], qids: Set[str]) -> Dict[str, Set[str]]:
    return {q: r for q, r in qrels.items() if q in qids}


def _metrics_block(run_result, qrels, cats):
    overall = metrics.evaluate_run(run_result, qrels,
                                   metrics=("recall", "mrr", "ndcg"), ks=(1, 5, 10))
    by_cat = {}
    for cat in sorted(set(cats.values())):
        qids = {q for q, c in cats.items() if c == cat}
        by_cat[cat] = metrics.evaluate_run(run_result, _subset(qrels, qids),
                                           metrics=("recall", "mrr", "ndcg"), ks=(1, 5, 10))
    return overall, by_cat


# ── эмбеддинг-кэш (раздельный на линию) ───────────────────────────────────────

def _emb_prefix(domain: str, model_key: str, line: str) -> str:
    safe_model = model_key.replace("/", "_")
    d = os.path.join(ROOT, "input-preprocessing", "emb")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{safe_model}_{domain}_{line}")


def _load_emb_cache(prefix: str, expected_ids):
    if not os.path.exists(prefix + ".npy") or not os.path.exists(prefix + ".ids.json"):
        return None
    ids = json.load(open(prefix + ".ids.json", encoding="utf-8"))
    if ids != list(expected_ids):
        return None
    return np.load(prefix + ".npy")


def _save_emb_cache(prefix: str, doc_ids, matrix):
    np.save(prefix + ".npy", matrix)
    json.dump(list(doc_ids), open(prefix + ".ids.json", "w", encoding="utf-8"))


# ── прогон одной линии ────────────────────────────────────────────────────────

def run_line(domain: str, model_key: str, line: str,
             top_k: int = 100, batch_size: int = 64,
             max_seq_len: int = 512, reuse_run: str = None) -> Dict:
    if line not in LINES:
        raise ValueError(f"линия {line!r} не из {LINES}")
    corpus, qmap, qrels, cats = load_domain(domain)

    # baseline можно переиспользовать из готовых ранжировок (без GPU)
    if reuse_run:
        with open(reuse_run, encoding="utf-8") as f:
            payload = json.load(f)
        run_result = payload["run"] if "run" in payload else payload
        # на всякий случай оставляем только запросы текущего домена
        run_result = {q: r for q, r in run_result.items() if q in qrels}
        overall, by_cat = _metrics_block(run_result, qrels, cats)
        return _result(domain, model_key, line, corpus, qmap,
                       overall, by_cat, run_result, reused=reuse_run)

    stemmer = None
    if needs_stemmer(line):
        stemmer = _make_stemmer((t for _, t in corpus), qmap.values())
    transform = build_transform(line, stemmer)

    corpus_t = [(doc_id, transform(text)) for doc_id, text in corpus]
    qmap_t = {qid: transform(text) for qid, text in qmap.items()}

    hf_id, qpref, dpref = MODELS.get(model_key, (model_key, "", ""))
    extra = MODEL_EXTRA.get(model_key, {})
    index = DenseIndex(model_name=hf_id, query_prefix=qpref, doc_prefix=dpref,
                       batch_size=batch_size, max_seq_len=max_seq_len,
                       query_task=extra.get("query_task"),
                       doc_task=extra.get("doc_task"),
                       trust_remote_code=extra.get("trust_remote_code", False))

    doc_ids = [d for d, _ in corpus_t]
    prefix = _emb_prefix(domain, model_key, line)
    cached = _load_emb_cache(prefix, doc_ids)
    if cached is not None:
        print(f"Эмбеддинги из кэша: {prefix}.npy", flush=True)
        index.set_embeddings(doc_ids, cached)
    else:
        print(f"Кодирование {len(corpus_t)} пассажей [{model_key}/{domain}/{line}]…",
              flush=True)
        index.index(corpus_t)
        _save_emb_cache(prefix, index.doc_ids, index.matrix)

    run_result = index.run(qmap_t, top_k=top_k)
    overall, by_cat = _metrics_block(run_result, qrels, cats)
    return _result(domain, model_key, line, corpus, qmap,
                   overall, by_cat, run_result, reused=None)


def _result(domain, model_key, line, corpus, qmap, overall, by_cat,
            run_result, reused):
    return {
        "system": f"preprocess:{model_key}:{line}",
        "domain": domain, "model": model_key, "line": line,
        "reused_run": reused,
        "n_passages": len(corpus), "n_queries": len(qmap),
        "overall": overall, "by_category": by_cat, "run": run_result,
    }


# ── сравнение линий против baseline ───────────────────────────────────────────

def _load_run(path: str) -> Dict[str, List[str]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["run"] if "run" in payload else payload


def compare(domain: str, model_key: str, runs_dir: str,
            resamples: int = 10000) -> Dict:
    """Δ nDCG@10 каждой линии против baseline + paired bootstrap, по срезам."""
    _, _, qrels, cats = load_domain(domain)
    safe_model = model_key.replace("/", "_")

    def _path(line):
        return os.path.join(runs_dir, f"{domain}_{safe_model}_{line}.json")

    base_run = _load_run(_path("baseline"))

    groups = {"ALL": set(qrels)}
    for c in sorted(set(cats.values())):
        groups[c] = {q for q, cc in cats.items() if cc == c}

    out = {"domain": domain, "model": model_key, "metric": "ndcg@10",
           "resamples": resamples, "baseline": {}, "lines": {}}

    # baseline-числа для контроля «бьются ли» (Δ = линия − baseline)
    for g, qids in groups.items():
        sub = _subset(qrels, qids)
        out["baseline"][g] = metrics.evaluate_run(
            base_run, sub, metrics=("ndcg",), ks=(10,))["ndcg@10"]

    for line in ("stem", "latin", "stem_latin"):
        p = _path(line)
        if not os.path.exists(p):
            continue
        line_run = _load_run(p)
        out["lines"][line] = {}
        for g, qids in groups.items():
            sub = _subset(qrels, qids)
            base_v = out["baseline"][g]
            line_v = metrics.evaluate_run(
                line_run, sub, metrics=("ndcg",), ks=(10,))["ndcg@10"]
            delta, pval, _ = metrics.paired_bootstrap(
                base_run, line_run, sub, metric="ndcg", k=10,
                n_resamples=resamples)
            out["lines"][line][g] = {
                "baseline": base_v, "line": line_v,
                "delta": delta, "p": pval, "sig": pval < 0.05,
            }
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def _save(path: str, obj: Dict, drop_run: bool = False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if drop_run:
        obj = {k: v for k, v in obj.items() if k != "run"}
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved → {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="input-preprocessing: 4 линии × модель × домен")
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("run", help="прогнать одну линию")
    rp.add_argument("--domain", required=True, choices=["wiki", "akorda"])
    rp.add_argument("--model", required=True, help="e5 | qwen3-0.6b | HF id")
    rp.add_argument("--line", required=True, choices=list(LINES))
    rp.add_argument("--out", required=True)
    rp.add_argument("--top-k", type=int, default=100)
    rp.add_argument("--batch-size", type=int, default=64)
    rp.add_argument("--max-seq-len", type=int, default=512)
    rp.add_argument("--reuse-run", default=None,
                    help="взять готовые ранжировки (для baseline без GPU)")

    cp = sub.add_parser("compare", help="Δ линий против baseline + bootstrap")
    cp.add_argument("--domain", required=True, choices=["wiki", "akorda"])
    cp.add_argument("--model", required=True)
    cp.add_argument("--runs-dir", default="input-preprocessing/runs")
    cp.add_argument("--out", required=True)
    cp.add_argument("--resamples", type=int, default=10000)

    args = ap.parse_args()

    if args.cmd == "run":
        result = run_line(args.domain, args.model, args.line,
                          top_k=args.top_k, batch_size=args.batch_size,
                          max_seq_len=args.max_seq_len, reuse_run=args.reuse_run)
        ov = result["overall"]
        print(f"\n[{args.model}/{args.domain}/{args.line}] "
              f"nDCG@10={ov['ndcg@10']:.4f}  recall@10={ov['recall@10']:.4f}",
              flush=True)
        for cat, m in sorted(result["by_category"].items()):
            print(f"  {cat:16s}: nDCG@10={m['ndcg@10']:.4f}", flush=True)
        _save(args.out, result)  # с per-query run для bootstrap
    elif args.cmd == "compare":
        result = compare(args.domain, args.model, args.runs_dir,
                         resamples=args.resamples)
        print(f"\nΔ nDCG@10 против baseline — {args.model} / {args.domain}")
        print(f"{'линия':<12}{'срез':<16}{'base':>8}{'line':>8}{'Δ':>8}{'p':>8}  знач.")
        print("-" * 64)
        for line, groups in result["lines"].items():
            for g, v in groups.items():
                mark = "да" if v["sig"] else ""
                print(f"{line:<12}{g:<16}{v['baseline']:>8.3f}{v['line']:>8.3f}"
                      f"{v['delta']:>+8.3f}{v['p']:>8.3f}  {mark}")
        _save(args.out, result)


if __name__ == "__main__":
    main()
