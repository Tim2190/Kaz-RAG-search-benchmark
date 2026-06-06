"""
RAG-бенчмарк галлюцинаций: меняет ли качество поиска (BM25 vs BM25+стеммер)
долю выдумываний у LLM на казахском.

Цепочка на каждый вопрос:
  поиск top-k пассажей  ->  контекст  ->  LLM (IBM Granite / Google Gemma)
  ->  ответ классифицируется: correct / hallucination / abstain.

Гипотеза: без стеммера поиск чаще не находит нужную статью -> модель получает
левый контекст -> выдумывает. Со стеммером контекст точный -> верный ответ.

ЗАПУСК (Colab, нужен transformers+torch и доступ к весам модели):
    python -m src.eval.run_rag --llm granite --top-k 3 --out results/rag_granite.json
    python -m src.eval.run_rag --llm gemma   --top-k 3 --out results/rag_gemma.json
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from ..preprocess.stemmer import get_stemmer
from ..retrieval.bm25 import BM25Index, default_analyzer
from ..queries import dataset
from ..rag import prompt as rag_prompt
from ..rag import score as rag_score
from ..rag.llm import EchoLLM, HFTransformersLLM


def run_system(corpus_pairs, queries, stemmer_name: str, llm, top_k: int = 3,
               max_passages: int = 5) -> Dict:
    """Прогон одной retrieval-конфигурации сквозь LLM, со скорингом."""
    text_by_id = {doc_id: text for doc_id, text in corpus_pairs}
    stemmer = get_stemmer(stemmer_name)
    index = BM25Index(analyzer=default_analyzer(stemmer)).index(corpus_pairs)

    labels: List[str] = []
    cats: List[str] = []
    hits: List[bool] = []
    per_query = []
    for q in queries:
        ranked = index.run({q["query_id"]: q["text"]}, top_k=top_k)[q["query_id"]]
        passages = [text_by_id[d] for d in ranked if d in text_by_id]
        gold_docs = set(q.get("gold_doc_ids", []))
        hit = bool(gold_docs & set(ranked))

        pr = rag_prompt.build_prompt(q["text"], passages, max_passages=max_passages)
        answer = llm.generate(pr)
        label = rag_score.classify(answer, q.get("answer", ""))

        labels.append(label)
        cats.append(q.get("category", "all"))
        hits.append(hit)
        per_query.append({"query_id": q["query_id"], "retrieval_hit": hit,
                          "label": label, "answer": answer})

    overall = rag_score.aggregate(labels)
    overall["retrieval_hit_rate"] = sum(hits) / (len(hits) or 1)

    by_cat = {}
    for cat in sorted(set(cats)):
        idx = [i for i, c in enumerate(cats) if c == cat]
        by_cat[cat] = rag_score.aggregate([labels[i] for i in idx])
        by_cat[cat]["retrieval_hit_rate"] = sum(hits[i] for i in idx) / (len(idx) or 1)

    # перекрёстная таблица: что делает модель, когда поиск ПРОМАХНУЛСЯ
    miss_labels = [labels[i] for i in range(len(labels)) if not hits[i]]
    miss = rag_score.aggregate(miss_labels) if miss_labels else None

    return {"stemmer": stemmer_name, "n_queries": len(queries),
            "overall": overall, "by_category": by_cat,
            "on_retrieval_miss": miss, "per_query": per_query}


def _fmt(res: Dict) -> str:
    o = res["overall"]
    return (f"[стеммер={res['stemmer']}] "
            f"accuracy={o['accuracy']:.3f}  "
            f"галлюцинации={o['hallucination_rate']:.3f}  "
            f"abstain={o['abstention_rate']:.3f}  "
            f"поиск-попал={o['retrieval_hit_rate']:.3f}")


def _mcnemar_accuracy(pq_identity: list, pq_kazakh: list):
    """Exact McNemar two-sided p-value on accuracy (correct vs not) between stemmers."""
    from math import comb

    def correct_map(per_query):
        return {pq["query_id"]: (pq["label"] == "correct") for pq in per_query}

    ci, ck = correct_map(pq_identity), correct_map(pq_kazakh)
    ids = set(ci) & set(ck)
    if not ids:
        return None
    b = sum(1 for i in ids if ci[i] and not ck[i])   # lost (was correct, now wrong)
    c = sum(1 for i in ids if not ci[i] and ck[i])    # gained (was wrong, now correct)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(p, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG-бенчмарк галлюцинаций (До/После стеммера)")
    ap.add_argument("--llm", default="granite", help="granite | gemma | echo | HF id")
    ap.add_argument("--corpus", default="data/corpus/corpus.jsonl")
    ap.add_argument("--queries", default="data/queries/queries.jsonl")
    ap.add_argument("--stemmers", nargs="+", default=["identity", "kazakh"])
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--show", type=int, default=8, help="сколько примеров ответов печатать")
    ap.add_argument("--4bit", dest="four_bit", action="store_true",
                    help="4-битная загрузка (для 7-9B моделей на T4)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    corpus_pairs = dataset.load_corpus(args.corpus)
    queries = dataset.load_queries(args.queries)
    llm = EchoLLM() if args.llm == "echo" else HFTransformersLLM(
        args.llm, max_new_tokens=args.max_new_tokens, load_in_4bit=args.four_bit)

    gold_by_id = {q["query_id"]: q.get("answer", "") for q in queries}
    results = {}
    for st in args.stemmers:
        print(f"\n=== RAG прогон: стеммер={st}, LLM={args.llm} ===")
        res = run_system(corpus_pairs, queries, st, llm, top_k=args.top_k)
        print(_fmt(res))
        # примеры для диагностики: что модель реально ответила
        print(f"  --- примеры ответов (стеммер={st}) ---")
        for pq in res["per_query"][:args.show]:
            print(f"  [{pq['label'][:4]}|hit={int(pq['retrieval_hit'])}] "
                  f"эталон='{gold_by_id.get(pq['query_id'])}' "
                  f"ответ='{pq['answer'][:90]}'")
        results[st] = res

    if "identity" in results and "kazakh" in results:
        b = results["identity"]["overall"]
        a = results["kazakh"]["overall"]
        dh = a["hallucination_rate"] - b["hallucination_rate"]
        da = a["accuracy"] - b["accuracy"]
        print(f"\n>>> ИТОГ (LLM={args.llm}): "
              f"галлюцинации {b['hallucination_rate']:.3f} → {a['hallucination_rate']:.3f} "
              f"({dh:+.3f}); accuracy {b['accuracy']:.3f} → {a['accuracy']:.3f} ({da:+.3f})")

        # McNemar paired significance test on accuracy (identity vs kazakh)
        p_val = _mcnemar_accuracy(results["identity"]["per_query"],
                                  results["kazakh"]["per_query"])
        if p_val is not None:
            verdict = "ЗНАЧИМО (p<0.05) ✅" if p_val < 0.05 else "не значимо ✗"
            print(f">>> McNemar p={p_val:.4f} — {verdict}")

    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        # сохраняем ПОЛНОСТЬЮ, включая сырые ответы — чтобы пере-оценивать офлайн
        out = dict(results)
        out["llm"] = args.llm
        json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\nСохранено → {args.out}")


if __name__ == "__main__":
    main()
