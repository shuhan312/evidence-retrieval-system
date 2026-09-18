"""
Stage 1: BM25 baseline (lightweight, keyword-matching retrieval).

What this does, in plain terms:
1. Loads the same BEIR dataset used by dense_retrieval.py.
2. Tokenizes every document (title + text) into lowercase words.
3. For each query, scores every document with BM25 (a classic term-frequency
   scoring formula — no embeddings, no neural network, purely word overlap
   and word rarity).
4. Keeps the top-100 highest-scoring documents per query, in the same
   {qid: {doc_id: score}} format BEIR's evaluator expects, so results are
   directly comparable to dense retrieval / reranking.
5. Evaluates against qrels (nDCG/Recall/MRR) and saves to results/.

Honest framing (see project README): this is `rank_bm25`, a pure-Python
implementation that scans the full corpus per query. It is a lightweight
baseline, not a production inverted-index BM25 — see latency.py for why we
do not report its query latency alongside the other methods.

Usage:
    python src/bm25_baseline.py                # defaults: scifact / test
    python src/bm25_baseline.py --dataset fiqa --split dev
"""

import argparse
import json
import os
import pathlib

from rank_bm25 import BM25Okapi
from beir.retrieval.evaluation import EvaluateRetrieval

from dense_retrieval import load_dataset

RESULTS_DIR = os.path.join(pathlib.Path(__file__).resolve().parent.parent, "results")


def run_bm25(dataset: str, split: str, top_k: int = 100):
    print(f"Loading dataset='{dataset}' split='{split}' ...")
    corpus, queries, qrels = load_dataset(dataset, split)
    print(f"  corpus documents: {len(corpus)}")
    print(f"  queries: {len(queries)}")

    # === NO TRANSFORMER ANYWHERE IN THIS FILE ===
    # `.lower().split()` is the entire "understanding" step here: it just
    # splits text into a plain list of lowercase words (e.g. "Credit rating"
    # -> ["credit", "rating"]). No neural network, no embeddings — this is
    # the deliberate contrast with dense_retrieval.py, which route both
    # queries and documents through a Transformer. BM25 only ever looks at
    # exact word overlap.
    print("Tokenizing corpus ...")
    doc_ids = list(corpus.keys())
    tokenized_corpus = [
        (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).lower().split()
        for d in doc_ids
    ]
    # BM25Okapi precomputes word-frequency statistics over the whole corpus
    # (how often each word appears overall, and in how many documents) so
    # it can later score "word overlap, weighted by how rare/informative
    # each word is" for any query.
    bm25 = BM25Okapi(tokenized_corpus)

    print("Scoring queries with BM25 (lightweight rank_bm25, full-corpus scan per query) ...")
    results_bm25 = {}
    for qid, qtext in queries.items():
        # get_scores() scans every single document in the corpus (this is
        # the "not a production inverted index" caveat from the docstring —
        # a real search engine would skip documents containing none of the
        # query words instead of scoring all of them every time) and returns
        # one BM25 score per document for this query.
        scores = bm25.get_scores(qtext.lower().split())
        # Keep only the top_k highest-scoring documents (default 100),
        # matching the same {doc_id: score} shape dense_retrieval.py
        # produces so both can be evaluated/compared with identical code.
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results_bm25[qid] = {doc_ids[i]: float(scores[i]) for i in top}

    print("Evaluating against qrels ...")
    k_values = [1, 3, 5, 10, 100, 1000]
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results_bm25, k_values)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results_bm25, k_values, metric="mrr")

    print("\n=== BM25 (lightweight baseline) results ===")
    print("nDCG:", ndcg)
    print("Recall:", recall)
    print("MRR:", mrr)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"bm25_{dataset}_{split}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": dataset,
                "split": split,
                "method": "BM25 (lightweight baseline, rank_bm25, not an inverted index)",
                "top_k": top_k,
                "num_docs": len(corpus),
                "num_queries": len(queries),
                "ndcg": ndcg,
                "map": _map,
                "recall": recall,
                "precision": precision,
                "mrr": mrr,
            },
            f,
            indent=2,
        )
    print(f"\nSaved metrics to {out_path}")
    return results_bm25, qrels, corpus, queries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BM25 baseline on a BEIR dataset.")
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name (default: scifact)")
    parser.add_argument("--split", default="test", help="Dataset split: train/dev/test (default: test)")
    parser.add_argument("--top-k", type=int, default=100, help="Number of top documents to keep per query")
    args = parser.parse_args()

    run_bm25(args.dataset, args.split, args.top_k)
