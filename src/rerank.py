"""
Stage 1: Cross-Encoder reranking on top of dense retrieval.

What this does, in plain terms:
1. Runs dense retrieval fresh (imports and calls dense_retrieval.run_dense_retrieval),
   getting back, for every query, its ranked list of documents with scores —
   this is the same "coarse" retrieval from stage 0.
2. For each query, takes the top-100 documents from that coarse ranking.
3. Feeds each (query, document) pair TOGETHER into a cross-encoder model,
   which reads both texts jointly and outputs one fine-grained relevance
   score per pair — this is much more accurate than comparing two
   separately-computed embeddings, but far more expensive, which is why it
   only runs on the top-100 shortlist instead of the whole corpus.
4. Re-sorts those 100 documents by the new cross-encoder scores.
5. Evaluates the reranked list against qrels and saves metrics, alongside
   the original (pre-rerank) dense metrics for direct before/after
   comparison.

Usage:
    python src/rerank.py                # defaults: scifact / test
    python src/rerank.py --dataset fiqa --split dev
"""

import argparse
import json
import os
import pathlib

from sentence_transformers import CrossEncoder
from beir.retrieval.evaluation import EvaluateRetrieval

from dense_retrieval import run_dense_retrieval, DEFAULT_MODEL

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RESULTS_DIR = os.path.join(pathlib.Path(__file__).resolve().parent.parent, "results")


def rerank(base_results, corpus, queries, ce, top_k: int = 100):
    """Rerank the top_k candidates of each query using a cross-encoder."""
    reranked = {}
    for qid, doc_scores in base_results.items():
        # From this query's dense results, keep only the top_k best-scoring
        # doc_ids (default 100) — these are the candidates worth the extra
        # compute cost of reranking; everything ranked below top_k is left
        # untouched from the dense stage.
        cand_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)[:top_k]
        # Build (query_text, document_text) PAIRS — this is the key
        # difference from dense_retrieval.py. Dense encodes the query and
        # each document SEPARATELY into two vectors and compares them
        # afterwards. Here, query text and document text go into the
        # Transformer TOGETHER, in the same input, so it can directly
        # cross-attend between every query word and every document word.
        pairs = [
            (
                queries[qid],
                (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip(),
            )
            for d in cand_ids
        ]
        # === THIS IS WHERE THE TRANSFORMER RUNS (2nd time in the project) ===
        # `ce` (CrossEncoder, loaded below) is also a Transformer — a
        # MiniLM-based BERT-style encoder, like the dense model, but used
        # differently: instead of producing one vector per text, it reads a
        # (query, document) pair jointly and outputs a single relevance
        # score for that specific pair. This joint reading is why it is
        # more accurate but much slower (see latency.py) than comparing
        # two independently-computed embeddings.
        ce_scores = ce.predict(pairs)
        reranked[qid] = {cand_ids[i]: float(ce_scores[i]) for i in range(len(cand_ids))}
    return reranked


def run_rerank(
    dataset: str,
    split: str,
    dense_model: str = DEFAULT_MODEL,
    ce_model: str = DEFAULT_CROSS_ENCODER,
    top_k: int = 100,
):
    # Step A reuses dense_retrieval.py's function wholesale — this calls the
    # FIRST Transformer (the sentence-embedding model) to produce the coarse
    # ranking over the entire corpus, exactly like stage 0.
    print("=== Step A: dense retrieval (coarse ranking) ===")
    dense_results, qrels, corpus, queries = run_dense_retrieval(dataset, split, dense_model)

    # Step B loads the SECOND Transformer used in this project: a
    # cross-encoder (default: cross-encoder/ms-marco-MiniLM-L-6-v2), also a
    # MiniLM Transformer, but fine-tuned specifically to score
    # (query, document) relevance jointly rather than to produce embeddings.
    print(f"\n=== Step B: cross-encoder reranking of top-{top_k} candidates ===")
    print(f"Loading cross-encoder '{ce_model}' ...")
    ce = CrossEncoder(ce_model)

    print("Reranking (this reads every query+document pair jointly, slower than dense) ...")
    reranked_results = rerank(dense_results, corpus, queries, ce, top_k=top_k)

    print("Evaluating reranked results against qrels ...")
    k_values = [1, 3, 5, 10, 100, 1000]
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, reranked_results, k_values)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, reranked_results, k_values, metric="mrr")

    print("\n=== Dense + Cross-Encoder reranked results ===")
    print("nDCG:", ndcg)
    print("Recall:", recall)
    print("MRR:", mrr)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"reranked_{dataset}_{split}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": dataset,
                "split": split,
                "method": "Dense + Cross-Encoder reranking",
                "dense_model": dense_model,
                "cross_encoder_model": ce_model,
                "rerank_top_k": top_k,
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
    return reranked_results, dense_results, qrels, corpus, queries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dense retrieval + cross-encoder reranking on a BEIR dataset.")
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name (default: scifact)")
    parser.add_argument("--split", default="test", help="Dataset split: train/dev/test (default: test)")
    parser.add_argument("--dense-model", default=DEFAULT_MODEL)
    parser.add_argument("--ce-model", default=DEFAULT_CROSS_ENCODER)
    parser.add_argument("--top-k", type=int, default=100, help="Number of dense candidates to rerank")
    args = parser.parse_args()

    run_rerank(args.dataset, args.split, args.dense_model, args.ce_model, args.top_k)
