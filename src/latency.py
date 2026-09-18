"""
Stage 1: Query latency measurement — Dense vs Dense + Cross-Encoder.

What this does, in plain terms:
For a sample of queries, time (in seconds) how long a SINGLE query takes to
run through:
  (a) dense retrieval alone (embed the query, compare to all doc vectors)
  (b) dense retrieval + cross-encoder reranking of its top-100 candidates
Taking the median across queries (median, not mean, so a couple of unusually
slow/fast queries don't skew the number) gives a fair "typical cost" figure
for each stage, and the GAP between (a) and (b) is exactly what reranking
costs you per query.

Why BM25 latency is intentionally NOT measured here: `rank_bm25` is a pure
Python implementation that linearly scans the whole corpus on every query —
it is not a production inverted index (e.g. Elasticsearch/Lucene). Timing it
against neural methods would produce a misleading "BM25 is slower than a
transformer" result caused by an unoptimized library, not by the BM25
algorithm itself. See the project README for the same caveat.

Usage:
    python src/latency.py                     # defaults: scifact / test, n=30
    python src/latency.py --dataset fiqa --split dev --n 30
"""

import argparse
import json
import os
import pathlib
import statistics
import time

from sentence_transformers import CrossEncoder
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from dense_retrieval import load_dataset, DEFAULT_MODEL
from rerank import DEFAULT_CROSS_ENCODER

RESULTS_DIR = os.path.join(pathlib.Path(__file__).resolve().parent.parent, "results")


def median_time(fn, items, n):
    """Run fn(item) for the first n items, return the median wall-clock time in seconds."""
    times = []
    for item in items[:n]:
        t0 = time.perf_counter()
        fn(item)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times


def run_latency(dataset: str, split: str, n: int, dense_model: str, ce_model: str, top_k: int = 100):
    print(f"Loading dataset='{dataset}' split='{split}' ...")
    corpus, queries, _qrels = load_dataset(dataset, split)
    doc_ids = list(corpus.keys())
    doc_texts = [
        (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids
    ]
    query_ids = list(queries.keys())[:n]
    print(f"Timing on {len(query_ids)} queries (median reported)")

    # Load the same embedding Transformer used in dense_retrieval.py.
    print(f"Loading dense model '{dense_model}' and pre-embedding the corpus ...")
    dense = DRES(models.SentenceBERT(dense_model), batch_size=64)
    # Pre-embed the WHOLE corpus once, up front, outside the timed loop below
    # — this mirrors a real production setup, where the document index is
    # built ahead of time offline, and only the query is embedded live at
    # request time. If we re-embedded all 5183 documents inside every timed
    # query, we'd be measuring index-building time, not query time.
    corpus_embeddings = dense.model.encode_corpus(
        [{"title": corpus[d].get("title", ""), "text": corpus[d].get("text", "")} for d in doc_ids],
        batch_size=64,
        show_progress_bar=False,
        convert_to_tensor=True,
    )

    def dense_query(qid):
        # This is the only Transformer work that happens PER QUERY in dense
        # retrieval: embed just this one query text into a vector, then
        # compare it (via matrix multiply = batched cosine-similarity-like
        # score) against the already-precomputed corpus vectors. This is why
        # dense queries are fast (~20ms): one short text through the
        # Transformer, then simple vector math.
        q_emb = dense.model.encode_queries([queries[qid]], convert_to_tensor=True, show_progress_bar=False)
        scores = (q_emb @ corpus_embeddings.T).squeeze(0)
        top = scores.topk(k=min(top_k, len(doc_ids)))
        return [(doc_ids[i], float(top.values[j])) for j, i in enumerate(top.indices.tolist())]

    print("Timing dense-only retrieval per query ...")
    dense_median, dense_times = median_time(dense_query, query_ids, n)

    # Load the cross-encoder Transformer (same model as rerank.py).
    print(f"Loading cross-encoder '{ce_model}' ...")
    ce = CrossEncoder(ce_model)

    def dense_plus_rerank_query(qid):
        # First redo the (fast) dense step to get this query's top-100
        # candidates, then run the (slow) cross-encoder over all 100
        # (query, document) pairs. Timing this whole function, and comparing
        # it to dense_query()'s time above, isolates exactly how much extra
        # latency the reranking Transformer adds per query.
        candidates = dense_query(qid)
        pairs = [(queries[qid], doc_texts[doc_ids.index(doc_id)]) for doc_id, _ in candidates]
        ce.predict(pairs, show_progress_bar=False)

    print("Timing dense + cross-encoder rerank per query (this is slower) ...")
    reranked_median, reranked_times = median_time(dense_plus_rerank_query, query_ids, n)

    rerank_overhead = reranked_median - dense_median

    print("\n=== Median per-query latency ===")
    print(f"Dense only:            {dense_median * 1000:.1f} ms")
    print(f"Dense + Cross-Encoder: {reranked_median * 1000:.1f} ms")
    print(f"Reranker overhead:     {rerank_overhead * 1000:.1f} ms  (+{rerank_overhead / dense_median:.1%})")
    print("BM25 (lightweight rank_bm25): not measured — see docstring / README for why")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"latency_{dataset}_{split}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": dataset,
                "split": split,
                "num_queries_timed": len(query_ids),
                "dense_median_seconds": dense_median,
                "dense_plus_rerank_median_seconds": reranked_median,
                "rerank_overhead_seconds": rerank_overhead,
                "bm25_latency": "not measured: rank_bm25 is a lightweight full-scan implementation, "
                "not a production inverted index — timing it would misrepresent the BM25 algorithm",
            },
            f,
            indent=2,
        )
    print(f"\nSaved latency results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure Dense vs Dense+Cross-Encoder query latency.")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=30, help="Number of queries to time")
    parser.add_argument("--dense-model", default=DEFAULT_MODEL)
    parser.add_argument("--ce-model", default=DEFAULT_CROSS_ENCODER)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    run_latency(args.dataset, args.split, args.n, args.dense_model, args.ce_model, args.top_k)
