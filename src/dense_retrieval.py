"""
Stage 0: Dense retrieval baseline on a BEIR dataset.

What this does, in plain terms:
1. Downloads a BEIR benchmark dataset (corpus + queries + qrels/gold labels).
2. Embeds every document and every query with a sentence-transformer model
   (turns text into vectors that capture meaning, not just keywords).
3. For each query, ranks all documents by cosine similarity to the query vector.
4. Compares the ranking against the gold labels (qrels) and computes standard
   IR metrics: nDCG, Recall, MRR.
5. Saves the raw metric numbers to results/ so they can be reused later
   (e.g. in the final comparison table).

Usage:
    python src/dense_retrieval.py                # defaults: scifact / test
    python src/dense_retrieval.py --dataset fiqa --split dev
"""

import argparse
import json
import os
import pathlib

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RESULTS_DIR = os.path.join(pathlib.Path(__file__).resolve().parent.parent, "results")


def load_dataset(dataset: str, split: str):
    """Download (if needed) and load a BEIR dataset's corpus/queries/qrels."""
    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
        f"{dataset}.zip"
    )
    out_dir = os.path.join(pathlib.Path(__file__).resolve().parent.parent, "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    return corpus, queries, qrels


def run_dense_retrieval(dataset: str, split: str, model_name: str, batch_size: int = 64):
    print(f"Loading dataset='{dataset}' split='{split}' ...")
    corpus, queries, qrels = load_dataset(dataset, split)
    print(f"  corpus documents: {len(corpus)}")
    print(f"  queries: {len(queries)}")

    # === THIS IS WHERE THE TRANSFORMER MODEL LOADS ===
    # `model_name` (default: "sentence-transformers/all-MiniLM-L6-v2") is a
    # sentence-transformers model — under the hood it IS a Transformer: a
    # distilled/compact BERT-style encoder (MiniLM), the same neural network
    # architecture behind BERT/GPT, just much smaller so it runs fast on CPU.
    # `models.SentenceBERT(...)` downloads its pretrained weights from
    # HuggingFace and wraps it so BEIR can call it. `DRES` (Dense Retrieval
    # Exact Search) is BEIR's helper that uses this Transformer to:
    #   1. Run every document's text through the Transformer -> one 384-dim
    #      vector per document (this is "embedding").
    #   2. Run every query's text through the SAME Transformer -> one
    #      384-dim vector per query.
    # No manual encoding loop is needed here — retriever.retrieve() below
    # calls the Transformer internally for both steps.
    print(f"Loading embedding model '{model_name}' ...")
    model = DRES(models.SentenceBERT(model_name), batch_size=batch_size)
    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    # This line is the actual "retrieval": it (a) embeds every doc + query
    # through the Transformer loaded above, then (b) for each query, ranks
    # all documents by cosine similarity between their vectors. High
    # cosine similarity = the Transformer judged them semantically close.
    print("Embedding + retrieving (this is the slow step, CPU-only) ...")
    results = retriever.retrieve(corpus, queries)

    # Compare the Transformer-produced ranking against qrels (the gold/
    # correct relevant documents for each query) and compute standard IR
    # metrics at multiple cutoffs (@1, @3, @5, @10, @100, @1000).
    print("Evaluating against qrels ...")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

    print("\n=== Dense retrieval results ===")
    print("nDCG:", ndcg)
    print("Recall:", recall)
    print("MRR:", mrr)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"dense_{dataset}_{split}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": dataset,
                "split": split,
                "model": model_name,
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
    return results, qrels, corpus, queries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dense retrieval on a BEIR dataset.")
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name (default: scifact)")
    parser.add_argument("--split", default="test", help="Dataset split: train/dev/test (default: test)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    run_dense_retrieval(args.dataset, args.split, args.model, args.batch_size)
