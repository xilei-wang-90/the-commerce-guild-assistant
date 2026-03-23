"""Evaluate retrieval quality against a labelled test set.

Loads a test set CSV (factoid, conceptual, or messy), retrieves the top-k
results for each question from a ChromaDB collection, and computes
evaluation metrics (e.g. Hit Rate@K).

Supports a ``--rerank`` flag to evaluate cross-encoder reranking.  When
enabled, the script fetches 10 results per query, reranks them, and
reports NDCG@K both before and after reranking for comparison.

If ``--testset`` or ``--collection`` is omitted the script prompts
interactively.

Requires Ollama running locally with ``all-minilm`` (embedding model).

Run with:

    python3 scripts/run_retrieval_eval.py --testset factoid --collection summary --k 5
    python3 scripts/run_retrieval_eval.py --testset factoid --collection summary --k 5 --rerank
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so tests.benchmark is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guild_assistant.rag.reranker import Reranker
from guild_assistant.rag.retriever import Retriever
from guild_assistant.utils.model_adapter import (
    CrossEncoderRerankerAdapter,
    OllamaEmbeddingAdapter,
)
from tests.benchmark.metrics import METRIC_REGISTRY
from tests.benchmark.retrieval_evaluator import RetrievalEvaluator
from tests.benchmark.testset_loader import load_testset

_ROOT = Path(__file__).parent.parent

TESTSET_DIR = _ROOT / "data" / "test-data"
DB_PATH = _ROOT / "sandrock_db"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "all-minilm"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_RETRIEVE_N = 10

VALID_TESTSETS = ("factoid", "conceptual", "messy")

_COLLECTION_MAP: dict[str, str] = {
    "summary": "sandrock_wiki_summary",
    "section-reverse-hyde": "sandrock_wiki_section_reverse_hyde",
}


def _prompt_choice(label: str, valid: tuple[str, ...] | list[str]) -> str:
    """Interactively ask the user to pick from *valid* choices."""
    choices = " / ".join(valid)
    while True:
        try:
            value = input(f"Choose {label} [{choices}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)
        if value in valid:
            return value
        print(f"Invalid choice '{value}'. Please enter one of: {choices}")


def main() -> None:
    available_metrics = ", ".join(sorted(METRIC_REGISTRY))

    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality against a labelled test set.",
    )
    parser.add_argument(
        "--testset",
        choices=VALID_TESTSETS,
        default=None,
        help="Which test set to evaluate: factoid, conceptual, or messy. Prompted if omitted.",
    )
    parser.add_argument(
        "--collection",
        choices=list(_COLLECTION_MAP),
        default=None,
        help="Which collection to test: summary or section-reverse-hyde. Prompted if omitted.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top results to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["hit_rate"],
        help=f"Metrics to compute (default: hit_rate). Available: {available_metrics}.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help="Enable rerank mode: fetch 10 results, rerank with a cross-encoder, and compare NDCG@K before/after.",
    )
    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_URL,
        help=f"Ollama server base URL (default: {OLLAMA_URL}).",
    )
    args = parser.parse_args()

    testset = args.testset if args.testset else _prompt_choice("test set", VALID_TESTSETS)
    collection = args.collection if args.collection else _prompt_choice(
        "collection", list(_COLLECTION_MAP),
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    csv_path = TESTSET_DIR / f"{testset}.csv"
    test_cases = load_testset(csv_path)
    if not test_cases:
        log.error("No test cases loaded from %s", csv_path)
        return

    collection_name = _COLLECTION_MAP[collection]
    embedding_adapter = OllamaEmbeddingAdapter(
        model_name=EMBEDDING_MODEL,
        base_url=args.ollama_url,
    )

    if args.rerank:
        # Rerank mode: fetch more results, then rerank down to k.
        retriever = Retriever(
            db_path=DB_PATH,
            collection_name=collection_name,
            embedding_adapter=embedding_adapter,
            n_results=RERANK_RETRIEVE_N,
        )
        reranker_adapter = CrossEncoderRerankerAdapter(model_name=RERANKER_MODEL)
        reranker = Reranker(adapter=reranker_adapter, top_n=args.k)
        evaluator = RetrievalEvaluator(
            retriever=retriever,
            collection_name=collection_name,
            k=args.k,
            reranker=reranker,
            retrieve_n=RERANK_RETRIEVE_N,
        )

        # Force ndcg into metrics if not already present.
        eval_metrics = list(args.metrics)
        if "ndcg" not in eval_metrics:
            eval_metrics.append("ndcg")

        log.info(
            "Evaluating %d %s questions against %s (k=%d, rerank from %d)",
            len(test_cases),
            testset,
            collection,
            args.k,
            RERANK_RETRIEVE_N,
        )

        before_scores, after_scores, before_results, after_results = evaluator.run_with_rerank(
            test_cases=test_cases, metrics=eval_metrics,
        )

        # Write rerank comparison CSV.
        metrics_tag = "_".join(sorted(eval_metrics))
        results_filename = f"eval_{testset}_{collection.replace('-', '_')}_rerank_{metrics_tag}_k{args.k}.csv"
        results_path = TESTSET_DIR / results_filename
        TESTSET_DIR.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["# parameter", "value"])
            writer.writerow(["# testset", testset])
            writer.writerow(["# collection", f"{collection} ({collection_name})"])
            writer.writerow(["# k", args.k])
            writer.writerow(["# retrieve_n", RERANK_RETRIEVE_N])
            writer.writerow(["# metrics", ", ".join(eval_metrics)])
            writer.writerow(["# questions", len(test_cases)])
            writer.writerow([])
            writer.writerow(["question", "expected", "before_hit", "after_hit", "before_retrieved", "after_retrieved"])
            for bqr, aqr in zip(before_results, after_results):
                b_hit = bqr.expected_id in bqr.retrieved_ids
                a_hit = aqr.expected_id in aqr.retrieved_ids
                writer.writerow([
                    bqr.question,
                    bqr.expected_id,
                    "HIT" if b_hit else "MISS",
                    "HIT" if a_hit else "MISS",
                    "; ".join(bqr.retrieved_ids),
                    "; ".join(aqr.retrieved_ids),
                ])
            writer.writerow([])
            for metric in eval_metrics:
                if metric in before_scores and metric in after_scores:
                    writer.writerow([
                        f"{metric}@{args.k} (before rerank)", f"{before_scores[metric]:.4f}",
                    ])
                    writer.writerow([
                        f"{metric}@{args.k} (after rerank)", f"{after_scores[metric]:.4f}",
                    ])
        log.info("Results written to %s", results_path)

        print(f"\n{'=' * 60}")
        print(f"Retrieval Evaluation Results (Rerank Comparison)")
        print(f"  Test set:    {testset} ({len(test_cases)} questions)")
        print(f"  Collection:  {collection} ({collection_name})")
        print(f"  K:           {args.k}")
        print(f"  Retrieve N:  {RERANK_RETRIEVE_N}")
        print(f"  Reranker:    {RERANKER_MODEL}")
        print(f"{'=' * 60}")
        for metric in eval_metrics:
            if metric in before_scores and metric in after_scores:
                b = before_scores[metric]
                a = after_scores[metric]
                delta = a - b
                sign = "+" if delta >= 0 else ""
                print(f"  {metric}@{args.k} before rerank: {b:.4f} ({b * 100:.1f}%)")
                print(f"  {metric}@{args.k} after  rerank: {a:.4f} ({a * 100:.1f}%)  [{sign}{delta:.4f}]")
        print(f"  Results:     {results_path}")
        print(f"{'=' * 60}\n")

    else:
        # Standard mode (no reranking).
        retriever = Retriever(
            db_path=DB_PATH,
            collection_name=collection_name,
            embedding_adapter=embedding_adapter,
            n_results=args.k,
        )
        evaluator = RetrievalEvaluator(
            retriever=retriever,
            collection_name=collection_name,
            k=args.k,
        )

        log.info(
            "Evaluating %d %s questions against %s (k=%d)",
            len(test_cases),
            testset,
            collection,
            args.k,
        )

        scores, query_results = evaluator.run(test_cases=test_cases, metrics=args.metrics)

        # Write per-query results to CSV.
        metrics_tag = "_".join(sorted(args.metrics))
        results_filename = f"eval_{testset}_{collection.replace('-', '_')}_{metrics_tag}_k{args.k}.csv"
        results_path = TESTSET_DIR / results_filename
        TESTSET_DIR.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Parameters section.
            writer.writerow(["# parameter", "value"])
            writer.writerow(["# testset", testset])
            writer.writerow(["# collection", f"{collection} ({collection_name})"])
            writer.writerow(["# k", args.k])
            writer.writerow(["# metrics", ", ".join(args.metrics)])
            writer.writerow(["# questions", len(test_cases)])
            writer.writerow([])
            # Per-query results.
            writer.writerow(["question", "expected", "hit", "retrieved"])
            for qr in query_results:
                hit = qr.expected_id in qr.retrieved_ids
                writer.writerow([
                    qr.question,
                    qr.expected_id,
                    "HIT" if hit else "MISS",
                    "; ".join(qr.retrieved_ids),
                ])
            # Summary scores.
            writer.writerow([])
            for metric, score in scores.items():
                writer.writerow([f"{metric}@{args.k}", f"{score:.4f}"])
        log.info("Results written to %s", results_path)

        print(f"\n{'=' * 50}")
        print(f"Retrieval Evaluation Results")
        print(f"  Test set:   {testset} ({len(test_cases)} questions)")
        print(f"  Collection: {collection} ({collection_name})")
        print(f"  K:          {args.k}")
        print(f"{'=' * 50}")
        for metric, score in scores.items():
            print(f"  {metric}@{args.k}: {score:.4f} ({score * 100:.1f}%)")
        print(f"  Results:    {results_path}")
        print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
