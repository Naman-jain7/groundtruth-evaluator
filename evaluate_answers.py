import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from config import RAW_ANSWERS_PATH, EVALUATED_RESULTS_PATH
from evaluator import DeepEvalEvaluator

load_dotenv()

# ============================================================================
# Phase 2: Evaluate Answers
# ============================================================================

def evaluate_answers(
    eval_model_name: str,
    raw_answers_path: Path = RAW_ANSWERS_PATH,
):
    """
    Phase 2: Load raw_answers.csv and run DeepEval metrics on each answer.
    Saves results to evaluated_results.csv.

    Args:
        eval_model_name: The OpenRouter model to use for evaluation
        raw_answers_path: Path to the raw answers CSV (default: outputs/raw_answers.csv)

    Returns:
        Tuple of (results list, evaluated_results_path)
    """
    print("\n" + "=" * 80)
    print("TruthfulQA Pipeline - Phase 2: Evaluating Answers")
    print("=" * 80 + "\n")

    if not raw_answers_path.exists():
        print(f"❌ Raw answers file not found: {raw_answers_path}")
        print("   Run Phase 1 first: python main.py --stage answers")
        return None, None

    # Load raw answers
    with open(raw_answers_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {raw_answers_path}\n")

    print(f"Initializing evaluator with model: {eval_model_name}...")
    evaluator = DeepEvalEvaluator(eval_model_name)
    print("✅ DeepEval metrics initialized\n")

    results = []

    for row in tqdm(rows, desc="Evaluating", unit="q"):
        question = row["question"]
        answer = row["answer"]
        correct_answer = row["correct_answer"]
        model = row["model"]

        print(f"\n  [{model}] Q: {question[:60]}...")

        # Skip rows where generation failed (no answer)
        if not answer:
            print("  ⚠️  No answer to evaluate, skipping.")
            results.append({
                **row,
                "hallucination_score": None,
                "hallucination_metric": None,
                "answer_relevancy_score": None,
                "answer_relevancy_metric": None,
                "faithfulness_score": None,
                "faithfulness_metric": None,
                "eval_error": "No answer to evaluate",
            })
            continue

        try:
            eval_results = evaluator.evaluate(question, answer, correct_answer)
            results.append({
                **row,
                "hallucination_score": eval_results.get("hallucination_score"),
                "hallucination_metric": eval_results.get("hallucination_pass"),
                "answer_relevancy_score": eval_results.get("relevancy_score"),
                "answer_relevancy_metric": eval_results.get("relevancy_pass"),
                "faithfulness_score": eval_results.get("faithfulness_score"),
                "faithfulness_metric": eval_results.get("faithfulness_pass"),
                "eval_error": None,
            })

        except Exception as e:
            print(f"  ❌ Evaluation error: {str(e)[:100]}")
            results.append({
                **row,
                "hallucination_score": None,
                "hallucination_metric": None,
                "answer_relevancy_score": None,
                "answer_relevancy_metric": None,
                "faithfulness_score": None,
                "faithfulness_metric": None,
                "eval_error": str(e)[:200],
            })

    # Save to evaluated_results.csv (overwrite)
    fieldnames = [
        "question_id", "question", "category", "model","answer", "correct_answer","hallucination_score", "hallucination_metric","answer_relevancy_score", "answer_relevancy_metric","faithfulness_score", "faithfulness_metric","error", "eval_error", "timestamp",
    ]

    with open(EVALUATED_RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n✅ Saved {len(results)} evaluated records to {EVALUATED_RESULTS_PATH}\n")

    return results, EVALUATED_RESULTS_PATH


if __name__ == "__main__":
    eval_model = os.getenv("EVAL_MODEL", "openai/gpt-oss-120b:free")
    results, path = evaluate_answers(eval_model_name=eval_model)
    print(f"Evaluated results saved to: {path}")
