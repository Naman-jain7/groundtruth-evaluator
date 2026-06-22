import csv
import time
from typing import Optional

# DeepEval imports
from dotenv import load_dotenv
from tqdm import tqdm

from config import MODELS, OLLAMA_API_KEY, OUTPUT_DIR
from evaluator import DeepEvalEvaluator
from load_data import load_truthful_qa
from ollama_client import OllamaCloudClient
from schemas.eval_result import EvaluationResult

load_dotenv()

# ============================================================================
# Main Pipeline
# ============================================================================

def generate_outputs(models: list = MODELS, dataset_limit: Optional[int] = None, sample_questions: int = 5):
    """
    Main pipeline: Load data → Generate outputs → Evaluate → Save CSV.
    """

    print("\n" + "=" * 80)
    print("TruthfulQA Evaluation Pipeline - Generate Outputs")
    print("=" * 80 + "\n")

    # Initialize clients
    print("Initializing Ollama Cloud client...")
    try:
        client = OllamaCloudClient(OLLAMA_API_KEY) # type: ignore
        print("✅ Ollama Cloud client initialized\n")
    except ValueError as e:
        print(f"❌ {e}")
        return

    evaluator = DeepEvalEvaluator()
    print("✅ DeepEval metrics initialized\n")

    # Load dataset
    data = load_truthful_qa(split="generation", limit=dataset_limit)

    if sample_questions:
        data = data[:sample_questions]
        print(f"Using sample of {len(data)} questions for testing\n")

    # Category grouping
    categories = {}
    for item in data:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)

    print(f"Categories found: {list(categories.keys())}")
    print(f"Distribution: {[(k, len(v)) for k, v in categories.items()]}\n")

    # Generate outputs and evaluate
    results = []

    for model in models:
        print(f"\n{'=' * 80}")
        print(f"Evaluating Model: {model}")
        print(f"{'=' * 80}\n")

        model_start = time.time()

        for item in tqdm(data, desc=f"{model}", unit="q"):
            question = item["question"]
            question_id = item["question_id"]
            category = item["category"]
            correct_answers = item["correct_answers"]

            # Use first correct answer as ground truth
            ground_truth = (correct_answers[0] if correct_answers else "No ground truth available")

            try:
                # Generate response
                print(f"\n  Q: {question[:60]}...")
                answer = client.generate(model, question)
                print(f"  A: {answer[:60]}...")

                # Evaluate
                eval_results = evaluator.evaluate(question, answer, ground_truth)

                # Store result
                result = EvaluationResult(
                    question_id=question_id,
                    question=question,
                    category=category,
                    model=model,
                    answer=answer,
                    correct_answer=ground_truth,
                    hallucination_score=eval_results.get("hallucination_score"),
                    hallucination_metric=eval_results.get("hallucination_pass"),
                    answer_relevancy_score=eval_results.get("relevancy_score"),
                    answer_relevancy_metric=eval_results.get("relevancy_pass"),
                    faithfulness_score=eval_results.get("faithfulness_score"),
                    faithfulness_metric=eval_results.get("faithfulness_pass"),
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                results.append(result)

            except Exception as e:
                print(f"  ❌ Error processing question: {str(e)[:100]}")
                result = EvaluationResult(
                    question_id=question_id,
                    question=question,
                    category=category,
                    model=model,
                    answer="",
                    correct_answer=ground_truth,
                    error=str(e)[:200],
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                results.append(result)

        elapsed = time.time() - model_start
        print(f"\n✅ Model {model} completed in {elapsed:.1f}s")

    # Save results to CSV
    csv_path = OUTPUT_DIR / "evaluation_results.csv"
    print(f"\n{'=' * 80}")
    print(f"Saving results to {csv_path}")
    print(f"{'=' * 80}\n")

    fieldnames = [
        "question_id",
        "question",
        "category",
        "model",
        "answer",
        "correct_answer",
        "hallucination_score",
        "hallucination_metric",
        "answer_relevancy_score",
        "answer_relevancy_metric",
        "faithfulness_score",
        "faithfulness_metric",
        "error",
        "timestamp",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "question_id": result.question_id,
                    "question": result.question,
                    "category": result.category,
                    "model": result.model,
                    "answer": result.answer,
                    "correct_answer": result.correct_answer,
                    "hallucination_score": result.hallucination_score,
                    "hallucination_metric": result.hallucination_metric,
                    "answer_relevancy_score": result.answer_relevancy_score,
                    "answer_relevancy_metric": result.answer_relevancy_metric,
                    "faithfulness_score": result.faithfulness_score,
                    "faithfulness_metric": result.faithfulness_metric,
                    "error": result.error,
                    "timestamp": result.timestamp,
                }
            )

    print(f"✅ Saved {len(results)} evaluation records")
    print(f"📊 CSV file: {csv_path}\n")

    return results, csv_path


if __name__ == "__main__":
    results, csv_path = generate_outputs(sample_questions=5) # type: ignore
    print(f"\n{'=' * 80}")
    print("Pipeline Complete!")
    print(f"Results saved to: {csv_path}")
    print(f"{'=' * 80}\n")
