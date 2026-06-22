import csv
import os
import time
from pathlib import Path
from typing import Optional
from schemas.eval_result import EvaluationResult
from ollama_client import OllamaCloudClient

from datasets import load_dataset

# DeepEval imports
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "https://api.ollama.ai/v1")

MODELS = [
    os.getenv("MODEL_A", "mistral"),
    os.getenv("MODEL_B", "neural-chat"),
]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Batch settings
BATCH_SIZE = 10
TIMEOUT = 60  # seconds per request

# ============================================================================
# Metrics Evaluation
# ============================================================================


class DeepEvalEvaluator:
    """Wrapper for DeepEval metrics with error handling."""

    def __init__(self):
        self.hallucination_metric = HallucinationMetric(threshold=0.5)
        self.relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
        self.faithfulness_metric = FaithfulnessMetric(threshold=0.5)

    def evaluate(self,question: str,answer: str,context: str) -> dict:
        """
        Evaluate answer using all three metrics.

        Args:
            question: Input question
            answer: Model-generated answer
            context: Ground truth answer (serves as context)

        Returns:
            Dictionary with metric scores and pass/fail status
        """
        try:
            # Create test case for evaluation
            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
                context=[context],  # Ground truth as context
            )

            # Evaluate with each metric
            results = {}

            # Hallucination metric
            try:
                self.hallucination_metric.measure(test_case)
                results["hallucination_score"] = self.hallucination_metric.score
                results["hallucination_pass"] = (
                    self.hallucination_metric.is_successful()
                )
            except Exception as e:
                print(f"    ⚠️  Hallucination metric error: {str(e)[:80]}")
                results["hallucination_score"] = None
                results["hallucination_pass"] = None

            # Answer relevancy metric
            try:
                self.relevancy_metric.measure(test_case)
                results["relevancy_score"] = self.relevancy_metric.score
                results["relevancy_pass"] = self.relevancy_metric.is_successful()
            except Exception as e:
                print(f"    ⚠️  Relevancy metric error: {str(e)[:80]}")
                results["relevancy_score"] = None
                results["relevancy_pass"] = None

            # Faithfulness metric
            try:
                self.faithfulness_metric.measure(test_case)
                results["faithfulness_score"] = self.faithfulness_metric.score
                results["faithfulness_pass"] = self.faithfulness_metric.is_successful()
            except Exception as e:
                print(f"    ⚠️  Faithfulness metric error: {str(e)[:80]}")
                results["faithfulness_score"] = None
                results["faithfulness_pass"] = None

            return results

        except Exception as e:
            print(f"    ❌ Evaluation failed: {str(e)[:80]}")
            return {
                "hallucination_score": None,
                "hallucination_pass": None,
                "relevancy_score": None,
                "relevancy_pass": None,
                "faithfulness_score": None,
                "faithfulness_pass": None,
            }


# ============================================================================
# Data Loading
# ============================================================================


def load_truthful_qa(split: str = "generation", limit: Optional[int] = None):
    """
    Load TruthfulQA dataset from Hugging Face.

    Args:
        split: Dataset split ('generation' or 'validation')
        limit: Maximum number of samples to load

    Returns:
        List of dictionaries with 'question', 'category', 'correct_answers'
    """
    print(f"Loading TruthfulQA ({split} split)...")
    try:
        dataset = load_dataset(
            "data/truthful_qa",
            split,
            trust_remote_code=True,
        )

        samples = []
        for idx, item in enumerate(dataset):
            if limit and idx >= limit:
                break

            samples.append(
                {
                    "question_id": f"{split}_{idx:04d}",
                    "question": item["question"], # type: ignore
                    "category": item.get("category", "unknown"), # type: ignore
                    "correct_answers": item.get("correct_answers", []), # type: ignore
                }
            )

        print(f"✅ Loaded {len(samples)} samples")
        return samples

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise


# ============================================================================
# Main Pipeline
# ============================================================================


def generate_outputs(models: list = MODELS,dataset_limit: Optional[int] = None,sample_questions: int = 5):
    """
    Main pipeline: Load data → Generate outputs → Evaluate → Save CSV.

    Args:
        models: List of model names to evaluate
        dataset_limit: Limit dataset size for testing
        sample_questions: If set, only evaluate this many questions
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
            ground_truth = (
                correct_answers[0] if correct_answers else "No ground truth available"
            )

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
    # Run with sample for testing (change sample_questions=None for full run)
    results, csv_path = generate_outputs(sample_questions=5) # type: ignore
    print(f"\n{'=' * 80}")
    print("Pipeline Complete!")
    print(f"Results saved to: {csv_path}")
    print(f"{'=' * 80}\n")
