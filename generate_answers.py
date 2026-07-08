import csv
import time
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

from config import AVAILABLE_MODELS, RAW_ANSWERS_PATH
from load_data import load_truthful_qa
from ollama_client import OllamaLocalClient

load_dotenv()


# ============================================================================
# Phase 1: Generate Answers
# ============================================================================

def generate_answers(
    models: list = AVAILABLE_MODELS,
    dataset_limit: Optional[int] = None,
    sample_questions: int = 0,
):
    """
    Phase 1: Generate raw answers using local Ollama models and save to raw_answers.csv.

    Args:
        models: List of local model names to generate answers with
        dataset_limit: Limit dataset size (for testing)
        sample_questions: If > 0, only use this many questions

    Returns:
        Tuple of (results list, csv_path)
    """
    models = [model for model in models if model]

    print("\n" + "=" * 80)
    print("TruthfulQA Pipeline - Phase 1: Generating Answers")
    print("=" * 80 + "\n")

    print("Initializing local Ollama client...")
    try:
        client = OllamaLocalClient()
        print("✅ Local Ollama client initialized\n")
    except ValueError as e:
        print(f"❌ {e}")
        return None, None

    data = load_truthful_qa(split="generation", limit=dataset_limit)

    if sample_questions:
        data = data[:sample_questions]
        print(f"Using sample of {len(data)} questions\n")

    categories = {}
    for item in data:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)

    print(f"Categories found: {list(categories.keys())}")
    print(f"Distribution: {[(k, len(v)) for k, v in categories.items()]}\n")

    results = []

    for model in models:
        print(f"\n{'=' * 80}")
        print(f"Generating answers for model: {model}")
        print(f"{'=' * 80}\n")

        model_start = time.time()

        for item in tqdm(data, desc=f"{model}", unit="q"):
            question = item["question"]
            question_id = item["question_id"]
            category = item["category"]
            correct_answers = item["correct_answers"]

            ground_truth = correct_answers[0] if correct_answers else "No ground truth available"

            try:
                print(f"\n  Q: {question[:60]}...")
                answer = client.generate(model, question)
                if not answer:
                    print(f"  ⚠️ Empty answer from {model}")
                print(f"  A: {answer[:60]}...")

                results.append({
                    "question_id": question_id,
                    "question": question,
                    "category": category,
                    "model": model,
                    "answer": answer,
                    "correct_answer": ground_truth,
                    "error": "",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })

            except Exception as e:
                print(f"  ❌ Error processing question: {str(e)[:100]}")
                results.append({
                    "question_id": question_id,
                    "question": question,
                    "category": category,
                    "model": model,
                    "answer": "",
                    "correct_answer": ground_truth,
                    "error": str(e)[:200],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })

        elapsed = time.time() - model_start
        print(f"\n✅ Model {model} completed in {elapsed:.1f}s")

    # Save to raw_answers.csv (overwrite)
    fieldnames = [
        "question_id", "question", "category", "model",
        "answer", "correct_answer", "error", "timestamp",
    ]

    with open(RAW_ANSWERS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n✅ Saved {len(results)} raw answers to {RAW_ANSWERS_PATH}\n")

    return results, RAW_ANSWERS_PATH


if __name__ == "__main__":
    results, path = generate_answers(sample_questions=5)
    print(f"Raw answers saved to: {path}")
